#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTOSVE2I8MM
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct SVE2I8MMDotOpCandidate {
  cpu::DotOp op;
};

static Value cstI64(Location loc, int64_t v, PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, v, 64);
}

static Value extractElem2D(Location loc, Value vec, int64_t r, int64_t c,
                           PatternRewriter &rewriter) {
  Value row = rewriter.create<vector::ExtractOp>(loc, vec, r);
  return rewriter.create<vector::ExtractElementOp>(loc, row,
                                                   cstI64(loc, c, rewriter));
}

static Value packVec8ToNxv16(Location loc, Value vec8, Type nxv16i8Ty,
                             PatternRewriter &rewriter) {
  // Widen vec<8xi8> to vec<16xi8> (zero-pad high bytes), then use
  // experimental.vector.insert (generates 0 extra instructions on aarch64
  // via the q0/z0 register alias).
  auto i8Ty = rewriter.getI8Type();
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto zeroAttr = rewriter.getZeroAttr(v16i8Ty);
  Value zero16 = rewriter.create<arith::ConstantOp>(loc, v16i8Ty, zeroAttr);
  // Widen by inserting vec8 at offset 0 in a zero 16-byte vector
  Value wide = rewriter.create<vector::InsertStridedSliceOp>(
      loc, vec8, zero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
  // Now use the zero-instruction NEON→SVE path
  Value undef = rewriter.create<LLVM::UndefOp>(loc, nxv16i8Ty);
  auto i64Ty = rewriter.getI64Type();
  Value zeroIdx =
      rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(0));
  return rewriter.create<LLVM::CallIntrinsicOp>(
             loc, nxv16i8Ty,
             StringAttr::get(rewriter.getContext(),
                             "llvm.vector.insert.nxv16i8.v16i8"),
             ValueRange{undef, wide, zeroIdx})
      .getResult(0);
}

static Value packVec16ToNxv16(Location loc, Value vec16, Type nxv16i8Ty,
                              PatternRewriter &rewriter) {
  // Use experimental.vector.insert: LLVM generates zero instructions on
  // aarch64 since q0 and z0 are the same physical register (register alias).
  Value undef = rewriter.create<LLVM::UndefOp>(loc, nxv16i8Ty);
  auto i64Ty = rewriter.getI64Type();
  Value zeroIdx =
      rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(0));
  return rewriter.create<LLVM::CallIntrinsicOp>(
                     loc, nxv16i8Ty,
                     StringAttr::get(rewriter.getContext(),
                                     "llvm.vector.insert.nxv16i8.v16i8"),
                     ValueRange{undef, vec16, zeroIdx})
      .getResult(0);
}

static Value zip1I64(Location loc, Value a, Value b, Type nxv16i8Ty,
                     PatternRewriter &rewriter) {
  Type nxv2i64Ty =
      LLVM::LLVMScalableVectorType::get(rewriter.getI64Type(), 2);
  Value a64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, a);
  Value b64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, b);
  StringAttr zip1 =
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.sve.zip1.nxv2i64");
  Value z = rewriter
                .create<LLVM::CallIntrinsicOp>(loc, nxv2i64Ty, zip1,
                                               ValueRange{a64, b64})
                .getResult(0);
  return rewriter.create<LLVM::BitcastOp>(loc, nxv16i8Ty, z);
}

static Value zip2I64(Location loc, Value a, Value b, Type nxv16i8Ty,
                     PatternRewriter &rewriter) {
  Type nxv2i64Ty =
      LLVM::LLVMScalableVectorType::get(rewriter.getI64Type(), 2);
  Value a64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, a);
  Value b64 = rewriter.create<LLVM::BitcastOp>(loc, nxv2i64Ty, b);
  StringAttr zip2 =
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.sve.zip2.nxv2i64");
  Value z = rewriter
                .create<LLVM::CallIntrinsicOp>(loc, nxv2i64Ty, zip2,
                                               ValueRange{a64, b64})
                .getResult(0);
  return rewriter.create<LLVM::BitcastOp>(loc, nxv16i8Ty, z);
}

static Value pack2x8i8ToNxv16(Location loc, Value tile, Type nxv16i8Ty,
                              PatternRewriter &rewriter) {
  // Pack rows as [row0(8), row1(8)] which matches zip1 on 64-bit lanes.
  Value row0 = rewriter.create<vector::ExtractOp>(loc, tile, 0);
  Value row1 = rewriter.create<vector::ExtractOp>(loc, tile, 1);
  Value v0 = packVec8ToNxv16(loc, row0, nxv16i8Ty, rewriter);
  Value v1 = packVec8ToNxv16(loc, row1, nxv16i8Ty, rewriter);
  return zip1I64(loc, v0, v1, nxv16i8Ty, rewriter);
}

static Value pack2x2i32ToNxv4(Location loc, Value tile, Type nxv4i32Ty,
                              PatternRewriter &rewriter) {
  // Flatten 2x2 tile to vec<4xi32>, then use zero-instruction q/z alias path.
  auto i32Ty = cast<VectorType>(tile.getType()).getElementType();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Value flat = rewriter.create<vector::ShapeCastOp>(loc, v4i32Ty, tile);
  Value undef = rewriter.create<LLVM::UndefOp>(loc, nxv4i32Ty);
  auto i64Ty = rewriter.getI64Type();
  Value zeroIdx =
      rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(0));
  return rewriter.create<LLVM::CallIntrinsicOp>(
                     loc, nxv4i32Ty,
                     StringAttr::get(rewriter.getContext(),
                                     "llvm.vector.insert.nxv4i32.v4i32"),
                     ValueRange{undef, flat, zeroIdx})
      .getResult(0);
}

static Value unpackNxv4To2x2i32(Location loc, Value vec, Type tileTy,
                                PatternRewriter &rewriter) {
  // Use zero-instruction z/q alias path to extract vec<4xi32> from scalable.
  auto i32Ty = cast<VectorType>(tileTy).getElementType();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto i64Ty = rewriter.getI64Type();
  Value zeroIdx =
      rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(0));
  Value flat =
      rewriter.create<LLVM::CallIntrinsicOp>(
          loc, v4i32Ty,
          StringAttr::get(rewriter.getContext(),
                          "llvm.vector.extract.v4i32.nxv4i32"),
          ValueRange{vec, zeroIdx})
          .getResult(0);
  return rewriter.create<vector::ShapeCastOp>(loc, tileTy, flat);
}

static Value build4x4FromAccVecs(Location loc, Value acc00, Value acc01,
                                 Value acc10, Value acc11, Type tileTy,
                                 PatternRewriter &rewriter) {
  auto elemTy = cast<VectorType>(tileTy).getElementType();
  auto tile2x2Ty = VectorType::get({2, 2}, elemTy);
  Value tile00 = unpackNxv4To2x2i32(loc, acc00, tile2x2Ty, rewriter);
  Value tile01 = unpackNxv4To2x2i32(loc, acc01, tile2x2Ty, rewriter);
  Value tile10 = unpackNxv4To2x2i32(loc, acc10, tile2x2Ty, rewriter);
  Value tile11 = unpackNxv4To2x2i32(loc, acc11, tile2x2Ty, rewriter);

  auto zeroAttr = rewriter.getZeroAttr(tileTy);
  Value tile = rewriter.create<arith::ConstantOp>(loc, tileTy, zeroAttr);
  tile = rewriter.create<vector::InsertStridedSliceOp>(
      loc, tile00, tile, ArrayRef<int64_t>{0, 0}, ArrayRef<int64_t>{1, 1});
  tile = rewriter.create<vector::InsertStridedSliceOp>(
      loc, tile01, tile, ArrayRef<int64_t>{0, 2}, ArrayRef<int64_t>{1, 1});
  tile = rewriter.create<vector::InsertStridedSliceOp>(
      loc, tile10, tile, ArrayRef<int64_t>{2, 0}, ArrayRef<int64_t>{1, 1});
  tile = rewriter.create<vector::InsertStridedSliceOp>(
      loc, tile11, tile, ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
  return tile;
}

bool isSVE2I8MMCandidate(cpu::DotOp op, SVE2I8MMDotOpCandidate &candidate) {
  auto aTy = dyn_cast<VectorType>(op.getA().getType());
  auto bTy = dyn_cast<VectorType>(op.getB().getType());
  auto cTy = dyn_cast<VectorType>(op.getC().getType());
  if (!aTy || !bTy || !cTy) {
    LDBG("Drop candidate with non-vector types.");
    return false;
  }

  if (aTy.getRank() != bTy.getRank() || aTy.getRank() != cTy.getRank() ||
      (aTy.getRank() != 2 && aTy.getRank() != 3)) {
    LDBG("Drop candidate with unsupported rank.");
    return false;
  }

  auto aElemTy = aTy.getElementType();
  auto bElemTy = bTy.getElementType();
  auto cElemTy = cTy.getElementType();

  if (!aElemTy.isInteger(8) || !bElemTy.isInteger(8)) {
    LDBG("Drop candidate with non-i8 inputs.");
    return false;
  }

  if (!cElemTy.isInteger(32)) {
    LDBG("Drop candidate with non-i32 accumulator.");
    return false;
  }

  candidate.op = op;
  return true;
}

// ─── M=2 specialized path (GEMV-like, e.g. BM=2 for decode) ──────────────────
//
// Optimization: A does not depend on n, so pack A tiles once before the N
// loop (hoisted A prepack), reusing them across all N steps.
// Uses 2 accumulators per N step instead of 4, avoiding unused computation.
static LogicalResult convertCandidateM2(cpu::DotOp op, Location loc,
                                        int64_t K, int64_t N,
                                        Type i32Ty, Type nxv16i8Ty,
                                        Type nxv4i32Ty, StringAttr smmla,
                                        PatternRewriter &rewriter) {
  Value A = op.getA();
  Value B = op.getB();
  Value res = op.getC();
  auto tile2x2Ty = VectorType::get({2, 2}, i32Ty);

  if (K % 16 == 0) {
    // ── Hoist A prepack: pack A[0:2, k:k+16] for each k step once ──────────
    int64_t numKSteps = K / 16;
    SmallVector<Value> aLo0s(numKSteps), aHi0s(numKSteps);
    for (int64_t ki = 0, k = 0; k < K; k += 16, ++ki) {
      Value aTile = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, A, ArrayRef<int64_t>{0, k}, ArrayRef<int64_t>{2, 16},
          ArrayRef<int64_t>{1, 1});
      Value row0 = rewriter.create<vector::ExtractOp>(loc, aTile, 0);
      Value row1 = rewriter.create<vector::ExtractOp>(loc, aTile, 1);
      Value v0 = packVec16ToNxv16(loc, row0, nxv16i8Ty, rewriter);
      Value v1 = packVec16ToNxv16(loc, row1, nxv16i8Ty, rewriter);
      aLo0s[ki] = zip1I64(loc, v0, v1, nxv16i8Ty, rewriter);
      aHi0s[ki] = zip2I64(loc, v0, v1, nxv16i8Ty, rewriter);
    }

    for (int64_t n = 0; n < N; n += 4) {
      // Prepack B for this n tile (depends on n, stays inside N loop)
      SmallVector<Value> bLo0s(numKSteps), bHi0s(numKSteps);
      SmallVector<Value> bLo1s(numKSteps), bHi1s(numKSteps);
      for (int64_t ki = 0, k = 0; k < K; k += 16, ++ki) {
        Value bTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, B, ArrayRef<int64_t>{k, n + 0}, ArrayRef<int64_t>{16, 2},
            ArrayRef<int64_t>{1, 1});
        Value bTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, B, ArrayRef<int64_t>{k, n + 2}, ArrayRef<int64_t>{16, 2},
            ArrayRef<int64_t>{1, 1});
        Value bTile0T = rewriter.create<vector::TransposeOp>(
            loc, bTile0, ArrayRef<int64_t>{1, 0});
        Value bTile1T = rewriter.create<vector::TransposeOp>(
            loc, bTile1, ArrayRef<int64_t>{1, 0});
        Value bRow00 = rewriter.create<vector::ExtractOp>(loc, bTile0T, 0);
        Value bRow01 = rewriter.create<vector::ExtractOp>(loc, bTile0T, 1);
        Value bRow10 = rewriter.create<vector::ExtractOp>(loc, bTile1T, 0);
        Value bRow11 = rewriter.create<vector::ExtractOp>(loc, bTile1T, 1);
        Value bVec00 = packVec16ToNxv16(loc, bRow00, nxv16i8Ty, rewriter);
        Value bVec01 = packVec16ToNxv16(loc, bRow01, nxv16i8Ty, rewriter);
        Value bVec10 = packVec16ToNxv16(loc, bRow10, nxv16i8Ty, rewriter);
        Value bVec11 = packVec16ToNxv16(loc, bRow11, nxv16i8Ty, rewriter);
        bLo0s[ki] = zip1I64(loc, bVec00, bVec01, nxv16i8Ty, rewriter);
        bHi0s[ki] = zip2I64(loc, bVec00, bVec01, nxv16i8Ty, rewriter);
        bLo1s[ki] = zip1I64(loc, bVec10, bVec11, nxv16i8Ty, rewriter);
        bHi1s[ki] = zip2I64(loc, bVec10, bVec11, nxv16i8Ty, rewriter);
      }

      // Load accumulators for rows 0-1, cols n:n+4
      Value accTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, res, ArrayRef<int64_t>{0, n + 0}, ArrayRef<int64_t>{2, 2},
          ArrayRef<int64_t>{1, 1});
      Value accTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, res, ArrayRef<int64_t>{0, n + 2}, ArrayRef<int64_t>{2, 2},
          ArrayRef<int64_t>{1, 1});
      Value accVec0 = pack2x2i32ToNxv4(loc, accTile0, nxv4i32Ty, rewriter);
      Value accVec1 = pack2x2i32ToNxv4(loc, accTile1, nxv4i32Ty, rewriter);

      // Accumulate using hoisted A packs
      for (int64_t ki = 0; ki < numKSteps; ++ki) {
        Value aLo0 = aLo0s[ki];
        Value aHi0 = aHi0s[ki];
        accVec0 = rewriter
                      .create<LLVM::CallIntrinsicOp>(loc, nxv4i32Ty, smmla,
                                                     ValueRange{accVec0, aLo0, bLo0s[ki]})
                      .getResult(0);
        accVec0 = rewriter
                      .create<LLVM::CallIntrinsicOp>(loc, nxv4i32Ty, smmla,
                                                     ValueRange{accVec0, aHi0, bHi0s[ki]})
                      .getResult(0);
        accVec1 = rewriter
                      .create<LLVM::CallIntrinsicOp>(loc, nxv4i32Ty, smmla,
                                                     ValueRange{accVec1, aLo0, bLo1s[ki]})
                      .getResult(0);
        accVec1 = rewriter
                      .create<LLVM::CallIntrinsicOp>(loc, nxv4i32Ty, smmla,
                                                     ValueRange{accVec1, aHi0, bHi1s[ki]})
                      .getResult(0);
      }

      // Store results back
      Value tile0 = unpackNxv4To2x2i32(loc, accVec0, tile2x2Ty, rewriter);
      Value tile1 = unpackNxv4To2x2i32(loc, accVec1, tile2x2Ty, rewriter);
      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, tile0, res, ArrayRef<int64_t>{0, n + 0}, ArrayRef<int64_t>{1, 1});
      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, tile1, res, ArrayRef<int64_t>{0, n + 2}, ArrayRef<int64_t>{1, 1});
    }
  } else {
    // K%8 path
    int64_t numKSteps = K / 8;
    SmallVector<Value> aVec0s(numKSteps);
    for (int64_t ki = 0, k = 0; k < K; k += 8, ++ki) {
      Value aTile = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, A, ArrayRef<int64_t>{0, k}, ArrayRef<int64_t>{2, 8},
          ArrayRef<int64_t>{1, 1});
      aVec0s[ki] = pack2x8i8ToNxv16(loc, aTile, nxv16i8Ty, rewriter);
    }

    for (int64_t n = 0; n < N; n += 4) {
      SmallVector<Value> bVec0s(numKSteps), bVec1s(numKSteps);
      for (int64_t ki = 0, k = 0; k < K; k += 8, ++ki) {
        Value bTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, B, ArrayRef<int64_t>{k, n + 0}, ArrayRef<int64_t>{8, 2},
            ArrayRef<int64_t>{1, 1});
        Value bTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, B, ArrayRef<int64_t>{k, n + 2}, ArrayRef<int64_t>{8, 2},
            ArrayRef<int64_t>{1, 1});
        Value bTile0T = rewriter.create<vector::TransposeOp>(
            loc, bTile0, ArrayRef<int64_t>{1, 0});
        Value bTile1T = rewriter.create<vector::TransposeOp>(
            loc, bTile1, ArrayRef<int64_t>{1, 0});
        bVec0s[ki] = pack2x8i8ToNxv16(loc, bTile0T, nxv16i8Ty, rewriter);
        bVec1s[ki] = pack2x8i8ToNxv16(loc, bTile1T, nxv16i8Ty, rewriter);
      }

      Value accTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, res, ArrayRef<int64_t>{0, n + 0}, ArrayRef<int64_t>{2, 2},
          ArrayRef<int64_t>{1, 1});
      Value accTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, res, ArrayRef<int64_t>{0, n + 2}, ArrayRef<int64_t>{2, 2},
          ArrayRef<int64_t>{1, 1});
      Value accVec0 = pack2x2i32ToNxv4(loc, accTile0, nxv4i32Ty, rewriter);
      Value accVec1 = pack2x2i32ToNxv4(loc, accTile1, nxv4i32Ty, rewriter);

      for (int64_t ki = 0; ki < numKSteps; ++ki) {
        accVec0 = rewriter
                      .create<LLVM::CallIntrinsicOp>(loc, nxv4i32Ty, smmla,
                                                     ValueRange{accVec0, aVec0s[ki], bVec0s[ki]})
                      .getResult(0);
        accVec1 = rewriter
                      .create<LLVM::CallIntrinsicOp>(loc, nxv4i32Ty, smmla,
                                                     ValueRange{accVec1, aVec0s[ki], bVec1s[ki]})
                      .getResult(0);
      }

      Value tile0 = unpackNxv4To2x2i32(loc, accVec0, tile2x2Ty, rewriter);
      Value tile1 = unpackNxv4To2x2i32(loc, accVec1, tile2x2Ty, rewriter);
      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, tile0, res, ArrayRef<int64_t>{0, n + 0}, ArrayRef<int64_t>{1, 1});
      res = rewriter.create<vector::InsertStridedSliceOp>(
          loc, tile1, res, ArrayRef<int64_t>{0, n + 2}, ArrayRef<int64_t>{1, 1});
    }
  }

  rewriter.replaceOp(op, res);
  return success();
}

// ─── Register-blocked M%4 path ────────────────────────────────────────────────
//
// ACL's a64_interleaved_s8s32_mmla_8x12 keeps 24 accumulator registers live
// during the entire K traversal.  The key is processing multiple (m, n) tiles
// together so that B vectors remain live across all m-tile computations.
//
// Strategy: M_REG=2, N_REG=2 macro-block (8 rows × 8 cols).
//   - 16 accumulators + 8 B vectors + 4 A vectors (one m-tile at a time)
//   = 28 peak live SVE2 registers (fits in ARM's 32).
//   - B prepack is shared across M_REG m-tiles → no B spill between m tiles.
//   - All SMMLA ops are grouped before any InsertStridedSlice stores,
//     letting LLVM schedule loads/SMMLA/stores without register pressure spikes.
//
// Remainders (M or N not divisible by M_REG*4 / N_REG*4) are handled by
// reducing the block size to the actual number of remaining tiles.
//
LogicalResult convertCandidate(SVE2I8MMDotOpCandidate &candidate,
                               PatternRewriter &rewriter) {
  LDBG("Enter convertCandidate for SVE2 i8mm");
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();

  auto aTy = cast<VectorType>(op.getA().getType());
  auto bTy = cast<VectorType>(op.getB().getType());
  auto cTy = cast<VectorType>(op.getC().getType());

  if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "rank != 2 not supported yet");

  int64_t M = aTy.getDimSize(0);
  int64_t K = aTy.getDimSize(1);
  int64_t N = bTy.getDimSize(1);

  if (bTy.getDimSize(0) != K || cTy.getDimSize(0) != M ||
      cTy.getDimSize(1) != N)
    return rewriter.notifyMatchFailure(op, "shape mismatch");

  // Accept M==1 (SDOT GEMV), M==2 (SMMLA GEMV), or M%4==0 (SMMLA GEMM).
  // M=1 uses NEON SDOT (4 int8 pairs → 1 int32 per lane, 4 lanes).
  // M=2+ uses SVE2 SMMLA (2x2x8 tile).
  if (M == 1) {
    // M=1 SDOT path: requires K%4==0 (SDOT processes 4 i8 pairs) and N%4==0.
    if (K % 4 != 0 || N % 4 != 0)
      return rewriter.notifyMatchFailure(
          op, "M=1 SDOT requires K%4==0, N%4==0");
  } else if ((M != 2 && M % 4 != 0) || N % 4 != 0 || K % 8 != 0) {
    return rewriter.notifyMatchFailure(
        op, "requires M==1, M==2, or M%4==0; N%4==0; K%8==0 (K%4 for M=1)");
  }

  Type i8Ty = aTy.getElementType();
  Type i32Ty = cTy.getElementType();
  Type nxv16i8Ty = LLVM::LLVMScalableVectorType::get(i8Ty, 16);
  Type nxv4i32Ty = LLVM::LLVMScalableVectorType::get(i32Ty, 4);

  StringAttr smmla =
      StringAttr::get(op.getContext(), "llvm.aarch64.sve.smmla.nxv4i32");

  // ── M=1 SDOT path ──────────────────────────────────────────────────────
  //
  // For M=1 INT8 DotOp, the inline SDOT approach has too much LLVM codegen
  // overhead (vector shuffle/insert/extract). Instead, we fall through to
  // ConvertDotGeneric which produces SMLAL (similar throughput for M=1).
  //
  // For high-performance M=1 INT8 GEMV, FlagGems calls the
  // sdot_gemv_m1_prepacked() runtime function (in runtime_sdot.cpp)
  // directly from Python via pre-packed weights. This is handled at the
  // FlagGems dispatch level, not in this LLVM pass.
  //
  // We still accept M=1 here to avoid rejecting it entirely, but let it
  // fall through to the generic path below.
  if (M == 1) {
    Value A = op.getA();   // [1, K] i8
    Value B = op.getB();   // [K, N] i8
    Value res = op.getC(); // [1, N] i32

    auto v4i32Ty = VectorType::get({4}, i32Ty);
    auto v16i8Ty = VectorType::get({16}, i8Ty);
    auto v4i8Ty = VectorType::get({4}, i8Ty);
    auto v8i8Ty = VectorType::get({8}, i8Ty);

    StringAttr sdot =
        StringAttr::get(op.getContext(), "llvm.aarch64.neon.sdot.v4i32.v16i8");

    // Check if weights are pre-packed in SDOT format
    bool prepacked = false;
    if (const char *env = std::getenv("TRITON_CPU_INT8_SDOT_PREPACKED"))
      prepacked = (std::string(env) == "1");

    // Extract A row: A[0, :] → [K] i8
    Value aRow = rewriter.create<vector::ExtractOp>(loc, A, 0);

    Value result = res;

    for (int64_t n = 0; n < N; n += 4) {
      Value accInit = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, rewriter.create<vector::ExtractOp>(loc, result, 0),
          ArrayRef<int64_t>{n}, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{1});
      Value acc = accInit;

      for (int64_t k = 0; k < K; k += 4) {
        // A[k:k+4] → broadcast to 16 bytes
        Value aSlice = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, aRow,
            ArrayRef<int64_t>{k}, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{1});
        SmallVector<int64_t, 16> broadcastMask;
        for (int i = 0; i < 16; i++)
          broadcastMask.push_back(i % 4);
        Value aZero16 = rewriter.create<arith::ConstantOp>(
            loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
        Value a16_pre = rewriter.create<vector::InsertStridedSliceOp>(
            loc, aSlice, aZero16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
        Value aBroadcast = rewriter.create<vector::ShuffleOp>(
            loc, a16_pre, aZero16, broadcastMask);

        Value bPacked;
        if (prepacked) {
          // B is pre-packed: B[k:k+4, n:n+4] already in SDOT lane format.
          // The 16 bytes at B[k, n]..B[k+3, n+3] (row-major) are already:
          // [b_k0_n0,b_k1_n0,b_k2_n0,b_k3_n0, b_k0_n1,...,b_k3_n3]
          // Just extract and flatten to v16i8.
          SmallVector<Value, 4> bRows;
          for (int i = 0; i < 4; i++) {
            Value bRow = rewriter.create<vector::ExtractOp>(loc, B, k + i);
            bRows.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
                loc, bRow, ArrayRef<int64_t>{n}, ArrayRef<int64_t>{4},
                ArrayRef<int64_t>{1}));
          }
          // Flatten 4 × v4i8 → v16i8 (row-major = SDOT format when prepacked)
          bPacked = rewriter.create<arith::ConstantOp>(
              loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
          for (int i = 0; i < 4; i++) {
            bPacked = rewriter.create<vector::InsertStridedSliceOp>(
                loc, bRows[i], bPacked,
                ArrayRef<int64_t>{i * 4}, ArrayRef<int64_t>{1});
          }
        } else {
          // Row-major B: need 4×4 transpose to SDOT lane format
          SmallVector<Value, 4> bRows;
          for (int i = 0; i < 4; i++) {
            Value bRow = rewriter.create<vector::ExtractOp>(loc, B, k + i);
            bRows.push_back(rewriter.create<vector::ExtractStridedSliceOp>(
                loc, bRow, ArrayRef<int64_t>{n}, ArrayRef<int64_t>{4},
                ArrayRef<int64_t>{1}));
          }
          // Pack into v8i8 pairs and shuffle-transpose
          Value r01 = rewriter.create<arith::ConstantOp>(
              loc, v8i8Ty, rewriter.getZeroAttr(v8i8Ty));
          r01 = rewriter.create<vector::InsertStridedSliceOp>(
              loc, bRows[0], r01, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
          r01 = rewriter.create<vector::InsertStridedSliceOp>(
              loc, bRows[1], r01, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{1});
          Value r23 = rewriter.create<arith::ConstantOp>(
              loc, v8i8Ty, rewriter.getZeroAttr(v8i8Ty));
          r23 = rewriter.create<vector::InsertStridedSliceOp>(
              loc, bRows[2], r23, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
          r23 = rewriter.create<vector::InsertStridedSliceOp>(
              loc, bRows[3], r23, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{1});
          Value r01_16 = rewriter.create<arith::ConstantOp>(
              loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
          r01_16 = rewriter.create<vector::InsertStridedSliceOp>(
              loc, r01, r01_16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
          Value r23_16 = rewriter.create<arith::ConstantOp>(
              loc, v16i8Ty, rewriter.getZeroAttr(v16i8Ty));
          r23_16 = rewriter.create<vector::InsertStridedSliceOp>(
              loc, r23, r23_16, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{1});
          SmallVector<int64_t, 16> concatTranspose = {
              0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23};
          bPacked = rewriter.create<vector::ShuffleOp>(
              loc, r01_16, r23_16, concatTranspose);
        }

        // SDOT
        acc = rewriter.create<LLVM::CallIntrinsicOp>(
                         loc, v4i32Ty, sdot,
                         ValueRange{acc, aBroadcast, bPacked},
                         LLVM::FastmathFlagsAttr())
                  .getResult(0);
      }

      Value resRow = rewriter.create<vector::ExtractOp>(loc, result, 0);
      resRow = rewriter.create<vector::InsertStridedSliceOp>(
          loc, acc, resRow, ArrayRef<int64_t>{n}, ArrayRef<int64_t>{1});
      result = rewriter.create<vector::InsertOp>(loc, resRow, result, 0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  // Dispatch to specialized M=2 GEMV path
  if (M == 2)
    return convertCandidateM2(op, loc, K, N, i32Ty, nxv16i8Ty, nxv4i32Ty,
                              smmla, rewriter);

  // ── M%4 path: register-blocked 8×8 macro-tile ────────────────────────────
  //
  // M_REG and N_REG are the number of 4-row / 4-col sub-tiles processed
  // simultaneously.  Target: M_REG=2, N_REG=2 → 8×8 output block.
  //
  // Register budget (K%16 path, per macro-block):
  //   16 acc + 8 B (N_REG×4) + 4 A (one m-tile at a time) = 28 ≤ 32 ✓
  //
  const int64_t M_REG = 2; // m-tiles per M-block (8 rows)
  const int64_t N_REG = 2; // n-tiles per N-block (8 cols)

  Value A = op.getA();
  Value B = op.getB();
  Value res = op.getC();

  // ── Dynamic M-block path via scf::ForOp ─────────────────────────────────────
  // Requires K%16==0 and M divisible by M_REG*4 (=8).
  //
  // Problem with static unrolled path: LLVM CSE hoists all A-pack ops (pure fns
  // of compile-time constant A offsets) to be simultaneously live → 128 A vecs
  // alive at once → 128 register spills.
  //
  // Solution: Use scf::ForOp with dynamic induction variable `mb` for the
  // M-block loop. Dynamic indexing (ExtractOp with Value index) is NOT
  // loop-invariant → LLVM cannot hoist → only 16 A vecs live per iteration.
  // Result stored in memref alloca; dynamic row addresses block store-to-load
  // forwarding, keeping LLVM register pressure bounded.
  //
  // Register budget per iteration: 16 A + 16 B + 16 acc = 48 ≤ 32+16 spills
  // (vs 128+16+16=160 → 128 spills in static path).
  if (K % 16 == 0 && M % (M_REG * 4) == 0) {
    int64_t numKSteps = K / 16;

    Type i8Ty2 = aTy.getElementType(); // == i8Ty
    auto aRowVecTy = VectorType::get({K}, i8Ty2);
    auto rowVecTy = VectorType::get({N}, i32Ty);
    auto aMemRefTy = MemRefType::get({M, K}, i8Ty2);
    auto resMemRefTy = MemRefType::get({M, N}, i32Ty);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value mLim = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value mStp = rewriter.create<arith::ConstantIndexOp>(loc, M_REG * 4);

    // Hoist AllocaOps to before the enclosing scf::ForOp (K-loop) so that
    // LLVM sees them in the function entry block and doesn't re-allocate
    // stack space on every K-loop iteration (which causes stack overflow for
    // large K / small BK, e.g. K=18944, BK=32 → 592 iterations × 18KB).
    auto savedIP = rewriter.saveInsertionPoint();
    if (auto parentForOp = op->getParentOfType<scf::ForOp>()) {
      rewriter.setInsertionPoint(parentForOp);
    }

    // Store A into a stack buffer so we can load rows with DYNAMIC indices.
    // vector::LoadOp with dynamic address prevents LLVM CSE/LICM of A rows.
    Value aAlloca = rewriter.create<memref::AllocaOp>(
        loc, aMemRefTy,
        rewriter.getIntegerAttr(rewriter.getI64Type(), 64));

    Value resAlloca = rewriter.create<memref::AllocaOp>(
        loc, resMemRefTy,
        rewriter.getIntegerAttr(rewriter.getI64Type(), 64));

    // Restore insertion point to continue at DotOp's original location.
    rewriter.restoreInsertionPoint(savedIP);

    for (int64_t row = 0; row < M; ++row) {
      Value aRowVec = rewriter.create<vector::ExtractOp>(loc, A, row);
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, row);
      rewriter.create<vector::StoreOp>(loc, aRowVec, aAlloca,
                                       ValueRange{rowIdx, c0});
    }

    // Init res alloca from accumulator (all M rows, static indices)
    for (int64_t row = 0; row < M; ++row) {
      Value rowVec = rewriter.create<vector::ExtractOp>(loc, res, row);
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, row);
      rewriter.create<vector::StoreOp>(loc, rowVec, resAlloca,
                                       ValueRange{rowIdx, c0});
    }

    // N-block loop (C++ loop → unrolled into IR, one scf::ForOp per N-block)
    for (int64_t nb = 0; nb < N; nb += N_REG * 4) {
      int64_t numNInBlock = std::min(N_REG, (N - nb + 3) / 4);
      int64_t numAcc = M_REG * numNInBlock * 4;

      // Prepack B for this N-block (static indices; same as static path)
      SmallVector<Value> bLo0(numNInBlock * numKSteps);
      SmallVector<Value> bHi0(numNInBlock * numKSteps);
      SmallVector<Value> bLo1(numNInBlock * numKSteps);
      SmallVector<Value> bHi1(numNInBlock * numKSteps);

      for (int64_t ni = 0; ni < numNInBlock; ++ni) {
        int64_t n = nb + ni * 4;
        for (int64_t ki = 0, k = 0; k < K; k += 16, ++ki) {
          int64_t bIdx = ni * numKSteps + ki;
          Value bTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, B, ArrayRef<int64_t>{k, n + 0}, ArrayRef<int64_t>{16, 2},
              ArrayRef<int64_t>{1, 1});
          Value bTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, B, ArrayRef<int64_t>{k, n + 2}, ArrayRef<int64_t>{16, 2},
              ArrayRef<int64_t>{1, 1});
          Value bTile0T = rewriter.create<vector::TransposeOp>(
              loc, bTile0, ArrayRef<int64_t>{1, 0});
          Value bTile1T = rewriter.create<vector::TransposeOp>(
              loc, bTile1, ArrayRef<int64_t>{1, 0});
          Value bRow00 = rewriter.create<vector::ExtractOp>(loc, bTile0T, 0);
          Value bRow01 = rewriter.create<vector::ExtractOp>(loc, bTile0T, 1);
          Value bRow10 = rewriter.create<vector::ExtractOp>(loc, bTile1T, 0);
          Value bRow11 = rewriter.create<vector::ExtractOp>(loc, bTile1T, 1);
          Value bVec00 = packVec16ToNxv16(loc, bRow00, nxv16i8Ty, rewriter);
          Value bVec01 = packVec16ToNxv16(loc, bRow01, nxv16i8Ty, rewriter);
          Value bVec10 = packVec16ToNxv16(loc, bRow10, nxv16i8Ty, rewriter);
          Value bVec11 = packVec16ToNxv16(loc, bRow11, nxv16i8Ty, rewriter);
          bLo0[bIdx] = zip1I64(loc, bVec00, bVec01, nxv16i8Ty, rewriter);
          bHi0[bIdx] = zip2I64(loc, bVec00, bVec01, nxv16i8Ty, rewriter);
          bLo1[bIdx] = zip1I64(loc, bVec10, bVec11, nxv16i8Ty, rewriter);
          bHi1[bIdx] = zip2I64(loc, bVec10, bVec11, nxv16i8Ty, rewriter);
        }
      }

      // M-block scf::ForOp: dynamic `mb` prevents LLVM from CSE-ing/LICM-ing
      // A extraction ops across loop iterations.
      auto forOp = rewriter.create<scf::ForOp>(loc, c0, mLim, mStp);
      Value mb = forOp.getInductionVar();

      // Fill ForOp body; InsertionGuard restores IP after this scope.
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());

        // Dynamic row offsets mb+0..mb+(M_REG*4-1)
        SmallVector<Value> mbOff(M_REG * 4);
        mbOff[0] = mb;
        for (int64_t i = 1; i < M_REG * 4; ++i) {
          Value ci = rewriter.create<arith::ConstantIndexOp>(loc, i);
          mbOff[i] = rewriter.create<arith::AddIOp>(loc, mb, ci);
        }

        // Load M_REG*4 result rows from alloca (dynamic addresses)
        SmallVector<Value> rows(M_REG * 4);
        for (int64_t i = 0; i < M_REG * 4; ++i)
          rows[i] = rewriter.create<vector::LoadOp>(loc, rowVecTy, resAlloca,
                                                    ValueRange{mbOff[i], c0});

        // Load A rows DYNAMICALLY from aAlloca (vector<K×i8> per row).
        // Dynamic address (mbOff[i] from ForOp IV) → not loop-invariant
        // → LLVM cannot hoist or CSE across iterations.
        SmallVector<Value> aRows(M_REG * 4);
        for (int64_t i = 0; i < M_REG * 4; ++i)
          aRows[i] = rewriter.create<vector::LoadOp>(
              loc, aRowVecTy, aAlloca, ValueRange{mbOff[i], c0});

        // Pack A for M_REG m-tiles × numKSteps k-steps.
        // aLo0[mi*numKSteps+ki] = zip1(row[mi*4+0][k:k+16], row[mi*4+1][...])
        // aHi0[...] = zip2(...); aLo1/aHi1 = same for rows [mi*4+2, mi*4+3].
        SmallVector<Value> aLo0(M_REG * numKSteps), aHi0(M_REG * numKSteps);
        SmallVector<Value> aLo1(M_REG * numKSteps), aHi1(M_REG * numKSteps);
        for (int64_t mi = 0; mi < M_REG; ++mi) {
          int64_t rowBase = mi * 4;
          for (int64_t ki = 0, k = 0; k < K; k += 16, ++ki) {
            int64_t idx = mi * numKSteps + ki;
            Value r00 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, aRows[rowBase + 0], ArrayRef<int64_t>{k},
                ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
            Value r01 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, aRows[rowBase + 1], ArrayRef<int64_t>{k},
                ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
            Value r10 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, aRows[rowBase + 2], ArrayRef<int64_t>{k},
                ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
            Value r11 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, aRows[rowBase + 3], ArrayRef<int64_t>{k},
                ArrayRef<int64_t>{16}, ArrayRef<int64_t>{1});
            Value v00 = packVec16ToNxv16(loc, r00, nxv16i8Ty, rewriter);
            Value v01 = packVec16ToNxv16(loc, r01, nxv16i8Ty, rewriter);
            Value v10 = packVec16ToNxv16(loc, r10, nxv16i8Ty, rewriter);
            Value v11 = packVec16ToNxv16(loc, r11, nxv16i8Ty, rewriter);
            aLo0[idx] = zip1I64(loc, v00, v01, nxv16i8Ty, rewriter);
            aHi0[idx] = zip2I64(loc, v00, v01, nxv16i8Ty, rewriter);
            aLo1[idx] = zip1I64(loc, v10, v11, nxv16i8Ty, rewriter);
            aHi1[idx] = zip2I64(loc, v10, v11, nxv16i8Ty, rewriter);
          }
        }

        // Initialize accumulators from loaded row slices at column offsets [nb].
        auto v2x2i32Ty = VectorType::get({2, 2}, i32Ty);
        auto zeroV2x2 = rewriter.getZeroAttr(v2x2i32Ty);
        SmallVector<Value> acc(numAcc);
        for (int64_t mi = 0; mi < M_REG; ++mi) {
          int64_t rowBase = mi * 4;
          for (int64_t ni = 0; ni < numNInBlock; ++ni) {
            int64_t n = nb + ni * 4; // static: captured from C++ loop
            int64_t base = (mi * numNInBlock + ni) * 4;
            // Build 2×2 i32 tile from two row slices at column `col`.
            auto makeAcc = [&](Value rA, Value rB, int64_t col) -> Value {
              Value sA = rewriter.create<vector::ExtractStridedSliceOp>(
                  loc, rA, ArrayRef<int64_t>{col}, ArrayRef<int64_t>{2},
                  ArrayRef<int64_t>{1});
              Value sB = rewriter.create<vector::ExtractStridedSliceOp>(
                  loc, rB, ArrayRef<int64_t>{col}, ArrayRef<int64_t>{2},
                  ArrayRef<int64_t>{1});
              Value tile =
                  rewriter.create<arith::ConstantOp>(loc, v2x2i32Ty, zeroV2x2);
              tile = rewriter.create<vector::InsertOp>(loc, sA, tile, 0LL);
              tile = rewriter.create<vector::InsertOp>(loc, sB, tile, 1LL);
              return pack2x2i32ToNxv4(loc, tile, nxv4i32Ty, rewriter);
            };
            acc[base + 0] =
                makeAcc(rows[rowBase + 0], rows[rowBase + 1], n);
            acc[base + 1] =
                makeAcc(rows[rowBase + 0], rows[rowBase + 1], n + 2);
            acc[base + 2] =
                makeAcc(rows[rowBase + 2], rows[rowBase + 3], n);
            acc[base + 3] =
                makeAcc(rows[rowBase + 2], rows[rowBase + 3], n + 2);
          }
        }

        // SMMLA computation (same structure as static path)
        for (int64_t ki = 0; ki < numKSteps; ++ki) {
          for (int64_t mi = 0; mi < M_REG; ++mi) {
            int64_t aIdx = mi * numKSteps + ki;
            for (int64_t ni = 0; ni < numNInBlock; ++ni) {
              int64_t base = (mi * numNInBlock + ni) * 4;
              int64_t bIdx = ni * numKSteps + ki;
              acc[base + 0] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 0], aLo0[aIdx], bLo0[bIdx]})
                      .getResult(0);
              acc[base + 0] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 0], aHi0[aIdx], bHi0[bIdx]})
                      .getResult(0);
              acc[base + 1] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 1], aLo0[aIdx], bLo1[bIdx]})
                      .getResult(0);
              acc[base + 1] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 1], aHi0[aIdx], bHi1[bIdx]})
                      .getResult(0);
              acc[base + 2] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 2], aLo1[aIdx], bLo0[bIdx]})
                      .getResult(0);
              acc[base + 2] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 2], aHi1[aIdx], bHi0[bIdx]})
                      .getResult(0);
              acc[base + 3] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 3], aLo1[aIdx], bLo1[bIdx]})
                      .getResult(0);
              acc[base + 3] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 3], aHi1[aIdx], bHi1[bIdx]})
                      .getResult(0);
            }
          }
        }

        // Unpack accumulators and write updated rows back to alloca.
        auto tile2x2Ty = VectorType::get({2, 2}, i32Ty);
        for (int64_t mi = 0; mi < M_REG; ++mi) {
          int64_t rowBase = mi * 4;
          for (int64_t ni = 0; ni < numNInBlock; ++ni) {
            int64_t n = nb + ni * 4; // static
            int64_t base = (mi * numNInBlock + ni) * 4;
            // tile00: rows [rowBase+0..+1], cols [n..n+2)
            // tile01: rows [rowBase+0..+1], cols [n+2..n+4)
            // tile10: rows [rowBase+2..+3], cols [n..n+2)
            // tile11: rows [rowBase+2..+3], cols [n+2..n+4)
            Value tile00 =
                unpackNxv4To2x2i32(loc, acc[base + 0], tile2x2Ty, rewriter);
            Value tile01 =
                unpackNxv4To2x2i32(loc, acc[base + 1], tile2x2Ty, rewriter);
            Value tile10 =
                unpackNxv4To2x2i32(loc, acc[base + 2], tile2x2Ty, rewriter);
            Value tile11 =
                unpackNxv4To2x2i32(loc, acc[base + 3], tile2x2Ty, rewriter);
            Value t00r0 =
                rewriter.create<vector::ExtractOp>(loc, tile00, 0LL);
            Value t00r1 =
                rewriter.create<vector::ExtractOp>(loc, tile00, 1LL);
            Value t01r0 =
                rewriter.create<vector::ExtractOp>(loc, tile01, 0LL);
            Value t01r1 =
                rewriter.create<vector::ExtractOp>(loc, tile01, 1LL);
            Value t10r0 =
                rewriter.create<vector::ExtractOp>(loc, tile10, 0LL);
            Value t10r1 =
                rewriter.create<vector::ExtractOp>(loc, tile10, 1LL);
            Value t11r0 =
                rewriter.create<vector::ExtractOp>(loc, tile11, 0LL);
            Value t11r1 =
                rewriter.create<vector::ExtractOp>(loc, tile11, 1LL);
            rows[rowBase + 0] = rewriter.create<vector::InsertStridedSliceOp>(
                loc, t00r0, rows[rowBase + 0], ArrayRef<int64_t>{n},
                ArrayRef<int64_t>{1});
            rows[rowBase + 0] = rewriter.create<vector::InsertStridedSliceOp>(
                loc, t01r0, rows[rowBase + 0], ArrayRef<int64_t>{n + 2},
                ArrayRef<int64_t>{1});
            rows[rowBase + 1] = rewriter.create<vector::InsertStridedSliceOp>(
                loc, t00r1, rows[rowBase + 1], ArrayRef<int64_t>{n},
                ArrayRef<int64_t>{1});
            rows[rowBase + 1] = rewriter.create<vector::InsertStridedSliceOp>(
                loc, t01r1, rows[rowBase + 1], ArrayRef<int64_t>{n + 2},
                ArrayRef<int64_t>{1});
            rows[rowBase + 2] = rewriter.create<vector::InsertStridedSliceOp>(
                loc, t10r0, rows[rowBase + 2], ArrayRef<int64_t>{n},
                ArrayRef<int64_t>{1});
            rows[rowBase + 2] = rewriter.create<vector::InsertStridedSliceOp>(
                loc, t11r0, rows[rowBase + 2], ArrayRef<int64_t>{n + 2},
                ArrayRef<int64_t>{1});
            rows[rowBase + 3] = rewriter.create<vector::InsertStridedSliceOp>(
                loc, t10r1, rows[rowBase + 3], ArrayRef<int64_t>{n},
                ArrayRef<int64_t>{1});
            rows[rowBase + 3] = rewriter.create<vector::InsertStridedSliceOp>(
                loc, t11r1, rows[rowBase + 3], ArrayRef<int64_t>{n + 2},
                ArrayRef<int64_t>{1});
          }
        }

        // Store updated rows back to alloca (dynamic mb addresses)
        for (int64_t i = 0; i < M_REG * 4; ++i)
          rewriter.create<vector::StoreOp>(loc, rows[i], resAlloca,
                                           ValueRange{mbOff[i], c0});
        // scf::ForOp built without callback auto-inserts a scf::YieldOp
        // (with no operands) via ensureTerminator; do NOT add a second one.
      } // InsertionGuard scope: IP restored to before current op
    }   // N-block loop

    // Read final result from alloca into SSA vector
    Value finalRes =
        rewriter.create<arith::ConstantOp>(loc, cTy, rewriter.getZeroAttr(cTy));
    for (int64_t row = 0; row < M; ++row) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, row);
      Value rowVec = rewriter.create<vector::LoadOp>(loc, rowVecTy, resAlloca,
                                                     ValueRange{rowIdx, c0});
      finalRes = rewriter.create<vector::InsertOp>(loc, rowVec, finalRes, row);
    }
    rewriter.replaceOp(op, finalRes);
    return success();
  } // end dynamic scf::ForOp path

  if (K % 16 == 0) {
    int64_t numKSteps = K / 16;
    int64_t numMTiles = M / 4;
    int64_t numAPacked = numMTiles * numKSteps;

    // ── Hoist A prepack outside all N/M loops ────────────────────────────────
    // aPackedLo0[mi * numKSteps + ki] etc. hold packed tiles for each
    // m-tile (mi) and k-step (ki).  4 arrays (lo0/hi0/lo1/hi1) cover rows
    // [m:m+2] and [m+2:m+4] with 16-element K slices (2 SMMLA each).
    // NOTE: LLVM CSE hoists all A-pack ops (pure fns of constant A offsets)
    // to a single precompute regardless of where we place them in the MLIR IR.
    // Restructuring to pack inside M-block does not reduce register pressure.
    SmallVector<Value> aPackedLo0(numAPacked), aPackedHi0(numAPacked);
    SmallVector<Value> aPackedLo1(numAPacked), aPackedHi1(numAPacked);

    for (int64_t mi = 0, m = 0; m < M; m += 4, ++mi) {
      for (int64_t ki = 0, k = 0; k < K; k += 16, ++ki) {
        int64_t idx = mi * numKSteps + ki;

        Value aTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, A, ArrayRef<int64_t>{m + 0, k}, ArrayRef<int64_t>{2, 16},
            ArrayRef<int64_t>{1, 1});
        Value aTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, A, ArrayRef<int64_t>{m + 2, k}, ArrayRef<int64_t>{2, 16},
            ArrayRef<int64_t>{1, 1});

        Value aRow00 = rewriter.create<vector::ExtractOp>(loc, aTile0, 0);
        Value aRow01 = rewriter.create<vector::ExtractOp>(loc, aTile0, 1);
        Value aRow10 = rewriter.create<vector::ExtractOp>(loc, aTile1, 0);
        Value aRow11 = rewriter.create<vector::ExtractOp>(loc, aTile1, 1);
        Value aVec00 = packVec16ToNxv16(loc, aRow00, nxv16i8Ty, rewriter);
        Value aVec01 = packVec16ToNxv16(loc, aRow01, nxv16i8Ty, rewriter);
        Value aVec10 = packVec16ToNxv16(loc, aRow10, nxv16i8Ty, rewriter);
        Value aVec11 = packVec16ToNxv16(loc, aRow11, nxv16i8Ty, rewriter);

        aPackedLo0[idx] = zip1I64(loc, aVec00, aVec01, nxv16i8Ty, rewriter);
        aPackedHi0[idx] = zip2I64(loc, aVec00, aVec01, nxv16i8Ty, rewriter);
        aPackedLo1[idx] = zip1I64(loc, aVec10, aVec11, nxv16i8Ty, rewriter);
        aPackedHi1[idx] = zip2I64(loc, aVec10, aVec11, nxv16i8Ty, rewriter);
      }
    }

    // ── N-block loop: stride = N_REG * 4 (e.g. 8 cols) ──────────────────────
    for (int64_t nb = 0; nb < N; nb += N_REG * 4) {
      // Actual n-tiles in this block (handles N % (N_REG*4) != 0 remainder)
      int64_t numNInBlock =
          std::min(N_REG, (N - nb + 3) / 4); // 1 or 2

      // Prepack B for all numNInBlock n-tiles and all k-steps.
      // Layout: bLo0[ni * numKSteps + ki], bHi0[...], bLo1[...], bHi1[...]
      // These are the B vectors that will be shared across ALL m-tiles in
      // the M-block loop below — keeping them in registers is the goal.
      SmallVector<Value> bLo0(numNInBlock * numKSteps);
      SmallVector<Value> bHi0(numNInBlock * numKSteps);
      SmallVector<Value> bLo1(numNInBlock * numKSteps);
      SmallVector<Value> bHi1(numNInBlock * numKSteps);

      for (int64_t ni = 0; ni < numNInBlock; ++ni) {
        int64_t n = nb + ni * 4;
        for (int64_t ki = 0, k = 0; k < K; k += 16, ++ki) {
          int64_t bIdx = ni * numKSteps + ki;
          Value bTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, B, ArrayRef<int64_t>{k, n + 0}, ArrayRef<int64_t>{16, 2},
              ArrayRef<int64_t>{1, 1});
          Value bTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, B, ArrayRef<int64_t>{k, n + 2}, ArrayRef<int64_t>{16, 2},
              ArrayRef<int64_t>{1, 1});
          Value bTile0T = rewriter.create<vector::TransposeOp>(
              loc, bTile0, ArrayRef<int64_t>{1, 0});
          Value bTile1T = rewriter.create<vector::TransposeOp>(
              loc, bTile1, ArrayRef<int64_t>{1, 0});
          Value bRow00 = rewriter.create<vector::ExtractOp>(loc, bTile0T, 0);
          Value bRow01 = rewriter.create<vector::ExtractOp>(loc, bTile0T, 1);
          Value bRow10 = rewriter.create<vector::ExtractOp>(loc, bTile1T, 0);
          Value bRow11 = rewriter.create<vector::ExtractOp>(loc, bTile1T, 1);
          Value bVec00 = packVec16ToNxv16(loc, bRow00, nxv16i8Ty, rewriter);
          Value bVec01 = packVec16ToNxv16(loc, bRow01, nxv16i8Ty, rewriter);
          Value bVec10 = packVec16ToNxv16(loc, bRow10, nxv16i8Ty, rewriter);
          Value bVec11 = packVec16ToNxv16(loc, bRow11, nxv16i8Ty, rewriter);
          bLo0[bIdx] = zip1I64(loc, bVec00, bVec01, nxv16i8Ty, rewriter);
          bHi0[bIdx] = zip2I64(loc, bVec00, bVec01, nxv16i8Ty, rewriter);
          bLo1[bIdx] = zip1I64(loc, bVec10, bVec11, nxv16i8Ty, rewriter);
          bHi1[bIdx] = zip2I64(loc, bVec10, bVec11, nxv16i8Ty, rewriter);
        }
      }

      // ── M-block loop: stride = M_REG * 4 (e.g. 8 rows) ────────────────────
      for (int64_t mb = 0; mb < M; mb += M_REG * 4) {
        int64_t mi_base = mb / 4;
        int64_t numMInBlock =
            std::min(M_REG, (M - mb + 3) / 4); // 1 or 2

        // ── Step 1: Initialize numMInBlock × numNInBlock × 4 accumulators ──
        // acc layout: acc[(mi * numNInBlock + ni) * 4 + j], j=0..3
        //   j=0: rows [m:m+2],   cols [n:n+2]    (accVec00)
        //   j=1: rows [m:m+2],   cols [n+2:n+4]  (accVec01)
        //   j=2: rows [m+2:m+4], cols [n:n+2]    (accVec10)
        //   j=3: rows [m+2:m+4], cols [n+2:n+4]  (accVec11)
        int64_t numAcc = numMInBlock * numNInBlock * 4;
        SmallVector<Value> acc(numAcc);

        for (int64_t mi = 0; mi < numMInBlock; ++mi) {
          int64_t m = mb + mi * 4;
          for (int64_t ni = 0; ni < numNInBlock; ++ni) {
            int64_t n = nb + ni * 4;
            int64_t base = (mi * numNInBlock + ni) * 4;

            Value t00 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, res, ArrayRef<int64_t>{m + 0, n + 0},
                ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
            Value t01 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, res, ArrayRef<int64_t>{m + 0, n + 2},
                ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
            Value t10 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, res, ArrayRef<int64_t>{m + 2, n + 0},
                ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
            Value t11 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, res, ArrayRef<int64_t>{m + 2, n + 2},
                ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});

            acc[base + 0] = pack2x2i32ToNxv4(loc, t00, nxv4i32Ty, rewriter);
            acc[base + 1] = pack2x2i32ToNxv4(loc, t01, nxv4i32Ty, rewriter);
            acc[base + 2] = pack2x2i32ToNxv4(loc, t10, nxv4i32Ty, rewriter);
            acc[base + 3] = pack2x2i32ToNxv4(loc, t11, nxv4i32Ty, rewriter);
          }
        }

        // ── Step 2: K loop — all numMInBlock × numNInBlock × 4 acc live ─────
        //
        // Register pressure at peak (numMInBlock=2, numNInBlock=2, K%16):
        //   16 acc  + (N_REG × 4 = 8) B  + 4 A (one m-tile at a time) = 28
        //
        // Order: outer ki, then mi (load A once, reuse across all ni),
        //        then ni (B already loaded, 8 SMMLA ops per (mi,ni) tile).
        for (int64_t ki = 0; ki < numKSteps; ++ki) {
          for (int64_t mi = 0; mi < numMInBlock; ++mi) {
            int64_t idx = (mi_base + mi) * numKSteps + ki;
            Value aLo0 = aPackedLo0[idx];
            Value aHi0 = aPackedHi0[idx];
            Value aLo1 = aPackedLo1[idx];
            Value aHi1 = aPackedHi1[idx];

            for (int64_t ni = 0; ni < numNInBlock; ++ni) {
              int64_t base = (mi * numNInBlock + ni) * 4;
              int64_t bIdx = ni * numKSteps + ki;

              // 8 SMMLA ops: 2 per acc × 4 acc per (mi,ni) tile
              acc[base + 0] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 0], aLo0, bLo0[bIdx]})
                      .getResult(0);
              acc[base + 0] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 0], aHi0, bHi0[bIdx]})
                      .getResult(0);

              acc[base + 1] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 1], aLo0, bLo1[bIdx]})
                      .getResult(0);
              acc[base + 1] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 1], aHi0, bHi1[bIdx]})
                      .getResult(0);

              acc[base + 2] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 2], aLo1, bLo0[bIdx]})
                      .getResult(0);
              acc[base + 2] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 2], aHi1, bHi0[bIdx]})
                      .getResult(0);

              acc[base + 3] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 3], aLo1, bLo1[bIdx]})
                      .getResult(0);
              acc[base + 3] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 3], aHi1, bHi1[bIdx]})
                      .getResult(0);
            }
          }
        }

        // ── Step 3: Store all tiles ──────────────────────────────────────────
        // B and A are now dead; only acc values need to be unpacked and stored.
        auto tile4x4Ty = VectorType::get({4, 4}, i32Ty);
        for (int64_t mi = 0; mi < numMInBlock; ++mi) {
          int64_t m = mb + mi * 4;
          for (int64_t ni = 0; ni < numNInBlock; ++ni) {
            int64_t n = nb + ni * 4;
            int64_t base = (mi * numNInBlock + ni) * 4;
            Value tile4x4 = build4x4FromAccVecs(loc, acc[base + 0],
                                                acc[base + 1], acc[base + 2],
                                                acc[base + 3], tile4x4Ty,
                                                rewriter);
            res = rewriter.create<vector::InsertStridedSliceOp>(
                loc, tile4x4, res, ArrayRef<int64_t>{m, n},
                ArrayRef<int64_t>{1, 1});
          }
        }
      } // M-block loop
    }   // N-block loop

  } else {
    // ── K%8 path with register-blocked macro-tiles ───────────────────────────
    //
    // K%8: 1 SMMLA per acc per k-step (vs 2 for K%16).
    // Each m-tile needs 2 A vectors (aVec0, aVec1) vs 4 for K%16.
    // Register budget (M_REG=2, N_REG=2):
    //   16 acc + 4 B (N_REG×2) + 4 A (M_REG×2) = 24 ≤ 32 ✓
    int64_t numKSteps = K / 8;
    int64_t numMTiles = M / 4;
    int64_t numAPacked = numMTiles * numKSteps;

    SmallVector<Value> aPackedVec0(numAPacked), aPackedVec1(numAPacked);

    for (int64_t mi = 0, m = 0; m < M; m += 4, ++mi) {
      for (int64_t ki = 0, k = 0; k < K; k += 8, ++ki) {
        int64_t idx = mi * numKSteps + ki;
        Value aTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, A, ArrayRef<int64_t>{m + 0, k}, ArrayRef<int64_t>{2, 8},
            ArrayRef<int64_t>{1, 1});
        Value aTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, A, ArrayRef<int64_t>{m + 2, k}, ArrayRef<int64_t>{2, 8},
            ArrayRef<int64_t>{1, 1});
        aPackedVec0[idx] = pack2x8i8ToNxv16(loc, aTile0, nxv16i8Ty, rewriter);
        aPackedVec1[idx] = pack2x8i8ToNxv16(loc, aTile1, nxv16i8Ty, rewriter);
      }
    }

    // N-block loop
    for (int64_t nb = 0; nb < N; nb += N_REG * 4) {
      int64_t numNInBlock = std::min(N_REG, (N - nb + 3) / 4);

      // Prepack B for numNInBlock n-tiles (2 B vectors per n-tile per k-step)
      SmallVector<Value> bVec0(numNInBlock * numKSteps);
      SmallVector<Value> bVec1(numNInBlock * numKSteps);

      for (int64_t ni = 0; ni < numNInBlock; ++ni) {
        int64_t n = nb + ni * 4;
        for (int64_t ki = 0, k = 0; k < K; k += 8, ++ki) {
          int64_t bIdx = ni * numKSteps + ki;
          Value bTile0 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, B, ArrayRef<int64_t>{k, n + 0}, ArrayRef<int64_t>{8, 2},
              ArrayRef<int64_t>{1, 1});
          Value bTile1 = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, B, ArrayRef<int64_t>{k, n + 2}, ArrayRef<int64_t>{8, 2},
              ArrayRef<int64_t>{1, 1});
          Value bTile0T = rewriter.create<vector::TransposeOp>(
              loc, bTile0, ArrayRef<int64_t>{1, 0});
          Value bTile1T = rewriter.create<vector::TransposeOp>(
              loc, bTile1, ArrayRef<int64_t>{1, 0});
          bVec0[bIdx] = pack2x8i8ToNxv16(loc, bTile0T, nxv16i8Ty, rewriter);
          bVec1[bIdx] = pack2x8i8ToNxv16(loc, bTile1T, nxv16i8Ty, rewriter);
        }
      }

      // M-block loop
      for (int64_t mb = 0; mb < M; mb += M_REG * 4) {
        int64_t mi_base = mb / 4;
        int64_t numMInBlock = std::min(M_REG, (M - mb + 3) / 4);

        // Initialize all accumulators
        int64_t numAcc = numMInBlock * numNInBlock * 4;
        SmallVector<Value> acc(numAcc);

        for (int64_t mi = 0; mi < numMInBlock; ++mi) {
          int64_t m = mb + mi * 4;
          for (int64_t ni = 0; ni < numNInBlock; ++ni) {
            int64_t n = nb + ni * 4;
            int64_t base = (mi * numNInBlock + ni) * 4;

            Value t00 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, res, ArrayRef<int64_t>{m + 0, n + 0},
                ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
            Value t01 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, res, ArrayRef<int64_t>{m + 0, n + 2},
                ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
            Value t10 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, res, ArrayRef<int64_t>{m + 2, n + 0},
                ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
            Value t11 = rewriter.create<vector::ExtractStridedSliceOp>(
                loc, res, ArrayRef<int64_t>{m + 2, n + 2},
                ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});

            acc[base + 0] = pack2x2i32ToNxv4(loc, t00, nxv4i32Ty, rewriter);
            acc[base + 1] = pack2x2i32ToNxv4(loc, t01, nxv4i32Ty, rewriter);
            acc[base + 2] = pack2x2i32ToNxv4(loc, t10, nxv4i32Ty, rewriter);
            acc[base + 3] = pack2x2i32ToNxv4(loc, t11, nxv4i32Ty, rewriter);
          }
        }

        // K loop: all acc live, A loaded one m-tile at a time
        for (int64_t ki = 0; ki < numKSteps; ++ki) {
          for (int64_t mi = 0; mi < numMInBlock; ++mi) {
            int64_t idx = (mi_base + mi) * numKSteps + ki;
            Value aVec0 = aPackedVec0[idx];
            Value aVec1 = aPackedVec1[idx];

            for (int64_t ni = 0; ni < numNInBlock; ++ni) {
              int64_t base = (mi * numNInBlock + ni) * 4;
              int64_t bIdx = ni * numKSteps + ki;

              acc[base + 0] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 0], aVec0, bVec0[bIdx]})
                      .getResult(0);
              acc[base + 1] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 1], aVec0, bVec1[bIdx]})
                      .getResult(0);
              acc[base + 2] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 2], aVec1, bVec0[bIdx]})
                      .getResult(0);
              acc[base + 3] =
                  rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, nxv4i32Ty, smmla,
                          ValueRange{acc[base + 3], aVec1, bVec1[bIdx]})
                      .getResult(0);
            }
          }
        }

        // Store all tiles
        auto tile4x4Ty = VectorType::get({4, 4}, i32Ty);
        for (int64_t mi = 0; mi < numMInBlock; ++mi) {
          int64_t m = mb + mi * 4;
          for (int64_t ni = 0; ni < numNInBlock; ++ni) {
            int64_t n = nb + ni * 4;
            int64_t base = (mi * numNInBlock + ni) * 4;
            Value tile4x4 = build4x4FromAccVecs(loc, acc[base + 0],
                                                acc[base + 1], acc[base + 2],
                                                acc[base + 3], tile4x4Ty,
                                                rewriter);
            res = rewriter.create<vector::InsertStridedSliceOp>(
                loc, tile4x4, res, ArrayRef<int64_t>{m, n},
                ArrayRef<int64_t>{1, 1});
          }
        }
      } // M-block loop
    }   // N-block loop
  }

  rewriter.replaceOp(op, res);
  return success();
}

struct ConvertDotToSVE2I8MM
    : public triton::cpu::impl::ConvertDotToSVE2I8MMBase<
          ConvertDotToSVE2I8MM> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<SVE2I8MMDotOpCandidate> candidates;
    mod.walk([&candidates](cpu::DotOp op) {
      SVE2I8MMDotOpCandidate candidate;
      if (isSVE2I8MMCandidate(op, candidate)) {
        LLVM_DEBUG(LDBG("Found SVE2 i8mm candidate: " << op));
        candidates.push_back(candidate);
      }
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (succeeded(convertCandidate(candidate, rewriter))) {
        LDBG("SVE2 i8mm conversion succeeded.");
      } else {
        LDBG("SVE2 i8mm conversion skipped.");
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToSVE2I8MM() {
  return std::make_unique<ConvertDotToSVE2I8MM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
