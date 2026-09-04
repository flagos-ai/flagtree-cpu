#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

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

// This lowering intentionally follows the shape of a conventional ACLE
// microkernel:
//
//   for n in 0..N step 8:
//     pack B[K, n:n+8]
//     for m in 0..M step 8:
//       acc[4][4] = load C[m:m+8, n:n+8]
//       for k in 0..K step 8:
//         load/pack four 2x8 A row pairs
//         load four packed 2x8 B column pairs
//         acc[4][4] = SVE2 SMMLA(acc, A, B)
//       store acc
//
// Keeping the loops in IR is important.  Expanding a complete Triton tile
// into SSA values makes LLVM keep hundreds of pack values live and produces
// a spill-heavy function even though SMMLA itself is selected.
struct SVE2I8MMCandidate {
  cpu::DotOp op;
  MemBuffer lhsBuf;
  MemBuffer rhsBuf;
  // Ordinary Triton loads an output-major [N,4] SDOT panel and transposes it
  // to the logical [4,N] dot operand.  Keep the physical flat buffer so M1
  // lowering can feed each already-packed sixteen-byte slice directly to
  // SDOT instead of round-tripping the complete transpose through stack.
  MemBuffer rhsPackedN4Buf;
  MemBuffer outputBuf;
  vector::TransferWriteOp directOutputWrite;
  bool keepAccInBuffer = false;
};

struct W4A8DotPair {
  cpu::DotOp lowDot;
  cpu::DotOp highDot;
  Value packed;
};

// Native KAI Q4 prefill is deliberately recognized from the ordinary
// load/reshape/transpose/nibble/tl.dot graph.  These records contain only the
// physical buffers proven by that graph; no frontend intrinsic or runtime
// symbol is required.
struct PackedQ4M4Dot {
  cpu::DotOp op;
  MemBuffer lhsPacked;
  MemBuffer rhsPacked;
};

struct PackedQ4M4Epilogue {
  PackedQ4M4Dot dot;
  // Optional token-asymmetric activation correction.  Empty values denote
  // the original symmetric KAI graph.  When present, the source computes
  // dot - lhsZeroPoint[:, none] * rhsSum[none, :] before scaling.
  Value lhsZeroPoint;
  Value rhsSum;
  Value lhsScale;
  Value rhsScale;
  arith::AddFOp add;
};

struct PackedQ4M4LoopCandidate {
  scf::ForOp loop;
  SmallVector<PackedQ4M4Epilogue, 4> panels;
};

// MiniCPM's native G128 source has two reductions: an outer FP32 loop over
// quantization groups and an inner INT32 loop over the four K32 fragments in
// each group.  Keeping four vector<4x4xf32> values live across the inner loop
// consumes half of Neon's register file before I8MM operands are considered.
// Record this exact nested shape so the outer carried values can be retired to
// a small local tile without changing the portable Triton source.
struct PackedQ4NestedG128LoopCandidate {
  scf::ForOp outerLoop;
  scf::ForOp innerLoop;
  SmallVector<cpu::DotOp, 4> dots;
};

struct PackedQ4M4IntegerLoopCandidate {
  scf::ForOp loop;
  SmallVector<PackedQ4M4Dot, 4> panels;
};

struct PackedW8PrefillCandidate {
  cpu::DotOp op;
  SmallVector<Value, 4> lhsPanels;
  Value rhsPanel;
};

static bool canClonePackedLoopValue(Value value, scf::ForOp loop);
static Value clonePackedLoopValue(Value value, scf::ForOp oldLoop,
                                  IRMapping &mapping,
                                  PatternRewriter &rewriter);
static bool matchPackedQ4M4Dot(cpu::DotOp op, PackedQ4M4Dot &candidate);
static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       int64_t row, int64_t col,
                                       PatternRewriter &rewriter);
static bool matchSplatIntegerConstant(Value value, int64_t expected);

// Packed-loop rewrites rebuild only the values needed by the native dot
// microkernel and then erase the original loop.  That is legal only when the
// old body has no observable effect other than reading its inputs.  Unknown
// effects are rejected as well: silently dropping a store, atomic or call is a
// miscompile even when the dot/layout graph itself is an exact match.
static bool hasOnlyReadOrNoEffects(scf::ForOp loop) {
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(loop);
  if (!effects)
    return false;
  return llvm::all_of(*effects,
                      [](const MemoryEffects::EffectInstance &effect) {
                        return isa<MemoryEffects::Read>(effect.getEffect());
                      });
}

static bool
matchPackedQ4NestedG128Loop(scf::ForOp outerLoop,
                            PackedQ4NestedG128LoopCandidate &candidate) {
  constexpr unsigned panelCount = 4;
  if (outerLoop.getNumRegionIterArgs() != panelCount ||
      outerLoop.getNumResults() != panelCount ||
      !hasOnlyReadOrNoEffects(outerLoop))
    return false;

  auto f32 = Float32Type::get(outerLoop.getContext());
  auto i32 = IntegerType::get(outerLoop.getContext(), 32);
  auto fpPanelTy = VectorType::get({4, 4}, f32);
  auto intPanelTy = VectorType::get({4, 4}, i32);
  for (auto [init, result] :
       llvm::zip(outerLoop.getInitArgs(), outerLoop.getResults()))
    if (init.getType() != fpPanelTy || result.getType() != fpPanelTy)
      return false;

  scf::ForOp innerLoop;
  for (Operation &operation : outerLoop.getBody()->without_terminator()) {
    auto nested = dyn_cast<scf::ForOp>(operation);
    if (!nested)
      continue;
    if (innerLoop)
      return false;
    innerLoop = nested;
  }
  if (!innerLoop || innerLoop.getNumRegionIterArgs() != panelCount ||
      innerLoop.getNumResults() != panelCount)
    return false;
  for (auto [init, result] :
       llvm::zip(innerLoop.getInitArgs(), innerLoop.getResults()))
    if (init.getType() != intPanelTy || result.getType() != intPanelTy)
      return false;

  SmallVector<cpu::DotOp, 4> dots;
  for (Operation &operation : innerLoop.getBody()->without_terminator()) {
    auto dot = dyn_cast<cpu::DotOp>(operation);
    if (!dot)
      continue;
    PackedQ4M4Dot packed;
    if (!matchPackedQ4M4Dot(dot, packed))
      return false;
    dots.push_back(dot);
  }
  if (dots.size() != panelCount)
    return false;

  candidate = {outerLoop, innerLoop, std::move(dots)};
  return true;
}

// Replace only the outer loop's FP32 iter_args with a cache-aligned local
// tile.  The complete original body, including the K32 loop and its packed-Q4
// dots, is cloned verbatim.  A subsequent walk in this pass then applies the
// normal packed-Q4/I8MM lowering to the cloned dots.
static LogicalResult
bufferPackedQ4NestedG128Outer(PackedQ4NestedG128LoopCandidate &candidate,
                              PatternRewriter &rewriter) {
  scf::ForOp oldLoop = candidate.outerLoop;
  Location loc = oldLoop.getLoc();
  constexpr int64_t panelCount = 4;
  auto f32 = rewriter.getF32Type();
  auto panelTy = VectorType::get({4, 4}, f32);
  auto allRowsTy = VectorType::get({panelCount * 4, 4}, f32);

  Operation *allocaPoint = oldLoop;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();
  MemBuffer buffer =
      allocateTmpBufferStack(loc, allRowsTy, allocaPoint, rewriter);

  rewriter.setInsertionPoint(oldLoop);
  for (int64_t panel = 0; panel < panelCount; ++panel) {
    SmallVector<Value> indices =
        get2DIndices(loc, buffer, panel * 4, 0, rewriter);
    op_write(oldLoop.getInitArgs()[panel], buffer.memRef, indices);
  }

  auto newLoop = scf::ForOp::create(rewriter, loc, oldLoop.getLowerBound(),
                                    oldLoop.getUpperBound(), oldLoop.getStep(),
                                    ValueRange{});
  // The zero-result builder inserts an empty yield.  Remove it before cloning
  // the original body, then recreate it after the buffer stores below.
  rewriter.eraseOp(newLoop.getBody()->getTerminator());
  IRMapping mapping;
  mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  rewriter.setInsertionPointToEnd(newLoop.getBody());
  for (int64_t panel = 0; panel < panelCount; ++panel) {
    SmallVector<Value> indices =
        get2DIndices(loc, buffer, panel * 4, 0, rewriter);
    Value loaded = op_read(panelTy, buffer.memRef, indices);
    mapping.map(oldLoop.getRegionIterArgs()[panel], loaded);
  }

  for (Operation &operation : oldLoop.getBody()->without_terminator())
    rewriter.clone(operation, mapping);
  auto oldYield = cast<scf::YieldOp>(oldLoop.getBody()->getTerminator());
  for (int64_t panel = 0; panel < panelCount; ++panel) {
    Value output = mapping.lookupOrDefault(oldYield.getOperand(panel));
    SmallVector<Value> indices =
        get2DIndices(loc, buffer, panel * 4, 0, rewriter);
    op_write(output, buffer.memRef, indices);
  }
  scf::YieldOp::create(rewriter, loc);

  rewriter.setInsertionPointAfter(oldLoop);
  for (int64_t panel = 0; panel < panelCount; ++panel) {
    SmallVector<Value> indices =
        get2DIndices(loc, buffer, panel * 4, 0, rewriter);
    Value output = op_read(panelTy, buffer.memRef, indices);
    rewriter.replaceAllUsesWith(oldLoop.getResult(panel), output);
  }
  rewriter.eraseOp(oldLoop);
  return success();
}

static Value addIndex(Location loc, Value base, Value offset,
                      PatternRewriter &rewriter) {
  return arith::AddIOp::create(rewriter, loc, base, offset);
}

static Value addIndex(Location loc, Value base, int64_t offset,
                      PatternRewriter &rewriter) {
  if (offset == 0)
    return base;
  Value c = arith::ConstantIndexOp::create(rewriter, loc, offset);
  return addIndex(loc, base, c, rewriter);
}

static MemBuffer findFlatVectorLoadBuffer(Value physical) {
  auto shape = physical.getDefiningOp<vector::ShapeCastOp>();
  if (!shape)
    return {};
  auto load = shape.getSource().getDefiningOp<vector::LoadOp>();
  if (!load)
    return {};
  return {load.getBase(), SmallVector<Value>(load.getIndices())};
}

static MemBuffer findPackedN4DotBuffer(Value logical, int64_t kSize,
                                       int64_t nSize) {
  auto transpose = logical.getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({1, 0}))
    return {};
  auto physicalTy = dyn_cast<VectorType>(transpose.getVector().getType());
  if (!physicalTy ||
      physicalTy.getShape() != ArrayRef<int64_t>({nSize, kSize}) ||
      kSize != 4 || !physicalTy.getElementType().isInteger(8))
    return {};
  MemBuffer buffer = findFlatVectorLoadBuffer(transpose.getVector());
  if (buffer.empty())
    return {};
  auto memRefTy = dyn_cast<MemRefType>(buffer.memRef.getType());
  if (!memRefTy || memRefTy.getRank() != 1 ||
      memRefTy.getDimSize(0) != nSize * kSize ||
      !memRefTy.getElementType().isInteger(8))
    return {};
  return buffer;
}

static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       Value row, Value col,
                                       PatternRewriter &rewriter) {
  assert(buf.indices.size() >= 2 && "expected a rank-2 input buffer");
  SmallVector<Value> indices(buf.indices);
  size_t rowIdx = indices.size() - 2;
  size_t colIdx = indices.size() - 1;
  indices[rowIdx] = addIndex(loc, indices[rowIdx], row, rewriter);
  indices[colIdx] = addIndex(loc, indices[colIdx], col, rewriter);
  return indices;
}

static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       int64_t row, int64_t col,
                                       PatternRewriter &rewriter) {
  assert(buf.indices.size() >= 2 && "expected a rank-2 input buffer");
  SmallVector<Value> indices(buf.indices);
  size_t rowIdx = indices.size() - 2;
  size_t colIdx = indices.size() - 1;
  indices[rowIdx] = shiftIndex(loc, indices[rowIdx], row, rewriter);
  indices[colIdx] = shiftIndex(loc, indices[colIdx], col, rewriter);
  return indices;
}

static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       Value row, int64_t col,
                                       PatternRewriter &rewriter) {
  assert(buf.indices.size() >= 2 && "expected a rank-2 input buffer");
  SmallVector<Value> indices(buf.indices);
  size_t rowIdx = indices.size() - 2;
  size_t colIdx = indices.size() - 1;
  indices[rowIdx] = addIndex(loc, indices[rowIdx], row, rewriter);
  indices[colIdx] = shiftIndex(loc, indices[colIdx], col, rewriter);
  return indices;
}

static SmallVector<Value> get2DIndices(Location loc, const MemBuffer &buf,
                                       int64_t row, Value col,
                                       PatternRewriter &rewriter) {
  assert(buf.indices.size() >= 2 && "expected a rank-2 input buffer");
  SmallVector<Value> indices(buf.indices);
  size_t rowIdx = indices.size() - 2;
  size_t colIdx = indices.size() - 1;
  indices[rowIdx] = shiftIndex(loc, indices[rowIdx], row, rewriter);
  indices[colIdx] = addIndex(loc, indices[colIdx], col, rewriter);
  return indices;
}

static Value insertFixedInScalable(Location loc, Value fixed, Type scalableTy,
                                   PatternRewriter &rewriter) {
  Value undef = LLVM::UndefOp::create(rewriter, loc, scalableTy);
  Value zero = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                        rewriter.getI64IntegerAttr(0));
  auto fixedTy = cast<VectorType>(fixed.getType());
  auto scalableVecTy = cast<VectorType>(scalableTy);
  std::string intrinsic =
      "llvm.vector.insert.nxv" +
      std::to_string(scalableVecTy.getNumElements()) + "i" +
      std::to_string(fixedTy.getElementType().getIntOrFloatBitWidth()) + ".v" +
      std::to_string(fixedTy.getNumElements()) + "i" +
      std::to_string(fixedTy.getElementType().getIntOrFloatBitWidth());
  return LLVM::CallIntrinsicOp::create(
             rewriter, loc, scalableTy,
             StringAttr::get(rewriter.getContext(), intrinsic),
             ValueRange{undef, fixed, zero})
      .getResult(0);
}

static Value extractFixedFromScalable(Location loc, Value scalable,
                                      VectorType fixedTy,
                                      PatternRewriter &rewriter) {
  Value zero = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                        rewriter.getI64IntegerAttr(0));
  auto scalableTy = cast<VectorType>(scalable.getType());
  std::string intrinsic =
      "llvm.vector.extract.v" + std::to_string(fixedTy.getNumElements()) + "i" +
      std::to_string(fixedTy.getElementType().getIntOrFloatBitWidth()) +
      ".nxv" + std::to_string(scalableTy.getNumElements()) + "i" +
      std::to_string(fixedTy.getElementType().getIntOrFloatBitWidth());
  return LLVM::CallIntrinsicOp::create(
             rewriter, loc, fixedTy,
             StringAttr::get(rewriter.getContext(), intrinsic),
             ValueRange{scalable, zero})
      .getResult(0);
}

static Value packRows2x8(Location loc, Value row0, Value row1,
                         VectorType v16i8Ty, Type nxv16i8Ty,
                         PatternRewriter &rewriter) {
  Type i8Ty = v16i8Ty.getElementType();
  auto tileTy = VectorType::get({2, 8}, i8Ty);
  Value tile = arith::ConstantOp::create(rewriter, loc, tileTy,
                                         rewriter.getZeroAttr(tileTy));
  tile = vector::InsertOp::create(rewriter, loc, row0, tile, 0LL);
  tile = vector::InsertOp::create(rewriter, loc, row1, tile, 1LL);
  Value flat = vector::ShapeCastOp::create(rewriter, loc, v16i8Ty, tile);
  return insertFixedInScalable(loc, flat, nxv16i8Ty, rewriter);
}

static std::pair<Value, Value> transposePair(Location loc, Value lhs, Value rhs,
                                             ArrayRef<int64_t> lowMask,
                                             ArrayRef<int64_t> highMask,
                                             PatternRewriter &rewriter) {
  Value low = vector::ShuffleOp::create(rewriter, loc, lhs, rhs, lowMask);
  Value high = vector::ShuffleOp::create(rewriter, loc, lhs, rhs, highMask);
  return {low, high};
}

static Value loadAcc2x2(Location loc, const MemBuffer &accBuf, Value row,
                        Value col, VectorType v2i32Ty, VectorType v4i32Ty,
                        Type nxv4i32Ty, PatternRewriter &rewriter) {
  Value row1 = addIndex(loc, row, 1, rewriter);
  Value r0 =
      vector::LoadOp::create(rewriter, loc, v2i32Ty, accBuf.memRef,
                             get2DIndices(loc, accBuf, row, col, rewriter));
  Value r1 =
      vector::LoadOp::create(rewriter, loc, v2i32Ty, accBuf.memRef,
                             get2DIndices(loc, accBuf, row1, col, rewriter));
  auto tileTy = VectorType::get({2, 2}, v2i32Ty.getElementType());
  Value tile = arith::ConstantOp::create(rewriter, loc, tileTy,
                                         rewriter.getZeroAttr(tileTy));
  tile = vector::InsertOp::create(rewriter, loc, r0, tile, 0LL);
  tile = vector::InsertOp::create(rewriter, loc, r1, tile, 1LL);
  Value flat = vector::ShapeCastOp::create(rewriter, loc, v4i32Ty, tile);
  return insertFixedInScalable(loc, flat, nxv4i32Ty, rewriter);
}

static void storeAcc2x2(Location loc, const MemBuffer &accBuf, Value row,
                        Value col, Value scalable, VectorType v2i32Ty,
                        VectorType v4i32Ty, PatternRewriter &rewriter) {
  Value flat = extractFixedFromScalable(loc, scalable, v4i32Ty, rewriter);
  auto tileTy = VectorType::get({2, 2}, v2i32Ty.getElementType());
  Value tile = vector::ShapeCastOp::create(rewriter, loc, tileTy, flat);
  Value r0 = vector::ExtractOp::create(rewriter, loc, tile, 0LL);
  Value r1 = vector::ExtractOp::create(rewriter, loc, tile, 1LL);
  Value row1 = addIndex(loc, row, 1, rewriter);
  vector::StoreOp::create(rewriter, loc, r0, accBuf.memRef,
                          get2DIndices(loc, accBuf, row, col, rewriter));
  vector::StoreOp::create(rewriter, loc, r1, accBuf.memRef,
                          get2DIndices(loc, accBuf, row1, col, rewriter));
}

static bool hasUnitMinorStride(const MemBuffer &buf) {
  auto memRefTy = dyn_cast<MemRefType>(buf.memRef.getType());
  if (!memRefTy || memRefTy.getRank() < 2)
    return false;
  auto [strides, offset] = memRefTy.getStridesAndOffset();
  (void)offset;
  return strides.back() == 1;
}

static bool isCandidate(cpu::DotOp op, SVE2I8MMCandidate &candidate) {
  auto lhsTy = dyn_cast<VectorType>(op.getA().getType());
  auto rhsTy = dyn_cast<VectorType>(op.getB().getType());
  auto accTy = dyn_cast<VectorType>(op.getC().getType());
  if (!lhsTy || !rhsTy || !accTy || lhsTy.getRank() != 2 ||
      rhsTy.getRank() != 2 || accTy.getRank() != 2)
    return false;
  if (!lhsTy.getElementType().isInteger(8) ||
      !rhsTy.getElementType().isInteger(8) ||
      !accTy.getElementType().isInteger(32))
    return false;

  int64_t m = lhsTy.getDimSize(0);
  int64_t k = lhsTy.getDimSize(1);
  int64_t n = rhsTy.getDimSize(1);
  if (rhsTy.getDimSize(0) != k || accTy.getDimSize(0) != m ||
      accTy.getDimSize(1) != n)
    return false;

  // Decode uses a rolled 1x4x4 NEON SDOT microkernel.  Prefill uses the
  // rolled 8x8x8 SVE2 SMMLA macro-tile below.  Tails remain on the generic
  // path until predicated packing is implemented.
  bool isM1Sdot = m == 1 && n >= 4 && k >= 4 && n % 4 == 0 && k % 4 == 0;
  bool isSVE2I8MM =
      m >= 8 && n >= 8 && k >= 8 && m % 8 == 0 && n % 8 == 0 && k % 8 == 0;
  if (!isM1Sdot && !isSVE2I8MM)
    return false;

  candidate.op = op;
  candidate.lhsBuf = findInputBuffer(op.getA());
  candidate.rhsBuf = findInputBuffer(op.getB());
  if (m == 1)
    candidate.rhsPackedN4Buf = findPackedN4DotBuffer(op.getB(), k, n);
  // Packed low-bit kernels naturally put cheap SSA transforms between a
  // load and dot (shape_cast for A; and/shr/sub for an unpacked Q4 B tile).
  // Do not reject those ordinary Triton programs merely because the input is
  // no longer a direct transfer_read.  convertCandidate() materializes such
  // values into compiler-owned stack buffers, after which the same compact
  // rolled SDOT lowering applies.  Direct load inputs retain the zero-copy
  // path.
  if ((!candidate.lhsBuf.empty() && !hasUnitMinorStride(candidate.lhsBuf)) ||
      (!candidate.rhsBuf.empty() && !hasUnitMinorStride(candidate.rhsBuf)))
    return false;

  candidate.keepAccInBuffer = isLoopCarriedAcc(op.getC());
  if (!candidate.keepAccInBuffer && matchPattern(op.getC(), m_Zero()) &&
      op->hasOneUse()) {
    auto write = dyn_cast<vector::TransferWriteOp>(*op->getUsers().begin());
    if (write && write.getValueToStore() == op.getResult() &&
        !hasMaskOrBoundsCheck(write) &&
        write.getPermutationMap().isMinorIdentity()) {
      candidate.outputBuf.memRef = write.getBase();
      candidate.outputBuf.indices.append(write.getIndices().begin(),
                                         write.getIndices().end());
      if (hasUnitMinorStride(candidate.outputBuf))
        candidate.directOutputWrite = write;
      else
        candidate.outputBuf = {};
    }
  }
  return true;
}

// Match the canonical ordinary-Triton Q4 expression:
//
//   lo = dot(x[0:4],  (packed & 15) - 8, acc)
//   hi = dot(x[16:20], (packed >> 4) - 8, lo)
//
// In addition to the graph shape, match every Q4_0 constant.  Type checking
// alone does not prove nibble mask, shift amount or zero point; accepting a
// lookalike expression and then emitting the hard-coded Q4 microkernel would
// silently change program semantics.
static bool matchW4A8DotPair(cpu::DotOp lowDot, W4A8DotPair &pair) {
  if (!lowDot->hasOneUse())
    return false;
  auto highDot = dyn_cast<cpu::DotOp>(*lowDot->getUsers().begin());
  if (!highDot || highDot.getC() != lowDot.getResult())
    return false;

  auto lhsTy = dyn_cast<VectorType>(lowDot.getA().getType());
  auto rhsTy = dyn_cast<VectorType>(lowDot.getB().getType());
  auto accTy = dyn_cast<VectorType>(lowDot.getC().getType());
  if (!lhsTy || !rhsTy || !accTy || lhsTy.getRank() != 2 ||
      lhsTy.getDimSize(0) != 1 || lhsTy.getDimSize(1) != 4 ||
      rhsTy.getRank() != 2 || rhsTy.getDimSize(0) != 4 ||
      rhsTy.getDimSize(1) < 4 || rhsTy.getDimSize(1) % 4 != 0 ||
      !lhsTy.getElementType().isInteger(8) ||
      !rhsTy.getElementType().isInteger(8) ||
      !accTy.getElementType().isInteger(32) ||
      highDot.getA().getType() != lowDot.getA().getType() ||
      highDot.getB().getType() != lowDot.getB().getType())
    return false;

  auto lowSub = lowDot.getB().getDefiningOp<arith::SubIOp>();
  auto highSub = highDot.getB().getDefiningOp<arith::SubIOp>();
  if (!lowSub || !highSub)
    return false;
  auto lowAnd = lowSub.getLhs().getDefiningOp<arith::AndIOp>();
  auto highShift = highSub.getLhs().getDefiningOp<arith::ShRUIOp>();
  if (!lowAnd || !highShift || lowAnd.getLhs() != highShift.getLhs() ||
      !matchSplatIntegerConstant(lowAnd.getRhs(), 15) ||
      !matchSplatIntegerConstant(highShift.getRhs(), 4) ||
      !matchSplatIntegerConstant(lowSub.getRhs(), 8) ||
      !matchSplatIntegerConstant(highSub.getRhs(), 8))
    return false;

  auto accArg = dyn_cast<BlockArgument>(lowDot.getC());
  if (!accArg || lowDot->getParentOp() != highDot->getParentOp())
    return false;
  auto outerLoop = dyn_cast<scf::ForOp>(lowDot->getParentOp());
  if (!outerLoop || !highDot->hasOneUse())
    return false;
  auto yield = dyn_cast<scf::YieldOp>(*highDot->getUsers().begin());
  int64_t iterArg = accArg.getArgNumber() - outerLoop.getNumInductionVars();
  if (!yield || iterArg < 0 ||
      static_cast<unsigned>(iterArg) >= yield.getNumOperands() ||
      yield.getOperand(iterArg) != highDot.getResult())
    return false;

  pair = {lowDot, highDot, lowAnd.getLhs()};
  return true;
}

// Fuse the matched Q4 pair into one packed-byte transpose and two SDOTs.
// This keeps the source program in ordinary tl.load/bitwise/tl.dot form while
// producing the same dataflow as a tuned ACLE W4A8 microkernel.
static LogicalResult convertW4A8DotPair(W4A8DotPair &pair,
                                        PatternRewriter &rewriter) {
  cpu::DotOp lowDot = pair.lowDot;
  cpu::DotOp highDot = pair.highDot;
  Location loc = lowDot.getLoc();
  auto lhsTy = cast<VectorType>(lowDot.getA().getType());
  auto rhsTy = cast<VectorType>(lowDot.getB().getType());
  auto accTy = cast<VectorType>(lowDot.getC().getType());
  Type i8Ty = lhsTy.getElementType();
  Type i32Ty = accTy.getElementType();
  int64_t nSize = rhsTy.getDimSize(1);

  auto v4i8Ty = VectorType::get({4}, i8Ty);
  auto v8i8Ty = VectorType::get({8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  StringAttr sdot = StringAttr::get(lowDot.getContext(),
                                    "llvm.aarch64.neon.sdot.v4i32.v16i8");

  Operation *allocaPoint = lowDot;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  // Materializing the original packed value (rather than each decoded
  // nibble matrix) gives LLVM a small, alias-free buffer.  SROA forwards the
  // original loads into the microkernel and removes the alloca.
  MemBuffer aLow = storeToTmpBuffer(loc, lowDot.getA(), allocaPoint, rewriter);
  MemBuffer aHigh =
      storeToTmpBuffer(loc, highDot.getA(), allocaPoint, rewriter);
  MemBuffer packed = storeToTmpBuffer(loc, pair.packed, allocaPoint, rewriter);

  auto outerLoop = cast<scf::ForOp>(lowDot->getParentOp());
  Value initialAcc = getInitAccValue(lowDot.getC());
  MemBuffer accBuf;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(outerLoop);
    accBuf = allocateTmpBufferStack(loc, accTy, allocaPoint, rewriter);
    op_write(initialAcc, accBuf.memRef, accBuf.indices);
  }

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c4 = arith::ConstantIndexOp::create(rewriter, loc, 4);
  Value nLimit = arith::ConstantIndexOp::create(rewriter, loc, nSize);
  Value zero8 = arith::ConstantOp::create(rewriter, loc, v8i8Ty,
                                          rewriter.getZeroAttr(v8i8Ty));
  Value zero16 = arith::ConstantOp::create(rewriter, loc, v16i8Ty,
                                           rewriter.getZeroAttr(v16i8Ty));
  Value maskScalar = arith::ConstantIntOp::create(rewriter, loc, 15, 8);
  Value zeroPointScalar = arith::ConstantIntOp::create(rewriter, loc, 8, 8);
  Value shiftScalar = arith::ConstantIntOp::create(rewriter, loc, 4, 8);
  Value mask = vector::BroadcastOp::create(rewriter, loc, v16i8Ty, maskScalar);
  Value zeroPoint =
      vector::BroadcastOp::create(rewriter, loc, v16i8Ty, zeroPointScalar);
  Value shift =
      vector::BroadcastOp::create(rewriter, loc, v16i8Ty, shiftScalar);

  auto broadcastA = [&](const MemBuffer &buf) {
    Value a4 = vector::LoadOp::create(rewriter, loc, v4i8Ty, buf.memRef,
                                      get2DIndices(loc, buf, 0, 0, rewriter));
    // Bitcast directly to a scalar word before broadcasting.  Using
    // vector.bitcast through vector<1xi32> made the vector-to-LLVM lowering
    // reconstruct the word byte-by-byte (USHLL + MOV + UZP1).  The scalar
    // LLVM bitcast folds with the four-byte load and selects a single DUP.
    Value scalar = LLVM::BitcastOp::create(rewriter, loc, i32Ty, a4);
    Value words = vector::BroadcastOp::create(rewriter, loc, v4i32Ty, scalar);
    return LLVM::BitcastOp::create(rewriter, loc, v16i8Ty, words).getResult();
  };
  Value aLowBroadcast = broadcastA(aLow);
  Value aHighBroadcast = broadcastA(aHigh);

  constexpr int64_t transposePacked[] = {0, 4, 16, 20, 1, 5, 17, 21,
                                         2, 6, 18, 22, 3, 7, 19, 23};

  auto nLoop = scf::ForOp::create(rewriter, loc, c0, nLimit, c4);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nBase = nLoop.getInductionVar();
    Value acc =
        vector::LoadOp::create(rewriter, loc, v4i32Ty, accBuf.memRef,
                               get2DIndices(loc, accBuf, 0, nBase, rewriter));

    SmallVector<Value> rows(4);
    for (int64_t r = 0; r < 4; ++r) {
      rows[r] =
          vector::LoadOp::create(rewriter, loc, v4i8Ty, packed.memRef,
                                 get2DIndices(loc, packed, r, nBase, rewriter));
    }
    Value rows01 = vector::InsertStridedSliceOp::create(
        rewriter, loc, rows[0], zero8, ArrayRef<int64_t>{0},
        ArrayRef<int64_t>{1});
    rows01 = vector::InsertStridedSliceOp::create(rewriter, loc, rows[1],
                                                  rows01, ArrayRef<int64_t>{4},
                                                  ArrayRef<int64_t>{1});
    Value rows23 = vector::InsertStridedSliceOp::create(
        rewriter, loc, rows[2], zero8, ArrayRef<int64_t>{0},
        ArrayRef<int64_t>{1});
    rows23 = vector::InsertStridedSliceOp::create(rewriter, loc, rows[3],
                                                  rows23, ArrayRef<int64_t>{4},
                                                  ArrayRef<int64_t>{1});
    Value rows01Wide = vector::InsertStridedSliceOp::create(
        rewriter, loc, rows01, zero16, ArrayRef<int64_t>{0},
        ArrayRef<int64_t>{1});
    Value rows23Wide = vector::InsertStridedSliceOp::create(
        rewriter, loc, rows23, zero16, ArrayRef<int64_t>{0},
        ArrayRef<int64_t>{1});
    Value transposed = vector::ShuffleOp::create(rewriter, loc, rows01Wide,
                                                 rows23Wide, transposePacked);
    Value weightLow = arith::AndIOp::create(rewriter, loc, transposed, mask);
    weightLow = arith::SubIOp::create(rewriter, loc, weightLow, zeroPoint);
    Value weightHigh = arith::ShRUIOp::create(rewriter, loc, transposed, shift);
    weightHigh = arith::SubIOp::create(rewriter, loc, weightHigh, zeroPoint);

    Value lowAcc =
        LLVM::CallIntrinsicOp::create(rewriter, loc, v4i32Ty, sdot,
                                      ValueRange{acc, aLowBroadcast, weightLow})
            .getResult(0);
    Value highAcc = LLVM::CallIntrinsicOp::create(
                        rewriter, loc, v4i32Ty, sdot,
                        ValueRange{lowAcc, aHighBroadcast, weightHigh})
                        .getResult(0);
    vector::StoreOp::create(rewriter, loc, highAcc, accBuf.memRef,
                            get2DIndices(loc, accBuf, 0, nBase, rewriter));
  }

  rewriter.setInsertionPointAfter(nLoop);
  Value loopResult =
      outerLoop.getTiedLoopResult(cast<BlockArgument>(lowDot.getC()));
  rewriter.replaceOp(highDot, lowDot.getC());
  rewriter.eraseOp(lowDot);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(outerLoop);
  Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
  rewriter.replaceAllUsesWith(loopResult, result);
  return success();
}

static SmallVector<Value> getFlatIndices(Location loc, const MemBuffer &buf,
                                         int64_t offset,
                                         PatternRewriter &rewriter) {
  assert(!buf.indices.empty() && "expected a flat packed input buffer");
  SmallVector<Value> indices(buf.indices);
  indices.back() = addIndex(loc, indices.back(), offset, rewriter);
  return indices;
}

// Match the exact native KAI graph used by the ordinary Triton Q4 prefill
// kernel.  Physical LHS panels are [K8, M4, 8] and RHS bytes are
// [K8-segment, N4, packed-K8].  The logical dot remains M4xK32 by K32xN4.
static bool matchSplatIntegerConstant(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto elements =
      constant ? dyn_cast<DenseIntElementsAttr>(constant.getValue()) : nullptr;
  if (!elements || !elements.isSplat())
    return false;
  APInt splat = elements.getSplatValue<APInt>();
  return splat.getBitWidth() <= 64 && splat.getSExtValue() == expected;
}

static bool matchPackedQ4M4Dot(cpu::DotOp op, PackedQ4M4Dot &candidate) {
  auto lhsTy = dyn_cast<VectorType>(op.getA().getType());
  auto rhsTy = dyn_cast<VectorType>(op.getB().getType());
  auto accTy = dyn_cast<VectorType>(op.getC().getType());
  if (!lhsTy || !rhsTy || !accTy ||
      lhsTy.getShape() != ArrayRef<int64_t>({4, 32}) ||
      rhsTy.getShape() != ArrayRef<int64_t>({32, 4}) ||
      accTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
      !lhsTy.getElementType().isInteger(8) ||
      !rhsTy.getElementType().isInteger(8) ||
      !accTy.getElementType().isInteger(32))
    return false;

  auto finalShape = op.getB().getDefiningOp<vector::ShapeCastOp>();
  if (!finalShape)
    return false;
  auto logicalTranspose =
      finalShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!logicalTranspose ||
      logicalTranspose.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return false;
  auto innerShape =
      logicalTranspose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!innerShape)
    return false;
  auto interleave =
      innerShape.getSource().getDefiningOp<vector::InterleaveOp>();
  if (!interleave)
    return false;
  auto lowShift = interleave.getLhs().getDefiningOp<arith::ShLIOp>();
  auto highMask = interleave.getRhs().getDefiningOp<arith::AndIOp>();
  if (!lowShift || !highMask || lowShift.getLhs() != highMask.getLhs() ||
      !matchSplatIntegerConstant(lowShift.getRhs(), 4) ||
      !matchSplatIntegerConstant(highMask.getRhs(), -16))
    return false;

  Value packed = lowShift.getLhs();
  auto packedTy = dyn_cast<VectorType>(packed.getType());
  if (!packedTy || packedTy.getShape() != ArrayRef<int64_t>({16, 4}) ||
      !packedTy.getElementType().isInteger(8))
    return false;
  auto packedShape = packed.getDefiningOp<vector::ShapeCastOp>();
  if (!packedShape)
    return false;
  auto packedTranspose =
      packedShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!packedTranspose ||
      packedTranspose.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return false;
  Value physicalRhs = packedTranspose.getVector();
  auto physicalRhsTy = dyn_cast<VectorType>(physicalRhs.getType());
  if (!physicalRhsTy ||
      physicalRhsTy.getShape() != ArrayRef<int64_t>({2, 4, 8}) ||
      !physicalRhsTy.getElementType().isInteger(8))
    return false;

  auto lhsShape = op.getA().getDefiningOp<vector::ShapeCastOp>();
  if (!lhsShape)
    return false;
  auto halvesTranspose =
      lhsShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!halvesTranspose ||
      halvesTranspose.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return false;
  auto halvesShape =
      halvesTranspose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!halvesShape)
    return false;
  vector::TransposeOp physicalTranspose =
      halvesShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!physicalTranspose) {
    auto sequentialShape =
        halvesShape.getSource().getDefiningOp<vector::ShapeCastOp>();
    if (sequentialShape)
      physicalTranspose =
          sequentialShape.getSource().getDefiningOp<vector::TransposeOp>();
  }
  if (!physicalTranspose ||
      physicalTranspose.getPermutation() != ArrayRef<int64_t>({1, 0, 2}))
    return false;
  Value physicalLhs = physicalTranspose.getVector();
  auto physicalLhsTy = dyn_cast<VectorType>(physicalLhs.getType());
  if (!physicalLhsTy ||
      physicalLhsTy.getShape() != ArrayRef<int64_t>({4, 4, 8}) ||
      !physicalLhsTy.getElementType().isInteger(8))
    return false;

  MemBuffer lhsPacked = findFlatVectorLoadBuffer(physicalLhs);
  MemBuffer rhsPacked = findFlatVectorLoadBuffer(physicalRhs);
  if (lhsPacked.empty() || rhsPacked.empty() || lhsPacked.indices.empty() ||
      rhsPacked.indices.empty())
    return false;
  candidate = {op, lhsPacked, rhsPacked};
  return true;
}

static Value getOtherMulOperand(arith::MulFOp op, Value known) {
  if (op.getLhs() == known)
    return op.getRhs();
  if (op.getRhs() == known)
    return op.getLhs();
  return {};
}

static bool matchQ4ScaleBroadcast(Value value, bool rowScale, Value &scale) {
  auto broadcast = value.getDefiningOp<vector::BroadcastOp>();
  if (!broadcast)
    return false;
  auto shape = broadcast.getSource().getDefiningOp<vector::ShapeCastOp>();
  // Triton 3.7 canonicalizes a [4] RHS scale directly to broadcast<4x4>,
  // while the row scale still carries the explicit [4,1] shape cast.
  if (!shape && !rowScale) {
    auto resultTy = dyn_cast<VectorType>(broadcast.getType());
    auto scaleTy = dyn_cast<VectorType>(broadcast.getSource().getType());
    if (!resultTy || resultTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
        !scaleTy || scaleTy.getShape() != ArrayRef<int64_t>({4}) ||
        !scaleTy.getElementType().isF32())
      return false;
    scale = broadcast.getSource();
    return true;
  }
  if (!shape)
    return false;
  auto shapedTy = dyn_cast<VectorType>(shape.getType());
  auto scaleTy = dyn_cast<VectorType>(shape.getSource().getType());
  SmallVector<int64_t, 2> expected =
      rowScale ? SmallVector<int64_t, 2>{4, 1} : SmallVector<int64_t, 2>{1, 4};
  if (!shapedTy || shapedTy.getShape() != ArrayRef<int64_t>(expected) ||
      !scaleTy || scaleTy.getShape() != ArrayRef<int64_t>({4}) ||
      !scaleTy.getElementType().isF32())
    return false;
  scale = shape.getSource();
  return true;
}

static bool matchQ4IntegerBroadcast(Value value, bool rowValue, Value &source) {
  auto broadcast = value.getDefiningOp<vector::BroadcastOp>();
  if (!broadcast)
    return false;
  auto shape = broadcast.getSource().getDefiningOp<vector::ShapeCastOp>();
  if (!shape && !rowValue) {
    auto resultTy = dyn_cast<VectorType>(broadcast.getType());
    auto sourceTy = dyn_cast<VectorType>(broadcast.getSource().getType());
    if (!resultTy || resultTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
        !sourceTy || sourceTy.getShape() != ArrayRef<int64_t>({4}) ||
        !sourceTy.getElementType().isInteger(32))
      return false;
    source = broadcast.getSource();
    return true;
  }
  if (!shape)
    return false;
  auto shapedTy = dyn_cast<VectorType>(shape.getType());
  auto sourceTy = dyn_cast<VectorType>(shape.getSource().getType());
  SmallVector<int64_t, 2> expected =
      rowValue ? SmallVector<int64_t, 2>{4, 1} : SmallVector<int64_t, 2>{1, 4};
  if (!shapedTy || shapedTy.getShape() != ArrayRef<int64_t>(expected) ||
      !sourceTy || sourceTy.getShape() != ArrayRef<int64_t>({4}) ||
      !sourceTy.getElementType().isInteger(32))
    return false;
  source = shape.getSource();
  return true;
}

static bool matchPackedQ4M4Epilogue(PackedQ4M4Dot dot, scf::ForOp loop,
                                    PackedQ4M4Epilogue &epilogue,
                                    int64_t &iterIndex) {
  if (!dot.op->hasOneUse())
    return false;
  Value integerResult = dot.op.getResult();
  Value lhsZeroPoint;
  Value rhsSum;
  if (auto subtract = dyn_cast<arith::SubIOp>(*dot.op->getUsers().begin())) {
    if (subtract.getLhs() != dot.op.getResult() || !subtract->hasOneUse())
      return false;
    auto correction = subtract.getRhs().getDefiningOp<arith::MulIOp>();
    if (!correction || !correction->hasOneUse())
      return false;
    Value rowSource;
    Value colSource;
    bool direct =
        matchQ4IntegerBroadcast(correction.getLhs(), true, rowSource) &&
        matchQ4IntegerBroadcast(correction.getRhs(), false, colSource);
    bool swapped =
        matchQ4IntegerBroadcast(correction.getRhs(), true, rowSource) &&
        matchQ4IntegerBroadcast(correction.getLhs(), false, colSource);
    if (!direct && !swapped)
      return false;
    lhsZeroPoint = rowSource;
    rhsSum = colSource;
    integerResult = subtract.getResult();
  }
  auto conversion =
      dyn_cast<arith::SIToFPOp>(*integerResult.getUsers().begin());
  if (!conversion || !conversion->hasOneUse())
    return false;
  auto fixedScale = dyn_cast<arith::MulFOp>(*conversion->getUsers().begin());
  if (!fixedScale || !fixedScale->hasOneUse())
    return false;
  Value fixedScaleValue = getOtherMulOperand(fixedScale, conversion.getOut());
  auto constant = fixedScaleValue.getDefiningOp<arith::ConstantOp>();
  auto scaleAttr =
      constant ? dyn_cast<DenseFPElementsAttr>(constant.getValue()) : nullptr;
  if (!scaleAttr || !scaleAttr.isSplat() ||
      scaleAttr.getSplatValue<APFloat>().convertToDouble() != 0.0625)
    return false;

  auto firstScaleMul = dyn_cast<arith::MulFOp>(*fixedScale->getUsers().begin());
  if (!firstScaleMul || !firstScaleMul->hasOneUse())
    return false;
  Value lhsScale;
  Value rhsScale;
  arith::MulFOp contribution;
  Value firstOther = getOtherMulOperand(firstScaleMul, fixedScale.getResult());
  if (matchQ4ScaleBroadcast(firstOther, true, lhsScale)) {
    // Non-reassociated source: ((dot / 16) * lhsScale) * rhsScale.
    auto rhsMul = dyn_cast<arith::MulFOp>(*firstScaleMul->getUsers().begin());
    if (!rhsMul || !rhsMul->hasOneUse() ||
        !matchQ4ScaleBroadcast(
            getOtherMulOperand(rhsMul, firstScaleMul.getResult()), false,
            rhsScale))
      return false;
    contribution = rhsMul;
  } else {
    // Fast-math canonicalization: (dot / 16) * (lhsScale * rhsScale).
    auto scaleProduct = firstOther.getDefiningOp<arith::MulFOp>();
    if (!scaleProduct ||
        !((matchQ4ScaleBroadcast(scaleProduct.getLhs(), true, lhsScale) &&
           matchQ4ScaleBroadcast(scaleProduct.getRhs(), false, rhsScale)) ||
          (matchQ4ScaleBroadcast(scaleProduct.getRhs(), true, lhsScale) &&
           matchQ4ScaleBroadcast(scaleProduct.getLhs(), false, rhsScale))))
      return false;
    contribution = firstScaleMul;
  }
  auto add = dyn_cast<arith::AddFOp>(*contribution->getUsers().begin());
  if (!add || !add->hasOneUse())
    return false;
  Value carried = add.getLhs() == contribution.getResult()   ? add.getRhs()
                  : add.getRhs() == contribution.getResult() ? add.getLhs()
                                                             : Value{};
  auto blockArg = dyn_cast<BlockArgument>(carried);
  if (!blockArg || blockArg.getOwner() != loop.getBody())
    return false;
  iterIndex = blockArg.getArgNumber() - loop.getNumInductionVars();
  auto yield = dyn_cast<scf::YieldOp>(*add->getUsers().begin());
  if (!yield || iterIndex < 0 ||
      static_cast<unsigned>(iterIndex) >= yield.getNumOperands() ||
      yield.getOperand(iterIndex) != add.getResult())
    return false;
  epilogue = {dot, lhsZeroPoint, rhsSum, lhsScale, rhsScale, add};
  return true;
}

static bool matchPackedQ4M4Loop(scf::ForOp loop,
                                PackedQ4M4LoopCandidate &candidate) {
  // The fusion clones one flat reduction loop.  A nested K32 reduction
  // carries its FP32 results into an enclosing G128 loop; rewriting that
  // shape as a flat loop leaves old region values crossing the new boundary
  // and breaks SSA dominance.  Its individual M4 dots still use the packed
  // Q4 lowering, so reject only the unsafe fusion.
  if (loop->getParentOfType<scf::ForOp>())
    return false;
  if (!hasOnlyReadOrNoEffects(loop))
    return false;
  unsigned panelCount = loop.getNumRegionIterArgs();
  if (panelCount < 1 || panelCount > 4 || loop.getNumResults() != panelCount)
    return false;
  SmallVector<std::pair<int64_t, PackedQ4M4Epilogue>, 4> matches;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    auto dotOp = dyn_cast<cpu::DotOp>(operation);
    if (!dotOp)
      continue;
    PackedQ4M4Dot dot;
    PackedQ4M4Epilogue epilogue;
    int64_t iterIndex = -1;
    if (!matchPackedQ4M4Dot(dotOp, dot) ||
        !matchPackedQ4M4Epilogue(dot, loop, epilogue, iterIndex))
      return false;
    matches.emplace_back(iterIndex, epilogue);
  }
  if (matches.size() != panelCount)
    return false;
  llvm::sort(matches, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (int64_t index = 0; index < panelCount; ++index) {
    if (matches[index].first != index)
      return false;
    candidate.panels.push_back(matches[index].second);
  }
  candidate.loop = loop;
  return true;
}

using PackedQ4M4Accumulators = SmallVector<Value, 4>;

// Emit several M4 panels against the same packed RHS.  Triton source naturally
// reuses one `weight` value for BLOCK_M=8/12/16, but lowering each dot in
// isolation used to reload and unpack those bytes once per panel.  Keeping the
// four RHS vectors live only for one K16 segment mirrors an I8MM microkernel:
// load/unpack B once, then visit every independent M4 row panel.
static SmallVector<PackedQ4M4Accumulators, 4> emitPackedQ4M4DotsSharedRhs(
    Location loc, ArrayRef<MemBuffer> lhsPanels, const MemBuffer &rhsPacked,
    PatternRewriter &rewriter, ArrayRef<PackedQ4M4Accumulators> initial = {}) {
  Type i8Ty = rewriter.getI8Type();
  Type i32Ty = rewriter.getI32Type();
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  StringAttr smmla = StringAttr::get(rewriter.getContext(),
                                     "llvm.aarch64.neon.smmla.v4i32.v16i8");
  Value shift4 = arith::ConstantOp::create(
      rewriter, loc, v16i8Ty,
      DenseElementsAttr::get(v16i8Ty, rewriter.getI8IntegerAttr(4)));
  Value highMask = arith::ConstantOp::create(
      rewriter, loc, v16i8Ty,
      DenseElementsAttr::get(v16i8Ty, rewriter.getI8IntegerAttr(-16)));
  Value zero = arith::ConstantOp::create(rewriter, loc, v4i32Ty,
                                         rewriter.getZeroAttr(v4i32Ty));
  SmallVector<PackedQ4M4Accumulators, 4> panelAcc;
  panelAcc.reserve(lhsPanels.size());
  assert((initial.empty() || initial.size() == lhsPanels.size()) &&
         "expected one native accumulator tile per M4 panel");
  for (size_t panel = 0; panel < lhsPanels.size(); ++panel) {
    if (initial.empty())
      panelAcc.emplace_back(4, zero);
    else
      panelAcc.push_back(initial[panel]);
  }

  for (int64_t segment = 0; segment < 2; ++segment) {
    SmallVector<Value, 2> rhsLow;
    SmallVector<Value, 2> rhsHigh;
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value packed = vector::LoadOp::create(
          rewriter, loc, v16i8Ty, rhsPacked.memRef,
          getFlatIndices(loc, rhsPacked, segment * 32 + colPair * 16,
                         rewriter));
      rhsLow.push_back(arith::ShLIOp::create(rewriter, loc, packed, shift4));
      rhsHigh.push_back(arith::AndIOp::create(rewriter, loc, packed, highMask));
    }
    for (size_t panel = 0; panel < lhsPanels.size(); ++panel) {
      const MemBuffer &lhsPacked = lhsPanels[panel];
      for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
        Value lhsLow = vector::LoadOp::create(
            rewriter, loc, v16i8Ty, lhsPacked.memRef,
            getFlatIndices(loc, lhsPacked, segment * 32 + rowPair * 16,
                           rewriter));
        Value lhsHigh = vector::LoadOp::create(
            rewriter, loc, v16i8Ty, lhsPacked.memRef,
            getFlatIndices(loc, lhsPacked, (segment + 2) * 32 + rowPair * 16,
                           rewriter));
        for (int64_t colPair = 0; colPair < 2; ++colPair) {
          int64_t index = rowPair * 2 + colPair;
          Value &acc = panelAcc[panel][index];
          acc = LLVM::CallIntrinsicOp::create(
                    rewriter, loc, v4i32Ty, smmla,
                    ValueRange{acc, lhsLow, rhsLow[colPair]})
                    .getResult(0);
          acc = LLVM::CallIntrinsicOp::create(
                    rewriter, loc, v4i32Ty, smmla,
                    ValueRange{acc, lhsHigh, rhsHigh[colPair]})
                    .getResult(0);
        }
      }
    }
  }
  return panelAcc;
}

static SmallVector<Value, 4> emitPackedQ4M4Dot(Location loc,
                                               const MemBuffer &lhsPacked,
                                               const MemBuffer &rhsPacked,
                                               PatternRewriter &rewriter) {
  SmallVector<MemBuffer, 1> lhsPanels{lhsPacked};
  return emitPackedQ4M4DotsSharedRhs(loc, lhsPanels, rhsPacked, rewriter)
      .front();
}

static SmallVector<Value, 4> formPackedQ4M4Rows(Location loc,
                                                ValueRange tileAccumulators,
                                                PatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto tile2x2Ty = VectorType::get({2, 2}, i32Ty);
  SmallVector<Value, 4> rows;
  for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
    Value left = vector::ShapeCastOp::create(rewriter, loc, tile2x2Ty,
                                             tileAccumulators[rowPair * 2]);
    Value right = vector::ShapeCastOp::create(
        rewriter, loc, tile2x2Ty, tileAccumulators[rowPair * 2 + 1]);
    for (int64_t row = 0; row < 2; ++row) {
      Value leftRow = vector::ExtractOp::create(rewriter, loc, left, row);
      Value rightRow = vector::ExtractOp::create(rewriter, loc, right, row);
      Value combined = arith::ConstantOp::create(rewriter, loc, v4i32Ty,
                                                 rewriter.getZeroAttr(v4i32Ty));
      combined = vector::InsertStridedSliceOp::create(
          rewriter, loc, leftRow, combined, ArrayRef<int64_t>{0},
          ArrayRef<int64_t>{1});
      combined = vector::InsertStridedSliceOp::create(
          rewriter, loc, rightRow, combined, ArrayRef<int64_t>{2},
          ArrayRef<int64_t>{1});
      rows.push_back(combined);
    }
  }
  return rows;
}

// Convert Triton's logical M4xN4 accumulator into the four native 2x2 I8MM
// accumulator registers.  Feeding these values directly to SMMLA is crucial:
// computing a fresh tile from zero and adding the carried value afterwards
// keeps two complete M16 accumulators live and necessarily spills on Neon.
static PackedQ4M4Accumulators
splitPackedQ4M4Accumulator(Location loc, Value accumulator,
                           PatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  PackedQ4M4Accumulators tiles;
  for (int64_t rowPair = 0; rowPair < 2; ++rowPair) {
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value tile = vector::ExtractStridedSliceOp::create(
          rewriter, loc, accumulator,
          ArrayRef<int64_t>{rowPair * 2, colPair * 2}, ArrayRef<int64_t>{2, 2},
          ArrayRef<int64_t>{1, 1});
      tiles.push_back(
          vector::ShapeCastOp::create(rewriter, loc, v4i32Ty, tile));
    }
  }
  return tiles;
}

// Lower one standalone packed-Q4 dot.  The established groupwise path fuses
// the complete FP32 accumulation loop below.  A native G128 source instead
// rolls four K32 dots into one integer accumulator before applying the common
// scale and zero-point correction, so each dot carries the preceding integer
// tile rather than a literal zero.  Preserve that C operand explicitly.
static LogicalResult convertPackedQ4M4Dot(PackedQ4M4Dot &candidate,
                                          PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  auto resultTy = cast<VectorType>(op.getC().getType());
  SmallVector<Value, 4> tileAcc = emitPackedQ4M4Dot(
      loc, candidate.lhsPacked, candidate.rhsPacked, rewriter);
  SmallVector<Value, 4> rows = formPackedQ4M4Rows(loc, tileAcc, rewriter);
  Value result = arith::ConstantOp::create(rewriter, loc, resultTy,
                                           rewriter.getZeroAttr(resultTy));
  for (int64_t row = 0; row < 4; ++row)
    result = vector::InsertOp::create(rewriter, loc, rows[row], result, row);
  result = arith::AddIOp::create(rewriter, loc, result, op.getC());
  rewriter.replaceOp(op, result);
  return success();
}

static bool samePackedBuffer(const MemBuffer &lhs, const MemBuffer &rhs) {
  return lhs.memRef == rhs.memRef && llvm::equal(lhs.indices, rhs.indices);
}

static bool canInsertSharedQ4DotBefore(ArrayRef<PackedQ4M4Dot> earlier,
                                       cpu::DotOp insertionPoint) {
  Operation *insertion = insertionPoint.getOperation();
  return llvm::all_of(earlier, [&](const PackedQ4M4Dot &candidate) {
    return llvm::all_of(candidate.op->getUsers(), [&](Operation *user) {
      if (user->getBlock() != insertion->getBlock())
        return false;
      return user == insertion || insertion->isBeforeInBlock(user);
    });
  });
}

// G128 carries four K32 integer dots before applying one common scale.  For
// BLOCK_M>4 those standalone dots have distinct LHS panels but the exact same
// RHS buffer.  Lower them as one microkernel fragment so RHS loads and nibble
// unpack are not duplicated.
static LogicalResult
convertPackedQ4M4DotGroup(ArrayRef<PackedQ4M4Dot> candidates,
                          PatternRewriter &rewriter) {
  if (candidates.empty())
    return failure();
  Operation *parent = candidates.front().op->getParentOp();
  const MemBuffer &rhsPacked = candidates.front().rhsPacked;
  SmallVector<MemBuffer, 4> lhsPanels;
  lhsPanels.reserve(candidates.size());
  for (const PackedQ4M4Dot &candidate : candidates) {
    if (candidate.op->getParentOp() != parent ||
        !samePackedBuffer(candidate.rhsPacked, rhsPacked))
      return failure();
    lhsPanels.push_back(candidate.lhsPacked);
  }

  cpu::DotOp firstOp = candidates.front().op;
  Location loc = firstOp.getLoc();
  SmallVector<PackedQ4M4Accumulators, 4> allAcc =
      emitPackedQ4M4DotsSharedRhs(loc, lhsPanels, rhsPacked, rewriter);
  for (auto [candidate, tileAcc] : llvm::zip(candidates, allAcc)) {
    cpu::DotOp op = candidate.op;
    auto resultTy = cast<VectorType>(op.getC().getType());
    SmallVector<Value, 4> rows = formPackedQ4M4Rows(loc, tileAcc, rewriter);
    Value result = arith::ConstantOp::create(rewriter, loc, resultTy,
                                             rewriter.getZeroAttr(resultTy));
    for (int64_t row = 0; row < 4; ++row)
      result = vector::InsertOp::create(rewriter, loc, rows[row], result, row);
    result = arith::AddIOp::create(rewriter, loc, result, op.getC());
    rewriter.replaceOp(op, result);
  }
  return success();
}

static bool canClonePackedLoopValue(Value value, scf::ForOp loop) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != loop.getBody())
      return true;
    return arg == loop.getInductionVar();
  }
  Operation *def = value.getDefiningOp();
  if (!def || !loop->isProperAncestor(def))
    return true;
  if (def->getBlock() != loop.getBody() || def->getNumRegions() != 0)
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return canClonePackedLoopValue(operand, loop);
  });
}

static Value clonePackedLoopValue(Value value, scf::ForOp oldLoop,
                                  IRMapping &mapping,
                                  PatternRewriter &rewriter) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    assert(arg.getOwner() != oldLoop.getBody() &&
           "unmapped packed-Q4 loop block argument");
    return value;
  }
  Operation *def = value.getDefiningOp();
  if (!def || !oldLoop->isProperAncestor(def))
    return value;
  for (Value operand : def->getOperands())
    (void)clonePackedLoopValue(operand, oldLoop, mapping, rewriter);
  rewriter.clone(*def, mapping);
  return mapping.lookup(value);
}

static bool
matchPackedQ4M4IntegerLoop(scf::ForOp loop,
                           PackedQ4M4IntegerLoopCandidate &candidate) {
  unsigned panelCount = loop.getNumRegionIterArgs();
  if (panelCount < 1 || panelCount > 4 || loop.getNumResults() != panelCount)
    return false;
  if (!hasOnlyReadOrNoEffects(loop))
    return false;
  auto panelTy =
      VectorType::get({4, 4}, IntegerType::get(loop.getContext(), 32));
  for (auto [init, result] : llvm::zip(loop.getInitArgs(), loop.getResults()))
    if (init.getType() != panelTy || result.getType() != panelTy)
      return false;

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<std::pair<unsigned, PackedQ4M4Dot>, 4> matches;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    auto dot = dyn_cast<cpu::DotOp>(operation);
    if (!dot)
      continue;
    PackedQ4M4Dot packed;
    if (!matchPackedQ4M4Dot(dot, packed))
      return false;
    auto carried = dyn_cast<BlockArgument>(dot.getC());
    if (!carried || carried.getOwner() != loop.getBody() ||
        carried.getArgNumber() == 0)
      return false;
    unsigned index = carried.getArgNumber() - 1;
    if (index >= panelCount || yield.getOperand(index) != dot.getResult())
      return false;
    matches.emplace_back(index, packed);
  }
  if (matches.size() != panelCount)
    return false;
  llvm::sort(matches, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (unsigned index = 0; index < panelCount; ++index) {
    if (matches[index].first != index)
      return false;
    candidate.panels.push_back(matches[index].second);
  }
  candidate.loop = loop;
  return true;
}

// Keep the K32 reduction in the native four-register-per-panel I8MM layout.
// This removes the logical M4xN4 rebuild/split (zip) sequence from every K32
// iteration and, more importantly, never creates a second accumulator tile.
static LogicalResult
convertPackedQ4M4IntegerLoop(PackedQ4M4IntegerLoopCandidate &candidate,
                             PatternRewriter &rewriter) {
  scf::ForOp oldLoop = candidate.loop;
  const MemBuffer &originalRhs = candidate.panels.front().rhsPacked;
  for (const PackedQ4M4Dot &panel : candidate.panels) {
    // Complete every fallible structural check before creating replacement
    // IR. PatternRewriter does not roll back arbitrary mutations when this
    // helper returns failure.
    if (!samePackedBuffer(panel.rhsPacked, originalRhs))
      return failure();
    if (!canClonePackedLoopValue(panel.lhsPacked.memRef, oldLoop) ||
        !canClonePackedLoopValue(panel.rhsPacked.memRef, oldLoop))
      return failure();
    for (Value index : panel.lhsPacked.indices)
      if (!canClonePackedLoopValue(index, oldLoop))
        return failure();
    for (Value index : panel.rhsPacked.indices)
      if (!canClonePackedLoopValue(index, oldLoop))
        return failure();
  }

  Location loc = oldLoop.getLoc();
  int64_t panelCount = candidate.panels.size();
  rewriter.setInsertionPoint(oldLoop);
  SmallVector<Value, 16> initial;
  for (Value panel : oldLoop.getInitArgs()) {
    PackedQ4M4Accumulators tiles =
        splitPackedQ4M4Accumulator(loc, panel, rewriter);
    initial.append(tiles.begin(), tiles.end());
  }
  auto newLoop =
      scf::ForOp::create(rewriter, loc, oldLoop.getLowerBound(),
                         oldLoop.getUpperBound(), oldLoop.getStep(), initial);
  IRMapping mapping;
  mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  rewriter.setInsertionPointToEnd(newLoop.getBody());

  SmallVector<MemBuffer, 4> lhsPanels;
  MemBuffer sharedRhs;
  for (int64_t panel = 0; panel < panelCount; ++panel) {
    MemBuffer lhs = candidate.panels[panel].lhsPacked;
    lhs.memRef = clonePackedLoopValue(lhs.memRef, oldLoop, mapping, rewriter);
    for (Value &index : lhs.indices)
      index = clonePackedLoopValue(index, oldLoop, mapping, rewriter);
    lhsPanels.push_back(lhs);

    MemBuffer rhs = candidate.panels[panel].rhsPacked;
    rhs.memRef = clonePackedLoopValue(rhs.memRef, oldLoop, mapping, rewriter);
    for (Value &index : rhs.indices)
      index = clonePackedLoopValue(index, oldLoop, mapping, rewriter);
    if (panel == 0)
      sharedRhs = rhs;
    else
      assert(samePackedBuffer(rhs, sharedRhs) &&
             "preflighted shared RHS must remain shared after cloning");
  }

  SmallVector<PackedQ4M4Accumulators, 4> carried;
  for (int64_t panel = 0; panel < panelCount; ++panel) {
    PackedQ4M4Accumulators tiles;
    for (int64_t tile = 0; tile < 4; ++tile)
      tiles.push_back(newLoop.getRegionIterArgs()[panel * 4 + tile]);
    carried.push_back(std::move(tiles));
  }
  SmallVector<PackedQ4M4Accumulators, 4> updated =
      emitPackedQ4M4DotsSharedRhs(loc, lhsPanels, sharedRhs, rewriter, carried);
  SmallVector<Value, 16> yielded;
  for (const PackedQ4M4Accumulators &tiles : updated)
    yielded.append(tiles.begin(), tiles.end());
  scf::YieldOp::create(rewriter, loc, yielded);

  rewriter.setInsertionPointAfter(oldLoop);
  auto panelTy = VectorType::get({4, 4}, rewriter.getI32Type());
  for (int64_t panel = 0; panel < panelCount; ++panel) {
    ValueRange native = newLoop.getResults().slice(panel * 4, 4);
    SmallVector<Value, 4> rows = formPackedQ4M4Rows(loc, native, rewriter);
    Value logical = arith::ConstantOp::create(rewriter, loc, panelTy,
                                              rewriter.getZeroAttr(panelTy));
    for (int64_t row = 0; row < 4; ++row)
      logical =
          vector::InsertOp::create(rewriter, loc, rows[row], logical, row);
    rewriter.replaceAllUsesWith(oldLoop.getResult(panel), logical);
  }
  rewriter.eraseOp(oldLoop);
  return success();
}

static LogicalResult convertPackedQ4M4Loop(PackedQ4M4LoopCandidate &candidate,
                                           bool fixedWidth,
                                           PatternRewriter &rewriter) {
  scf::ForOp oldLoop = candidate.loop;
  for (const PackedQ4M4Epilogue &panel : candidate.panels) {
    if (!canClonePackedLoopValue(panel.dot.lhsPacked.memRef, oldLoop) ||
        !canClonePackedLoopValue(panel.dot.rhsPacked.memRef, oldLoop) ||
        (panel.lhsZeroPoint &&
         !canClonePackedLoopValue(panel.lhsZeroPoint, oldLoop)) ||
        (panel.rhsSum && !canClonePackedLoopValue(panel.rhsSum, oldLoop)) ||
        !canClonePackedLoopValue(panel.lhsScale, oldLoop) ||
        !canClonePackedLoopValue(panel.rhsScale, oldLoop))
      return failure();
    for (Value index : panel.dot.lhsPacked.indices)
      if (!canClonePackedLoopValue(index, oldLoop))
        return failure();
    for (Value index : panel.dot.rhsPacked.indices)
      if (!canClonePackedLoopValue(index, oldLoop))
        return failure();
  }

  Location loc = oldLoop.getLoc();
  Type f32Ty = rewriter.getF32Type();
  Type i32Ty = rewriter.getI32Type();
  auto v4f32Ty = VectorType::get({4}, f32Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  rewriter.setInsertionPoint(oldLoop);
  SmallVector<Value, 16> initialRows;
  for (Value panel : oldLoop.getInitArgs()) {
    for (int64_t row = 0; row < 4; ++row)
      initialRows.push_back(
          vector::ExtractOp::create(rewriter, loc, panel, row));
  }
  int64_t panelCount = candidate.panels.size();
  // Fixed-width Neon has 32 vector registers.  Carrying all sixteen M16
  // FP32 output rows as scf.for iter_args while materializing the I8MM dot
  // leaves too little room for packed operands and forces LLVM to spill in
  // the hot loop.  Keep those long-lived rows in an explicit, cache-aligned
  // local slot instead.  This mirrors a conventional 16x4 microkernel: the
  // INT32 SMMLA fragment owns the register file and each scaled G128 result
  // is retired to the local FP32 accumulation tile.
  const bool bufferFixedM16 = fixedWidth && panelCount == 4;
  MemBuffer rowBuffer;
  if (bufferFixedM16) {
    Operation *allocaPoint = oldLoop;
    while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
      allocaPoint = allocaPoint->getParentOp();
    auto allRowsTy = VectorType::get({panelCount * 4, 4}, f32Ty);
    rowBuffer = allocateTmpBufferStack(loc, allRowsTy, allocaPoint, rewriter);
    for (int64_t row = 0; row < panelCount * 4; ++row)
      vector::StoreOp::create(rewriter, loc, initialRows[row], rowBuffer.memRef,
                              get2DIndices(loc, rowBuffer, row, 0, rewriter));
  }
  Value fractionalBits = arith::ConstantIntOp::create(rewriter, loc, 4, 32);
  StringAttr scvtf =
      StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.vcvtfxs2fp");

  rewriter.setInsertionPoint(oldLoop);
  ValueRange loopInit = bufferFixedM16 ? ValueRange{} : ValueRange(initialRows);
  auto newLoop =
      scf::ForOp::create(rewriter, loc, oldLoop.getLowerBound(),
                         oldLoop.getUpperBound(), oldLoop.getStep(), loopInit);
  // scf.for with no iter_args is built with an empty terminator.  The M16
  // buffered form emits the rewritten body itself, so remove that terminator
  // before appending operations and the final yield.
  if (bufferFixedM16)
    rewriter.eraseOp(newLoop.getBody()->getTerminator());
  IRMapping mapping;
  mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  rewriter.setInsertionPointToEnd(newLoop.getBody());
  SmallVector<Value, 16> outputRows;
  if (!bufferFixedM16)
    outputRows.append(newLoop.getRegionIterArgs().begin(),
                      newLoop.getRegionIterArgs().end());

  for (int64_t panelOrder = 0; panelOrder < panelCount; ++panelOrder) {
    // LLVM 20's two-address coalescer is sensitive to update order. SVE M16
    // uses natural order, fixed-width M16 and M8 use reverse order, and M12
    // uses the measured rotation.
    int64_t panelIndex =
        panelCount == 3 ? (panelOrder + 2) % 3
        : panelCount == 4
            ? (fixedWidth ? panelCount - 1 - panelOrder : panelOrder)
            : panelCount - 1 - panelOrder;
    const PackedQ4M4Epilogue &panel = candidate.panels[panelIndex];
    MemBuffer lhsPacked = panel.dot.lhsPacked;
    lhsPacked.memRef =
        clonePackedLoopValue(lhsPacked.memRef, oldLoop, mapping, rewriter);
    for (Value &index : lhsPacked.indices)
      index = clonePackedLoopValue(index, oldLoop, mapping, rewriter);
    MemBuffer rhsPacked = panel.dot.rhsPacked;
    rhsPacked.memRef =
        clonePackedLoopValue(rhsPacked.memRef, oldLoop, mapping, rewriter);
    for (Value &index : rhsPacked.indices)
      index = clonePackedLoopValue(index, oldLoop, mapping, rewriter);
    Value lhsScale =
        clonePackedLoopValue(panel.lhsScale, oldLoop, mapping, rewriter);
    Value rhsScale =
        clonePackedLoopValue(panel.rhsScale, oldLoop, mapping, rewriter);
    Value lhsZeroPoint = panel.lhsZeroPoint
                             ? clonePackedLoopValue(panel.lhsZeroPoint, oldLoop,
                                                    mapping, rewriter)
                             : Value{};
    Value rhsSum = panel.rhsSum ? clonePackedLoopValue(panel.rhsSum, oldLoop,
                                                       mapping, rewriter)
                                : Value{};

    SmallVector<Value, 4> tileAcc =
        emitPackedQ4M4Dot(loc, lhsPacked, rhsPacked, rewriter);
    SmallVector<Value, 4> dotRows = formPackedQ4M4Rows(loc, tileAcc, rewriter);
    for (int64_t row = 0; row < 4; ++row) {
      if (lhsZeroPoint) {
        Value zpScalar =
            vector::ExtractOp::create(rewriter, loc, lhsZeroPoint, row);
        Value zpBroadcast =
            vector::BroadcastOp::create(rewriter, loc, v4i32Ty, zpScalar);
        Value correction =
            arith::MulIOp::create(rewriter, loc, zpBroadcast, rhsSum);
        dotRows[row] =
            arith::SubIOp::create(rewriter, loc, dotRows[row], correction);
      }
      Value converted = LLVM::CallIntrinsicOp::create(
                            rewriter, loc, v4f32Ty, scvtf,
                            ValueRange{dotRows[row], fractionalBits})
                            .getResult(0);
      Value lhsScalar = vector::ExtractOp::create(rewriter, loc, lhsScale, row);
      Value lhsBroadcast =
          vector::BroadcastOp::create(rewriter, loc, v4f32Ty, lhsScalar);
      Value scale =
          arith::MulFOp::create(rewriter, loc, lhsBroadcast, rhsScale);
      Value contribution =
          arith::MulFOp::create(rewriter, loc, converted, scale);
      int64_t outputIndex = panelIndex * 4 + row;
      Value oldRow =
          bufferFixedM16
              ? vector::LoadOp::create(
                    rewriter, loc, v4f32Ty, rowBuffer.memRef,
                    get2DIndices(loc, rowBuffer, outputIndex, 0, rewriter))
              : outputRows[outputIndex];
      Value updated =
          arith::AddFOp::create(rewriter, loc, oldRow, contribution);
      if (bufferFixedM16) {
        vector::StoreOp::create(
            rewriter, loc, updated, rowBuffer.memRef,
            get2DIndices(loc, rowBuffer, outputIndex, 0, rewriter));
      } else {
        outputRows[outputIndex] = updated;
      }
    }
  }
  scf::YieldOp::create(rewriter, loc,
                       bufferFixedM16 ? ValueRange{} : ValueRange(outputRows));

  rewriter.setInsertionPointAfter(oldLoop);
  auto panelTy = VectorType::get({4, 4}, f32Ty);
  for (int64_t panel = 0; panel < panelCount; ++panel) {
    Value rebuilt = arith::ConstantOp::create(rewriter, loc, panelTy,
                                              rewriter.getZeroAttr(panelTy));
    for (int64_t row = 0; row < 4; ++row) {
      Value rowValue =
          bufferFixedM16
              ? vector::LoadOp::create(
                    rewriter, loc, v4f32Ty, rowBuffer.memRef,
                    get2DIndices(loc, rowBuffer, panel * 4 + row, 0, rewriter))
              : newLoop.getResult(panel * 4 + row);
      rebuilt = vector::InsertOp::create(rewriter, loc, rowValue, rebuilt, row);
    }
    rewriter.replaceAllUsesWith(oldLoop.getResult(panel), rebuilt);
  }
  rewriter.eraseOp(oldLoop);
  return success();
}

// Match the exact ordinary join/permute graph that exposes KleidiAI's W8
// prefill layout.  Each leaf is one physical M4/K32 panel and the RHS is one
// physical N4/K32 panel.  The Triton source remains a portable tl.dot.
static bool matchKaiW8LhsLeaf(Value value, Value &physical) {
  auto logicalShape = value.getDefiningOp<vector::ShapeCastOp>();
  if (!logicalShape)
    return false;
  auto transpose =
      logicalShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({1, 0, 2}))
    return false;
  auto physicalShape =
      transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!physicalShape)
    return false;
  physical = physicalShape.getSource();
  auto physicalTy = dyn_cast<VectorType>(physical.getType());
  return physicalTy && physicalTy.getShape() == ArrayRef<int64_t>({128}) &&
         physicalTy.getElementType().isInteger(8);
}

static bool matchKaiW8LhsPair(Value value, Value &low, Value &high) {
  auto finalShape = value.getDefiningOp<vector::ShapeCastOp>();
  if (!finalShape)
    return false;
  auto transpose = finalShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>({2, 0, 1}))
    return false;
  auto joinShape = transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!joinShape)
    return false;
  auto join = joinShape.getSource().getDefiningOp<vector::InterleaveOp>();
  if (!join)
    return false;
  low = join.getLhs();
  high = join.getRhs();
  return true;
}

static bool matchPackedW8Prefill(cpu::DotOp op,
                                 PackedW8PrefillCandidate &candidate) {
  auto lhsTy = dyn_cast<VectorType>(op.getA().getType());
  auto rhsTy = dyn_cast<VectorType>(op.getB().getType());
  auto accTy = dyn_cast<VectorType>(op.getC().getType());
  if (!lhsTy || !rhsTy || !accTy || lhsTy.getRank() != 2 ||
      (lhsTy.getDimSize(0) != 4 && lhsTy.getDimSize(0) != 8 &&
       lhsTy.getDimSize(0) != 16) ||
      lhsTy.getDimSize(1) != 32 ||
      rhsTy.getShape() != ArrayRef<int64_t>({32, 4}) || accTy.getRank() != 2 ||
      accTy.getDimSize(0) != lhsTy.getDimSize(0) || accTy.getDimSize(1) != 4 ||
      !lhsTy.getElementType().isInteger(8) ||
      !rhsTy.getElementType().isInteger(8) ||
      !accTy.getElementType().isInteger(32))
    return false;

  SmallVector<Value, 4> logicalPanels;
  int64_t mSize = lhsTy.getDimSize(0);
  if (mSize == 4) {
    logicalPanels.push_back(op.getA());
  } else if (mSize == 8) {
    Value lhs0;
    Value lhs1;
    if (!matchKaiW8LhsPair(op.getA(), lhs0, lhs1))
      return false;
    logicalPanels.push_back(lhs0);
    logicalPanels.push_back(lhs1);
  } else {
    Value lhs01;
    Value lhs23;
    if (!matchKaiW8LhsPair(op.getA(), lhs01, lhs23))
      return false;
    Value lhs0;
    Value lhs1;
    Value lhs2;
    Value lhs3;
    if (!matchKaiW8LhsPair(lhs01, lhs0, lhs1) ||
        !matchKaiW8LhsPair(lhs23, lhs2, lhs3))
      return false;
    logicalPanels.push_back(lhs0);
    logicalPanels.push_back(lhs1);
    logicalPanels.push_back(lhs2);
    logicalPanels.push_back(lhs3);
  }
  SmallVector<Value, 4> panels;
  panels.reserve(logicalPanels.size());
  for (Value logical : logicalPanels) {
    Value physical;
    if (!matchKaiW8LhsLeaf(logical, physical))
      return false;
    panels.push_back(physical);
  }

  auto rhsShape = op.getB().getDefiningOp<vector::ShapeCastOp>();
  if (!rhsShape)
    return false;
  auto rhsTranspose = rhsShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!rhsTranspose ||
      rhsTranspose.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return false;
  auto rhsPhysicalShape =
      rhsTranspose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!rhsPhysicalShape)
    return false;
  Value rhsPanel = rhsPhysicalShape.getSource();
  auto rhsPanelTy = dyn_cast<VectorType>(rhsPanel.getType());
  if (!rhsPanelTy || rhsPanelTy.getShape() != ArrayRef<int64_t>({128}) ||
      !rhsPanelTy.getElementType().isInteger(8))
    return false;

  candidate = {op, panels, rhsPanel};
  return true;
}

static SmallVector<Value, 16>
emitPackedW8PrefillSmmla(Location loc, ArrayRef<Value> lhsPanels,
                         Value rhsPanel, ValueRange initialAccumulators,
                         bool fixedWidth, PatternRewriter &rewriter) {
  Type i8Ty = rewriter.getI8Type();
  Type i32Ty = rewriter.getI32Type();
  auto panelTy = VectorType::get({4, 4, 8}, i8Ty);
  auto packed2x8Ty = VectorType::get({2, 8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Type nxv16i8Ty = VectorType::get({16}, i8Ty, {true});
  Type nxv4i32Ty = VectorType::get({4}, i32Ty, {true});
  Type operandTy = fixedWidth ? Type(v16i8Ty) : nxv16i8Ty;
  Type accumulatorTy = fixedWidth ? Type(v4i32Ty) : nxv4i32Ty;
  StringAttr smmla = StringAttr::get(
      rewriter.getContext(), fixedWidth ? "llvm.aarch64.neon.smmla.v4i32.v16i8"
                                        : "llvm.aarch64.sve.smmla.nxv4i32");

  SmallVector<Value, 4> lhsPhysical;
  int64_t panelCount = lhsPanels.size();
  lhsPhysical.resize(panelCount);
  for (int64_t panel = 0; panel < panelCount; ++panel)
    lhsPhysical[panel] =
        vector::ShapeCastOp::create(rewriter, loc, panelTy, lhsPanels[panel]);
  Value rhsPhysical =
      vector::ShapeCastOp::create(rewriter, loc, panelTy, rhsPanel);
  SmallVector<Value, 16> accumulators(initialAccumulators.begin(),
                                      initialAccumulators.end());

  int64_t rowPairs = panelCount * 2;
  assert(initialAccumulators.size() == static_cast<size_t>(rowPairs * 2) &&
         "packed W8 accumulator count does not match LHS panels");
  for (int64_t kStep = 0; kStep < 4; ++kStep) {
    SmallVector<Value, 8> lhsPacked;
    for (int64_t rowPair = 0; rowPair < rowPairs; ++rowPair) {
      int64_t panel = rowPair / 2;
      int64_t firstRow = (rowPair % 2) * 2;
      Value rows = vector::ExtractStridedSliceOp::create(
          rewriter, loc, lhsPhysical[panel],
          ArrayRef<int64_t>{kStep, firstRow, 0}, ArrayRef<int64_t>{1, 2, 8},
          ArrayRef<int64_t>{1, 1, 1});
      Value packed =
          vector::ShapeCastOp::create(rewriter, loc, packed2x8Ty, rows);
      Value flat = vector::ShapeCastOp::create(rewriter, loc, v16i8Ty, packed);
      lhsPacked.push_back(
          fixedWidth ? flat
                     : insertFixedInScalable(loc, flat, operandTy, rewriter));
    }

    SmallVector<Value, 2> rhsPacked;
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value columns = vector::ExtractStridedSliceOp::create(
          rewriter, loc, rhsPhysical, ArrayRef<int64_t>{kStep, colPair * 2, 0},
          ArrayRef<int64_t>{1, 2, 8}, ArrayRef<int64_t>{1, 1, 1});
      Value packed =
          vector::ShapeCastOp::create(rewriter, loc, packed2x8Ty, columns);
      Value flat = vector::ShapeCastOp::create(rewriter, loc, v16i8Ty, packed);
      rhsPacked.push_back(
          fixedWidth ? flat
                     : insertFixedInScalable(loc, flat, operandTy, rewriter));
    }

    for (int64_t rowPair = 0; rowPair < rowPairs; ++rowPair) {
      for (int64_t colPair = 0; colPair < 2; ++colPair) {
        int64_t index = rowPair * 2 + colPair;
        accumulators[index] =
            LLVM::CallIntrinsicOp::create(rewriter, loc, accumulatorTy, smmla,
                                          ValueRange{accumulators[index],
                                                     lhsPacked[rowPair],
                                                     rhsPacked[colPair]})
                .getResult(0);
      }
    }
  }
  return accumulators;
}

static SmallVector<Value, 16>
splitPackedW8Accumulator(Location loc, Value accumulator, bool fixedWidth,
                         PatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  auto accTy = cast<VectorType>(accumulator.getType());
  int64_t mSize = accTy.getDimSize(0);
  assert(accTy.getShape().back() == 4 && mSize % 2 == 0 &&
         "expected packed W8 Mx4 accumulator");
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Type nxv4i32Ty = VectorType::get({4}, i32Ty, {true});
  SmallVector<Value, 16> tiles;
  tiles.reserve(mSize);
  for (int64_t rowPair = 0; rowPair < mSize / 2; ++rowPair) {
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value tile = vector::ExtractStridedSliceOp::create(
          rewriter, loc, accumulator,
          ArrayRef<int64_t>{rowPair * 2, colPair * 2}, ArrayRef<int64_t>{2, 2},
          ArrayRef<int64_t>{1, 1});
      Value flat = vector::ShapeCastOp::create(rewriter, loc, v4i32Ty, tile);
      tiles.push_back(
          fixedWidth ? flat
                     : insertFixedInScalable(loc, flat, nxv4i32Ty, rewriter));
    }
  }
  return tiles;
}

static Value joinPackedW8Accumulator(Location loc, VectorType accTy,
                                     ValueRange accumulators, bool fixedWidth,
                                     PatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  int64_t mSize = accTy.getDimSize(0);
  assert(accTy.getDimSize(1) == 4 && mSize % 2 == 0 &&
         accumulators.size() == static_cast<size_t>(mSize) &&
         "expected one packed W8 accumulator per output row");
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  auto tile2x2Ty = VectorType::get({2, 2}, i32Ty);
  Value result = arith::ConstantOp::create(rewriter, loc, accTy,
                                           rewriter.getZeroAttr(accTy));
  for (int64_t rowPair = 0; rowPair < mSize / 2; ++rowPair) {
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value native = accumulators[rowPair * 2 + colPair];
      Value flat =
          fixedWidth ? native
                     : extractFixedFromScalable(loc, native, v4i32Ty, rewriter);
      Value tile = vector::ShapeCastOp::create(rewriter, loc, tile2x2Ty, flat);
      result = vector::InsertStridedSliceOp::create(
          rewriter, loc, tile, result,
          ArrayRef<int64_t>{rowPair * 2, colPair * 2}, ArrayRef<int64_t>{1, 1});
    }
  }
  return result;
}

// Carry native 2x2 SMMLA accumulators through the K loop and rebuild the
// logical Mx4 result only once.  This also handles M12, whose Triton source
// carries an M8 and an M4 dot as two region iter arguments.
static LogicalResult
convertPackedW8PrefillLoop(PackedW8PrefillCandidate &candidate, bool fixedWidth,
                           PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  auto oldLoop = dyn_cast<scf::ForOp>(op->getParentOp());
  if (!oldLoop || oldLoop.getNumRegionIterArgs() == 0 ||
      oldLoop.getNumResults() != oldLoop.getNumRegionIterArgs() ||
      !hasOnlyReadOrNoEffects(oldLoop))
    return failure();
  auto yield = dyn_cast<scf::YieldOp>(oldLoop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != oldLoop.getNumRegionIterArgs())
    return failure();

  // loop_unroll_factor creates a linear chain
  //
  //   acc_arg -> dot(chunk + 0) -> ... -> dot(chunk + U - 1) -> yield
  //
  // for each region accumulator.  Recover that complete chain so every
  // unrolled copy stays on the native SMMLA path.  Previously the matcher
  // required yield(dot(acc_arg)) and only the first copy was converted;
  // later copies then fell through to the generic vector expansion.
  SmallVector<SmallVector<PackedW8PrefillCandidate, 4>, 4> chains;
  chains.reserve(oldLoop.getNumRegionIterArgs());
  for (auto [index, accArg] : llvm::enumerate(oldLoop.getRegionIterArgs())) {
    SmallVector<PackedW8PrefillCandidate, 4> reverseChain;
    Value current = yield.getOperand(index);
    while (current != accArg) {
      auto dot = current.getDefiningOp<cpu::DotOp>();
      if (!dot || dot->getParentOp() != oldLoop || !dot->hasOneUse())
        return failure();
      PackedW8PrefillCandidate sibling;
      if (!matchPackedW8Prefill(dot, sibling))
        return failure();
      for (Value panel : sibling.lhsPanels)
        if (!canClonePackedLoopValue(panel, oldLoop))
          return failure();
      if (!canClonePackedLoopValue(sibling.rhsPanel, oldLoop))
        return failure();
      reverseChain.push_back(std::move(sibling));
      current = dot.getC();
    }
    if (reverseChain.empty())
      return failure();
    std::reverse(reverseChain.begin(), reverseChain.end());
    size_t panelCount = reverseChain.front().lhsPanels.size();
    if (llvm::any_of(reverseChain, [&](const auto &packed) {
          return packed.lhsPanels.size() != panelCount;
        }))
      return failure();
    chains.push_back(std::move(reverseChain));
  }

  Location loc = op.getLoc();
  rewriter.setInsertionPoint(oldLoop);
  SmallVector<Value, 32> initial;
  SmallVector<size_t, 4> offsets;
  for (Value init : oldLoop.getInitArgs()) {
    offsets.push_back(initial.size());
    SmallVector<Value, 16> tiles =
        splitPackedW8Accumulator(loc, init, fixedWidth, rewriter);
    initial.append(tiles.begin(), tiles.end());
  }
  auto newLoop =
      scf::ForOp::create(rewriter, loc, oldLoop.getLowerBound(),
                         oldLoop.getUpperBound(), oldLoop.getStep(), initial);

  IRMapping mapping;
  mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  rewriter.setInsertionPointToEnd(newLoop.getBody());
  SmallVector<Value, 32> next;
  for (auto [index, chain] : llvm::enumerate(chains)) {
    size_t count = chain.front().lhsPanels.size() * 4;
    ValueRange current =
        newLoop.getRegionIterArgs().slice(offsets[index], count);
    SmallVector<Value, 16> accumulators(current.begin(), current.end());
    for (auto &packed : chain) {
      SmallVector<Value, 4> lhsPanels;
      for (Value panel : packed.lhsPanels)
        lhsPanels.push_back(
            clonePackedLoopValue(panel, oldLoop, mapping, rewriter));
      Value rhsPanel =
          clonePackedLoopValue(packed.rhsPanel, oldLoop, mapping, rewriter);
      accumulators = emitPackedW8PrefillSmmla(
          loc, lhsPanels, rhsPanel, accumulators, fixedWidth, rewriter);
    }
    next.append(accumulators.begin(), accumulators.end());
  }
  scf::YieldOp::create(rewriter, loc, next);

  rewriter.setInsertionPointAfter(newLoop);
  for (auto [index, chain] : llvm::enumerate(chains)) {
    auto accTy = cast<VectorType>(chain.back().op.getC().getType());
    size_t count = chain.front().lhsPanels.size() * 4;
    ValueRange nativeResults =
        newLoop.getResults().slice(offsets[index], count);
    Value result = joinPackedW8Accumulator(loc, accTy, nativeResults,
                                           fixedWidth, rewriter);
    rewriter.replaceAllUsesWith(oldLoop.getResult(index), result);
  }
  rewriter.eraseOp(oldLoop);
  return success();
}

static LogicalResult convertPackedW8Prefill(PackedW8PrefillCandidate &candidate,
                                            bool fixedWidth,
                                            PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  if (succeeded(convertPackedW8PrefillLoop(candidate, fixedWidth, rewriter)))
    return success();
  Location loc = op.getLoc();
  auto accTy = cast<VectorType>(op.getC().getType());
  int64_t mSize = accTy.getDimSize(0);
  Type i32Ty = rewriter.getI32Type();
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Type nxv4i32Ty = VectorType::get({4}, i32Ty, {true});

  SmallVector<Value, 16> accumulators;
  accumulators.reserve(mSize);
  for (int64_t rowPair = 0; rowPair < mSize / 2; ++rowPair) {
    for (int64_t colPair = 0; colPair < 2; ++colPair) {
      Value tile = vector::ExtractStridedSliceOp::create(
          rewriter, loc, op.getC(), ArrayRef<int64_t>{rowPair * 2, colPair * 2},
          ArrayRef<int64_t>{2, 2}, ArrayRef<int64_t>{1, 1});
      Value flat = vector::ShapeCastOp::create(rewriter, loc, v4i32Ty, tile);
      accumulators.push_back(
          fixedWidth ? flat
                     : insertFixedInScalable(loc, flat, nxv4i32Ty, rewriter));
    }
  }
  accumulators =
      emitPackedW8PrefillSmmla(loc, candidate.lhsPanels, candidate.rhsPanel,
                               accumulators, fixedWidth, rewriter);
  Value result =
      joinPackedW8Accumulator(loc, accTy, accumulators, fixedWidth, rewriter);
  rewriter.replaceOp(op, result);
  return success();
}

// Lower an ordinary Triton M=1 int8 tl.dot to a rolled SDOT GEMV.
//
// This deliberately consumes the buffers discovered behind cpu::DotOp
// instead of expanding the complete [1,K] x [K,N] tile into SSA values.
// Consequently the generated LLVM IR retains compact N/K loops even for
// model-sized dimensions.  Four row-major B rows are transposed in registers
// into SDOT's four groups of four K bytes:
//
//   [b(k+0,n+0), b(k+1,n+0), b(k+2,n+0), b(k+3,n+0), ... n+3]
//
// A prepacked variant can later replace those four loads and the shuffle with
// one 16-byte load without changing the accumulator structure.
static LogicalResult convertCandidateM1(SVE2I8MMCandidate &candidate,
                                        PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  auto lhsTy = cast<VectorType>(op.getA().getType());
  auto rhsTy = cast<VectorType>(op.getB().getType());
  auto accTy = cast<VectorType>(op.getC().getType());
  Type i8Ty = lhsTy.getElementType();
  Type i32Ty = accTy.getElementType();
  int64_t kSize = lhsTy.getDimSize(1);
  int64_t nSize = rhsTy.getDimSize(1);

  auto v4i8Ty = VectorType::get({4}, i8Ty);
  auto v8i8Ty = VectorType::get({8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  StringAttr sdot =
      StringAttr::get(op.getContext(), "llvm.aarch64.neon.sdot.v4i32.v16i8");

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  scf::ForOp outerKLoop;
  Value initialAcc = op.getC();
  if (candidate.keepAccInBuffer) {
    outerKLoop = cast<scf::ForOp>(op->getParentOp());
    initialAcc = getInitAccValue(op.getC());
  }
  bool zeroInitialAcc =
      !candidate.keepAccInBuffer && matchPattern(initialAcc, m_Zero());

  MemBuffer accBuf;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    if (candidate.keepAccInBuffer)
      rewriter.setInsertionPoint(outerKLoop);
    accBuf = allocateTmpBufferStack(loc, accTy, allocaPoint, rewriter);
    if (!zeroInitialAcc)
      op_write(initialAcc, accBuf.memRef, accBuf.indices);
  }

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value c4 = arith::ConstantIndexOp::create(rewriter, loc, 4);
  Value nLimit = arith::ConstantIndexOp::create(rewriter, loc, nSize);
  Value kLimit = arith::ConstantIndexOp::create(rewriter, loc, kSize);
  Value zero8 = arith::ConstantOp::create(rewriter, loc, v8i8Ty,
                                          rewriter.getZeroAttr(v8i8Ty));
  Value zero16 = arith::ConstantOp::create(rewriter, loc, v16i8Ty,
                                           rewriter.getZeroAttr(v16i8Ty));
  Value zeroAcc = arith::ConstantOp::create(rewriter, loc, v4i32Ty,
                                            rewriter.getZeroAttr(v4i32Ty));

  constexpr int64_t transposeB[] = {0, 4, 16, 20, 1, 5, 17, 21,
                                    2, 6, 18, 22, 3, 7, 19, 23};

  auto nLoop = scf::ForOp::create(rewriter, loc, c0, nLimit, c4);
  {
    OpBuilder::InsertionGuard nGuard(rewriter);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nBase = nLoop.getInductionVar();
    Value initAcc = zeroAcc;
    if (!zeroInitialAcc) {
      initAcc =
          vector::LoadOp::create(rewriter, loc, v4i32Ty, accBuf.memRef,
                                 get2DIndices(loc, accBuf, 0, nBase, rewriter));
    }

    auto kLoop =
        scf::ForOp::create(rewriter, loc, c0, kLimit, c4, ValueRange{initAcc});
    {
      OpBuilder::InsertionGuard kGuard(rewriter);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value kBase = kLoop.getInductionVar();

      Value a4 = vector::LoadOp::create(
          rewriter, loc, v4i8Ty, candidate.lhsBuf.memRef,
          get2DIndices(loc, candidate.lhsBuf, 0, kBase, rewriter));
      // SDOT needs the same four activation bytes in every 32-bit lane.
      // Express that as a word broadcast rather than an i8 shuffle padded
      // through two zero vectors.  LLVM then selects LD1R/DUP instead of an
      // extend/insert/shuffle sequence.
      Value aScalar = LLVM::BitcastOp::create(rewriter, loc, i32Ty, a4);
      Value aWords =
          vector::BroadcastOp::create(rewriter, loc, v4i32Ty, aScalar);
      Value aBroadcast =
          LLVM::BitcastOp::create(rewriter, loc, v16i8Ty, aWords);

      Value bPacked;
      if (!candidate.rhsPackedN4Buf.empty()) {
        Value packedOffset = arith::MulIOp::create(rewriter, loc, nBase, c4);
        SmallVector<Value> indices(candidate.rhsPackedN4Buf.indices);
        indices.back() = addIndex(loc, indices.back(), packedOffset, rewriter);
        bPacked = vector::LoadOp::create(
            rewriter, loc, v16i8Ty, candidate.rhsPackedN4Buf.memRef, indices);
      } else {
        SmallVector<Value> bRows(4);
        for (int64_t r = 0; r < 4; ++r) {
          Value row = addIndex(loc, kBase, r, rewriter);
          bRows[r] = vector::LoadOp::create(
              rewriter, loc, v4i8Ty, candidate.rhsBuf.memRef,
              get2DIndices(loc, candidate.rhsBuf, row, nBase, rewriter));
        }
        Value b01 = vector::InsertStridedSliceOp::create(
            rewriter, loc, bRows[0], zero8, ArrayRef<int64_t>{0},
            ArrayRef<int64_t>{1});
        b01 = vector::InsertStridedSliceOp::create(rewriter, loc, bRows[1], b01,
                                                   ArrayRef<int64_t>{4},
                                                   ArrayRef<int64_t>{1});
        Value b23 = vector::InsertStridedSliceOp::create(
            rewriter, loc, bRows[2], zero8, ArrayRef<int64_t>{0},
            ArrayRef<int64_t>{1});
        b23 = vector::InsertStridedSliceOp::create(rewriter, loc, bRows[3], b23,
                                                   ArrayRef<int64_t>{4},
                                                   ArrayRef<int64_t>{1});
        Value b01Wide = vector::InsertStridedSliceOp::create(
            rewriter, loc, b01, zero16, ArrayRef<int64_t>{0},
            ArrayRef<int64_t>{1});
        Value b23Wide = vector::InsertStridedSliceOp::create(
            rewriter, loc, b23, zero16, ArrayRef<int64_t>{0},
            ArrayRef<int64_t>{1});
        bPacked = vector::ShuffleOp::create(rewriter, loc, b01Wide, b23Wide,
                                            transposeB);
      }

      Value next =
          LLVM::CallIntrinsicOp::create(
              rewriter, loc, v4i32Ty, sdot,
              ValueRange{kLoop.getRegionIterArgs()[0], aBroadcast, bPacked})
              .getResult(0);
      rewriter.setInsertionPointToEnd(kLoop.getBody());
      scf::YieldOp::create(rewriter, loc, next);
    }

    rewriter.setInsertionPointAfter(kLoop);
    vector::StoreOp::create(rewriter, loc, kLoop.getResult(0), accBuf.memRef,
                            get2DIndices(loc, accBuf, 0, nBase, rewriter));
  }

  rewriter.setInsertionPointAfter(nLoop);
  if (candidate.keepAccInBuffer) {
    Value loopResult =
        outerKLoop.getTiedLoopResult(cast<BlockArgument>(op.getC()));
    rewriter.replaceOp(op, op.getC());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(outerKLoop);
    Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
    rewriter.replaceAllUsesWith(loopResult, result);
  } else {
    Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
    rewriter.replaceOp(op, result);
  }
  return success();
}

static LogicalResult convertCandidate(SVE2I8MMCandidate &candidate,
                                      PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  auto lhsTy = cast<VectorType>(op.getA().getType());
  auto rhsTy = cast<VectorType>(op.getB().getType());
  auto accTy = cast<VectorType>(op.getC().getType());
  Type i8Ty = lhsTy.getElementType();
  Type i32Ty = accTy.getElementType();
  int64_t mSize = lhsTy.getDimSize(0);
  int64_t kSize = lhsTy.getDimSize(1);
  int64_t nSize = rhsTy.getDimSize(1);

  // See isCandidate(): preserve the fast direct-buffer case, but provide a
  // general compiler path for computed vectors such as Q4 nibble unpack.
  // The allocas live at function scope while the stores remain immediately
  // before the replacement microkernel, so loop-varying values are correct.
  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();
  if (candidate.lhsBuf.empty())
    candidate.lhsBuf = storeToTmpBuffer(loc, op.getA(), allocaPoint, rewriter);
  if (candidate.rhsBuf.empty() && candidate.rhsPackedN4Buf.empty())
    candidate.rhsBuf = storeToTmpBuffer(loc, op.getB(), allocaPoint, rewriter);

  if (mSize == 1)
    return convertCandidateM1(candidate, rewriter);

  auto v8i8Ty = VectorType::get({8}, i8Ty);
  auto v16i8Ty = VectorType::get({16}, i8Ty);
  auto v2i32Ty = VectorType::get({2}, i32Ty);
  auto v4i32Ty = VectorType::get({4}, i32Ty);
  Type nxv16i8Ty = VectorType::get({16}, i8Ty, {true});
  Type nxv4i32Ty = VectorType::get({4}, i32Ty, {true});
  StringAttr smmla =
      StringAttr::get(op.getContext(), "llvm.aarch64.sve.smmla.nxv4i32");

  scf::ForOp outerKLoop;
  Value initialAcc = op.getC();
  if (candidate.keepAccInBuffer) {
    outerKLoop = cast<scf::ForOp>(op->getParentOp());
    initialAcc = getInitAccValue(op.getC());
  }
  bool zeroInitialAcc =
      !candidate.keepAccInBuffer && matchPattern(initialAcc, m_Zero());

  MemBuffer accBuf;
  Value bPack;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    if (candidate.keepAccInBuffer)
      rewriter.setInsertionPoint(outerKLoop);

    accBuf = candidate.directOutputWrite
                 ? candidate.outputBuf
                 : allocateTmpBufferStack(loc, accTy, allocaPoint, rewriter);
    if (!zeroInitialAcc)
      op_write(initialAcc, accBuf.memRef, accBuf.indices);

    auto bPackTy = MemRefType::get({kSize / 8, 4, 16}, i8Ty);
    {
      OpBuilder::InsertionGuard allocaGuard(rewriter);
      rewriter.setInsertionPoint(allocaPoint);
      bPack = memref::AllocaOp::create(
          rewriter, loc, bPackTy,
          rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
    }
  }

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value c8 = arith::ConstantIndexOp::create(rewriter, loc, 8);
  Value mLimit = arith::ConstantIndexOp::create(rewriter, loc, mSize);
  Value nLimit = arith::ConstantIndexOp::create(rewriter, loc, nSize);
  Value kSteps = arith::ConstantIndexOp::create(rewriter, loc, kSize / 8);

  // N is outermost so each packed B panel is reused by every M block.
  auto nLoop = scf::ForOp::create(rewriter, loc, c0, nLimit, c8);
  {
    OpBuilder::InsertionGuard nGuard(rewriter);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value nBase = nLoop.getInductionVar();

    // Pack one Kx8 B panel.  Loading an 8x8 block by rows keeps source memory
    // contiguous; the local transpose creates four SMMLA 2x8 column pairs.
    auto packLoop = scf::ForOp::create(rewriter, loc, c0, kSteps, c1);
    {
      OpBuilder::InsertionGuard packGuard(rewriter);
      rewriter.setInsertionPointToStart(packLoop.getBody());
      Value ki = packLoop.getInductionVar();
      Value kBase = arith::MulIOp::create(rewriter, loc, ki, c8);
      SmallVector<Value> rows(8);
      for (int64_t r = 0; r < 8; ++r) {
        Value row = addIndex(loc, kBase, r, rewriter);
        rows[r] = vector::LoadOp::create(
            rewriter, loc, v8i8Ty, candidate.rhsBuf.memRef,
            get2DIndices(loc, candidate.rhsBuf, row, nBase, rewriter));
      }

      // Spell out an 8x8 byte transpose as the same three-level TRN network
      // used by an optimized NEON C implementation.  A generic
      // vector.transpose was scalarized into dozens of lane moves by LLVM;
      // these shuffle masks select trn1/trn2 (and zip1/zip2 for 32-bit lanes).
      constexpr int64_t trn1b[] = {0, 8, 2, 10, 4, 12, 6, 14};
      constexpr int64_t trn2b[] = {1, 9, 3, 11, 5, 13, 7, 15};
      constexpr int64_t trn1h[] = {0, 4, 2, 6};
      constexpr int64_t trn2h[] = {1, 5, 3, 7};
      constexpr int64_t trn1s[] = {0, 2};
      constexpr int64_t trn2s[] = {1, 3};
      constexpr int64_t concat[] = {0, 1, 2,  3,  4,  5,  6,  7,
                                    8, 9, 10, 11, 12, 13, 14, 15};

      SmallVector<Value> bytes(8);
      for (int64_t pair = 0; pair < 4; ++pair) {
        auto [lo, hi] = transposePair(loc, rows[pair * 2], rows[pair * 2 + 1],
                                      trn1b, trn2b, rewriter);
        bytes[pair * 2] = lo;
        bytes[pair * 2 + 1] = hi;
      }

      auto v4i16Ty = VectorType::get({4}, rewriter.getI16Type());
      for (Value &value : bytes)
        value = vector::BitCastOp::create(rewriter, loc, v4i16Ty, value);
      SmallVector<Value> halfwords(8);
      constexpr std::pair<int64_t, int64_t> halfwordPairs[] = {
          {0, 2}, {1, 3}, {4, 6}, {5, 7}};
      for (auto [idx, halfwordPair] : llvm::enumerate(halfwordPairs)) {
        auto [lo, hi] =
            transposePair(loc, bytes[halfwordPair.first],
                          bytes[halfwordPair.second], trn1h, trn2h, rewriter);
        halfwords[idx * 2] = lo;
        halfwords[idx * 2 + 1] = hi;
      }

      auto v2i32Ty = VectorType::get({2}, rewriter.getI32Type());
      for (Value &value : halfwords)
        value = vector::BitCastOp::create(rewriter, loc, v2i32Ty, value);
      constexpr std::pair<int64_t, int64_t> wordPairs[] = {
          {0, 4}, {2, 6}, {1, 5}, {3, 7}};
      SmallVector<Value> columns(8);
      for (auto [idx, wordPair] : llvm::enumerate(wordPairs)) {
        auto [lo, hi] =
            transposePair(loc, halfwords[wordPair.first],
                          halfwords[wordPair.second], trn1s, trn2s, rewriter);
        columns[idx] = lo;
        columns[idx + 4] = hi;
      }
      for (Value &value : columns)
        value = vector::BitCastOp::create(rewriter, loc, v8i8Ty, value);

      for (int64_t pair = 0; pair < 4; ++pair) {
        Value packed = vector::ShuffleOp::create(
            rewriter, loc, columns[pair * 2], columns[pair * 2 + 1], concat);
        Value pairIdx = arith::ConstantIndexOp::create(rewriter, loc, pair);
        vector::StoreOp::create(rewriter, loc, packed, bPack,
                                ValueRange{ki, pairIdx, c0});
      }
    }

    auto mLoop = scf::ForOp::create(rewriter, loc, c0, mLimit, c8);
    {
      OpBuilder::InsertionGuard mGuard(rewriter);
      rewriter.setInsertionPointToStart(mLoop.getBody());
      Value mBase = mLoop.getInductionVar();

      SmallVector<Value> initAcc(16);
      if (zeroInitialAcc) {
        Value zero = LLVM::ZeroOp::create(rewriter, loc, nxv4i32Ty);
        std::fill(initAcc.begin(), initAcc.end(), zero);
      } else {
        for (int64_t mr = 0; mr < 4; ++mr) {
          Value row = addIndex(loc, mBase, mr * 2, rewriter);
          for (int64_t nr = 0; nr < 4; ++nr) {
            Value col = addIndex(loc, nBase, nr * 2, rewriter);
            initAcc[mr * 4 + nr] = loadAcc2x2(loc, accBuf, row, col, v2i32Ty,
                                              v4i32Ty, nxv4i32Ty, rewriter);
          }
        }
      }

      auto computeLoop =
          scf::ForOp::create(rewriter, loc, c0, kSteps, c1, initAcc);
      {
        OpBuilder::InsertionGuard computeGuard(rewriter);
        rewriter.setInsertionPointToStart(computeLoop.getBody());
        Value ki = computeLoop.getInductionVar();
        Value kBase = arith::MulIOp::create(rewriter, loc, ki, c8);

        SmallVector<Value> aPacked(4);
        SmallVector<Value> bPacked(4);
        for (int64_t pair = 0; pair < 4; ++pair) {
          Value row0 = addIndex(loc, mBase, pair * 2, rewriter);
          Value row1 = addIndex(loc, row0, 1, rewriter);
          Value a0 = vector::LoadOp::create(
              rewriter, loc, v8i8Ty, candidate.lhsBuf.memRef,
              get2DIndices(loc, candidate.lhsBuf, row0, kBase, rewriter));
          Value a1 = vector::LoadOp::create(
              rewriter, loc, v8i8Ty, candidate.lhsBuf.memRef,
              get2DIndices(loc, candidate.lhsBuf, row1, kBase, rewriter));
          aPacked[pair] =
              packRows2x8(loc, a0, a1, v16i8Ty, nxv16i8Ty, rewriter);

          Value pairIdx = arith::ConstantIndexOp::create(rewriter, loc, pair);
          Value bFixed = vector::LoadOp::create(rewriter, loc, v16i8Ty, bPack,
                                                ValueRange{ki, pairIdx, c0});
          bPacked[pair] =
              insertFixedInScalable(loc, bFixed, nxv16i8Ty, rewriter);
        }

        SmallVector<Value> nextAcc(16);
        ValueRange carried = computeLoop.getRegionIterArgs();
        for (int64_t mr = 0; mr < 4; ++mr) {
          for (int64_t nr = 0; nr < 4; ++nr) {
            int64_t idx = mr * 4 + nr;
            nextAcc[idx] =
                LLVM::CallIntrinsicOp::create(
                    rewriter, loc, nxv4i32Ty, smmla,
                    ValueRange{carried[idx], aPacked[mr], bPacked[nr]})
                    .getResult(0);
          }
        }

        // The scf.for builder cannot synthesize a terminator for non-empty
        // iter_args because there is no identity yield to infer.  Finish the
        // newly-created body explicitly.
        rewriter.setInsertionPointToEnd(computeLoop.getBody());
        scf::YieldOp::create(rewriter, loc, nextAcc);
      }

      rewriter.setInsertionPointAfter(computeLoop);
      ValueRange finalAcc = computeLoop.getResults();
      for (int64_t mr = 0; mr < 4; ++mr) {
        Value row = addIndex(loc, mBase, mr * 2, rewriter);
        for (int64_t nr = 0; nr < 4; ++nr) {
          Value col = addIndex(loc, nBase, nr * 2, rewriter);
          storeAcc2x2(loc, accBuf, row, col, finalAcc[mr * 4 + nr], v2i32Ty,
                      v4i32Ty, rewriter);
        }
      }
    }
  }

  rewriter.setInsertionPointAfter(nLoop);
  if (candidate.directOutputWrite) {
    rewriter.eraseOp(candidate.directOutputWrite);
    rewriter.eraseOp(op);
  } else if (candidate.keepAccInBuffer) {
    Value loopResult =
        outerKLoop.getTiedLoopResult(cast<BlockArgument>(op.getC()));
    rewriter.replaceOp(op, op.getC());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(outerKLoop);
    Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
    rewriter.replaceAllUsesWith(loopResult, result);
  } else {
    Value result = op_read(accTy, accBuf.memRef, accBuf.indices);
    rewriter.replaceOp(op, result);
  }
  return success();
}

struct ConvertDotToSVE2I8MM
    : public triton::cpu::impl::ConvertDotToSVE2I8MMBase<ConvertDotToSVE2I8MM> {
  ConvertDotToSVE2I8MM() = default;
  explicit ConvertDotToSVE2I8MM(bool fixedOnlyValue) {
    this->fixedOnly = fixedOnlyValue;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // On fixed-width Arm, free the Neon register file before collecting dot
    // candidates.  The rewrite clones the nested dots, so all later matching
    // must deliberately walk the post-rewrite IR.
    if (this->fixedOnly) {
      SmallVector<PackedQ4NestedG128LoopCandidate, 1> nestedG128Loops;
      mod.walk([&](scf::ForOp loop) {
        PackedQ4NestedG128LoopCandidate candidate;
        if (matchPackedQ4NestedG128Loop(loop, candidate))
          nestedG128Loops.push_back(std::move(candidate));
        return WalkResult::advance();
      });
      for (auto &candidate : nestedG128Loops) {
        PatternRewriter rewriter(context);
        rewriter.setInsertionPoint(candidate.outerLoop);
        if (failed(bufferPackedQ4NestedG128Outer(candidate, rewriter)))
          candidate.outerLoop.emitRemark(
              "packed Q4 nested G128 buffering skipped");
      }

      SmallVector<PackedQ4M4IntegerLoopCandidate, 1> integerLoops;
      mod.walk([&](scf::ForOp loop) {
        PackedQ4M4IntegerLoopCandidate candidate;
        if (matchPackedQ4M4IntegerLoop(loop, candidate))
          integerLoops.push_back(std::move(candidate));
        return WalkResult::advance();
      });
      for (auto &candidate : integerLoops) {
        PatternRewriter rewriter(context);
        rewriter.setInsertionPoint(candidate.loop);
        if (failed(convertPackedQ4M4IntegerLoop(candidate, rewriter)))
          candidate.loop.emitRemark(
              "packed Q4 native integer loop fusion skipped");
      }
    }

    SmallVector<W4A8DotPair, 1> w4Pairs;
    llvm::SmallPtrSet<Operation *, 4> pairedDots;
    mod.walk([&](cpu::DotOp op) {
      if (pairedDots.contains(op))
        return WalkResult::advance();
      W4A8DotPair pair;
      if (matchW4A8DotPair(op, pair)) {
        w4Pairs.push_back(pair);
        pairedDots.insert(pair.lowDot);
        pairedDots.insert(pair.highDot);
      }
      return WalkResult::advance();
    });

    SmallVector<PackedQ4M4LoopCandidate, 1> packedQ4Loops;
    llvm::SmallPtrSet<Operation *, 8> packedQ4LoopDots;
    mod.walk([&](scf::ForOp loop) {
      PackedQ4M4LoopCandidate candidate;
      if (matchPackedQ4M4Loop(loop, candidate)) {
        for (const PackedQ4M4Epilogue &panel : candidate.panels)
          packedQ4LoopDots.insert(panel.dot.op);
        packedQ4Loops.push_back(std::move(candidate));
      }
      return WalkResult::advance();
    });

    SmallVector<PackedQ4M4Dot, 4> standalonePackedQ4Dots;
    mod.walk([&](cpu::DotOp op) {
      if (packedQ4LoopDots.contains(op))
        return WalkResult::advance();
      PackedQ4M4Dot candidate;
      if (matchPackedQ4M4Dot(op, candidate))
        standalonePackedQ4Dots.push_back(candidate);
      return WalkResult::advance();
    });

    SmallVector<PackedW8PrefillCandidate, 1> packedW8Prefill;
    llvm::SmallPtrSet<Operation *, 4> packedW8Dots;
    llvm::SmallPtrSet<Operation *, 2> packedW8Loops;
    SmallVector<SVE2I8MMCandidate, 1> candidates;
    mod.walk([&](cpu::DotOp op) {
      if (pairedDots.contains(op))
        return WalkResult::advance();
      PackedW8PrefillCandidate candidate;
      if (matchPackedW8Prefill(op, candidate)) {
        packedW8Dots.insert(op);
        Operation *parent = op->getParentOp();
        if (!isa_and_nonnull<scf::ForOp>(parent) ||
            packedW8Loops.insert(parent).second)
          packedW8Prefill.push_back(candidate);
      }
      return WalkResult::advance();
    });
    mod.walk([&](cpu::DotOp op) {
      if (pairedDots.contains(op) || packedW8Dots.contains(op))
        return WalkResult::advance();
      SVE2I8MMCandidate candidate;
      if (!isCandidate(op, candidate))
        return WalkResult::advance();
      auto lhsTy = cast<VectorType>(op.getA().getType());
      if (!this->fixedOnly || lhsTy.getDimSize(0) == 1)
        candidates.push_back(candidate);
      return WalkResult::advance();
    });

    for (auto &pair : w4Pairs) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(pair.lowDot);
      if (failed(convertW4A8DotPair(pair, rewriter)))
        pair.lowDot.emitRemark("packed W4A8 SDOT fusion skipped");
    }

    for (auto &candidate : packedQ4Loops) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.loop);
      if (failed(convertPackedQ4M4Loop(candidate, this->fixedOnly, rewriter)))
        candidate.loop.emitRemark("packed Q4 M4 loop fusion skipped");
    }

    SmallVector<bool, 4> convertedStandalone(standalonePackedQ4Dots.size(),
                                             false);
    for (size_t index = 0; index < standalonePackedQ4Dots.size(); ++index) {
      if (convertedStandalone[index])
        continue;
      SmallVector<PackedQ4M4Dot, 4> group;
      group.push_back(standalonePackedQ4Dots[index]);
      convertedStandalone[index] = true;
      for (size_t other = index + 1;
           other < standalonePackedQ4Dots.size() && group.size() < 2; ++other) {
        if (convertedStandalone[other])
          continue;
        const PackedQ4M4Dot &candidate = standalonePackedQ4Dots[other];
        if (candidate.op->getParentOp() == group.front().op->getParentOp() &&
            samePackedBuffer(candidate.rhsPacked, group.front().rhsPacked) &&
            canInsertSharedQ4DotBefore(group, candidate.op)) {
          group.push_back(candidate);
          convertedStandalone[other] = true;
        }
      }

      PatternRewriter rewriter(context);
      // Later panel operands may be formed after the first dot.  Insert the
      // shared microkernel immediately before the last dot, where all packed
      // LHS values dominate, then replace every member together.
      rewriter.setInsertionPoint(group.back().op);
      LogicalResult result = group.size() == 1
                                 ? convertPackedQ4M4Dot(group.front(), rewriter)
                                 : convertPackedQ4M4DotGroup(group, rewriter);
      if (failed(result))
        group.front().op.emitRemark("standalone packed Q4 M4 lowering skipped");
    }

    for (auto &candidate : packedW8Prefill) {
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (failed(convertPackedW8Prefill(candidate, this->fixedOnly, rewriter)))
        candidate.op.emitRemark("packed W8 prefill i8mm lowering skipped");
    }

    for (auto &candidate : candidates) {
      PatternRewriter rewriter(context);
      if (candidate.directOutputWrite)
        rewriter.setInsertionPoint(candidate.directOutputWrite);
      else
        rewriter.setInsertionPoint(candidate.op);
      if (failed(convertCandidate(candidate, rewriter)))
        candidate.op.emitRemark("SVE2 i8mm rolled lowering skipped");
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

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToSVE2I8MM(bool fixedOnly) {
  return std::make_unique<ConvertDotToSVE2I8MM>(fixedOnly);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
