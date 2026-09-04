#include "cpu/include/TritonCPUTransforms/OptCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "include/triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include <iostream>
#include <utility>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTPRODUCT
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// TODO: support SVE and different vector width
// We currently only supported Arm Neon (128 bit vector).
// To support scalable vectors in SVE, we need to generate
// vector-length agnostic (VLA) code using vector.vscale.
// To support other platform (AVX512 for X86), we need to
// change the vectorBitWidth and the intrinsics.
constexpr int vectorBitWidth = 128;

// Recover the original eight-byte vector behind tl.cat(x, x, dim=0) after
// Triton's element-manipulation lowering.  The exact sequence is intentionally
// checked so an unrelated 4x4 operand is never treated as a replicated load.
static Value findRepeated8ByteInput(Value value) {
  auto outerShape = value.getDefiningOp<vector::ShapeCastOp>();
  if (!outerShape)
    return {};
  auto transpose = outerShape.getSource().getDefiningOp<vector::TransposeOp>();
  if (!transpose)
    return {};

  // Triton 3.7 lowers the Q4 source expression
  //
  //   tl.join(x8, x8).permute(1, 0).reshape(4, 4)
  //
  // to interleave -> shape_cast<8x2> -> transpose<1,0> -> shape_cast.
  // Preserve the original vector<8xi8> load so the i64 broadcast below is
  // selected as LD1R rather than LDR(D) plus a lane-duplication MOV.  W8 uses
  // the distinct 3-D graph retained in the second branch.
  if (transpose.getPermutation() == ArrayRef<int64_t>({1, 0})) {
    auto outerSourceTy = dyn_cast<VectorType>(outerShape.getSource().getType());
    if (!outerSourceTy || outerSourceTy.getShape() != ArrayRef<int64_t>({2, 8}))
      return {};
    auto innerShape =
        transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
    if (!innerShape)
      return {};
    auto innerSourceTy = dyn_cast<VectorType>(innerShape.getSource().getType());
    if (!innerSourceTy || innerSourceTy.getShape() != ArrayRef<int64_t>({16}))
      return {};
    auto interleave =
        innerShape.getSource().getDefiningOp<vector::InterleaveOp>();
    if (!interleave || interleave.getLhs() != interleave.getRhs())
      return {};
    Value input = interleave.getLhs();
    auto inputTy = dyn_cast<VectorType>(input.getType());
    if (!inputTy || inputTy.getShape() != ArrayRef<int64_t>({8}) ||
        !inputTy.getElementType().isInteger(8))
      return {};
    return input;
  }

  if (transpose.getPermutation() != ArrayRef<int64_t>({2, 0, 1}) ||
      cast<VectorType>(outerShape.getSource().getType()).getShape() !=
          ArrayRef<int64_t>({2, 2, 4}))
    return {};
  auto middleShape = transpose.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!middleShape ||
      cast<VectorType>(middleShape.getSource().getType()).getShape() !=
          ArrayRef<int64_t>({2, 8}))
    return {};
  auto interleave =
      middleShape.getSource().getDefiningOp<vector::InterleaveOp>();
  if (!interleave || interleave.getLhs() != interleave.getRhs())
    return {};
  auto innerShape = interleave.getLhs().getDefiningOp<vector::ShapeCastOp>();
  if (!innerShape)
    return {};
  Value input = innerShape.getSource();
  auto inputTy = dyn_cast<VectorType>(input.getType());
  if (!inputTy || inputTy.getShape() != ArrayRef<int64_t>({8}) ||
      !inputTy.getElementType().isInteger(8))
    return {};
  return input;
}

// Fuse four independent four-byte signed dot products into one SDOT.  This
// shape is useful when a packed GEMV kernel carries two native v4i32
// accumulators explicitly, avoiding multidimensional-vector ABI shuffles in
// the K loop.
struct ConvertI8RowDotAccumulate : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Value carried;
    vector::MultiDimReductionOp reduction;
    if ((reduction =
             op.getRhs().getDefiningOp<vector::MultiDimReductionOp>())) {
      carried = op.getLhs();
    } else if ((reduction =
                    op.getLhs().getDefiningOp<vector::MultiDimReductionOp>())) {
      carried = op.getRhs();
    } else {
      return failure();
    }

    auto resultTy = dyn_cast<VectorType>(op.getType());
    auto sourceTy = dyn_cast<VectorType>(reduction.getSource().getType());
    if (!resultTy || !sourceTy ||
        resultTy.getShape() != ArrayRef<int64_t>({4}) ||
        !resultTy.getElementType().isInteger(32) ||
        sourceTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
        !sourceTy.getElementType().isInteger(32) ||
        reduction.getType() != resultTy ||
        reduction.getKind() != vector::CombiningKind::ADD ||
        reduction.isReducedDim(0) || !reduction.isReducedDim(1) ||
        !isZeroConst(reduction.getAcc()) || !reduction->hasOneUse()) {
      return failure();
    }

    auto multiply = reduction.getSource().getDefiningOp<arith::MulIOp>();
    if (!multiply || !multiply->hasOneUse())
      return failure();
    auto lhsExt = multiply.getLhs().getDefiningOp<arith::ExtSIOp>();
    auto rhsExt = multiply.getRhs().getDefiningOp<arith::ExtSIOp>();
    if (!lhsExt || !rhsExt)
      return failure();
    auto lhsTy = dyn_cast<VectorType>(lhsExt.getIn().getType());
    auto rhsTy = dyn_cast<VectorType>(rhsExt.getIn().getType());
    if (!lhsTy || !rhsTy || lhsTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
        rhsTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
        !lhsTy.getElementType().isInteger(8) ||
        !rhsTy.getElementType().isInteger(8)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto v16i8Ty = VectorType::get({16}, rewriter.getI8Type());
    auto makeOperand = [&](Value input) -> Value {
      if (Value repeated = findRepeated8ByteInput(input)) {
        Value scalar = LLVM::BitcastOp::create(rewriter, loc,
                                               rewriter.getI64Type(), repeated);
        auto v2i64Ty = VectorType::get({2}, rewriter.getI64Type());
        Value broadcast =
            vector::BroadcastOp::create(rewriter, loc, v2i64Ty, scalar);
        return LLVM::BitcastOp::create(rewriter, loc, v16i8Ty, broadcast);
      }
      return vector::ShapeCastOp::create(rewriter, loc, v16i8Ty, input);
    };
    Value lhs = makeOperand(lhsExt.getIn());
    Value rhs = makeOperand(rhsExt.getIn());
    auto sdot =
        StringAttr::get(op.getContext(), "llvm.aarch64.neon.sdot.v4i32.v16i8");
    Value result = LLVM::CallIntrinsicOp::create(rewriter, loc, resultTy, sdot,
                                                 ValueRange{carried, lhs, rhs})
                       .getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// The first dot product in an explicitly carried accumulator has no preceding
// arith.addi to anchor ConvertI8RowDotAccumulate.  Lower that seed reduction to
// SDOT as well.  Reductions still feeding an add are deliberately left alone
// first, so the accumulate pattern can retain the native accumulator operand.
struct ConvertI8RowDot : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasOneUse() && isa<arith::AddIOp>(*op->getUsers().begin()))
      return failure();

    auto resultTy = dyn_cast<VectorType>(op.getType());
    auto sourceTy = dyn_cast<VectorType>(op.getSource().getType());
    if (!resultTy || !sourceTy ||
        resultTy.getShape() != ArrayRef<int64_t>({4}) ||
        !resultTy.getElementType().isInteger(32) ||
        sourceTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
        !sourceTy.getElementType().isInteger(32) ||
        op.getKind() != vector::CombiningKind::ADD || op.isReducedDim(0) ||
        !op.isReducedDim(1) || !isZeroConst(op.getAcc()) || !op->hasOneUse()) {
      return failure();
    }

    auto multiply = op.getSource().getDefiningOp<arith::MulIOp>();
    if (!multiply || !multiply->hasOneUse())
      return failure();
    auto lhsExt = multiply.getLhs().getDefiningOp<arith::ExtSIOp>();
    auto rhsExt = multiply.getRhs().getDefiningOp<arith::ExtSIOp>();
    if (!lhsExt || !rhsExt)
      return failure();
    auto lhsTy = dyn_cast<VectorType>(lhsExt.getIn().getType());
    auto rhsTy = dyn_cast<VectorType>(rhsExt.getIn().getType());
    if (!lhsTy || !rhsTy || lhsTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
        rhsTy.getShape() != ArrayRef<int64_t>({4, 4}) ||
        !lhsTy.getElementType().isInteger(8) ||
        !rhsTy.getElementType().isInteger(8)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto v16i8Ty = VectorType::get({16}, rewriter.getI8Type());
    auto makeOperand = [&](Value input) -> Value {
      if (Value repeated = findRepeated8ByteInput(input)) {
        Value scalar = LLVM::BitcastOp::create(rewriter, loc,
                                               rewriter.getI64Type(), repeated);
        auto v2i64Ty = VectorType::get({2}, rewriter.getI64Type());
        Value broadcast =
            vector::BroadcastOp::create(rewriter, loc, v2i64Ty, scalar);
        return LLVM::BitcastOp::create(rewriter, loc, v16i8Ty, broadcast);
      }
      return vector::ShapeCastOp::create(rewriter, loc, v16i8Ty, input);
    };
    Value lhs = makeOperand(lhsExt.getIn());
    Value rhs = makeOperand(rhsExt.getIn());
    Value zero = arith::ConstantOp::create(rewriter, loc, resultTy,
                                           rewriter.getZeroAttr(resultTy));
    auto sdot =
        StringAttr::get(op.getContext(), "llvm.aarch64.neon.sdot.v4i32.v16i8");
    Value result = LLVM::CallIntrinsicOp::create(rewriter, loc, resultTy, sdot,
                                                 ValueRange{zero, lhs, rhs})
                       .getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Four adjacent pairs map exactly to NEON ADDP(a, b):
// [a0+a1, a2+a3, b0+b1, b2+b3].  Preserve that relationship instead of
// lowering the multidimensional reduction into four scalar reductions and
// rebuilding a vector with lane moves.
struct ConvertI32PairReduction
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceTy = dyn_cast<VectorType>(op.getSource().getType());
    auto resultTy = dyn_cast<VectorType>(op.getType());
    if (!sourceTy || !resultTy ||
        sourceTy.getShape() != ArrayRef<int64_t>({4, 2}) ||
        !sourceTy.getElementType().isInteger(32) ||
        resultTy.getShape() != ArrayRef<int64_t>({4}) ||
        !resultTy.getElementType().isInteger(32) ||
        op.getKind() != vector::CombiningKind::ADD || op.isReducedDim(0) ||
        !op.isReducedDim(1)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto v8i32Ty = VectorType::get({8}, rewriter.getI32Type());
    auto v4i32Ty = VectorType::get({4}, rewriter.getI32Type());
    Value flat =
        vector::ShapeCastOp::create(rewriter, loc, v8i32Ty, op.getSource());
    Value low = vector::ExtractStridedSliceOp::create(
        rewriter, loc, flat, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{4},
        ArrayRef<int64_t>{1});
    Value high = vector::ExtractStridedSliceOp::create(
        rewriter, loc, flat, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{4},
        ArrayRef<int64_t>{1});
    auto addp =
        StringAttr::get(op.getContext(), "llvm.aarch64.neon.addp.v4i32");
    Value result = LLVM::CallIntrinsicOp::create(rewriter, loc, v4i32Ty, addp,
                                                 ValueRange{low, high})
                       .getResult(0);
    if (!isZeroConst(op.getAcc()))
      result = arith::AddIOp::create(rewriter, loc, result, op.getAcc());
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Preserve AArch64's fixed-point form of SCVTF for the signed-int conversion
// followed by an exact 1/16 scale used by KAI-style int4 nibble decoding.  The
// generic LLVM combine currently leaves this as SCVTF plus a separate FMUL.
struct ConvertI32ToF32Scale16 : public OpRewritePattern<arith::MulFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulFOp op,
                                PatternRewriter &rewriter) const override {
    arith::SIToFPOp conversion;
    arith::ConstantOp constant;
    if ((conversion = op.getLhs().getDefiningOp<arith::SIToFPOp>()) &&
        (constant = op.getRhs().getDefiningOp<arith::ConstantOp>())) {
      // Matched in source order.
    } else if ((conversion = op.getRhs().getDefiningOp<arith::SIToFPOp>()) &&
               (constant = op.getLhs().getDefiningOp<arith::ConstantOp>())) {
      // Matched the commuted multiply.
    } else {
      return failure();
    }

    auto resultTy = dyn_cast<VectorType>(op.getType());
    auto inputTy = dyn_cast<VectorType>(conversion.getIn().getType());
    auto scale = dyn_cast<DenseFPElementsAttr>(constant.getValue());
    if (!resultTy || !inputTy ||
        resultTy.getShape() != ArrayRef<int64_t>({4}) ||
        !resultTy.getElementType().isF32() ||
        inputTy.getShape() != ArrayRef<int64_t>({4}) ||
        !inputTy.getElementType().isInteger(32) || !scale || !scale.isSplat() ||
        scale.getSplatValue<APFloat>().convertToDouble() != 0.0625) {
      return failure();
    }

    Location loc = op.getLoc();
    Value fractionalBits = arith::ConstantIntOp::create(rewriter, loc, 4, 32);
    auto scvtf =
        StringAttr::get(op.getContext(), "llvm.aarch64.neon.vcvtfxs2fp");
    Value result = LLVM::CallIntrinsicOp::create(
                       rewriter, loc, resultTy, scvtf,
                       ValueRange{conversion.getIn(), fractionalBits})
                       .getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Match the ordinary Triton expression used for a KleidiAI-compatible W8
// microtile:
//
//   partial += tl.sum(
//       weight.reshape((4, 2, 4)).to(tl.int32) *
//       x.reshape((1, 2, 4)).to(tl.int32), axis=2)
//
// The result keeps two four-byte partial sums per output.  That is exactly
// SDOT's physical accumulator layout for two adjacent output channels:
//
//   [out0.k0_3, out0.k4_7, out1.k0_3, out1.k4_7]
//
// Keeping the operation as eight independent scalar reductions prevents
// LLVM's SLP vectorizer from recovering this layout.  Fuse the loop-carried
// add while the MultiDimReduction still records it, producing two SDOTs for
// the contiguous 32-byte packed weight microtile.  The source remains plain
// tl.load/integer arithmetic/tl.sum; no frontend or runtime intrinsic is
// required.
struct ConvertPackedI8MulSumAccumulate
    : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Value carried;
    vector::MultiDimReductionOp reduction;
    if ((reduction =
             op.getRhs().getDefiningOp<vector::MultiDimReductionOp>())) {
      carried = op.getLhs();
    } else if ((reduction =
                    op.getLhs().getDefiningOp<vector::MultiDimReductionOp>())) {
      carried = op.getRhs();
    } else {
      return failure();
    }

    auto resultTy = dyn_cast<VectorType>(op.getType());
    auto sourceTy = dyn_cast<VectorType>(reduction.getSource().getType());
    auto reductionTy = dyn_cast<VectorType>(reduction.getType());
    if (!resultTy || !sourceTy || !reductionTy || resultTy != reductionTy ||
        resultTy.getShape() != ArrayRef<int64_t>({4, 2}) ||
        !resultTy.getElementType().isInteger(32) ||
        sourceTy.getShape() != ArrayRef<int64_t>({4, 2, 4}) ||
        !sourceTy.getElementType().isInteger(32) ||
        reduction.getKind() != vector::CombiningKind::ADD ||
        reduction.isReducedDim(0) || reduction.isReducedDim(1) ||
        !reduction.isReducedDim(2) || !isZeroConst(reduction.getAcc()) ||
        !reduction->hasOneUse()) {
      return failure();
    }

    auto multiply = reduction.getSource().getDefiningOp<arith::MulIOp>();
    if (!multiply || !multiply->hasOneUse())
      return failure();

    Value weightInput;
    Value activationInput;
    auto matchInputs = [&](Value weightCandidate,
                           Value activationCandidate) -> bool {
      auto weightExt = weightCandidate.getDefiningOp<arith::ExtSIOp>();
      auto activationBroadcast =
          activationCandidate.getDefiningOp<vector::BroadcastOp>();
      if (!weightExt || !activationBroadcast)
        return false;
      auto activationExt =
          activationBroadcast.getSource().getDefiningOp<arith::ExtSIOp>();
      if (!activationExt)
        return false;

      auto weightTy = dyn_cast<VectorType>(weightExt.getIn().getType());
      auto activationTy = dyn_cast<VectorType>(activationExt.getIn().getType());
      auto broadcastSourceTy =
          dyn_cast<VectorType>(activationBroadcast.getSource().getType());
      if (!weightTy || !activationTy || !broadcastSourceTy ||
          weightTy.getShape() != ArrayRef<int64_t>({4, 2, 4}) ||
          !weightTy.getElementType().isInteger(8) ||
          activationTy.getShape() != ArrayRef<int64_t>({1, 2, 4}) ||
          !activationTy.getElementType().isInteger(8) ||
          broadcastSourceTy.getShape() != ArrayRef<int64_t>({1, 2, 4}) ||
          !broadcastSourceTy.getElementType().isInteger(32)) {
        return false;
      }
      weightInput = weightExt.getIn();
      activationInput = activationExt.getIn();
      return true;
    };

    if (!matchInputs(multiply.getLhs(), multiply.getRhs()) &&
        !matchInputs(multiply.getRhs(), multiply.getLhs())) {
      return failure();
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    auto v8i8Ty = VectorType::get({8}, i8Ty);
    auto v16i8Ty = VectorType::get({16}, i8Ty);
    auto v32i8Ty = VectorType::get({32}, i8Ty);
    auto v4i32Ty = VectorType::get({4}, i32Ty);
    auto v8i32Ty = VectorType::get({8}, i32Ty);

    Value weights =
        vector::ShapeCastOp::create(rewriter, loc, v32i8Ty, weightInput);
    Value weights01 = vector::ExtractStridedSliceOp::create(
        rewriter, loc, weights, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{16},
        ArrayRef<int64_t>{1});
    Value weights23 = vector::ExtractStridedSliceOp::create(
        rewriter, loc, weights, ArrayRef<int64_t>{16}, ArrayRef<int64_t>{16},
        ArrayRef<int64_t>{1});

    Value activation =
        vector::ShapeCastOp::create(rewriter, loc, v8i8Ty, activationInput);
    constexpr int64_t repeatActivation[] = {0, 1, 2, 3, 4, 5, 6, 7,
                                            0, 1, 2, 3, 4, 5, 6, 7};
    Value activationRepeated = vector::ShuffleOp::create(
        rewriter, loc, activation, activation, repeatActivation);

    Value carriedFlat =
        vector::ShapeCastOp::create(rewriter, loc, v8i32Ty, carried);
    Value carried01 = vector::ExtractStridedSliceOp::create(
        rewriter, loc, carriedFlat, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{4},
        ArrayRef<int64_t>{1});
    Value carried23 = vector::ExtractStridedSliceOp::create(
        rewriter, loc, carriedFlat, ArrayRef<int64_t>{4}, ArrayRef<int64_t>{4},
        ArrayRef<int64_t>{1});

    auto sdot = StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");
    Value result01 = LLVM::CallIntrinsicOp::create(
                         rewriter, loc, v4i32Ty, sdot,
                         ValueRange{carried01, weights01, activationRepeated})
                         .getResult(0);
    Value result23 = LLVM::CallIntrinsicOp::create(
                         rewriter, loc, v4i32Ty, sdot,
                         ValueRange{carried23, weights23, activationRepeated})
                         .getResult(0);
    constexpr int64_t concatenate[] = {0, 1, 2, 3, 4, 5, 6, 7};
    Value resultFlat = vector::ShuffleOp::create(rewriter, loc, result01,
                                                 result23, concatenate);
    Value result =
        vector::ShapeCastOp::create(rewriter, loc, resultTy, resultFlat);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// This function is used to identify bf16 dot product (expressed by elementwise
// multiplication follwed by a sum).
// For example, the following pattern: tl.sum(a * x[None, :], axis=1)
// is used to express a dot product.
// Since x is broadcated for the elementwise multiplication. And tl.sum will
// cast its bf16 input to fp32.
// The pattern in MLIR will be:
// BroadcastOp -> MulFOp -> ExtFOp -> MultiDimReductionOp
bool isBf16DotProduct(vector::MultiDimReductionOp op, bool useHorizontalSum,
                      Value &matInput, Value &vecInput,
                      PatternRewriter &rewriter) {
  Value src = op.getSource();
  Value acc = op.getAcc();
  auto srcTy = cast<VectorType>(src.getType());
  auto accTy = cast<VectorType>(acc.getType());
  auto resTy = cast<VectorType>(op.getType());

  auto srcRank = srcTy.getRank();
  auto outNum = srcTy.getDimSize(0);

  if (resTy != accTy || srcRank != 2 || !isFp32(srcTy))
    return false;

  if (op.isReducedDim(0) || !op.isReducedDim(1))
    return false;

  if (op.getKind() != vector::CombiningKind::ADD)
    return false;

  auto extFOp = src.getDefiningOp<arith::ExtFOp>();

  if (!extFOp || !extFOp->hasOneUse())
    return false;

  auto mulFOp = extFOp.getIn().getDefiningOp<arith::MulFOp>();

  if (!mulFOp || !mulFOp->hasOneUse())
    return false;

  Value lhs = mulFOp.getLhs();
  Value rhs = mulFOp.getRhs();

  auto lhsTy = cast<VectorType>(lhs.getType());
  auto rhsTy = cast<VectorType>(rhs.getType());

  if (!isBf16(lhsTy) || !isBf16(rhsTy))
    return false;

  const int lanes =
      vectorBitWidth / lhsTy.getElementType().getIntOrFloatBitWidth();
  const int resultLanes =
      vectorBitWidth / resTy.getElementType().getIntOrFloatBitWidth();
  int64_t kVal = lhsTy.getDimSize(1);

  if (outNum < 1)
    return false;

  if (!useHorizontalSum) {
    // TODO: masking is not currrently supported
    if (outNum % resultLanes != 0)
      return false;
  }

  // TODO: masking is not currrently supported
  if (kVal % lanes != 0)
    return false;

  if (outNum == 1) {
    matInput = lhs;
    vecInput = rhs;
  } else {
    vector::BroadcastOp broadCastOp;
    if (rhs.getDefiningOp<vector::BroadcastOp>()) {
      matInput = lhs;
      broadCastOp = rhs.getDefiningOp<vector::BroadcastOp>();
    } else {
      matInput = rhs;
      broadCastOp = lhs.getDefiningOp<vector::BroadcastOp>();
    }
    if (!broadCastOp || !broadCastOp->hasOneUse())
      return false;
    vecInput = broadCastOp.getSource();
  }

  if (cast<VectorType>(vecInput.getType()).getDimSize(0) != 1 ||
      cast<VectorType>(matInput.getType()).getDimSize(0) != outNum)
    return false;

  return true;
}

struct ConvertMulSumToDotHorizontalSum
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Value acc = op.getAcc();
    auto resTy = cast<VectorType>(op.getType());

    Value matInput;
    Value vecInput;

    bool isMatch = isBf16DotProduct(op, /*useHorizontalSum=*/true, matInput,
                                    vecInput, rewriter);
    if (!isMatch)
      return failure();

    // Once we get the matrix input (NxK) and vector input (K),
    // where N is the output channel dimension
    // and K is the reduction dimension.
    // We will generate the following code to perform the dot product.
    // For each output channel:
    // we will pull 8 bf16 elements from the vector and matrix each time when
    // we iterate over the K dimension.
    // We will then use bfdot to perform sum-of-products on pairs of
    // bf16 elements, accumulate and get 4 fp32 outputs.
    // After the iteration over the K dimension finishes, we will use a
    // horizontal sum (faddv) to sum the 4 fp32 into a single fp32.
    // We will also share the vector input across the output channels
    // to reduce the number of loads.
    // For example, if we dot product a size 2x16 matrix with a size 16 vector,
    // the pseudo code will be:
    // matrix = shapecast(matrix, 2x2x8)
    // vector = shapecast(vector, 2x8)
    // out    = zeros(2x4, fp32)
    // out[0] = bfdot(out[0], matrix[0][0], vector[0])
    // out[1] = bfdot(out[1], matrix[1][0], vector[0])
    // out[0] = bfdot(out[0], matrix[0][1], vector[1])
    // out[1] = bfdot(out[1], matrix[1][1], vector[1])
    // out_0  = faddv(out[0]) : 4xfp32 -> fp32
    // out_1  = faddv(out[1]) : 4xfp32 -> fp32

    auto matInputTy = cast<VectorType>(matInput.getType());
    auto vecInputTy = cast<VectorType>(vecInput.getType());

    const int lanes =
        vectorBitWidth / matInputTy.getElementType().getIntOrFloatBitWidth();
    const int resultLanes =
        vectorBitWidth / resTy.getElementType().getIntOrFloatBitWidth();
    int64_t kVal = matInputTy.getDimSize(1);

    // numOfOutputChannels is the number of output channels (N)
    const int numOfOutputChannels = matInputTy.getDimSize(0);
    // numOfBfdotOps is the number of bfdots needed for each output channel.
    const int numOfBfdotOps = kVal / lanes;

    matInput = shapeCast(loc, matInput,
                         {numOfOutputChannels, numOfBfdotOps, lanes}, rewriter);
    vecInput = shapeCast(loc, vecInput, {numOfBfdotOps, lanes}, rewriter);

    SmallVector<Value> outRes(numOfOutputChannels);
    SmallVector<Value> mats(numOfOutputChannels);

    Type outResTy = VectorType::get(resultLanes, resTy.getElementType());

    Value zeroRes = arith::ConstantOp::create(rewriter, loc, outResTy,
                                              rewriter.getZeroAttr(outResTy));
    for (int64_t outIdx = 0; outIdx < numOfOutputChannels; outIdx += 1) {
      outRes[outIdx] = zeroRes;
      // Intermediate array to store each row of the input matrix.
      mats[outIdx] = vector::ExtractOp::create(rewriter, loc, matInput, outIdx);
    }

    SmallVector<Type> resultTypes = {outResTy};
    // TODO: this intrinsic is hard-coded for Arm Neon
    auto bfdot = StringAttr::get(ctx, "llvm.aarch64.neon.bfdot.v4f32.v8bf16");
    SmallVector<Value> args;

    for (int64_t idx = 0; idx < numOfBfdotOps; idx += 1) {
      auto subVec = vector::ExtractOp::create(rewriter, loc, vecInput, idx);
      for (int64_t outIdx = 0; outIdx < numOfOutputChannels; outIdx += 1) {
        auto subMat =
            vector::ExtractOp::create(rewriter, loc, mats[outIdx], idx);
        args = {outRes[outIdx], subMat, subVec};
        // bfdot instruction:
        // https://developer.arm.com/documentation/ddi0602/2024-06/SIMD-FP-Instructions/BFDOT--vector---BFloat16-floating-point-dot-product--vector--
        // LLVM fast math flags:
        // https://llvm.org/docs/LangRef.html#fast-math-flags
        // This bfdot intrinsic will perform an unfused sum-of-products of each
        // pair of adjacent bf16 elements in the source vectors (8 bf16), and
        // output 4 fp32 elements.
        auto callIntrOp = LLVM::CallIntrinsicOp::create(
            rewriter, loc, resultTypes, bfdot, args,
            LLVM::FastmathFlagsAttr::get(ctx, LLVM::FastmathFlags::fast));
        outRes[outIdx] = callIntrOp.getResult(0);
      }
    }

    Value res = arith::ConstantOp::create(rewriter, loc, resTy,
                                          rewriter.getZeroAttr(resTy));

    resultTypes = {resTy.getElementType()};
    // TODO: this intrinsic is hard-coded for Arm Neon
    auto horzSum = StringAttr::get(ctx, "llvm.aarch64.neon.faddv.f32.v4f32");
    for (int64_t outIdx = 0; outIdx < numOfOutputChannels; outIdx += 1) {
      args = {outRes[outIdx]};
      // This horizontal sum intrinsic will sum all fp32 elements in the source
      // vector into a single fp32 element
      auto callIntrOp = LLVM::CallIntrinsicOp::create(
          rewriter, loc, resultTypes, horzSum, args,
          LLVM::FastmathFlagsAttr::get(ctx, LLVM::FastmathFlags::fast));
      res = vector::InsertOp::create(rewriter, loc, callIntrOp.getResult(0),
                                     res, outIdx);
    }

    if (!isZeroConst(acc)) {
      res = arith::AddFOp::create(rewriter, loc, res, acc);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertMulSumToDotPack
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Value acc = op.getAcc();
    auto resTy = cast<VectorType>(op.getType());

    Value matInput;
    Value vecInput;

    bool isMatch = isBf16DotProduct(op, /*useHorizontalSum=*/false, matInput,
                                    vecInput, rewriter);
    if (!isMatch)
      return failure();

    // Once we get the matrix input (NxK) and vector input (K),
    // where N is the output channel dimension
    // and K is the reduction dimension.
    // We will generate the following code to perform the dot product.
    // We will first transpose the matrix so that the output channel dimension
    // is continuous, so we can store multiple output channels in one
    // SIMD register.
    // Then we will loop over the K dimension.
    // For each iteration over K, we will pull 2 bf16 from the input vector.
    // Inside the K loop, we will also iterate over the output channels.
    // For each iteration over the output channel, we will pull
    // 4 output channel (each containing 2 bf16).
    // Then we will broadcast the 2 bf16 from the input vector,
    // dot product it with the 4 output channels (each containing 2 bf16),
    // and accumulate it with 4 outputs.
    // We will iterate over N until all output channels are processed.
    // Then we will move to the next 2 bf16 from the input vector (the K loop).
    // We will also share the vector input across the output channels.
    // For example, if we dot product a size 8x8 matrix with a size 8 vector,
    // the generated pseudo code will be:
    // Dimension:
    //            N:  the output channel dimension
    //            n0: the number of SIMD registers needed to store the output
    //                -- N / 4 (2 in this case)
    //            n1: the number of outputs stored per SIMD register
    //                -- 4
    //            K:  the reduction dimension
    //            k0: the number of SIMD registers needed for the input vector
    //                -- K / 8 (1 in this case)
    //            k1: the number of lanes per SIMD register
    //                -- 4
    //            k2: the number of bf16 elements per SIMD lane
    //                -- 2
    // matrix = shapecast(matrix, 8x4x2)
    //          shape: NxK -> Nx(k0xk1)xk2
    // matrix = tranpose(matrix, 1, 0, 2) : 8x4x2xbf16 -> 4x8x2xbf16
    //          shape: Nx(k0xk1)xk2 -> (k0xk1)xNxk2
    // matrix = shapecast(matrix, 1x4x2x4x2xbf16)
    //          shape: (k0xk1)xNxk2 -> k0xk1xn0xn1xk2
    // vector = shapecast(vector, 1x4x2)
    //          shape: K -> k0xk1xk2
    // out    = zeros(2x4, fp32)
    //          shape: n0xn1
    // subvec = broadcast(vector[0][0]) : 2xbf16 -> 4x2xbf16
    //          shape: k2 -> k1xk2
    // out[0] = bfdot(out[0], matrix[0][0][0], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out[1] = bfdot(out[1], matrix[0][0][1], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // subvec = broadcast(vector[0][1]) : 2xbf16 -> 4x2xbf16
    //          shape: k2 -> k1xk2
    // out[0] = bfdot(out[0], matrix[0][1][0], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out[1] = bfdot(out[1], matrix[0][1][1], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // subvec = broadcast(vector[0][2]) : 2xbf16 -> 4x2xbf16
    //          shape: k2 -> k1xk2
    // out[0] = bfdot(out[0], matrix[0][2][0], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out[1] = bfdot(out[1], matrix[0][2][1], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // subvec = broadcast(vector[0][3]) : 2xbf16 -> 4x2xbf16
    //          shape: k2 -> k1xk2
    // out[0] = bfdot(out[0], matrix[0][3][0], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out[1] = bfdot(out[1], matrix[0][3][1], subvec)
    //          shape: (n1, n1xk2, k1xk2) -> n1
    // out    = shapecast(out, 8) : 2x4xfp32 -> 8xfp32
    //          shape: n0xn1 -> N

    auto matInputTy = cast<VectorType>(matInput.getType());
    auto vecInputTy = cast<VectorType>(vecInput.getType());

    const int lanes =
        vectorBitWidth / matInputTy.getElementType().getIntOrFloatBitWidth();
    const int resultLanes =
        vectorBitWidth / resTy.getElementType().getIntOrFloatBitWidth();
    int64_t kVal = matInputTy.getDimSize(1);

    // numOfOutputChannels is the number of output channels (N)
    const int numOfOutputChannels = matInputTy.getDimSize(0);
    // numOfOutputRegs is the number of SIMD registers needed to store the
    // output.
    const int numOfOutputRegs = numOfOutputChannels / resultLanes;
    // numOfVecRegs is the number of SIMD registers needed for the
    // input vector.
    const int numOfVecRegs = kVal / lanes;
    // numOfVecPairs is the number of pairs (pair of bf16 elements) for the
    // input vector.
    const int numOfVecPairs = numOfVecRegs * resultLanes;

    VectorType fullResTy =
        VectorType::get({numOfOutputRegs, resultLanes}, resTy.getElementType());

    VectorType subResTy = VectorType::get(resultLanes, resTy.getElementType());

    acc = shapeCast(loc, acc, fullResTy, rewriter);

    Type inElemTy = matInputTy.getElementType();
    // Integer type for a pair of bf16 elements
    Type pairTy = IntegerType::get(ctx, 32);

    vecInput =
        shapeCast(loc, vecInput, {numOfVecRegs, resultLanes, 2}, rewriter);
    // We bitcast here because we are pulling pairs of bf16 each time.
    vecInput = vector::BitCastOp::create(
        rewriter, loc, VectorType::get({numOfVecRegs, resultLanes, 1}, pairTy),
        vecInput);
    vecInput = shapeCast(loc, vecInput, {numOfVecRegs, resultLanes}, rewriter);

    matInput = shapeCast(loc, matInput, {numOfOutputChannels, numOfVecPairs, 2},
                         rewriter);
    // We bitcast here because we are pulling pairs of bf16 each time.
    matInput = vector::BitCastOp::create(
        rewriter, loc,
        VectorType::get({numOfOutputChannels, numOfVecPairs, 1}, pairTy),
        matInput);
    matInput = shapeCast(loc, matInput, {numOfOutputChannels, numOfVecPairs},
                         rewriter);
    // Packing/Transposing the weight matrix so that
    // the output channel is continuous
    matInput = vector::TransposeOp::create(rewriter, loc, matInput,
                                           SmallVector<int64_t, 2>{1, 0});
    matInput = shapeCast(
        loc, matInput,
        {numOfVecRegs, resultLanes, numOfOutputRegs, resultLanes}, rewriter);

    Value res = arith::ConstantOp::create(rewriter, loc, fullResTy,
                                          rewriter.getZeroAttr(fullResTy));
    SmallVector<Type> resultTypes = {subResTy};
    // TODO: this intrinsic is hard-coded for Arm Neon
    auto bfdot = StringAttr::get(ctx, "llvm.aarch64.neon.bfdot.v4f32.v8bf16");
    SmallVector<Value> args;

    SmallVector<Value> subRes(numOfOutputRegs);
    for (int64_t outIdx = 0; outIdx < numOfOutputRegs; outIdx += 1) {
      subRes[outIdx] = vector::ExtractOp::create(rewriter, loc, acc, outIdx);
    }
    for (int64_t idx = 0; idx < numOfVecRegs; idx += 1) {
      Value fullVec = vector::ExtractOp::create(rewriter, loc, vecInput, idx);
      for (int64_t vecIdx = 0; vecIdx < resultLanes; vecIdx += 1) {
        // shuffle mask used to broadcast the 'vecIdx'th lane of fullVec
        SmallVector<int64_t> shuffleMask(resultLanes, vecIdx);
        // Broadcasting the 'vecIdx'th lane of fullVec
        Value subVec = vector::ShuffleOp::create(rewriter, loc, fullVec,
                                                 fullVec, shuffleMask);
        subVec = vector::BitCastOp::create(
            rewriter, loc, VectorType::get({lanes}, inElemTy), subVec);
        for (int64_t outIdx = 0; outIdx < numOfOutputRegs; outIdx += 1) {
          Value subMat = vector::ExtractOp::create(
              rewriter, loc, matInput,
              SmallVector<int64_t, 3>{idx, vecIdx, outIdx});
          subMat = vector::BitCastOp::create(
              rewriter, loc, VectorType::get({lanes}, inElemTy), subMat);
          args = {subRes[outIdx], subMat, subVec};
          // bfdot instruction:
          // https://developer.arm.com/documentation/ddi0602/2024-06/SIMD-FP-Instructions/BFDOT--vector---BFloat16-floating-point-dot-product--vector--
          // LLVM fast math flags:
          // https://llvm.org/docs/LangRef.html#fast-math-flags
          // This bfdot intrinsic will perform an unfused sum-of-products of
          // each pair of adjacent bf16 elements in the source vectors
          // (8 bf16), and output 4 fp32 elements.
          auto callIntrOp = LLVM::CallIntrinsicOp::create(
              rewriter, loc, resultTypes, bfdot, args,
              LLVM::FastmathFlagsAttr::get(ctx, LLVM::FastmathFlags::fast));
          subRes[outIdx] = callIntrOp.getResult(0);
        }
      }
    }

    for (int64_t outIdx = 0; outIdx < numOfOutputRegs; outIdx += 1) {
      res =
          vector::InsertOp::create(rewriter, loc, subRes[outIdx], res, outIdx);
    }

    res = shapeCast(loc, res, resTy, rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertDotProduct
    : public triton::cpu::impl::ConvertDotProductBase<ConvertDotProduct> {
  ConvertDotProduct() = default;
  ConvertDotProduct(bool useHorizontalSum) {
    this->useHorizontalSum = useHorizontalSum;
  }
  ConvertDotProduct(bool useHorizontalSum, bool enableBf16) {
    this->useHorizontalSum = useHorizontalSum;
    this->enableBf16 = enableBf16;
  }
  ConvertDotProduct(bool useHorizontalSum, bool enableBf16, bool enableI8) {
    this->useHorizontalSum = useHorizontalSum;
    this->enableBf16 = enableBf16;
    this->enableI8 = enableI8;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);

    if (enableI8) {
      patterns.add<ConvertI8RowDotAccumulate, ConvertI8RowDot,
                   ConvertI32PairReduction, ConvertI32ToF32Scale16,
                   ConvertPackedI8MulSumAccumulate>(context);
    }

    if (enableBf16) {
      if (useHorizontalSum) {
        patterns.add<ConvertMulSumToDotHorizontalSum>(context);
      } else {
        patterns.add<ConvertMulSumToDotPack>(context);
      }
    }

    if (failed(mlir::applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotProduct() {
  return std::make_unique<ConvertDotProduct>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotProduct(bool useHorizontalSum) {
  return std::make_unique<ConvertDotProduct>(useHorizontalSum);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotProduct(bool useHorizontalSum, bool enableBf16) {
  return std::make_unique<ConvertDotProduct>(useHorizontalSum, enableBf16);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotProduct(bool useHorizontalSum, bool enableBf16, bool enableI8) {
  return std::make_unique<ConvertDotProduct>(useHorizontalSum, enableBf16,
                                             enableI8);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
