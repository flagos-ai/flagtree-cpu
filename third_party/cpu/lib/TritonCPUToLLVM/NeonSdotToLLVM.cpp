/*
 * Lower NeonSdotOp → LLVM SDOT intrinsic
 * Lower CpuSdotGemvOp → runtime call to sdot_gemv_m1_prepacked()
 */

#include "cpu/include/TritonCPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

// ---------- NeonSdotOp → LLVM intrinsic ----------

struct NeonSdotOpLowering : public OpRewritePattern<NeonSdotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NeonSdotOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto v4i32Ty = VectorType::get({4}, rewriter.getI32Type());
    auto sdotName = StringAttr::get(ctx, "llvm.aarch64.neon.sdot.v4i32.v16i8");
    auto result = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, v4i32Ty, sdotName,
        ValueRange{op.getAcc(), op.getA(), op.getB()},
        LLVM::FastmathFlagsAttr());
    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

// ---------- CpuSdotGemvOp → runtime function call ----------

struct SdotGemvOpLowering : public OpRewritePattern<triton::cpu::SdotGemvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SdotGemvOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();

    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    // Get or declare the runtime function
    auto funcName = "sdot_gemv_m1_prepacked";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    // Convert tt.ptr to llvm.ptr via unrealized_conversion_cast
    auto castToLLVMPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType()))
        return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    auto aPtr = castToLLVMPtr(op.getAPtr());
    auto bPtr = castToLLVMPtr(op.getBPackedPtr());
    auto cPtr = castToLLVMPtr(op.getCPtr());

    auto four = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(4));
    auto N4 = rewriter.create<LLVM::SDivOp>(loc, i64Ty, op.getN(), four);

    rewriter.create<LLVM::CallOp>(
        loc, funcOp, ValueRange{aPtr, bPtr, cPtr, op.getK(), op.getN(), N4});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuFusedDecodeStepOp → runtime call (returns i64) ----------

struct FusedDecodeStepOpLowering
    : public OpRewritePattern<triton::cpu::FusedDecodeStepOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::FusedDecodeStepOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "fused_decode_step";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      // 2 i64 + 9 ptr + 8 i64 + 1 f32 = 20 args, returns i64
      SmallVector<Type, 20> argTypes;
      argTypes.push_back(i64Ty);  // token_id
      argTypes.push_back(i64Ty);  // pos
      for (int i = 0; i < 9; i++) argTypes.push_back(ptrTy);  // 9 ptrs
      for (int i = 0; i < 8; i++) argTypes.push_back(i64Ty);  // 8 dims
      argTypes.push_back(f32Ty);  // rms_eps
      auto funcType = LLVM::LLVMFunctionType::get(i64Ty, argTypes, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };
    auto toI64 = [&](Value v) -> Value {
      if (v.getType() == i64Ty) return v;
      return rewriter.create<LLVM::SExtOp>(loc, i64Ty, v);
    };

    auto epsVal = rewriter.create<LLVM::ConstantOp>(loc, f32Ty, op.getRmsEpsAttr());

    SmallVector<Value, 20> args;
    args.push_back(toI64(op.getTokenId()));
    args.push_back(toI64(op.getPos()));
    args.push_back(castPtr(op.getEmbedTable()));
    args.push_back(castPtr(op.getLayerPtrs()));
    args.push_back(castPtr(op.getKCache()));
    args.push_back(castPtr(op.getVCache()));
    args.push_back(castPtr(op.getRopeCos()));
    args.push_back(castPtr(op.getRopeSin()));
    args.push_back(castPtr(op.getFinalNormW()));
    args.push_back(castPtr(op.getLmHeadPacked()));
    args.push_back(castPtr(op.getLmHeadScale()));
    args.push_back(toI64(op.getHiddenDim()));
    args.push_back(toI64(op.getHeadDim()));
    args.push_back(toI64(op.getNHeads()));
    args.push_back(toI64(op.getNKvHeads()));
    args.push_back(toI64(op.getIntermediate()));
    args.push_back(toI64(op.getVocabSize()));
    args.push_back(toI64(op.getNLayers()));
    args.push_back(toI64(op.getMaxSeq()));
    args.push_back(epsVal);

    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

// ---------- CpuFusedTransformerLayerOp → runtime call ----------

struct FusedTransformerLayerOpLowering
    : public OpRewritePattern<triton::cpu::FusedTransformerLayerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::FusedTransformerLayerOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "fused_transformer_decode_layer";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      // 26 ptr + 5 i64 + 1 f32 = 32 args
      SmallVector<Type, 32> argTypes;
      // hidden_states
      argTypes.push_back(ptrTy);
      // wq wk wv wo wq_s wk_s wv_s wo_s (8 ptrs)
      for (int i = 0; i < 8; i++) argTypes.push_back(ptrTy);
      // q_norm k_norm cos sin (4 ptrs)
      for (int i = 0; i < 4; i++) argTypes.push_back(ptrTy);
      // k_cache v_cache (2 ptrs)
      argTypes.push_back(ptrTy); argTypes.push_back(ptrTy);
      // cache_pos max_seq (2 i64)
      argTypes.push_back(i64Ty); argTypes.push_back(i64Ty);
      // gate up down gate_s up_s down_s (6 ptrs)
      for (int i = 0; i < 6; i++) argTypes.push_back(ptrTy);
      // input_norm post_norm (2 ptrs)
      argTypes.push_back(ptrTy); argTypes.push_back(ptrTy);
      // hidden head_dim n_heads n_kv_heads intermediate (5 i64)
      for (int i = 0; i < 5; i++) argTypes.push_back(i64Ty);
      // rms_eps (1 f32)
      argTypes.push_back(f32Ty);

      auto funcType = LLVM::LLVMFunctionType::get(voidTy, argTypes, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    auto epsVal = rewriter.create<LLVM::ConstantOp>(loc, f32Ty, op.getRmsEpsAttr());

    SmallVector<Value, 32> args;
    args.push_back(castPtr(op.getHiddenStates()));
    args.push_back(castPtr(op.getWq()));
    args.push_back(castPtr(op.getWk()));
    args.push_back(castPtr(op.getWv()));
    args.push_back(castPtr(op.getWo()));
    args.push_back(castPtr(op.getWqS()));
    args.push_back(castPtr(op.getWkS()));
    args.push_back(castPtr(op.getWvS()));
    args.push_back(castPtr(op.getWoS()));
    args.push_back(castPtr(op.getQNormW()));
    args.push_back(castPtr(op.getKNormW()));
    args.push_back(castPtr(op.getCosEmb()));
    args.push_back(castPtr(op.getSinEmb()));
    args.push_back(castPtr(op.getKCache()));
    args.push_back(castPtr(op.getVCache()));
    args.push_back(op.getCachePos());
    args.push_back(op.getMaxSeqLen());
    args.push_back(castPtr(op.getGateW()));
    args.push_back(castPtr(op.getUpW()));
    args.push_back(castPtr(op.getDownW()));
    args.push_back(castPtr(op.getGateS()));
    args.push_back(castPtr(op.getUpS()));
    args.push_back(castPtr(op.getDownS()));
    args.push_back(castPtr(op.getInputNormW()));
    args.push_back(castPtr(op.getPostNormW()));
    args.push_back(op.getHiddenDim());
    args.push_back(op.getHeadDim());
    args.push_back(op.getNHeads());
    args.push_back(op.getNKvHeads());
    args.push_back(op.getIntermediate());
    args.push_back(epsVal);

    rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSdotGemvFusedBf16Op → runtime call ----------

struct SdotGemvFusedBf16OpLowering
    : public OpRewritePattern<triton::cpu::SdotGemvFusedBf16Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SdotGemvFusedBf16Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "sdot_gemv_m1_fused_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      // 7 args: x_bf16, B_packed, w_scale, out_bf16, K, N, N4
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    // N4 = N / 4
    auto four = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(4));
    auto N4 = rewriter.create<LLVM::SDivOp>(loc, i64Ty, op.getN(), four);

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getXPtr()), castPtr(op.getBPackedPtr()),
                   castPtr(op.getWScalePtr()), castPtr(op.getOutPtr()),
                   op.getK(), op.getN(), N4});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSdotPackWeightsOp → runtime call ----------

struct SdotPackWeightsOpLowering
    : public OpRewritePattern<triton::cpu::SdotPackWeightsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SdotPackWeightsOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "sdot_pack_weights";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getBPtr()), castPtr(op.getBPackedPtr()),
                   op.getK(), op.getN()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuFusedMlpOp → runtime call ----------

struct FusedMlpOpLowering : public OpRewritePattern<triton::cpu::FusedMlpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::FusedMlpOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "fused_mlp_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty},
          false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getXPtr()),
                   castPtr(op.getGatePackedPtr()),
                   castPtr(op.getUpPackedPtr()),
                   castPtr(op.getGateScalePtr()),
                   castPtr(op.getUpScalePtr()),
                   castPtr(op.getOutPtr()),
                   op.getK(), op.getN()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSmeGemmOp → runtime call (ARM SME int8 GEMM) ----------

struct SmeGemmOpLowering : public OpRewritePattern<triton::cpu::SmeGemmOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SmeGemmOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "sme_gemm_int32";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getApPtr()), castPtr(op.getBpPtr()),
                   castPtr(op.getCPtr()), op.getMp(), op.getNp(), op.getK4()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSmmlaUkOp → runtime call (TLE-Struct micro-kernel) ----------

struct SmmlaUkOpLowering : public OpRewritePattern<triton::cpu::SmmlaUkOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SmmlaUkOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "smmla_uk";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty},
          false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getApPtr()), castPtr(op.getWpPtr()), castPtr(op.getCPtr()),
                   castPtr(op.getXsPtr()), castPtr(op.getWsPtr()),
                   op.getK8(), op.getMP(), op.getN(), op.getMp0(), op.getNp0()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- More TLE-Struct micro-kernel ops → runtime calls ----------
namespace {
LLVM::LLVMFuncOp getOrDeclare(PatternRewriter &rewriter, ModuleOp module,
                              StringRef name, ArrayRef<Type> argTys) {
  auto ctx = rewriter.getContext();
  auto f = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (f) return f;
  auto ft = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argTys, false);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), name, ft);
}
} // namespace

#define UK_LOWERING(OpName, RtName)                                            \
  struct OpName##Lowering : public OpRewritePattern<triton::cpu::OpName> {      \
    using OpRewritePattern::OpRewritePattern;                                   \
    LogicalResult matchAndRewrite(triton::cpu::OpName op,                       \
                                  PatternRewriter &rewriter) const override {   \
      auto loc = op.getLoc(); auto ctx = rewriter.getContext();                 \
      auto module = op->getParentOfType<ModuleOp>();                           \
      auto i64Ty = IntegerType::get(ctx, 64);                                   \
      auto ptrTy = LLVM::LLVMPointerType::get(ctx);                             \
      SmallVector<Type> tys;                                                    \
      for (auto v : op->getOperands())                                          \
        tys.push_back(isa<IntegerType>(v.getType()) ? (Type)i64Ty : (Type)ptrTy);\
      auto fn = getOrDeclare(rewriter, module, RtName, tys);                    \
      auto castPtr = [&](Value v) -> Value {                                    \
        if (isa<LLVM::LLVMPointerType>(v.getType())) return v;                  \
        if (isa<IntegerType>(v.getType())) return v;                            \
        return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v).getResult(0);\
      };                                                                        \
      SmallVector<Value> args;                                                  \
      for (auto v : op->getOperands()) args.push_back(castPtr(v));              \
      rewriter.create<LLVM::CallOp>(loc, fn, args);                             \
      rewriter.eraseOp(op);                                                     \
      return success();                                                         \
    }                                                                           \
  };
UK_LOWERING(SmeUkOp, "sme_uk")
UK_LOWERING(SdotGemvUkOp, "sdot_gemv_uk")
UK_LOWERING(SwigluUkOp, "swiglu_uk")
UK_LOWERING(RmsnormUkOp, "rmsnorm_uk")
UK_LOWERING(ResidualUkOp, "residual_uk")

// ---------- CpuFlashAttnDecodeOp → runtime call ----------

struct FlashAttnDecodeOpLowering
    : public OpRewritePattern<triton::cpu::FlashAttnDecodeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::FlashAttnDecodeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "flash_attn_decode_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, ptrTy,
                   i64Ty, i64Ty, f32Ty, i64Ty, i64Ty, i64Ty, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    auto smScaleVal = rewriter.create<LLVM::ConstantOp>(
        loc, f32Ty, op.getSmScaleAttr());

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getQPtr()), castPtr(op.getKPtr()),
                   castPtr(op.getVPtr()), castPtr(op.getOutPtr()),
                   op.getSeqLen(), op.getHeadDim(), smScaleVal,
                   op.getNumHeads(), op.getNumKvHeads(),
                   op.getStrideKn(), op.getStrideVn()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuRmsNormOp → runtime call ----------

struct RmsNormOpLowering : public OpRewritePattern<triton::cpu::RmsNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::RmsNormOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto f32Ty = Float32Type::get(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "standalone_rms_norm_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty, f32Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    // Extract eps from F32Attr
    auto epsVal = rewriter.create<LLVM::ConstantOp>(
        loc, f32Ty, op.getEpsAttr());

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getXPtr()), castPtr(op.getWeightPtr()),
                   castPtr(op.getOutPtr()), op.getD(), epsVal});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- CpuSwigluOp → runtime call ----------

struct SwigluOpLowering : public OpRewritePattern<triton::cpu::SwigluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::cpu::SwigluOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Ty = IntegerType::get(ctx, 64);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);

    auto funcName = "swiglu_bf16";
    auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!funcOp) {
      auto voidTy = LLVM::LLVMVoidType::get(ctx);
      auto funcType = LLVM::LLVMFunctionType::get(
          voidTy, {ptrTy, ptrTy, ptrTy, i64Ty}, false);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(
          UnknownLoc::get(ctx), funcName, funcType);
    }

    auto castPtr = [&](Value v) -> Value {
      if (isa<LLVM::LLVMPointerType>(v.getType())) return v;
      return rewriter.create<UnrealizedConversionCastOp>(loc, ptrTy, v)
          .getResult(0);
    };

    rewriter.create<LLVM::CallOp>(
        loc, funcOp,
        ValueRange{castPtr(op.getGatePtr()), castPtr(op.getUpPtr()),
                   castPtr(op.getOutPtr()), op.getN()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------- Pass ----------

namespace mlir::triton::cpu {

std::unique_ptr<Pass> createNeonSdotToLLVMPass() {
  struct NeonSdotToLLVMPass : public OperationPass<ModuleOp> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NeonSdotToLLVMPass)
    NeonSdotToLLVMPass()
        : OperationPass<ModuleOp>(TypeID::get<NeonSdotToLLVMPass>()) {}
    StringRef getName() const override { return "NeonSdotToLLVM"; }
    std::unique_ptr<Pass> clonePass() const override {
      return std::make_unique<NeonSdotToLLVMPass>();
    }
    StringRef getArgument() const override {
      return "convert-neon-sdot-to-llvm";
    }
    void runOnOperation() override {
      auto *ctx = &getContext();
      RewritePatternSet patterns(ctx);
      patterns.add<NeonSdotOpLowering>(ctx);
      patterns.add<SdotGemvOpLowering>(ctx);
      patterns.add<SdotGemvFusedBf16OpLowering>(ctx);
      patterns.add<SdotPackWeightsOpLowering>(ctx);
      patterns.add<RmsNormOpLowering>(ctx);
      patterns.add<SwigluOpLowering>(ctx);
      patterns.add<FlashAttnDecodeOpLowering>(ctx);
      patterns.add<FusedMlpOpLowering>(ctx);
      patterns.add<FusedTransformerLayerOpLowering>(ctx);
      patterns.add<FusedDecodeStepOpLowering>(ctx);
      patterns.add<SmeGemmOpLowering>(ctx);
      patterns.add<SmmlaUkOpLowering>(ctx);
      patterns.add<SmeUkOpLowering>(ctx);
      patterns.add<SdotGemvUkOpLowering>(ctx);
      patterns.add<SwigluUkOpLowering>(ctx);
      patterns.add<RmsnormUkOpLowering>(ctx);
      patterns.add<ResidualUkOpLowering>(ctx);
      if (failed(applyPatternsGreedily(getOperation(),
                                               std::move(patterns))))
        signalPassFailure();
    }
  };
  return std::make_unique<NeonSdotToLLVMPass>();
}

} // namespace mlir::triton::cpu
