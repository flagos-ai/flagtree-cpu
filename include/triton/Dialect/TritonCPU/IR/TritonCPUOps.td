#ifndef TRITONCPU_OPS
#define TRITONCPU_OPS

include "triton/Dialect/TritonCPU/IR/TritonCPUDialect.td"
include "triton/Dialect/TritonCPU/IR/TritonCPUTypes.td"
include "triton/Dialect/TritonCPU/IR/TritonCPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class TTC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonCPU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

//
// External Elementwise op
//
def TTC_ExternElementwiseOp : TTC_Op<"extern_elementwise", [Elementwise,
                                                            SameOperandsAndResultEncoding,
                                                            SameVariadicOperandSize,
                                                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

    let description = [{
        Similar to TT_ExternElementwiseOp, but only supports calls to libsleef at the moment.
        The string "%s(numel)" in $symbol will be interpolated with the number of elements of
        the vector argument(s).
    }];

    let arguments = (ins Variadic<TTC_Type>:$srcs, StrAttr:$symbol, BoolAttr:$pure);

    let results = (outs TTC_Type:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";
}

//
// NEON SDOT: signed int8 dot product (4-lane, 4 pairs each)
// acc[i] += sum_{k=0}^{3}(a[i*4+k] * b[i*4+k])  for i in 0..3
//
def TTC_NeonSdotOp : TTC_Op<"neon_sdot", [Pure]> {
    let summary = "NEON SDOT: 4-lane signed int8 dot product accumulate";

    let description = [{
        Computes NEON SDOT (signed dot product) on ARM64.
        Takes accumulator (vector<4xi32>), two operands (vector<16xi8>),
        and returns updated accumulator (vector<4xi32>).

        Each lane computes: acc[i] += sum_{k=0}^{3}(a[i*4+k] * b[i*4+k])

        Lowered to: llvm.aarch64.neon.sdot.v4i32.v16i8 intrinsic.

        Usage in TLE CPU:
          acc = tle.cpu.sdot(acc, a_broadcast, b_packed)
    }];

    let arguments = (ins
        TTC_Type:$acc,
        TTC_Type:$a,
        TTC_Type:$b
    );

    let results = (outs TTC_Type:$result);

    let assemblyFormat = "$acc `,` $a `,` $b attr-dict `:` type($acc) `,` type($a) `,` type($b) `->` type($result)";
}

def TTC_ExtractMemRefOp : TTC_Op<"extract_memref", [NoMemoryEffect]> {
  let summary = "Extract base memref from a block pointer";

  let description = [{
    Extract base memref from a block pointer. It covers whole base tensor memory,
    not only the block referenced. Base pointer, shape, and strides are used
    in the resulting memref. Offsets and block shape are ignored.

  }];

  let arguments = (ins TT_TensorPtr:$src);

  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTC_ExtractIndicesOp : TTC_Op<"extract_indices", [NoMemoryEffect]> {
  let summary = "Extract indices from a block pointer.";

  let description = [{
    Extract indices that can be used to access the block using its base memref.
    Indices are supposed to be used for vector loads/stores with the base
    memref extracted from the same block pointer.
  }];

  let arguments = (ins TT_TensorPtr:$src);

  let results = (outs Variadic<Index>:$result);

  let builders = [
    OpBuilder<(ins "Value":$src)>
  ];

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTC_PtrToMemRefOp : TTC_Op<"ptr_to_memref", [NoMemoryEffect]> {
  let summary = "Build a memref for a pointer.";

  let description = [{
    Build memref with static shape, offset, strides, and specified base pointer.
  }];

  let arguments = (ins TT_Ptr:$src);

  let results = (outs AnyStaticShapeMemRef:$result);

  let hasCanonicalizer = 0;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;


def TTC_LoadOp : TTC_Op<"load", [
  MemoryEffects<[MemRead<GlobalMemory>]>,
]> {
  let summary = "Load from a memref to triton tensor";

  let description = [{
    Operation to allow load from allocated temporary buffer to triton tensor.
  }];

  let arguments = (ins AnyMemRef:$src);

  let results = (outs TT_Type:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTC_StoreOp : TTC_Op<"store", [
  MemoryEffects<[MemWrite<GlobalMemory>]>,
]> {
  let summary = "Store triton tensor to memref";

  let description = [{
    Operation to allow store triton tensor to allocated temporary buffer.
  }];

  let arguments = (
    ins
    TT_Type:$src,
    AnyMemRef:$dst
  );

  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def TTC_PrintOp : TTC_Op<"print", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Print at most a single scalar or vector (converted from tensor) on each line";

  let description = [{
    For converting tensor types to vector types.
    It only takes a single scalar or vector (tensor) element.
  }];

  let arguments = (ins
    StrAttr:$prefix,
    BoolAttr:$hex,
    Variadic<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr, AnyRankedOrUnrankedMemRef]>>:$val,
    DenseI32ArrayAttr:$isSigned
  );

  let assemblyFormat = [{
    $prefix attr-dict (`:` $val^ `:` type($val))?
  }];

  let hasVerifier = 1;
}

def TTC_AssertOp : TTC_Op<"assert", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "For correctness checking";
  let description = [{
    Takes a condition tensor, a message string, a file string, a function string, and a line number.
    If the condition is false, the message is printed, and the program is aborted.
  }];
  let arguments = (ins I1:$condition, StrAttr:$message);
  let assemblyFormat = "$condition `,` $message attr-dict `:` type($condition)";
}

def TTC_DotOp : TTC_Op<"dot", [Pure,
                               DeclareOpInterfaceMethods<InferTypeOpInterface>,
                               TypesMatchWith<"result's type matches accumulator's type",
                                              "d", "c", "$_self">]> {
    let summary = "dot";

    let description = [{Same as tt.dot but on vectors.}];

    let arguments = (
      ins
      TTC_Vector:$a,
      TTC_Vector:$b,
      TTC_Vector:$c,
      DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
      DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc
    );

    let results = (outs TTC_Vector:$d);

    let assemblyFormat = [{
      $a`,` $b`,` $c (`,` `inputPrecision` `=` $inputPrecision^)? attr-dict `:`
      type($a) `*` type($b) `->` type($d)
    }];
}

def TTC_BrgemmCreate : TTC_Op<"brgemm_create", [NoMemoryEffect]> {
  let summary = "Crete ukernels handles";

  let description = [{For creation of ukernels, that can be used to replace op with dot-like sematnics}];

  // M, N, K_k, batch_size, lda, ldb, ldc, dtypeA, dtypeB, dtypeC
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$M,
    AnyTypeOf<[AnyInteger, Index]>:$N,
    AnyTypeOf<[AnyInteger, Index]>:$K_k,
    AnyTypeOf<[AnyInteger, Index]>:$batch_size,
    AnyTypeOf<[AnyInteger, Index]>:$lda,
    AnyTypeOf<[AnyInteger, Index]>:$ldb,
    AnyTypeOf<[AnyInteger, Index]>:$ldc,
    AnyTypeOf<[AnyInteger, Index]>:$stepA,
    AnyTypeOf<[AnyInteger, Index]>:$stepB,
    // TODO: Maybe Use properties
    TypeAttr:$dtypeA,
    TypeAttr:$dtypeB,
    TypeAttr:$dtypeC,
    I1:$skipPacking
  );

  let results = (outs Index:$result);
}

def TTC_BrgemmExecute : TTC_Op<"brgemm_execute",
    [MemoryEffects<[MemRead<GlobalMemory>,
                    MemWrite<GlobalMemory>]>]> {
  let summary = "Call ukernel with existing handle for passed ops";

  let arguments = (ins
    Index:$brgemm_kernel_hash,
    Arg<AnyMemRef, "Lhs/In buffer data ptr", [MemRead]>:$A_ptr,
    Arg<AnyMemRef, "Rhs/Weight buffer data ptr", [MemRead]>:$B_ptr,
    Arg<AnyMemRef, "Result/Acc buffer data ptr", [MemWrite]>:$C_ptr,
    AnyTypeOf<[AnyInteger, Index]>:$stepA,
    AnyTypeOf<[AnyInteger, Index]>:$stepB,
    AnyTypeOf<[AnyInteger, Index]>:$blockedBsize,
    AnyTypeOf<[AnyInteger, Index]>:$numBatches,
    I1:$skipPacking
  );
}

// ─────────────────────────────────────────────────────────────────────
// ARM64 TLE extended operations (moved from core Triton dialect).
// Emitted from @triton.jit via the tle_arm64 plugin's create_cpu_* builders.
// ─────────────────────────────────────────────────────────────────────

def TTC_FusedDecodeStepOp : TTC_Op<"fused_decode_step", []> {
    let summary = "TLE-CPU: Full decode step in one C call, returns next token ID";
    let arguments = (ins
        I64:$token_id, I64:$pos,
        TT_Ptr:$embed_table,
        TT_Ptr:$layer_ptrs,
        TT_Ptr:$k_cache, TT_Ptr:$v_cache,
        TT_Ptr:$rope_cos, TT_Ptr:$rope_sin,
        TT_Ptr:$final_norm_w,
        TT_Ptr:$lm_head_packed, TT_Ptr:$lm_head_scale,
        I64:$hidden_dim, I64:$head_dim, I64:$n_heads, I64:$n_kv_heads,
        I64:$intermediate, I64:$vocab_size, I64:$n_layers,
        I64:$max_seq, F32Attr:$rms_eps
    );
    let results = (outs I64:$next_token);
    let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def TTC_FusedTransformerLayerOp : TTC_Op<"fused_transformer_layer", []> {
    let summary = "TLE-CPU: Full transformer decode layer in one C call";
    let description = [{
      RMSNorm → QKV GEMV → QK_Norm → RoPE → KV_cache_write →
      Attention → O_GEMV → Residual → RMSNorm → Gate+Up+SWIGLU → Down_GEMV → Residual.
      Zero tensor allocation. Zero Python dispatch within layer.
    }];
    let arguments = (ins
        TT_Ptr:$hidden_states,
        TT_Ptr:$wq, TT_Ptr:$wk, TT_Ptr:$wv, TT_Ptr:$wo,
        TT_Ptr:$wq_s, TT_Ptr:$wk_s, TT_Ptr:$wv_s, TT_Ptr:$wo_s,
        TT_Ptr:$q_norm_w, TT_Ptr:$k_norm_w,
        TT_Ptr:$cos_emb, TT_Ptr:$sin_emb,
        TT_Ptr:$k_cache, TT_Ptr:$v_cache,
        I64:$cache_pos, I64:$max_seq_len,
        TT_Ptr:$gate_w, TT_Ptr:$up_w, TT_Ptr:$down_w,
        TT_Ptr:$gate_s, TT_Ptr:$up_s, TT_Ptr:$down_s,
        TT_Ptr:$input_norm_w, TT_Ptr:$post_norm_w,
        I64:$hidden_dim, I64:$head_dim, I64:$n_heads, I64:$n_kv_heads,
        I64:$intermediate, F32Attr:$rms_eps
    );
    let results = (outs);
    let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def TTC_FusedMlpOp : TTC_Op<"fused_mlp", []> {
    let summary = "TLE-CPU: Fused MLP = gate SDOT GEMV + up SDOT GEMV + SWIGLU";
    let arguments = (ins TT_Ptr:$x_ptr,
                         TT_Ptr:$gate_packed_ptr, TT_Ptr:$up_packed_ptr,
                         TT_Ptr:$gate_scale_ptr, TT_Ptr:$up_scale_ptr,
                         TT_Ptr:$out_ptr,
                         I64:$K, I64:$N);
    let results = (outs);
    let assemblyFormat = "$x_ptr `,` $gate_packed_ptr `,` $up_packed_ptr `,` $gate_scale_ptr `,` $up_scale_ptr `,` $out_ptr `,` $K `,` $N attr-dict `:` type($x_ptr) `,` type($gate_packed_ptr) `,` type($up_packed_ptr) `,` type($gate_scale_ptr) `,` type($up_scale_ptr) `,` type($out_ptr)";
}

def TTC_FlashAttnDecodeOp : TTC_Op<"flash_attn_decode", []> {
    let summary = "TLE-CPU: M=1 Flash Attention with online softmax";
    let arguments = (ins TT_Ptr:$q_ptr, TT_Ptr:$k_ptr, TT_Ptr:$v_ptr,
                         TT_Ptr:$out_ptr,
                         I64:$seq_len, I64:$head_dim,
                         F32Attr:$sm_scale,
                         I64:$num_heads, I64:$num_kv_heads,
                         I64:$stride_kn, I64:$stride_vn);
    let results = (outs);
    let assemblyFormat = "$q_ptr `,` $k_ptr `,` $v_ptr `,` $out_ptr `,` $seq_len `,` $head_dim `,` $num_heads `,` $num_kv_heads `,` $stride_kn `,` $stride_vn attr-dict `:` type($q_ptr) `,` type($k_ptr) `,` type($v_ptr) `,` type($out_ptr)";
}

def TTC_RmsNormOp : TTC_Op<"rms_norm", []> {
    let summary = "TLE-CPU: RMSNorm (x / rms(x) * weight)";
    let arguments = (ins TT_Ptr:$x_ptr, TT_Ptr:$weight_ptr, TT_Ptr:$out_ptr, I64:$D, F32Attr:$eps);
    let results = (outs);
    let assemblyFormat = "$x_ptr `,` $weight_ptr `,` $out_ptr `,` $D attr-dict `:` type($x_ptr) `,` type($weight_ptr) `,` type($out_ptr)";
}

def TTC_SwigluOp : TTC_Op<"swiglu", []> {
    let summary = "TLE-CPU: Fused SWIGLU activation (silu(gate) * up)";
    let arguments = (ins TT_Ptr:$gate_ptr, TT_Ptr:$up_ptr, TT_Ptr:$out_ptr, I64:$N);
    let results = (outs);
    let assemblyFormat = "$gate_ptr `,` $up_ptr `,` $out_ptr `,` $N attr-dict `:` type($gate_ptr) `,` type($up_ptr) `,` type($out_ptr)";
}

def TTC_NeonSdotTensorOp : TTC_Op<"neon_sdot_tensor", [Pure]> {
    let summary = "NEON SDOT (tensor entry): 4-lane signed int8 dot product accumulate";
    let description = [{
      TLE-CPU tensor-typed entry for NEON SDOT, emitted inside @triton.jit kernels.
      acc: tensor<4xi32>, a: tensor<16xi8>, b: tensor<16xi8> -> tensor<4xi32>
      Converted to the vector-typed triton_cpu.neon_sdot in TritonToTritonCPU,
      then lowered to llvm.aarch64.neon.sdot.v4i32.v16i8.
    }];
    let arguments = (ins TT_Type:$acc, TT_Type:$a, TT_Type:$b);
    let results = (outs TT_Type:$result);
    let assemblyFormat = "$acc `,` $a `,` $b attr-dict `:` type($acc) `,` type($a) `,` type($b) `->` type($result)";
}

def TTC_SdotGemvOp : TTC_Op<"sdot_gemv", []> {
    let summary = "TLE-CPU: M=1 INT8 GEMV micro-kernel using NEON SDOT";
    let description = [{
      Performs M=1 INT8 GEMV with pre-packed weights via NEON SDOT.
      A: [K] int8, B_packed: [K//4, N//4, 4, 4] int8, C: [N] int32 output.
      Calls the sdot_gemv_m1_prepacked runtime function in libTritonCPURuntime.so.
      OMP parallelized internally.
    }];
    let arguments = (ins
        TT_Ptr:$a_ptr,
        TT_Ptr:$b_packed_ptr,
        TT_Ptr:$c_ptr,
        I64:$K,
        I64:$N
    );
    let results = (outs);
    let assemblyFormat = "$a_ptr `,` $b_packed_ptr `,` $c_ptr `,` $K `,` $N attr-dict `:` type($a_ptr) `,` type($b_packed_ptr) `,` type($c_ptr)";
}

def TTC_SdotGemvFusedBf16Op : TTC_Op<"sdot_gemv_fused_bf16", []> {
    let summary = "TLE-CPU: Fused BF16 INT8 GEMV with dynamic activation quantization";
    let description = [{
      Fused: BF16 activation → dynamic INT8 quantization → SDOT GEMV → dequant → BF16 output.
      Eliminates Python-side abs/div/clamp/to overhead.
      x: [K] bf16, B_packed: SDOT format, w_scale: [N] fp32, out: [N] bf16.
    }];
    let arguments = (ins TT_Ptr:$x_ptr, TT_Ptr:$b_packed_ptr,
                         TT_Ptr:$w_scale_ptr, TT_Ptr:$out_ptr,
                         I64:$K, I64:$N);
    let results = (outs);
    let assemblyFormat = "$x_ptr `,` $b_packed_ptr `,` $w_scale_ptr `,` $out_ptr `,` $K `,` $N attr-dict `:` type($x_ptr) `,` type($b_packed_ptr) `,` type($w_scale_ptr) `,` type($out_ptr)";
}

def TTC_SdotPackWeightsOp : TTC_Op<"sdot_pack_weights", []> {
    let summary = "TLE-CPU: Pack INT8 weights from row-major to SDOT lane format";
    let arguments = (ins TT_Ptr:$b_ptr, TT_Ptr:$b_packed_ptr,
                         I64:$K, I64:$N);
    let results = (outs);
    let assemblyFormat = "$b_ptr `,` $b_packed_ptr `,` $K `,` $N attr-dict `:` type($b_ptr) `,` type($b_packed_ptr)";
}

#endif
