// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s

// Compressed-tensors W4A8 uses an asymmetric, per-token activation.  The
// ordinary Triton kernel expresses its group integer result as
//
//   dot(q_activation, q_weight * 16) - activation_zp * sum(q_weight * 16)
//
// before applying the activation and weight scales.  Keep this graph on the
// packed M4 I8MM path and apply the zero-point correction in native i32 rows.
//
// CHECK-LABEL: @kai_q4_asymmetric_prefill
// CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
// CHECK: arith.subi %{{.*}}, %{{.*}} : vector<4xi32>
// CHECK: llvm.call_intrinsic "llvm.aarch64.neon.vcvtfxs2fp"
// CHECK-NOT: triton_cpu.dot

module {
  tt.func public @kai_q4_asymmetric_prefill(
      %lhs: memref<?xi8>, %rhs: memref<?xi8>,
      %lhs_scale: memref<?xf16>, %rhs_scale: memref<?xf16>,
      %lhs_zp: memref<?xi32>, %rhs_sum: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %shift = arith.constant dense<4> : vector<16x4xi8>
    %mask = arith.constant dense<-16> : vector<16x4xi8>
    %fixed = arith.constant dense<6.250000e-02> : vector<4x4xf32>
    %zero_i32 = arith.constant dense<0> : vector<4x4xi32>
    %zero_f32 = arith.constant dense<0.000000e+00> : vector<4x4xf32>

    %result = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%out = %zero_f32) -> vector<4x4xf32> {
      %rhs_raw = vector.load %rhs[%iv] : memref<?xi8>, vector<64xi8>
      %rhs_physical = vector.shape_cast %rhs_raw :
        vector<64xi8> to vector<2x4x8xi8>
      %rhs_transpose = vector.transpose %rhs_physical, [0, 2, 1] :
        vector<2x4x8xi8> to vector<2x8x4xi8>
      %rhs_packed = vector.shape_cast %rhs_transpose :
        vector<2x8x4xi8> to vector<16x4xi8>
      %rhs_low = arith.shli %rhs_packed, %shift : vector<16x4xi8>
      %rhs_high = arith.andi %rhs_packed, %mask : vector<16x4xi8>
      %rhs_join = vector.interleave %rhs_low, %rhs_high :
        vector<16x4xi8> -> vector<16x8xi8>
      %rhs_join_shape = vector.shape_cast %rhs_join :
        vector<16x8xi8> to vector<16x4x2xi8>
      %rhs_logical_transpose = vector.transpose %rhs_join_shape, [0, 2, 1] :
        vector<16x4x2xi8> to vector<16x2x4xi8>
      %rhs_logical = vector.shape_cast %rhs_logical_transpose :
        vector<16x2x4xi8> to vector<32x4xi8>

      %lhs_raw = vector.load %lhs[%iv] : memref<?xi8>, vector<128xi8>
      %lhs_physical = vector.shape_cast %lhs_raw :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs_physical_transpose = vector.transpose %lhs_physical, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs_halves = vector.shape_cast %lhs_physical_transpose :
        vector<4x4x8xi8> to vector<4x2x16xi8>
      %lhs_halves_transpose = vector.transpose %lhs_halves, [0, 2, 1] :
        vector<4x2x16xi8> to vector<4x16x2xi8>
      %lhs_logical = vector.shape_cast %lhs_halves_transpose :
        vector<4x16x2xi8> to vector<4x32xi8>

      %dot = triton_cpu.dot %lhs_logical, %rhs_logical, %zero_i32,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>

      %zp = vector.load %lhs_zp[%iv] : memref<?xi32>, vector<4xi32>
      %zp_shape = vector.shape_cast %zp :
        vector<4xi32> to vector<4x1xi32>
      %zp_broadcast = vector.broadcast %zp_shape :
        vector<4x1xi32> to vector<4x4xi32>
      %sum = vector.load %rhs_sum[%iv] : memref<?xi32>, vector<4xi32>
      %sum_broadcast = vector.broadcast %sum :
        vector<4xi32> to vector<4x4xi32>
      %correction = arith.muli %zp_broadcast, %sum_broadcast :
        vector<4x4xi32>
      %corrected = arith.subi %dot, %correction : vector<4x4xi32>

      %fp = arith.sitofp %corrected : vector<4x4xi32> to vector<4x4xf32>
      %scaled = arith.mulf %fp, %fixed : vector<4x4xf32>
      %ls_f16 = vector.load %lhs_scale[%iv] :
        memref<?xf16>, vector<4xf16>
      %ls = arith.extf %ls_f16 : vector<4xf16> to vector<4xf32>
      %ls_shape = vector.shape_cast %ls :
        vector<4xf32> to vector<4x1xf32>
      %ls_broadcast = vector.broadcast %ls_shape :
        vector<4x1xf32> to vector<4x4xf32>
      %rs_f16 = vector.load %rhs_scale[%iv] :
        memref<?xf16>, vector<4xf16>
      %rs = arith.extf %rs_f16 : vector<4xf16> to vector<4xf32>
      %rs_broadcast = vector.broadcast %rs :
        vector<4xf32> to vector<4x4xf32>
      %lhs_scaled = arith.mulf %scaled, %ls_broadcast : vector<4x4xf32>
      %group_scaled = arith.mulf %lhs_scaled, %rs_broadcast : vector<4x4xf32>
      %next = arith.addf %out, %group_scaled : vector<4x4xf32>
      scf.yield %next : vector<4x4xf32>
    }
    tt.return
  }
}
