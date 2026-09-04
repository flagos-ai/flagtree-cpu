// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s

// A TTIR loop-unroll factor creates a linear chain of packed dots for each
// loop-carried accumulator. Every copy must remain on the native path; it is
// invalid to convert only the first dot and generically expand the second.
// CHECK-LABEL: @kai_w8_prefill_m4_unroll2
// CHECK-COUNT-32: llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
// CHECK-NOT: triton_cpu.dot

module {
  tt.func public @kai_w8_prefill_m4_unroll2(
      %lhs: memref<?xi8>, %rhs: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant dense<0> : vector<4x4xi32>
    %result = scf.for %iv = %c0 to %c32 step %c2
        iter_args(%acc = %zero) -> (vector<4x4xi32>) {
      %lhs_raw0 = vector.load %lhs[%iv] : memref<?xi8>, vector<128xi8>
      %lhs_panel0 = vector.shape_cast %lhs_raw0 :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs_transpose0 = vector.transpose %lhs_panel0, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs_logical0 = vector.shape_cast %lhs_transpose0 :
        vector<4x4x8xi8> to vector<4x32xi8>
      %rhs_raw0 = vector.load %rhs[%iv] : memref<?xi8>, vector<128xi8>
      %rhs_panel0 = vector.shape_cast %rhs_raw0 :
        vector<128xi8> to vector<4x4x8xi8>
      %rhs_transpose0 = vector.transpose %rhs_panel0, [0, 2, 1] :
        vector<4x4x8xi8> to vector<4x8x4xi8>
      %rhs_logical0 = vector.shape_cast %rhs_transpose0 :
        vector<4x8x4xi8> to vector<32x4xi8>
      %next0 = triton_cpu.dot %lhs_logical0, %rhs_logical0, %acc,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>

      %iv1 = arith.addi %iv, %c1 : index
      %lhs_raw1 = vector.load %lhs[%iv1] : memref<?xi8>, vector<128xi8>
      %lhs_panel1 = vector.shape_cast %lhs_raw1 :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs_transpose1 = vector.transpose %lhs_panel1, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs_logical1 = vector.shape_cast %lhs_transpose1 :
        vector<4x4x8xi8> to vector<4x32xi8>
      %rhs_raw1 = vector.load %rhs[%iv1] : memref<?xi8>, vector<128xi8>
      %rhs_panel1 = vector.shape_cast %rhs_raw1 :
        vector<128xi8> to vector<4x4x8xi8>
      %rhs_transpose1 = vector.transpose %rhs_panel1, [0, 2, 1] :
        vector<4x4x8xi8> to vector<4x8x4xi8>
      %rhs_logical1 = vector.shape_cast %rhs_transpose1 :
        vector<4x8x4xi8> to vector<32x4xi8>
      %next1 = triton_cpu.dot %lhs_logical1, %rhs_logical1, %next0,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      scf.yield %next1 : vector<4x4xi32>
    }
    tt.return
  }

  // A packed-dot loop with an observable side effect must not be replaced as
  // a whole. Individual dots may still use the semantics-preserving native
  // fallback, but the original loop and store must survive.
  // CHECK-LABEL: @kai_w8_prefill_m4_unroll2_with_store
  // CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
  // CHECK-COUNT-1: triton_cpu.dot
  // CHECK: memref.store
  // CHECK: tt.return
  tt.func public @kai_w8_prefill_m4_unroll2_with_store(
      %lhs: memref<?xi8>, %rhs: memref<?xi8>, %sink: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %one = arith.constant 1 : i32
    %zero = arith.constant dense<0> : vector<4x4xi32>
    %result = scf.for %iv = %c0 to %c32 step %c2
        iter_args(%acc = %zero) -> (vector<4x4xi32>) {
      %lhs_raw0 = vector.load %lhs[%iv] : memref<?xi8>, vector<128xi8>
      %lhs_panel0 = vector.shape_cast %lhs_raw0 :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs_transpose0 = vector.transpose %lhs_panel0, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs_logical0 = vector.shape_cast %lhs_transpose0 :
        vector<4x4x8xi8> to vector<4x32xi8>
      %rhs_raw0 = vector.load %rhs[%iv] : memref<?xi8>, vector<128xi8>
      %rhs_panel0 = vector.shape_cast %rhs_raw0 :
        vector<128xi8> to vector<4x4x8xi8>
      %rhs_transpose0 = vector.transpose %rhs_panel0, [0, 2, 1] :
        vector<4x4x8xi8> to vector<4x8x4xi8>
      %rhs_logical0 = vector.shape_cast %rhs_transpose0 :
        vector<4x8x4xi8> to vector<32x4xi8>
      %next0 = triton_cpu.dot %lhs_logical0, %rhs_logical0, %acc,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>

      %iv1 = arith.addi %iv, %c1 : index
      %lhs_raw1 = vector.load %lhs[%iv1] : memref<?xi8>, vector<128xi8>
      %lhs_panel1 = vector.shape_cast %lhs_raw1 :
        vector<128xi8> to vector<4x4x8xi8>
      %lhs_transpose1 = vector.transpose %lhs_panel1, [1, 0, 2] :
        vector<4x4x8xi8> to vector<4x4x8xi8>
      %lhs_logical1 = vector.shape_cast %lhs_transpose1 :
        vector<4x4x8xi8> to vector<4x32xi8>
      %rhs_raw1 = vector.load %rhs[%iv1] : memref<?xi8>, vector<128xi8>
      %rhs_panel1 = vector.shape_cast %rhs_raw1 :
        vector<128xi8> to vector<4x4x8xi8>
      %rhs_transpose1 = vector.transpose %rhs_panel1, [0, 2, 1] :
        vector<4x4x8xi8> to vector<4x8x4xi8>
      %rhs_logical1 = vector.shape_cast %rhs_transpose1 :
        vector<4x8x4xi8> to vector<32x4xi8>
      %next1 = triton_cpu.dot %lhs_logical1, %rhs_logical1, %next0,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      memref.store %one, %sink[%iv] : memref<?xi32>
      scf.yield %next1 : vector<4x4xi32>
    }
    tt.return
  }
}
