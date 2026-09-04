// RUN: triton-opt %s \
// RUN:   -triton-cpu-convert-dot-to-sve2-i8mm="fixed-only=true" \
// RUN:   | FileCheck %s

// A native G128 Q4 kernel rolls four physical K32 panels into one integer
// accumulator before applying the common scale and zero-point correction.
// The packed-Q4 matcher must therefore accept a loop-carried dot C operand,
// not only a literal zero.
//
// CHECK-LABEL: @kai_q4_g128_integer_acc
// CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
// The loop-carried C fragments feed SMMLA directly. Do not materialize a
// zero-based dot followed by a logical 4x4 vector add on every K32 group.
// CHECK-NOT: arith.addi %{{.*}}, %{{.*}} : vector<4x4xi32>
// CHECK-NOT: triton_cpu.dot

module {
  tt.func public @kai_q4_g128_integer_acc(
      %lhs: memref<?xi8>, %rhs: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %shift = arith.constant dense<4> : vector<16x4xi8>
    %mask = arith.constant dense<-16> : vector<16x4xi8>
    %zero = arith.constant dense<0> : vector<4x4xi32>

    %result = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%acc = %zero) -> vector<4x4xi32> {
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

      %dot = triton_cpu.dot %lhs_logical, %rhs_logical, %acc,
        inputPrecision = ieee :
        vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
      scf.yield %dot : vector<4x4xi32>
    }
    tt.return
  }

  // Two M4 panels using one weight value model the BLOCK_M=8 G128 source.
  // The lowering must form one shared RHS microkernel fragment while still
  // preserving each dot's independent C accumulator.
  // CHECK-LABEL: @kai_q4_g128_shared_rhs
  // CHECK-COUNT-32: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
  // Standalone dots retain the established zero-dot-plus-C form so decode
  // kernels do not inherit the prefill-only loop-carried accumulator rewrite.
  // CHECK-COUNT-2: arith.addi %{{.*}}, %{{.*}} : vector<4x4xi32>
  // CHECK-NOT: triton_cpu.dot
  tt.func public @kai_q4_g128_shared_rhs(
      %lhs0: memref<?xi8>, %lhs1: memref<?xi8>, %rhs: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %shift = arith.constant dense<4> : vector<16x4xi8>
    %mask = arith.constant dense<-16> : vector<16x4xi8>
    %zero = arith.constant dense<0> : vector<4x4xi32>

    %rhs_raw = vector.load %rhs[%c0] : memref<?xi8>, vector<64xi8>
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

    %lhs0_raw = vector.load %lhs0[%c0] : memref<?xi8>, vector<128xi8>
    %lhs0_physical = vector.shape_cast %lhs0_raw :
      vector<128xi8> to vector<4x4x8xi8>
    %lhs0_transpose = vector.transpose %lhs0_physical, [1, 0, 2] :
      vector<4x4x8xi8> to vector<4x4x8xi8>
    %lhs0_halves = vector.shape_cast %lhs0_transpose :
      vector<4x4x8xi8> to vector<4x2x16xi8>
    %lhs0_halves_transpose = vector.transpose %lhs0_halves, [0, 2, 1] :
      vector<4x2x16xi8> to vector<4x16x2xi8>
    %lhs0_logical = vector.shape_cast %lhs0_halves_transpose :
      vector<4x16x2xi8> to vector<4x32xi8>

    %lhs1_raw = vector.load %lhs1[%c0] : memref<?xi8>, vector<128xi8>
    %lhs1_physical = vector.shape_cast %lhs1_raw :
      vector<128xi8> to vector<4x4x8xi8>
    %lhs1_transpose = vector.transpose %lhs1_physical, [1, 0, 2] :
      vector<4x4x8xi8> to vector<4x4x8xi8>
    %lhs1_halves = vector.shape_cast %lhs1_transpose :
      vector<4x4x8xi8> to vector<4x2x16xi8>
    %lhs1_halves_transpose = vector.transpose %lhs1_halves, [0, 2, 1] :
      vector<4x2x16xi8> to vector<4x16x2xi8>
    %lhs1_logical = vector.shape_cast %lhs1_halves_transpose :
      vector<4x16x2xi8> to vector<4x32xi8>

    %dot0 = triton_cpu.dot %lhs0_logical, %rhs_logical, %zero,
      inputPrecision = ieee :
      vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
    %dot1 = triton_cpu.dot %lhs1_logical, %rhs_logical, %zero,
      inputPrecision = ieee :
      vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
    tt.return
  }

  // If the first dot is consumed before the second one, a shared lowering
  // cannot be inserted at the second dot without violating SSA dominance.
  // Fall back to two independent M4 fragments and preserve the early store.
  // CHECK-LABEL: @kai_q4_g128_shared_rhs_early_use
  // CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
  // CHECK: vector.store
  // CHECK-COUNT-16: llvm.call_intrinsic "llvm.aarch64.neon.smmla.v4i32.v16i8"
  // CHECK-NOT: triton_cpu.dot
  tt.func public @kai_q4_g128_shared_rhs_early_use(
      %lhs0: memref<?xi8>, %lhs1: memref<?xi8>, %rhs: memref<?xi8>,
      %output: memref<?x4x4xi32>) {
    %c0 = arith.constant 0 : index
    %shift = arith.constant dense<4> : vector<16x4xi8>
    %mask = arith.constant dense<-16> : vector<16x4xi8>
    %zero = arith.constant dense<0> : vector<4x4xi32>

    %rhs_raw = vector.load %rhs[%c0] : memref<?xi8>, vector<64xi8>
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

    %lhs0_raw = vector.load %lhs0[%c0] : memref<?xi8>, vector<128xi8>
    %lhs0_physical = vector.shape_cast %lhs0_raw :
      vector<128xi8> to vector<4x4x8xi8>
    %lhs0_transpose = vector.transpose %lhs0_physical, [1, 0, 2] :
      vector<4x4x8xi8> to vector<4x4x8xi8>
    %lhs0_halves = vector.shape_cast %lhs0_transpose :
      vector<4x4x8xi8> to vector<4x2x16xi8>
    %lhs0_halves_transpose = vector.transpose %lhs0_halves, [0, 2, 1] :
      vector<4x2x16xi8> to vector<4x16x2xi8>
    %lhs0_logical = vector.shape_cast %lhs0_halves_transpose :
      vector<4x16x2xi8> to vector<4x32xi8>

    %lhs1_raw = vector.load %lhs1[%c0] : memref<?xi8>, vector<128xi8>
    %lhs1_physical = vector.shape_cast %lhs1_raw :
      vector<128xi8> to vector<4x4x8xi8>
    %lhs1_transpose = vector.transpose %lhs1_physical, [1, 0, 2] :
      vector<4x4x8xi8> to vector<4x4x8xi8>
    %lhs1_halves = vector.shape_cast %lhs1_transpose :
      vector<4x4x8xi8> to vector<4x2x16xi8>
    %lhs1_halves_transpose = vector.transpose %lhs1_halves, [0, 2, 1] :
      vector<4x2x16xi8> to vector<4x16x2xi8>
    %lhs1_logical = vector.shape_cast %lhs1_halves_transpose :
      vector<4x16x2xi8> to vector<4x32xi8>

    %dot0 = triton_cpu.dot %lhs0_logical, %rhs_logical, %zero,
      inputPrecision = ieee :
      vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
    vector.store %dot0, %output[%c0, %c0, %c0] :
      memref<?x4x4xi32>, vector<4x4xi32>
    %dot1 = triton_cpu.dot %lhs1_logical, %rhs_logical, %zero,
      inputPrecision = ieee :
      vector<4x32xi8> * vector<32x4xi8> -> vector<4x4xi32>
    tt.return
  }
}
