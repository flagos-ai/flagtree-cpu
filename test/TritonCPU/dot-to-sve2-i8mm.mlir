// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm | FileCheck %s
// RUN: triton-opt %s -triton-cpu-convert-dot-to-sve2-i8mm='fixed-only=true' | FileCheck %s --check-prefix=FIXED

// The SVE2 path must remain a rolled 8x8x8 microkernel.  In particular, do
// not expand the complete Triton tile into hundreds of simultaneously-live
// fixed-vector SSA values.
//
// CHECK-LABEL: @s8_gemm_16x16x32
// CHECK-NOT:       vector.transfer_write
// CHECK:       memref.alloca() {alignment = 64 : i64} : memref<4x4x16xi8>
// CHECK:       scf.for {{.*}} step {{.*}} {
// CHECK:         scf.for {{.*}} step {{.*}} {
// CHECK-COUNT-8:   vector.load {{.*}} vector<8xi8>
// CHECK:           vector.shuffle
// CHECK:           vector.bitcast
// CHECK-NOT:       vector.transpose
// CHECK:         scf.for {{.*}} step {{.*}} {
// CHECK:           llvm.mlir.zero
// CHECK:           scf.for {{.*}} iter_args(
// CHECK-COUNT-16:    llvm.call_intrinsic "llvm.aarch64.sve.smmla.nxv4i32"
// CHECK-NOT:     triton_cpu.dot
// FIXED-LABEL: @s8_gemm_16x16x32
// FIXED-NOT:   llvm.aarch64.sve
// FIXED:       triton_cpu.dot

#loc = loc(unknown)
module {
  tt.func public @s8_gemm_16x16x32(
      %lhs: memref<16x32xi8, strided<[?, 1]>>,
      %rhs: memref<32x16xi8, strided<[?, 1]>>) {
    %c0 = arith.constant 0 : index
    %zero_i8 = arith.constant 0 : i8
    %zero = arith.constant dense<0> : vector<16x16xi32>
    %a = vector.transfer_read %lhs[%c0, %c0], %zero_i8
      {in_bounds = [true, true]} :
      memref<16x32xi8, strided<[?, 1]>>, vector<16x32xi8>
    %b = vector.transfer_read %rhs[%c0, %c0], %zero_i8
      {in_bounds = [true, true]} :
      memref<32x16xi8, strided<[?, 1]>>, vector<32x16xi8>
    %result = triton_cpu.dot %a, %b, %zero, inputPrecision = ieee :
      vector<16x32xi8> * vector<32x16xi8> -> vector<16x16xi32>
    tt.return
  }

  // CHECK-LABEL: @s8_gemv_packed_n4_128
  // CHECK-NOT:   memref.alloca() {{.*}} memref<4x128xi8>
  // CHECK:       vector.load %{{.*}}[{{.*}}] : memref<512xi8>, vector<16xi8>
  // CHECK:       llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // CHECK-NOT:   vector.transfer_write {{.*}} memref<4x128xi8>
  // CHECK-NOT:   triton_cpu.dot
  // FIXED-LABEL: @s8_gemv_packed_n4_128
  // FIXED-NOT:   llvm.aarch64.sve
  // FIXED:       llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // FIXED-NOT:   triton_cpu.dot
  tt.func public @s8_gemv_packed_n4_128(
      %lhs: memref<1x4xi8, strided<[?, 1]>>,
      %rhs: memref<512xi8>) {
    %c0 = arith.constant 0 : index
    %zero_i8 = arith.constant 0 : i8
    %zero = arith.constant dense<0> : vector<1x128xi32>
    %a = vector.transfer_read %lhs[%c0, %c0], %zero_i8
      {in_bounds = [true, true]} :
      memref<1x4xi8, strided<[?, 1]>>, vector<1x4xi8>
    %flat = vector.load %rhs[%c0] : memref<512xi8>, vector<512xi8>
    %physical = vector.shape_cast %flat :
      vector<512xi8> to vector<128x4xi8>
    %logical = vector.transpose %physical, [1, 0] :
      vector<128x4xi8> to vector<4x128xi8>
    %result = triton_cpu.dot %a, %logical, %zero, inputPrecision = ieee :
      vector<1x4xi8> * vector<4x128xi8> -> vector<1x128xi32>
    tt.return
  }

  // Match only the exact Q4_0 nibble expression.  The lowering emits two
  // fixed SDOTs and therefore must reject lookalike mask/shift/zero-point
  // expressions instead of changing their semantics.
  // CHECK-LABEL: @w4_nibble_pair_exact
  // CHECK-COUNT-2: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // CHECK-NOT: triton_cpu.dot
  // FIXED-LABEL: @w4_nibble_pair_exact
  // FIXED-COUNT-2: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // FIXED-NOT: triton_cpu.dot
  tt.func public @w4_nibble_pair_exact(%packed: vector<4x4xi8>,
                                       %lhs0: vector<1x4xi8>,
                                       %lhs1: vector<1x4xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mask = arith.constant dense<15> : vector<4x4xi8>
    %shift = arith.constant dense<4> : vector<4x4xi8>
    %zp = arith.constant dense<8> : vector<4x4xi8>
    %zero = arith.constant dense<0> : vector<1x4xi32>
    %result = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%acc = %zero) -> vector<1x4xi32> {
      %low_bits = arith.andi %packed, %mask : vector<4x4xi8>
      %low = arith.subi %low_bits, %zp : vector<4x4xi8>
      %high_bits = arith.shrui %packed, %shift : vector<4x4xi8>
      %high = arith.subi %high_bits, %zp : vector<4x4xi8>
      %low_dot = triton_cpu.dot %lhs0, %low, %acc, inputPrecision = ieee :
        vector<1x4xi8> * vector<4x4xi8> -> vector<1x4xi32>
      %high_dot = triton_cpu.dot %lhs1, %high, %low_dot,
        inputPrecision = ieee :
        vector<1x4xi8> * vector<4x4xi8> -> vector<1x4xi32>
      scf.yield %high_dot : vector<1x4xi32>
    }
    tt.return
  }

  // CHECK-LABEL: @w4_nibble_pair_wrong_mask
  // CHECK: %[[MASKED:.*]] = arith.andi
  // CHECK: %[[LOW:.*]] = arith.subi %[[MASKED]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui
  // CHECK: %[[HIGH:.*]] = arith.subi %[[SHIFTED]]
  // CHECK: vector.transfer_write %[[LOW]],
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // CHECK: vector.transfer_write %[[HIGH]],
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // FIXED-LABEL: @w4_nibble_pair_wrong_mask
  // FIXED: %[[MASKED:.*]] = arith.andi
  // FIXED: %[[LOW:.*]] = arith.subi %[[MASKED]]
  // FIXED: %[[SHIFTED:.*]] = arith.shrui
  // FIXED: %[[HIGH:.*]] = arith.subi %[[SHIFTED]]
  // FIXED: vector.transfer_write %[[LOW]],
  // FIXED: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // FIXED: vector.transfer_write %[[HIGH]],
  // FIXED: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  tt.func public @w4_nibble_pair_wrong_mask(%packed: vector<4x4xi8>,
                                            %lhs0: vector<1x4xi8>,
                                            %lhs1: vector<1x4xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mask = arith.constant dense<7> : vector<4x4xi8>
    %shift = arith.constant dense<4> : vector<4x4xi8>
    %zp = arith.constant dense<8> : vector<4x4xi8>
    %zero = arith.constant dense<0> : vector<1x4xi32>
    %result = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%acc = %zero) -> vector<1x4xi32> {
      %low_bits = arith.andi %packed, %mask : vector<4x4xi8>
      %low = arith.subi %low_bits, %zp : vector<4x4xi8>
      %high_bits = arith.shrui %packed, %shift : vector<4x4xi8>
      %high = arith.subi %high_bits, %zp : vector<4x4xi8>
      %low_dot = triton_cpu.dot %lhs0, %low, %acc, inputPrecision = ieee :
        vector<1x4xi8> * vector<4x4xi8> -> vector<1x4xi32>
      %high_dot = triton_cpu.dot %lhs1, %high, %low_dot,
        inputPrecision = ieee :
        vector<1x4xi8> * vector<4x4xi8> -> vector<1x4xi32>
      scf.yield %high_dot : vector<1x4xi32>
    }
    tt.return
  }

  // CHECK-LABEL: @w4_nibble_pair_wrong_shift
  // CHECK: %[[MASKED:.*]] = arith.andi
  // CHECK: %[[LOW:.*]] = arith.subi %[[MASKED]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui
  // CHECK: %[[HIGH:.*]] = arith.subi %[[SHIFTED]]
  // CHECK: vector.transfer_write %[[LOW]],
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // CHECK: vector.transfer_write %[[HIGH]],
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // FIXED-LABEL: @w4_nibble_pair_wrong_shift
  // FIXED: %[[MASKED:.*]] = arith.andi
  // FIXED: %[[LOW:.*]] = arith.subi %[[MASKED]]
  // FIXED: %[[SHIFTED:.*]] = arith.shrui
  // FIXED: %[[HIGH:.*]] = arith.subi %[[SHIFTED]]
  // FIXED: vector.transfer_write %[[LOW]],
  // FIXED: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // FIXED: vector.transfer_write %[[HIGH]],
  // FIXED: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  tt.func public @w4_nibble_pair_wrong_shift(%packed: vector<4x4xi8>,
                                             %lhs0: vector<1x4xi8>,
                                             %lhs1: vector<1x4xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mask = arith.constant dense<15> : vector<4x4xi8>
    %shift = arith.constant dense<3> : vector<4x4xi8>
    %zp = arith.constant dense<8> : vector<4x4xi8>
    %zero = arith.constant dense<0> : vector<1x4xi32>
    %result = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%acc = %zero) -> vector<1x4xi32> {
      %low_bits = arith.andi %packed, %mask : vector<4x4xi8>
      %low = arith.subi %low_bits, %zp : vector<4x4xi8>
      %high_bits = arith.shrui %packed, %shift : vector<4x4xi8>
      %high = arith.subi %high_bits, %zp : vector<4x4xi8>
      %low_dot = triton_cpu.dot %lhs0, %low, %acc, inputPrecision = ieee :
        vector<1x4xi8> * vector<4x4xi8> -> vector<1x4xi32>
      %high_dot = triton_cpu.dot %lhs1, %high, %low_dot,
        inputPrecision = ieee :
        vector<1x4xi8> * vector<4x4xi8> -> vector<1x4xi32>
      scf.yield %high_dot : vector<1x4xi32>
    }
    tt.return
  }

  // CHECK-LABEL: @w4_nibble_pair_wrong_zero_point
  // CHECK: %[[MASKED:.*]] = arith.andi
  // CHECK: %[[LOW:.*]] = arith.subi %[[MASKED]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui
  // CHECK: %[[HIGH:.*]] = arith.subi %[[SHIFTED]]
  // CHECK: vector.transfer_write %[[LOW]],
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // CHECK: vector.transfer_write %[[HIGH]],
  // CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // FIXED-LABEL: @w4_nibble_pair_wrong_zero_point
  // FIXED: %[[MASKED:.*]] = arith.andi
  // FIXED: %[[LOW:.*]] = arith.subi %[[MASKED]]
  // FIXED: %[[SHIFTED:.*]] = arith.shrui
  // FIXED: %[[HIGH:.*]] = arith.subi %[[SHIFTED]]
  // FIXED: vector.transfer_write %[[LOW]],
  // FIXED: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  // FIXED: vector.transfer_write %[[HIGH]],
  // FIXED: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
  tt.func public @w4_nibble_pair_wrong_zero_point(
      %packed: vector<4x4xi8>, %lhs0: vector<1x4xi8>,
      %lhs1: vector<1x4xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mask = arith.constant dense<15> : vector<4x4xi8>
    %shift = arith.constant dense<4> : vector<4x4xi8>
    %low_zp = arith.constant dense<7> : vector<4x4xi8>
    %high_zp = arith.constant dense<8> : vector<4x4xi8>
    %zero = arith.constant dense<0> : vector<1x4xi32>
    %result = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%acc = %zero) -> vector<1x4xi32> {
      %low_bits = arith.andi %packed, %mask : vector<4x4xi8>
      %low = arith.subi %low_bits, %low_zp : vector<4x4xi8>
      %high_bits = arith.shrui %packed, %shift : vector<4x4xi8>
      %high = arith.subi %high_bits, %high_zp : vector<4x4xi8>
      %low_dot = triton_cpu.dot %lhs0, %low, %acc, inputPrecision = ieee :
        vector<1x4xi8> * vector<4x4xi8> -> vector<1x4xi32>
      %high_dot = triton_cpu.dot %lhs1, %high, %low_dot,
        inputPrecision = ieee :
        vector<1x4xi8> * vector<4x4xi8> -> vector<1x4xi32>
      scf.yield %high_dot : vector<1x4xi32>
    }
    tt.return
  }
}
