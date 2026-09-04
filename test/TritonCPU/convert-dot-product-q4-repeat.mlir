// RUN: triton-opt %s --triton-cpu-convert-dot-product="enable-i8=true" | FileCheck %s

module {
  // Protect the Triton 3.7 lowering of
  // tl.join(x8, x8).permute(1, 0).reshape(4, 4).  The conversion must retain
  // the original vector<8xi8> value and broadcast it through i64 so AArch64
  // instruction selection can use LD1R instead of LDR(D) plus lane moves.
  tt.func public @q4_repeated_k8(
      %activation: vector<8xi8>, %weight: vector<4x4xi8>,
      %accumulator: vector<4xi32>) -> vector<4xi32> {
    %joined = vector.interleave %activation, %activation
      : vector<8xi8> -> vector<16xi8>
    %joined_2d = vector.shape_cast %joined
      : vector<16xi8> to vector<8x2xi8>
    %transposed = vector.transpose %joined_2d, [1, 0]
      : vector<8x2xi8> to vector<2x8xi8>
    %matrix = vector.shape_cast %transposed
      : vector<2x8xi8> to vector<4x4xi8>
    %activation_i32 = arith.extsi %matrix
      : vector<4x4xi8> to vector<4x4xi32>
    %weight_i32 = arith.extsi %weight
      : vector<4x4xi8> to vector<4x4xi32>
    %products = arith.muli %activation_i32, %weight_i32
      : vector<4x4xi32>
    %zero = arith.constant dense<0> : vector<4xi32>
    %dot = vector.multi_reduction <add>, %products, %zero [1]
      : vector<4x4xi32> to vector<4xi32>
    %result = arith.addi %accumulator, %dot : vector<4xi32>
    tt.return %result : vector<4xi32>
  }

  // The shape alone is insufficient: two different eight-byte inputs must
  // retain the ordinary interleave graph rather than becoming a broadcast.
  tt.func public @q4_distinct_k8(
      %activation0: vector<8xi8>, %activation1: vector<8xi8>,
      %weight: vector<4x4xi8>, %accumulator: vector<4xi32>)
      -> vector<4xi32> {
    %joined = vector.interleave %activation0, %activation1
      : vector<8xi8> -> vector<16xi8>
    %joined_2d = vector.shape_cast %joined
      : vector<16xi8> to vector<8x2xi8>
    %transposed = vector.transpose %joined_2d, [1, 0]
      : vector<8x2xi8> to vector<2x8xi8>
    %matrix = vector.shape_cast %transposed
      : vector<2x8xi8> to vector<4x4xi8>
    %activation_i32 = arith.extsi %matrix
      : vector<4x4xi8> to vector<4x4xi32>
    %weight_i32 = arith.extsi %weight
      : vector<4x4xi8> to vector<4x4xi32>
    %products = arith.muli %activation_i32, %weight_i32
      : vector<4x4xi32>
    %zero = arith.constant dense<0> : vector<4xi32>
    %dot = vector.multi_reduction <add>, %products, %zero [1]
      : vector<4x4xi32> to vector<4xi32>
    %result = arith.addi %accumulator, %dot : vector<4xi32>
    tt.return %result : vector<4xi32>
  }
}

// CHECK-LABEL: tt.func public @q4_repeated_k8
// CHECK: %[[SCALAR:.*]] = llvm.bitcast %arg0 : vector<8xi8> to i64
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[SCALAR]] : i64 to vector<2xi64>
// CHECK: %[[REPEATED:.*]] = llvm.bitcast %[[BROADCAST]] : vector<2xi64> to vector<16xi8>
// CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"{{.*}}%[[REPEATED]]
// CHECK-NOT: vector.interleave %arg0, %arg0

// CHECK-LABEL: tt.func public @q4_distinct_k8
// CHECK-NOT: vector.broadcast
// CHECK: vector.interleave %arg0, %arg1
// CHECK: llvm.call_intrinsic "llvm.aarch64.neon.sdot.v4i32.v16i8"
