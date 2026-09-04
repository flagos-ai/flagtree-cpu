// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-ops=use-gather-scatter=true -cse | FileCheck %s
// RUN: triton-opt %s -split-input-file -triton-cpu-mark-wide-bf16-stores-volatile | FileCheck %s --check-prefix=A720

// Convert strided masked loads to gather.

// CHECK-LABEL: @strided_masked_loads
// CHECK:       %[[PTR:.+]] = triton_cpu.ptr_to_memref %[[BASE:.+]] : <i32> -> memref<i32>
// CHECK:       %[[VAL:.+]] = vector.gather %[[PTR]][] [%[[INDEX_VEC:.+]]], %[[MASK:.+]], %[[OTHER:.+]] : memref<i32>, vector<32xi32>, vector<32xi1>, vector<32xi32> into vector<32xi32>

module {
  tt.func public @strided_masked_loads(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<2> : tensor<32xi32>
    %cst_0 = arith.constant dense<16> : tensor<32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = arith.cmpi slt, %0, %cst_0 : tensor<32xi32>
    %2 = arith.muli %0, %cst : tensor<32xi32>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %5 = tt.load %4, %1 : tensor<32x!tt.ptr<i32>>
      tt.store %4, %5 : tensor<32x!tt.ptr<i32>>
    }
    tt.return
  }
}

// -----

// An in-place loop loads and stores the same SSA address, so it must retain a
// normal store.  This keeps RoPE-style kernels out of the A720 workaround.

// A720-LABEL: llvm.func @keep_in_place_store
// A720:       %[[VALUE:.*]] = llvm.load %[[PTR:.*]] : !llvm.ptr -> vector<16xbf16>
// A720-NEXT:  llvm.store %[[VALUE]], %[[PTR]] : vector<16xbf16>, !llvm.ptr
// A720-NOT:   llvm.store volatile

module {
  llvm.func @keep_in_place_store(%ptr: !llvm.ptr, %condition: i1) {
    llvm.br ^loop
  ^loop:
    %value = llvm.load %ptr : !llvm.ptr -> vector<16xbf16>
    llvm.store %value, %ptr : vector<16xbf16>, !llvm.ptr
    llvm.cond_br %condition, ^loop, ^exit
  ^exit:
    llvm.return
  }
}

// -----

// The A720 workaround is deliberately exact: only a fixed 16-lane BF16 store
// in a loop becomes volatile.  Neighboring widths, element types, and the same
// wide store outside a loop retain ordinary LLVM memory semantics.

// A720-LABEL: llvm.func @mark_only_wide_bf16
// A720:       llvm.store volatile %{{.*}}, %{{.*}} : vector<16xbf16>, !llvm.ptr
// A720:       llvm.store %{{.*}}, %{{.*}} : vector<8xbf16>, !llvm.ptr
// A720:       llvm.store %{{.*}}, %{{.*}} : vector<16xf16>, !llvm.ptr
// A720:       llvm.store %{{.*}}, %{{.*}} : vector<16xbf16>, !llvm.ptr

module {
  llvm.func @mark_only_wide_bf16(
      %ptr0: !llvm.ptr, %ptr1: !llvm.ptr, %ptr2: !llvm.ptr,
      %wide: vector<16xbf16>, %narrow: vector<8xbf16>,
      %fp16: vector<16xf16>, %condition: i1) {
    llvm.br ^loop
  ^loop:
    llvm.store %wide, %ptr0 : vector<16xbf16>, !llvm.ptr
    llvm.store %narrow, %ptr1 : vector<8xbf16>, !llvm.ptr
    llvm.store %fp16, %ptr2 : vector<16xf16>, !llvm.ptr
    llvm.cond_br %condition, ^loop, ^exit
  ^exit:
    llvm.store %wide, %ptr0 : vector<16xbf16>, !llvm.ptr
    llvm.return
  }
}

// -----

// Convert strided masked stores to scatter.

// CHECK-LABEL: @strided_masked_stores
// CHECK:       %[[PTR:.+]] = triton_cpu.ptr_to_memref %[[BASE:.+]] : <i32> -> memref<i32>
// CHECK:       vector.scatter %[[PTR]][] [%[[INDEX_VEC:.+]]], %[[MASK:.+]], %[[VALS:.+]] : memref<i32>, vector<32xi32>, vector<32xi1>, vector<32xi32>

module {
  tt.func public @strided_masked_stores(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) {
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<32xi32>
    %cst_0 = arith.constant dense<2> : tensor<32xi32>
    %cst_1 = arith.constant dense<16> : tensor<32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = arith.cmpi slt, %0, %cst_1 : tensor<32xi32>
    %2 = arith.muli %0, %cst_0 : tensor<32xi32>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %5 = arith.subi %cst, %2 : tensor<32xi32>
    %6 = tt.addptr %3, %5 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %7 = tt.load %4 : tensor<32x!tt.ptr<i32>>
      tt.store %6, %7, %1 : tensor<32x!tt.ptr<i32>>
    }
    tt.return
  }
}

// -----

// Check that pointer for vector load/store is not extracted from a vector

// CHECK-LABEL: @scalar_ptrs
// CHECK-NOT:   vector.extract {{.+}} : i64 from vector<128xi64>
// CHECK:       {{.+}} = vector.load {{.+}} : memref<128xf32>, vector<128xf32>
// CHECK-NOT:   vector.extract {{.+}} : i64 from vector<128xi64>
// CHECK:       vector.store {{.+}}, {{.+}} : memref<128xf32>, vector<128xf32>

module {
  tt.func public @scalar_ptrs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %5, %3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
