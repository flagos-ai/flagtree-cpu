#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_MARKWIDEBF16STORESVOLATILE
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

static bool blockIsInCycle(Block *block) {
  Operation *terminator = block->getTerminator();
  if (!terminator)
    return false;

  llvm::SmallVector<Block *, 8> worklist(terminator->getSuccessors());
  llvm::SmallPtrSet<Block *, 16> visited;
  while (!worklist.empty()) {
    Block *current = worklist.pop_back_val();
    if (current == block)
      return true;
    if (!visited.insert(current).second)
      continue;
    Operation *currentTerminator = current->getTerminator();
    if (!currentTerminator)
      continue;
    for (Block *successor : currentTerminator->getSuccessors())
      worklist.push_back(successor);
  }
  return false;
}

static bool blockLoadsFromStoreAddress(LLVM::StoreOp store) {
  for (LLVM::LoadOp load : store->getBlock()->getOps<LLVM::LoadOp>())
    if (load.getAddr() == store.getAddr())
      return true;
  return false;
}

struct MarkWideBf16StoresVolatile
    : public triton::cpu::impl::MarkWideBf16StoresVolatileBase<
          MarkWideBf16StoresVolatile> {
  void runOnOperation() override {
    getOperation().walk([](LLVM::StoreOp store) {
      auto vectorTy = dyn_cast<VectorType>(store.getValue().getType());
      if (!vectorTy || vectorTy.isScalable() || vectorTy.getRank() != 1 ||
          vectorTy.getShape()[0] != 16 || !vectorTy.getElementType().isBF16() ||
          !blockIsInCycle(store->getBlock()) ||
          blockLoadsFromStoreAddress(store))
        return;
      store.setVolatile_(true);
    });
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createMarkWideBf16StoresVolatile() {
  return std::make_unique<MarkWideBf16StoresVolatile>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
