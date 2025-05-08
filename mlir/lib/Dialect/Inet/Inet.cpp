#include "mlir/Dialect/Inet/Inet.h"

// #include "llvm/ADT/TypeSwitch.h"
// #include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;


//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "InetCanonicalization.inc"
} // namespace



void inet::EraseOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
}

void inet::CoEraseOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<CoEraseEraseAnnihilation>(context);
  patterns.add<CoEraseConstructCommutation>(context);
}

void inet::ConstructOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add<ConstructEraseLeftSimplification>(context);
  patterns.add<ConstructEraseRightSimplification>(context);
}

void inet::CoConstructOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                      MLIRContext *context) {
  patterns.add<CoConstructConstructAnnihilation>(context);
  patterns.add<CoConstructEraseCommutation>(context);
  patterns.add<CoConstructDuplicateCommutation>(context);
}

void inet::DuplicateOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add<DuplicateEraseLeftSimplification>(context);
  patterns.add<DuplicateEraseRightSimplification>(context);
}

void inet::CoDuplicateOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                      MLIRContext *context) {
  patterns.add<CoDuplicateDuplicateAnnihilation>(context);
  patterns.add<CoDuplicateConstructCommutation>(context);
}

void inet::CapOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<CapConstructDuplicateCommutation>(context);
}


#include "mlir/Dialect/Inet/InetDialect.cpp.inc"

void inet::InetDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Inet/InetTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Inet/Inet.cpp.inc"
      >();
}
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Inet/InetTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Inet/Inet.cpp.inc"
