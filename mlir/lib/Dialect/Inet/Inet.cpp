#include "mlir/Dialect/Inet/Inet.h"

// #include "llvm/ADT/TypeSwitch.h"
// #include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;

#include "mlir/Dialect/Inet/InetDialect.cpp.inc"

void inet::InetDialect::initialize() {
//   addTypes<
// #define GET_TYPEDEF_LIST
// #include "mlir/Dialect/Inet/Inet.cpp.inc"
//       >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Inet/Inet.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

// #define GET_TYPEDEF_CLASSES
// #include "mlir/Dialect/Inet/Inet.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Inet/Inet.cpp.inc"
