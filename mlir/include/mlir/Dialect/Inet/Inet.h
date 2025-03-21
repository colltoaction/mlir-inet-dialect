#ifndef MLIR_DIALECT_INET_INET_H_
#define MLIR_DIALECT_INET_INET_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Inet/InetDialect.h.inc"

// #define GET_TYPEDEF_CLASSES
// #include "mlir/Dialect/Inet/Inet.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Inet/Inet.h.inc"

#endif // MLIR_DIALECT_INET_INET_H_
