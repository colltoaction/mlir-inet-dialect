#ifndef MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
#define MLIR_LIB_DIALECT_INET_INETCANONICALIZATION

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Inet/Inet.td"


def EraseEraseAnnihilation :
  Pat<(Inet_EraOp (Inet_EmptyOp)),
      (Inet_EmptyOp)>;


#endif // MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
