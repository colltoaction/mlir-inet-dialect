#ifndef MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
#define MLIR_LIB_DIALECT_INET_INETCANONICALIZATION

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Inet/Inet.td"


def EraseCoEraseAnnihilation :
  Pattern<
    (Inet_CoEraseOp (Inet_EraseOp)),
    [(Inet_InetOp)]>;


def ConstructCoEraseCommutation :
  Pattern<
    (Inet_CoEraseOp (Inet_ConstructOp)),
    [(Inet_InetOp)]>;


#endif // MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
