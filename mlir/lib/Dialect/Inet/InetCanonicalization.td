#ifndef MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
#define MLIR_LIB_DIALECT_INET_INETCANONICALIZATION

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Inet/Inet.td"


def EraseCoEraseAnnihilation :
  Pattern<
    (Inet_CoEraseOp (Inet_EraseOp)),
    [(Inet_EmptyOp)]>;


def ConstructCoConstructAnnihilation :
  Pattern<
    (Inet_CoConstructOp (Inet_ConstructOp), (Inet_ConstructOp), (Inet_ConstructOp)),
    [(Inet_EmptyOp)]>;


def DuplicateCoDuplicateAnnihilation :
  Pattern<
    (Inet_CoDuplicateOp (Inet_DuplicateOp), (Inet_DuplicateOp), (Inet_DuplicateOp)),
    [(Inet_EmptyOp)]>;


#endif // MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
