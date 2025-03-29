#ifndef MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
#define MLIR_LIB_DIALECT_INET_INETCANONICALIZATION

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Inet/Inet.td"


def CoEraseEraseAnnihilation :
  Pattern<
    (Inet_CoEraseOp (Inet_EraseOp)),
    [(Inet_EmptyOp)]>;


def CoConstructConstructAnnihilation :
  Pattern<
    (Inet_CoConstructOp (Inet_ConstructOp $a, $b)),
    [(Inet_SwapOp $a, $b)]>;


// def DuplicateCoDuplicateAnnihilation :
//   Pattern<
//     (Inet_CoDuplicateOp (Inet_DuplicateOp $a, $b)),
//     [(Inet_SwapOp $a, $b)]>;


// def CoEraseDuplicateCommutation :
//   Pattern<
//     (Inet_CoEraseOp (Inet_DuplicateOp $a, $b)),
//     [(Inet_SwapOp $a, $b)]>;


// def CoEraseConstructCommutation :
//   Pattern<
//     (Inet_CoEraseOp (Inet_ConstructOp $a, $b)),
//     [(Inet_SwapOp $a, $b)]>;


// def CoConstructEraseCommutation :
//   Pattern<
//     (Inet_CoConstructOp (Inet_EraseOp:$a)),
//     [(Inet_TensorOp $a, $a)]>;


// def CoDuplicateEraseCommutation :
//   Pattern<
//     (Inet_CoDuplicateOp (Inet_EraseOp:$a)),
//     [(Inet_TensorOp $a, $a)]>;



#endif // MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
