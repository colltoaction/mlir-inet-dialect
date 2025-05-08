#ifndef MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
#define MLIR_LIB_DIALECT_INET_INETCANONICALIZATION

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Inet/Inet.td"


//===----------------------------------------------------------------------===//
// Standard SIC reduction rules
//===----------------------------------------------------------------------===//

def CoEraseEraseAnnihilation :
  Pattern<
    (Inet_CoEraseOp (Inet_EraseOp)),
    []>;


def CoConstructConstructAnnihilation :
  Pattern<
    (Inet_CoConstructOp (Inet_ConstructOp $a, $b)),
    [(replaceWithValue $b), (replaceWithValue $a)]>;


def CoDuplicateDuplicateAnnihilation :
  Pattern<
    (Inet_CoDuplicateOp (Inet_DuplicateOp $a, $b)),
    [(replaceWithValue $b), (replaceWithValue $a)]>;


def CoEraseConstructCommutation :
  Pattern<
    (Inet_CoEraseOp:$a (Inet_ConstructOp $b, $c)),
    []>;


def CoEraseDuplicateCommutation :
  Pattern<
    (Inet_CoEraseOp:$a (Inet_DuplicateOp $b, $c)),
    []>;


def CoConstructEraseCommutation :
  Pattern<
    (Inet_CoConstructOp (Inet_EraseOp:$a)),
    [(replaceWithValue $a), (replaceWithValue $a)]>;


def CoDuplicateConstructCommutation :
  Pattern<
    (Inet_CoDuplicateOp (Inet_ConstructOp:$a $c, $d)),
    [
      (Inet_CoDuplicateOp:$e $c, (returnType $c, $d)),
      (Inet_CoDuplicateOp:$f $d, (returnType $c, $d)),
      (Inet_ConstructOp $e__0, $f__0),
      (Inet_ConstructOp $e__1, $f__1)
    ]>;


def CoConstructDuplicateCommutation :
  Pattern<
    (Inet_CoConstructOp (Inet_DuplicateOp:$a $c, $d)),
    [
      (Inet_CoConstructOp:$e $c, (returnType $c, $d)),
      (Inet_CoConstructOp:$f $d, (returnType $c, $d)),
      (Inet_DuplicateOp $e__0, $f__0),
      (Inet_DuplicateOp $e__1, $f__1)
    ]>;


def CapConstructDuplicateCommutation :
  Pattern<
    (Inet_CapOp:$a
      (Inet_ConstructOp:$b $c, $d),
      (Inet_DuplicateOp:$e $f, $g)),
    [
      (Inet_CoConstructOp:$h $f, (returnType $c, $d)),
      (Inet_CoConstructOp:$i $g, (returnType $c, $d)),
      (Inet_CoDuplicateOp:$j $c, (returnType $f, $g)),
      (Inet_CoDuplicateOp:$k $d, (returnType $f, $g)),
      (Inet_CapOp $h__0, $k__1),
      (Inet_CapOp $h__1, $j__1),
      (Inet_CapOp $j__0, $i__1),
      (Inet_CapOp $k__0, $i__0)
    ]>;

//===----------------------------------------------------------------------===//
// Simplifications
//===----------------------------------------------------------------------===//

def ConstructEraseLeftSimplification :
  Pattern<
    (Inet_ConstructOp (Inet_EraseOp:$a), $b),
    [(replaceWithValue $b)]>;


def ConstructEraseRightSimplification :
  Pattern<
    (Inet_ConstructOp $a, (Inet_EraseOp:$b)),
    [(replaceWithValue $a)]>;

def DuplicateEraseLeftSimplification :
  Pattern<
    (Inet_DuplicateOp (Inet_EraseOp:$a), $b),
    [(replaceWithValue $b)]>;


def DuplicateEraseRightSimplification :
  Pattern<
    (Inet_DuplicateOp $a, (Inet_EraseOp:$b)),
    [(replaceWithValue $a)]>;


#endif // MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
