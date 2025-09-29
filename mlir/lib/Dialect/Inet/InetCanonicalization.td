#ifndef MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
#define MLIR_LIB_DIALECT_INET_INETCANONICALIZATION

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Inet/Inet.td"


//===----------------------------------------------------------------------===//
// Standard SIC reduction rules
//===----------------------------------------------------------------------===//

def CoEraseEraseAnnihilation :
  Pattern<
    (Inet_EraseOp (Inet_CoEraseOp)),
    []>;


def CoConstructConstructAnnihilation :
  Pattern<
    (Inet_ConstructOp (Inet_CoConstructOp $a, $b)),
    [(replaceWithValue $b), (replaceWithValue $a)]>;


def CoDuplicateDuplicateAnnihilation :
  Pattern<
    (Inet_DuplicateOp (Inet_CoDuplicateOp $a, $b)),
    [(replaceWithValue $b), (replaceWithValue $a)]>;


def CoEraseConstructCommutation :
  Pattern<
    (Inet_EraseOp:$a (Inet_CoConstructOp $b, $c)),
    [
      (Inet_EraseOp $b),
      (Inet_EraseOp $c),
    ]>;


def CoEraseDuplicateCommutation :
  Pattern<
    (Inet_EraseOp:$a (Inet_CoDuplicateOp $b, $c)),
    [
      (Inet_EraseOp $b),
      (Inet_EraseOp $c),
    ]>;


def CoConstructEraseCommutation :
  Pattern<
    (Inet_ConstructOp (Inet_CoEraseOp:$a)),
    [(replaceWithValue $a), (replaceWithValue $a)]>;


def CoDuplicateConstructCommutation :
  Pattern<
    (Inet_DuplicateOp (Inet_CoConstructOp:$a $c, $d)),
    [
      (Inet_DuplicateOp:$e $c, (returnType $c, $d)),
      (Inet_DuplicateOp:$f $d, (returnType $c, $d)),
      (Inet_CoConstructOp $e__0, $f__0),
      (Inet_CoConstructOp $e__1, $f__1)
    ]>;


def CoConstructDuplicateCommutation :
  Pattern<
    (Inet_ConstructOp (Inet_CoDuplicateOp:$a $c, $d)),
    [
      (Inet_ConstructOp:$e $c, (returnType $c, $d)),
      (Inet_ConstructOp:$f $d, (returnType $c, $d)),
      (Inet_CoDuplicateOp $e__0, $f__0),
      (Inet_CoDuplicateOp $e__1, $f__1)
    ]>;


def CapConstructDuplicateCommutation :
  Pattern<
    (Inet_CapOp:$a
      (Inet_CoConstructOp:$b $c, $d),
      (Inet_CoDuplicateOp:$e $f, $g)),
    [
      (Inet_ConstructOp:$h $f, (returnType $c, $d)),
      (Inet_ConstructOp:$i $g, (returnType $c, $d)),
      (Inet_DuplicateOp:$j $c, (returnType $f, $g)),
      (Inet_DuplicateOp:$k $d, (returnType $f, $g)),
      (Inet_CapOp $h__0, $k__1),
      (Inet_CapOp $h__1, $j__1),
      (Inet_CapOp $j__0, $i__1),
      (Inet_CapOp $k__0, $i__0)
    ]>;


def CapDuplicateConstructCommutation :
  Pattern<
    (Inet_CapOp:$a
      (Inet_CoDuplicateOp:$b $c, $d),
      (Inet_CoConstructOp:$e $f, $g)),
    [
      (Inet_DuplicateOp:$h $f, (returnType $c, $d)),
      (Inet_DuplicateOp:$i $g, (returnType $c, $d)),
      (Inet_ConstructOp:$j $c, (returnType $f, $g)),
      (Inet_ConstructOp:$k $d, (returnType $f, $g)),
      (Inet_CapOp $h__0, $k__1),
      (Inet_CapOp $h__1, $j__1),
      (Inet_CapOp $j__0, $i__1),
      (Inet_CapOp $k__0, $i__0)
    ]>;

def CoEraseCupCommutation :
  Pattern<
    (Inet_EraseOp:$a
      (Inet_CupOp:$e)),
    [
      (Inet_CoEraseOp:$b (returnType $e__1)),
      (Inet_CapOp $b, $e__1)
    ]>;

def CapCupStraighten :
  Pattern<
    (Inet_CapOp:$a
      (Inet_CupOp:$e), $f),
    [
      (replaceWithValue $e__1),
      (replaceWithValue $f)
    ]>;


// This is key to implement recursion
def CupCoConstructCoConstructAnnihilation :
  Pattern<
    (Inet_ConstructOp:$a
      (Inet_CupOp:$e)),
    [
      (Inet_CupOp:$f (returnType $e__0, $e__1)),
      (Inet_CupOp:$g (returnType $e__0, $e__1)),
      (Inet_CoConstructOp:$h $f__0, $g__0, (returnType $e__0)),
      (Inet_CapOp:$i $h, $e__0),
      (replaceWithValue $f__1),
      (replaceWithValue $g__1)
    ]>;

def CupCoDuplicateAnnihilation :
  Pattern<
    (Inet_DuplicateOp:$a
      (Inet_CupOp:$e)),
    [
      (Inet_CupOp:$f (returnType $e__0, $e__1)),
      (Inet_CupOp:$g (returnType $e__0, $e__1)),
      (Inet_CoDuplicateOp:$h $f__0, $g__0, (returnType $e__0)),
      // (Inet_CapOp:$i $h, $e__1),
      (replaceWithValue $f__1),
      (replaceWithValue $g__1)
    ]>;


#endif // MLIR_LIB_DIALECT_INET_INETCANONICALIZATION
