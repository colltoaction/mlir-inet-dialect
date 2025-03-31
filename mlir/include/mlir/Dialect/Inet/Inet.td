#ifndef MLIR_DIALECT_INET_INET
#define MLIR_DIALECT_INET_INET

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Inet_Dialect : Dialect {
  let name = "inet";
  let cppNamespace = "::mlir::inet";
  // let hasCanonicalizer = 1;
  // let hasConstantMaterializer = 1;
  // let useDefaultAttributePrinterParser = 1;
  // let useDefaultTypePrinterParser = 1;
}

class Inet_Type<string name, string typeMnemonic, list<Trait> traits = [Pure]>
    : TypeDef<Inet_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Inet Operations
//===----------------------------------------------------------------------===//


class Inet_Op<string mnemonic, list<Trait> traits =[]> :
    Op<Inet_Dialect, mnemonic, traits> {
  let hasCanonicalizer = 1;
}

def Inet_InetOp : Inet_Op<"inet"> {
  // let regions = (region AnyRegion:$region);
  // let arguments = (ins Variadic<AnyType>:$ports);
  let assemblyFormat = "attr-dict-with-keyword";
}

def Inet_EmptyOp : Inet_Op<"empty"> {
  let assemblyFormat = "attr-dict-with-keyword";
}

def Inet_EraseOp : Inet_Op<"erase"> {
  let results = (outs AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type(results) attr-dict-with-keyword";
}

def Inet_CoEraseOp : Inet_Op<"coerase"> {
  let arguments = (ins AnyType:$principal);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($principal) $principal attr-dict-with-keyword";
}

def Inet_ConstructOp : Inet_Op<"construct"> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($left) $left type($right) $right type(results) attr-dict-with-keyword";
}

def Inet_CoConstructOp : Inet_Op<"coconstruct"> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType, AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}

def Inet_DuplicateOp : Inet_Op<"duplicate"> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($left) $left type($right) $right type(results) attr-dict-with-keyword";
}

def Inet_CoDuplicateOp : Inet_Op<"coduplicate"> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType, AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}

//===----------------------------------------------------------------------===//
// Monoidal Operations
//===----------------------------------------------------------------------===//

def Inet_IdOp : Inet_Op<"id"> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}

def Inet_TensorOp : Inet_Op<"tensor", [ElementwiseMappable, Commutative]> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType, AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($left) $left type($right) $right type(results) attr-dict-with-keyword";
}

def Inet_SwapOp : Inet_Op<"swap"> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType, AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($left) $left type($right) $right type(results) attr-dict-with-keyword";
}

def Inet_CupOp : Inet_Op<"cup"> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type($left) $left type($right) $right attr-dict-with-keyword";
}

def Inet_CapOp : Inet_Op<"cap"> {
  let results = (outs AnyType, AnyType);
  // let regions = (region AnyRegion:$region);
  let assemblyFormat = "type(results) attr-dict-with-keyword";
}


#endif // MLIR_DIALECT_INET_INET
