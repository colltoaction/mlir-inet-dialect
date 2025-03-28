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

class Inet_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Inet_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Inet Operations
//===----------------------------------------------------------------------===//


class Inet_Op<string mnemonic, list<Trait> traits =[]> :
    Op<Inet_Dialect, mnemonic, traits # GraphRegionNoTerminator.traits> {
  let hasCanonicalizer = 1;
}

def Inet_InetOp : Inet_Op<"inet"> {
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$region attr-dict-with-keyword";
}

def Inet_EmptyOp : Inet_Op<"empty"> {
}

def Inet_EraseOp : Inet_Op<"erase"> {
  let results = (outs F64);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$region attr-dict-with-keyword";
}

def Inet_CoEraseOp : Inet_Op<"coerase"> {
  let arguments = (ins F64:$principal);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$principal $region attr-dict-with-keyword";
}

def Inet_ConstructOp : Inet_Op<"construct"> {
  let regions = (region AnyRegion:$region);
  let results = (outs F64, F64, F64);
  let assemblyFormat = "$region attr-dict-with-keyword";
}

def Inet_CoConstructOp : Inet_Op<"coconstruct"> {
  let arguments = (ins F64:$principal, F64:$left, F64:$right);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$principal $left $right $region attr-dict-with-keyword";
}

def Inet_DuplicateOp : Inet_Op<"duplicate"> {
  let regions = (region AnyRegion:$region);
  let results = (outs F64, F64, F64);
  let assemblyFormat = "$region attr-dict-with-keyword";
}

def Inet_CoDuplicateOp : Inet_Op<"coduplicate"> {
  let arguments = (ins F64:$principal, F64:$left, F64:$right);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$principal $left $right $region attr-dict-with-keyword";
}

//===----------------------------------------------------------------------===//
// Monoidal Operations
//===----------------------------------------------------------------------===//

def Inet_CupOp : Inet_Op<"cup"> {
  let arguments = (ins F64:$left, F64:$right);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$left $right $region attr-dict-with-keyword";
}

def Inet_CapOp : Inet_Op<"cap"> {
  let results = (outs F64, F64);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$region attr-dict-with-keyword";
}


#endif // MLIR_DIALECT_INET_INET
