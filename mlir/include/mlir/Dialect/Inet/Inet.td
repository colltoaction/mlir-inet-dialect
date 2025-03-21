#ifndef MLIR_DIALECT_INET_INET
#define MLIR_DIALECT_INET_INET

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"
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

// def Inet_EraType : Inet_Type<"Inet", "era"> {
//   let hasCustomAssemblyFormat = 1;
// }
// def Inet_ConType : Inet_Type<"Inet", "dup"> {}
// def Inet_DupType : Inet_Type<"Inet", "con"> {}

// def Inet_PortType : Inet_Type<"Inet", "port"> {
//   let parameters = (ins "Type":$portType);
//   let assemblyFormat = "`<` $portType `>`";
//   // let assemblyFormat = "";
// }

// def InetEra : Inet_Type<"Inet", "era"> {
  // let parameters = (ins "asd":$degreeBound);
  // let hasCustomAssemblyFormat = 1;
// }

class Inet_Op<string mnemonic, list<Trait> traits =[]> :
    Op<Inet_Dialect, mnemonic, traits> {
//   let arguments = (ins Inet_InetOPSType:$principal);
}

def Inet_EraOp : Inet_Op<"era", [Pure]> {
  let arguments = (ins F64:$principal);
  let results = (outs F64);
  let assemblyFormat = "$principal attr-dict";
}

//===----------------------------------------------------------------------===//
// Inet Operations
//===----------------------------------------------------------------------===//

// class EraOp<Inet_Op principal> : Inet_Op<"era">;
// class ConOp<Inet_Op principal, list<Inet_Op> secondary> : Inet_Op<"dup">;
// class DupOp<Inet_Op principal, list<Inet_Op> secondary> : Inet_Op<"con">;

// def EraOp2 : EraOp<> {
//   let summary = "era";
//   let arguments = (ins EraOpAttr:$principal);
//   let results = (outs F64Tensor);
// }


#endif // MLIR_DIALECT_INET_INET
