#ifndef MLIR_DIALECT_INET_TYPES
#define MLIR_DIALECT_INET_TYPES

include "mlir/Dialect/Inet/InetDialect.td"

// class Inet_Type<string name, string typeMnemonic>
//     : TypeDef<Inet_Dialect, name> {
//   let mnemonic = typeMnemonic;
// }

// def Inet_InetType : Inet_Type<"Inet", "inet"> {
//   // let parameters = (ins Inet_RingAttr:$ring);
//   // let assemblyFormat = "`<` struct(params) `>`";
// }

// def InetEra : Inet_Type<"Inet", "era"> {
//   // let parameters = (ins "asd":$degreeBound);
//   // let hasCustomAssemblyFormat = 1;
// }



#endif // MLIR_DIALECT_INET_TYPES
