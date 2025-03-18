#ifndef MLIR_DIALECT_INET_INET
#define MLIR_DIALECT_INET_INET

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Inet/InetDialect.td"
include "mlir/Dialect/Inet/InetTypes.td"

class Inet_Op<string mnemonic, list<Trait> traits =[]> :
    Op<Inet_Dialect, mnemonic, traits> {
//   let arguments = (ins Inet_InetType:$principal);
}

def Inet_EraOp : Inet_Op<"era", [Pure]> {
//   let arguments = (ins InetEra:$degreeBound);
  // let hasCustomAssemblyFormat = 1;
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
