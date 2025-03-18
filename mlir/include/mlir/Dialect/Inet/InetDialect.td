#ifndef MLIR_DIALECT_INET_DIALECT
#define MLIR_DIALECT_INET_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Inet_Dialect : Dialect {
  let name = "inet";
  let cppNamespace = "::mlir::inet";
  // let hasCanonicalizer = 1;
  // let hasConstantMaterializer = 1;
  // let useDefaultAttributePrinterParser = 1;
}

#endif // MLIR_DIALECT_INET_DIALECT
