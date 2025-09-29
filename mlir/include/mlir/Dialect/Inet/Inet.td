#ifndef MLIR_DIALECT_INET_INET
#define MLIR_DIALECT_INET_INET

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Inet_Dialect : Dialect {
  let name = "inet";
  let cppNamespace = "::mlir::inet";
}

class Inet_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Inet_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Inet Operations
//===----------------------------------------------------------------------===//


class Inet_Op<string mnemonic, list<Trait> traits =[Pure]> :
    Op<Inet_Dialect, mnemonic, traits> {
  let hasCanonicalizer = 1;
}


def Inet_InetOp : Inet_Op<"inet", [
  FunctionOpInterface,
  IsolatedFromAbove,
]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$functionType,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this inet.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  
    /// Returns the number of results of this inet
    unsigned getNumResults() {return getResultTypes().size();}
  }];
  let assemblyFormat = [{
    $sym_name $functionType attr-dict-with-keyword $body
  }];
}

def Inet_ReturnOp : Inet_Op<"return", [
  Terminator,
  ReturnLike,
  HasParent<"InetOp">,
]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() {
      auto parentOp = cast<InetOp>((*this)->getParentOp());
      if (parentOp.getNumResults() != getNumOperands())
        return emitOpError() << "parent of inet.return must have same number of "
                                "results as the inet.inet operands";

      const auto &results = parentOp.getResultTypes();
      auto operands = getOperands();
      for (auto it : llvm::zip(results, operands)) {
        if (std::get<0>(it) != std::get<1>(it).getType())
          return emitOpError() << "types mismatch between inet.return op and its parent inet.inet";
      }

      return success();
    }
  }];
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict-with-keyword";
  let hasVerifier = 1;
}

def Inet_Erase2Op : Inet_Op<"erase2", [
  Terminator,
]> {
  let arguments = (ins AnyType:$principal);
  let successors = (successor AnySuccessor:$target);
  let assemblyFormat = "$target $principal type($principal) attr-dict-with-keyword";
}

def Inet_EraseOp : Inet_Op<"erase", []> {
  let results = (outs AnyType);
  let assemblyFormat = "type(results) attr-dict-with-keyword";
}

def Inet_CoEraseOp : Inet_Op<"coerase", []> {
  let arguments = (ins AnyType:$principal);
  let assemblyFormat = "type($principal) $principal attr-dict-with-keyword";
}

def Inet_CoErase2Op : Inet_Op<"coerase2", [
  Terminator,
  HasParent<"InetOp">,
]> {
  let regions = (region AnyRegion:$defaultRegion);
  let assemblyFormat = [{
    attr-dict-with-keyword $defaultRegion
  }];
}

def Inet_Construct2Op : Inet_Op<"construct2", [
  Terminator,
  HasParent<"InetOp">,
]> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType:$left, AnyType:$right);
  let successors = (successor AnySuccessor:$target);
  let assemblyFormat = "$target $principal`:` type($principal) type(results) attr-dict-with-keyword";
}

def Inet_Duplicate2Op : Inet_Op<"duplicate2", [
  Terminator,
]> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType:$left, AnyType:$right);
  let successors = (successor AnySuccessor:$target);
  let assemblyFormat = "$target $principal`:` type($principal) type(results) attr-dict-with-keyword";
}

def Inet_ConstructOp : Inet_Op<"construct"> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType);
  let assemblyFormat = "type($left) $left type($right) $right type(results) attr-dict-with-keyword";
}

def Inet_CoConstructOp : Inet_Op<"coconstruct"> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType, AnyType);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}

def Inet_CoConstruct2Op : Inet_Op<"coconstruct2", [
  Terminator,
]> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType);
  let successors = (successor AnySuccessor:$target);
  let assemblyFormat = "$target $left`:` type($left) $right `:` type($right) type(results) attr-dict-with-keyword";
}

def Inet_CoDuplicate2Op : Inet_Op<"coduplicate2", [
  Terminator,
]> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType);
  let successors = (successor AnySuccessor:$target);
  let assemblyFormat = "$target $left`:` type($left) $right `:` type($right) type(results) attr-dict-with-keyword";
}

def Inet_DuplicateOp : Inet_Op<"duplicate"> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType);
  let assemblyFormat = "type($left) $left type($right) $right type(results) attr-dict-with-keyword";
}

def Inet_CoDuplicateOp : Inet_Op<"coduplicate"> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType, AnyType);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}

def Inet_Cap2Op : Inet_Op<"cap2", [
  Terminator,
]> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let assemblyFormat = "$left`:` type($left) $right `:` type($right) attr-dict-with-keyword";
}

def Inet_CapOp : Inet_Op<"cap", []> {
  let arguments = (ins AnyType:$left, AnyType:$right);
  let assemblyFormat = "type($left) $left type($right) $right attr-dict-with-keyword";
}

def Inet_CupOp : Inet_Op<"cup"> {
  let results = (outs AnyType, AnyType);
  let assemblyFormat = "type(results) attr-dict-with-keyword";
}

def Inet_Cup2Op : Inet_Op<"cup2", [
]> {
  let results = (outs AnyType:$left, AnyType:$right);
  let assemblyFormat = "type($left) type($right) attr-dict-with-keyword";
}

def Inet_CapLeftOp : Inet_Op<"cap_left", []> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}

def Inet_CupLeftOp : Inet_Op<"cup_left", []> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}

def Inet_CapRightOp : Inet_Op<"cap_right", []> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}

def Inet_CupRightOp : Inet_Op<"cup_right", []> {
  let arguments = (ins AnyType:$principal);
  let results = (outs AnyType);
  let assemblyFormat = "type($principal) $principal type(results) attr-dict-with-keyword";
}


#endif // MLIR_DIALECT_INET_INET
