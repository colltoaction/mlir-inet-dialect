// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

// CHECK-LABEL: @erase_coerase_annihilation
// CHECK-NEXT: inet.inet {
// CHECK-NEXT:   inet.erase {
// CHECK-NEXT:   }
// CHECK-NEXT:   "inet.empty"() : () -> ()
// CHECK-NEXT: }
func.func @erase_coerase_annihilation() -> () {
  inet.inet {
    %a = inet.erase {}
    inet.coerase %a {}
  }
  return
}

// CHECK-LABEL: @construct_coconstruct_annihilation
// CHECK-NEXT: inet.inet {
// CHECK-NEXT:   %0:3 = inet.construct {
// CHECK-NEXT:   }
// CHECK-NEXT:   "inet.empty"() : () -> ()
// CHECK-NEXT: }
func.func @construct_coconstruct_annihilation() -> () {
  inet.inet {
    %a, %b, %e = inet.construct {}
    inet.coconstruct %a %b %e {}
  }
  return
}

// CHECK-LABEL: @duplicate_coduplicate_annihilation
// CHECK-NEXT: inet.inet {
// CHECK-NEXT:   %0:3 = inet.duplicate {
// CHECK-NEXT:   }
// CHECK-NEXT:   "inet.empty"() : () -> ()
// CHECK-NEXT: }
func.func @duplicate_coduplicate_annihilation() -> () {
  inet.inet {
    %a, %b, %e = inet.duplicate {}
    inet.coduplicate %a %b %e {}
  }
  return
}

