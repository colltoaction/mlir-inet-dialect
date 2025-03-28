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
// CHECK-NEXT:   %0 = inet.erase {
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = inet.construct %0 %0 {
// CHECK-NEXT:   }
// CHECK-NEXT:   %2:2 = inet.swap %0 %0 {
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @construct_coconstruct_annihilation() -> () {
  inet.inet {
    %e = inet.erase {}
    %a = inet.construct %e %e {}
    inet.coconstruct %a {}
  }
  return
}

