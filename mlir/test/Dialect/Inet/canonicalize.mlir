// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

// CHECK-LABEL: @erase_coerase_annihilation
// CHECK-NEXT: inet.inet {
// CHECK-NEXT:   inet.erase {
// CHECK-NEXT:   }
// CHECK-NEXT:   inet.inet
// CHECK-NEXT: }
func.func @erase_coerase_annihilation() -> () {
  inet.inet {
    %a = inet.erase {}
    inet.coerase %a {}
  }
  return
}

// CHECK-LABEL: @erase_era_era
// CHECK-NEXT: inet.inet {
// CHECK-NEXT:   %0:3 = inet.construct {
// CHECK-NEXT:   }
// CHECK-NEXT:   inet.coconstruct %0#0 %0#1 %0#2 {
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @erase_era_era() -> () {
  inet.inet {
    %a, %b, %e = inet.construct {}
    inet.coconstruct %a %b %e {}
  }
  return
}

