// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

// CHECK-LABEL: @erase_era_era
// CHECK-NEXT: inet.inet {
// CHECK-NEXT:   inet.construct {
// CHECK-NEXT:     inet.erase {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   inet.erase {
// CHECK-NEXT:     inet.inet {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   inet.inet {
// CHECK-NEXT:   }
// CHECK-NEXT:   inet.inet {
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @erase_era_era() -> () {
  inet.inet {
    %b = inet.construct {
      %c = inet.erase {}
    }
    %d = inet.erase {
      inet.coerase %d {}
    }
    inet.coerase %b {}
    inet.coerase %d {}
  }
  return
}

