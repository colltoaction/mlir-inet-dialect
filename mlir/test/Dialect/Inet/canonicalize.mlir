// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

// CHECK-LABEL: func @erase_era_era
func.func @erase_era_era() -> () {
  %e = inet.empty
//       CHECK-NEXT: inet.empty
  %d = inet.era %e
  inet.inet {
    %a = inet.era %b
    %b = inet.era %a
  }
  return
}

