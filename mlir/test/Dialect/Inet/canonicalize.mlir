// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func @erase_era_era
//       CHECK-NEXT: return
func.func @erase_era_era() {
  %a = inet.era %b
  %b = inet.era %a
  return
}

