// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func @erase_era_era
//       CHECK-NEXT: return
func.func @erase_era_era() -> () {
  // %arg0 = "inet.era"(%arg1) : (f64) -> (f64)
  // %arg1 = "inet.era"(%arg0) : (f64) -> (f64)
  "inet.inet"() ({
    %a = inet.era %b
    %b = inet.era %a
  }) : () -> ()
  return
}

