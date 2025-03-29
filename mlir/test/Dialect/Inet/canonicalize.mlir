// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

// CHECK-LABEL: @erase_coerase_annihilation
// CHECK-NEXT: inet.inet {
// CHECK-NEXT:   %0 = inet.erase {
// CHECK-NEXT:   } f64
// CHECK-NEXT:   inet.empty
// CHECK-NEXT: }
func.func @erase_coerase_annihilation() -> () {
  inet.inet {
    %a = inet.erase {} f64
    inet.coerase f64 %a {}
  }
  return
}

// CHECK-LABEL: @construct_coconstruct_annihilation
// CHECK-NEXT: inet.inet {
// CHECK-NEXT:   %0 = inet.erase {
// CHECK-NEXT:   } f64
// CHECK-NEXT:   %1 = inet.construct f64 %0 f64 %0 f64 {
// CHECK-NEXT:   }
// CHECK-NEXT:   %2:2 = inet.swap f64 %0 f64 %0 f64, f64 {
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @construct_coconstruct_annihilation() -> () {
  inet.inet {
    %e = inet.erase {} f64
    %a = inet.construct f64 %e f64 %e f64 {}
    inet.coconstruct f64 %a f64, f64 {}
  }
  return
}

