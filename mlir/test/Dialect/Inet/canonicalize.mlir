// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

// CHECK-LABEL: @erase_coerase_annihilation
// CHECK-NEXT:  %0 = inet.erase f64
// CHECK-NEXT:  inet.empty
func.func @erase_coerase_annihilation() -> () {
  %a = inet.erase f64
  inet.coerase f64 %a
  return
}

// CHECK-LABEL: @construct_coconstruct_annihilation
// CHECK:       (%arg0: f64) -> (f64, f64) {
// CHECK-NEXT:  %0 = inet.construct f64 %arg0 f64 %arg0 f64
// CHECK-NEXT:  %1:2 = inet.swap f64 %arg0 f64 %arg0 f64, f64
// CHECK-NEXT:  return %1#0, %1#1 : f64, f64
func.func @construct_coconstruct_annihilation(%arg0 : f64) -> (f64, f64) {
  %a = inet.construct f64 %arg0 f64 %arg0 f64
  %b, %c = inet.coconstruct f64 %a f64, f64
  return %b, %c : f64, f64
}

// CHECK-LABEL: @coconstruct_erase_commutation
// CHECK:       () -> (f64, f64) {
// CHECK-NEXT:  %0 = inet.erase f64
// CHECK-NEXT:  %1:2 = inet.tensor f64 %0 f64 %0 f64, f64
// CHECK-NEXT:  return %1#0, %1#1 : f64, f64
func.func @coconstruct_erase_commutation() -> (f64, f64) {
  %e = inet.erase f64
  %a, %b = inet.coconstruct f64 %e f64, f64
  return %a, %b : f64, f64
}

