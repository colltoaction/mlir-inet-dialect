// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

// CHECK-LABEL: @coerase_erase_annihilation
// CHECK-NEXT:  return
func.func @coerase_erase_annihilation() -> () {
  %a = inet.erase f64
  inet.coerase f64 %a
  return
}

// CHECK-LABEL: @coconstruct_construct_annihilation
// CHECK:       (%arg0: f64, %arg1: f64) -> (f64, f64) {
// CHECK-NEXT:  return %arg1, %arg0 : f64, f64
func.func @coconstruct_construct_annihilation(%arg0 : f64, %arg1 : f64) -> (f64, f64) {
  %a = inet.construct f64 %arg0 f64 %arg1 f64
  %b, %c = inet.coconstruct f64 %a f64, f64
  return %b, %c : f64, f64
}

// CHECK-LABEL: @coerase_construct_commutation
// CHECK:       (%arg0: f64, %arg1: f64) {
// CHECK-NEXT:  return
func.func @coerase_construct_commutation(%arg0 : f64, %arg1 : f64) -> () {
  %a = inet.construct f64 %arg0 f64 %arg1 f64
  inet.coerase f64 %a
  return
}

// CHECK-LABEL: @coconstruct_erase_commutation
// CHECK:       () -> (f64, f64) {
// CHECK-NEXT:  %0 = inet.erase f64
// CHECK-NEXT:  return %0, %0 : f64, f64
func.func @coconstruct_erase_commutation() -> (f64, f64) {
  %e = inet.erase f64
  %a, %b = inet.coconstruct f64 %e f64, f64
  return %a, %b : f64, f64
}

// CHECK-LABEL: @construct_erase_left_simplification
// CHECK:       (%arg0: f64) -> f64 {
// CHECK-NEXT:  return %arg0 : f64
func.func @construct_erase_left_simplification(%arg0 : f64) -> f64 {
  %e = inet.erase f64
  %a = inet.construct f64 %e f64 %arg0 f64
  return %a : f64
}


// CHECK-LABEL: @construct_erase_right_simplification
// CHECK:       (%arg0: f64) -> f64 {
// CHECK-NEXT:  return %arg0 : f64
func.func @construct_erase_right_simplification(%arg0 : f64) -> f64 {
  %e = inet.erase f64
  %a = inet.construct f64 %arg0 f64 %e f64
  return %a : f64
}

// CHECK-LABEL: @coduplicate_construct_commutation
// CHECK:       (%arg0: f64, %arg1: f64) -> (f64, f64) {
// CHECK-NEXT:  %0:2 = inet.coduplicate f64 %arg0 f64, f64
// CHECK-NEXT:  %1:2 = inet.coduplicate f64 %arg1 f64, f64
// CHECK-NEXT:  %2 = inet.construct f64 %0#0 f64 %1#0 f64
// CHECK-NEXT:  %3 = inet.construct f64 %0#1 f64 %1#1 f64
// CHECK-NEXT:  return %2, %3 : f64, f64
func.func @coduplicate_construct_commutation(%arg0 : f64, %arg1 : f64) -> (f64, f64) {
  %a = inet.construct f64 %arg0 f64 %arg1 f64
  %b, %c = inet.coduplicate f64 %a f64, f64
  return %b, %c : f64, f64
}

