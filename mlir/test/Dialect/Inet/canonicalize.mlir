// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Standard SIC reduction rules
//===----------------------------------------------------------------------===//

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

// CHECK-LABEL: @coduplicate_duplicate_annihilation
// CHECK:       (%arg0: f64, %arg1: f64) -> (f64, f64) {
// CHECK-NEXT:  return %arg1, %arg0 : f64, f64
func.func @coduplicate_duplicate_annihilation(%arg0 : f64, %arg1 : f64) -> (f64, f64) {
  %a = inet.duplicate f64 %arg0 f64 %arg1 f64
  %b, %c = inet.coduplicate f64 %a f64, f64
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

// CHECK-LABEL: @coerase_duplicate_commutation
// CHECK:       (%arg0: f64, %arg1: f64) {
// CHECK-NEXT:  return
func.func @coerase_duplicate_commutation(%arg0 : f64, %arg1 : f64) -> () {
  %a = inet.duplicate f64 %arg0 f64 %arg1 f64
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

// CHECK-LABEL: @coconstruct_duplicate_commutation
// CHECK:       (%arg0: f64, %arg1: f64) -> (f64, f64) {
// CHECK-NEXT:  %0:2 = inet.coconstruct f64 %arg0 f64, f64
// CHECK-NEXT:  %1:2 = inet.coconstruct f64 %arg1 f64, f64
// CHECK-NEXT:  %2 = inet.duplicate f64 %0#0 f64 %1#0 f64
// CHECK-NEXT:  %3 = inet.duplicate f64 %0#1 f64 %1#1 f64
// CHECK-NEXT:  return %2, %3 : f64, f64
func.func @coconstruct_duplicate_commutation(%arg0 : f64, %arg1 : f64) -> (f64, f64) {
  %a = inet.duplicate f64 %arg0 f64 %arg1 f64
  %b, %c = inet.coconstruct f64 %a f64, f64
  return %b, %c : f64, f64
}

// CHECK-LABEL: @cap_construct_duplicate_commutation
// CHECK:       (%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) {
// CHECK-NEXT:  %0:2 = inet.coconstruct f64 %arg2 f64, f64
// CHECK-NEXT:  %1:2 = inet.coconstruct f64 %arg3 f64, f64
// CHECK-NEXT:  %2:2 = inet.coduplicate f64 %arg0 f64, f64
// CHECK-NEXT:  %3:2 = inet.coduplicate f64 %arg1 f64, f64
// CHECK-NEXT:  inet.cap f64 %0#0 f64 %3#1
// CHECK-NEXT:  inet.cap f64 %0#1 f64 %2#1
// CHECK-NEXT:  inet.cap f64 %2#0 f64 %1#1
// CHECK-NEXT:  inet.cap f64 %3#0 f64 %1#0
// CHECK-NEXT:  return
func.func @cap_construct_duplicate_commutation(%arg0 : f64, %arg1 : f64, %arg2 : f64, %arg3 : f64) {
  %a = inet.construct f64 %arg0 f64 %arg1 f64
  %b = inet.duplicate f64 %arg2 f64 %arg3 f64
  inet.cap f64 %a f64 %b
  return
}
