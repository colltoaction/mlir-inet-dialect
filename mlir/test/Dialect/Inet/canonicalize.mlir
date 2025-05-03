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

//===----------------------------------------------------------------------===//
// Simplifications
//===----------------------------------------------------------------------===//

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

// CHECK-LABEL: @duplicate_erase_left_simplification
// CHECK:       (%arg0: f64) -> f64 {
// CHECK-NEXT:  return %arg0 : f64
func.func @duplicate_erase_left_simplification(%arg0 : f64) -> f64 {
  %e = inet.erase f64
  %a = inet.duplicate f64 %e f64 %arg0 f64
  return %a : f64
}

// CHECK-LABEL: @duplicate_erase_right_simplification
// CHECK:       (%arg0: f64) -> f64 {
// CHECK-NEXT:  return %arg0 : f64
func.func @duplicate_erase_right_simplification(%arg0 : f64) -> f64 {
  %e = inet.erase f64
  %a = inet.duplicate f64 %arg0 f64 %e f64
  return %a : f64
}

// CHECK-LABEL: @sum_to_20() -> i32 {
// CHECK-NEXT:  %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:  %c20_i32 = arith.constant 20 : i32
// CHECK-NEXT:  %0:2 = inet.coduplicate i32 %c20_i32 i32, i32
// CHECK-NEXT:  %1 = inet.construct i32 %0#0 i32 %0#1 i32
// CHECK-NEXT:  return %1 : i32
func.func @sum_to_20() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // arithmetic constants will help with compile-time rewriting.
  // adding and subtracting constants gives new constants.
  // constant comparison unfolds if statements.

  %n = arith.constant 4 : i32

  // the first element has to be an operation
  // because it exposes a principal.
  %1 = arith.subi %n, %c1 : i32
  %d1 = inet.duplicate i32 %n i32 %1 i32
  %2 = arith.subi %1, %c1 : i32
  %d2 = inet.duplicate i32 %2 i32 %d1 i32
  %3 = arith.subi %2, %c1 : i32
  %d3 = inet.duplicate i32 %3 i32 %d2 i32
  %4 = arith.subi %3, %c1 : i32
  %d5 = inet.duplicate i32 %4 i32 %d3 i32
  // return %4 : i32

  // every time we loop back to "s(n)=n+s(n-1)"
  // we duplicate the branch of code that contains "if ... subi addi"

  %9:2 = inet.coduplicate i32 %d5 i32, i32
  %10:2 = inet.coduplicate i32 %9#0 i32, i32
  %11:2 = inet.coduplicate i32 %10#0 i32, i32
  %12:2 = inet.coduplicate i32 %11#0 i32, i32
  %5 = arith.addi %9#1, %10#1 : i32
  %6 = arith.addi %5, %11#1 : i32
  %7 = arith.addi %6, %12#0 : i32
  %8 = arith.addi %7, %12#1 : i32
  return %8#0 : i32
}