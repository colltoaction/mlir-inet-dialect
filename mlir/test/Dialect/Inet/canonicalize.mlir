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
  %c_param = arith.constant 20 : i32
  %cond = arith.cmpi eq, %c_param, %c0 : i32
  %result = scf.if %cond -> i32 {
    // base case n=0, result=partial
    %r = inet.construct i32 %c_param i32 %c_param i32
    scf.yield %r : i32
  } else {
    // recursive case TODO
    %cn4 = arith.subi %c_param, %c1 : i32
    %partial3 = arith.addi %c_param, %cn4 : i32
    %partial4 = inet.construct i32 %cn4 i32 %partial3 i32
    scf.yield %partial4 : i32
  }

  %total, %_ = inet.coconstruct i32 %result i32, i32
  return %total : i32
}
