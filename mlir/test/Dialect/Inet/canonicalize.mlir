// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Standard SIC reduction rules
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @coerase_erase_annihilation
// CHECK-NEXT:  inet.return
inet.inet @coerase_erase_annihilation() -> () {
  %0 = inet.coerase f64
  inet.erase %0 : f64
  inet.return
}

// CHECK-LABEL: @coconstruct_construct_annihilation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64):
// CHECK-NEXT:  inet.return %arg1, %arg0 : f64, f64
inet.inet @coconstruct_construct_annihilation(f64, f64) -> (f64, f64) {
^bb0(%arg0: f64, %arg1: f64):
  %a = inet.coconstruct %arg0, %arg1 : f64, f64 -> f64
  %b, %c = inet.construct %a: f64 -> f64, f64
  inet.return %b, %c : f64, f64
}

// CHECK-LABEL: @coduplicate_duplicate_annihilation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64):
// CHECK-NEXT:  inet.return %arg1, %arg0 : f64, f64
inet.inet @coduplicate_duplicate_annihilation(f64, f64) -> (f64, f64) {
^bb0(%arg0: f64, %arg1: f64):
  %a = inet.coduplicate %arg0, %arg1 : f64, f64 -> f64
  %b, %c = inet.duplicate %a: f64 -> f64, f64
  inet.return %b, %c : f64, f64
}

// CHECK-LABEL: @coerase_construct_commutation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64):
// CHECK-NEXT:  inet.erase %arg0 : f64
// CHECK-NEXT:  inet.erase %arg1 : f64
// CHECK-NEXT:  inet.return
inet.inet @coerase_construct_commutation(f64, f64) -> () {
^bb0(%arg0: f64, %arg1: f64):
  %a = inet.coconstruct %arg0, %arg1 : f64, f64 -> f64
  inet.erase %a : f64
  inet.return
}

// CHECK-LABEL: @coerase_duplicate_commutation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64):
// CHECK-NEXT:  inet.erase %arg0 : f64
// CHECK-NEXT:  inet.erase %arg1 : f64
// CHECK-NEXT:  inet.return
inet.inet @coerase_duplicate_commutation(f64, f64) -> () {
^bb0(%arg0: f64, %arg1: f64):
  %a = inet.coduplicate %arg0, %arg1 : f64, f64 -> f64
  inet.erase %a : f64
  inet.return
}

// CHECK-LABEL: @coconstruct_erase_commutation
// CHECK:       () -> (f64, f64) {
// CHECK-NEXT:  %0 = inet.coerase f64
// CHECK-NEXT:  inet.return %0, %0 : f64, f64
inet.inet @coconstruct_erase_commutation() -> (f64, f64) {
  %e = inet.coerase f64
  %a, %b = inet.construct %e: f64 -> f64, f64
  inet.return %a, %b : f64, f64
}

// CHECK-LABEL: @coduplicate_construct_commutation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64):
// CHECK-NEXT:  %left, %right = inet.duplicate %arg0 : f64 -> f64, f64
// CHECK-NEXT:  %left_0, %right_1 = inet.duplicate %arg1 : f64 -> f64, f64
// CHECK-NEXT:  %0 = inet.coconstruct %left, %left_0 : f64, f64 -> f64
// CHECK-NEXT:  %1 = inet.coconstruct %right, %right_1 : f64, f64 -> f64
// CHECK-NEXT:  inet.return %0, %1 : f64, f64
inet.inet @coduplicate_construct_commutation(f64, f64) -> (f64, f64) {
^bb0(%arg0: f64, %arg1: f64):
  %a = inet.coconstruct %arg0, %arg1 : f64, f64 -> f64
  %b, %c = inet.duplicate %a: f64 -> f64, f64
  inet.return %b, %c : f64, f64
}

// CHECK-LABEL: @coconstruct_duplicate_commutation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64):
// CHECK-NEXT:  %left, %right = inet.construct %arg0 : f64 -> f64, f64
// CHECK-NEXT:  %left_0, %right_1 = inet.construct %arg1 : f64 -> f64, f64
// CHECK-NEXT:  %0 = inet.coduplicate %left, %left_0 : f64, f64 -> f64
// CHECK-NEXT:  %1 = inet.coduplicate %right, %right_1 : f64, f64 -> f64
// CHECK-NEXT:  inet.return %0, %1 : f64, f64
inet.inet @coconstruct_duplicate_commutation(f64, f64) -> (f64, f64) {
^bb0(%arg0: f64, %arg1: f64):
  %a = inet.coduplicate %arg0, %arg1 : f64, f64 -> f64
  %b, %c = inet.construct %a: f64 -> f64, f64
  inet.return %b, %c : f64, f64
}

// CHECK-LABEL: @cap_construct_duplicate_commutation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
// CHECK-NEXT:  %left, %right = inet.construct %arg2 : f64 -> f64, f64
// CHECK-NEXT:  %left_0, %right_1 = inet.construct %arg3 : f64 -> f64, f64
// CHECK-NEXT:  %left_2, %right_3 = inet.duplicate %arg0 : f64 -> f64, f64
// CHECK-NEXT:  %left_4, %right_5 = inet.duplicate %arg1 : f64 -> f64, f64
// CHECK-NEXT:  inet.cap %left, %right_5 : f64, f64
// CHECK-NEXT:  inet.cap %right, %right_3 : f64, f64
// CHECK-NEXT:  inet.cap %left_2, %right_1 : f64, f64
// CHECK-NEXT:  inet.cap %left_4, %left_0 : f64, f64
// CHECK-NEXT:  inet.return
inet.inet @cap_construct_duplicate_commutation(f64, f64, f64, f64) -> () {
^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
  %a = inet.coconstruct %arg0, %arg1 : f64, f64 -> f64
  %b = inet.coduplicate %arg2, %arg3 : f64, f64 -> f64
  inet.cap %a, %b : f64, f64
  inet.return
}

// CHECK-LABEL: @cap_duplicate_construct_commutation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
// CHECK-NEXT:  %left, %right = inet.duplicate %arg2 : f64 -> f64, f64
// CHECK-NEXT:  %left_0, %right_1 = inet.duplicate %arg3 : f64 -> f64, f64
// CHECK-NEXT:  %left_2, %right_3 = inet.construct %arg0 : f64 -> f64, f64
// CHECK-NEXT:  %left_4, %right_5 = inet.construct %arg1 : f64 -> f64, f64
// CHECK-NEXT:  inet.cap %left, %right_5 : f64, f64
// CHECK-NEXT:  inet.cap %right, %right_3 : f64, f64
// CHECK-NEXT:  inet.cap %left_2, %right_1 : f64, f64
// CHECK-NEXT:  inet.cap %left_4, %left_0 : f64, f64
// CHECK-NEXT:  inet.return
inet.inet @cap_duplicate_construct_commutation(f64, f64, f64, f64) -> () {
^bb0(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64):
  %a = inet.coduplicate %arg0, %arg1 : f64, f64 -> f64
  %b = inet.coconstruct %arg2, %arg3 : f64, f64 -> f64
  inet.cap %a, %b : f64, f64
  inet.return
}

// CHECK-LABEL: @cap_construct_erase_commutation
// CHECK:       ^bb0(%arg0: f64, %arg1: f64):
// CHECK-NEXT:  %0 = inet.coerase f64
// CHECK-NEXT:  %1 = inet.coconstruct %arg0, %arg1 : f64, f64 -> f64
// CHECK-NEXT:  inet.cap %0, %1 : f64, f64
// CHECK-NEXT:  inet.return
inet.inet @cap_construct_erase_commutation(f64, f64) -> () {
^bb0(%arg0: f64, %arg1: f64):
  %a = inet.coerase f64
  %b = inet.coconstruct %arg0, %arg1 : f64, f64 -> f64
  inet.cap %a, %b : f64, f64
  inet.return
}
