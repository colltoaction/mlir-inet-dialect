// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s


// CHECK-LABEL: @cup_coerase_coconstruct_commutation
// CHECK:       () -> (f64, f64, f64, f64) {
// CHECK-NEXT:  %0:2 = inet.cup f64, f64 
// CHECK-NEXT:  %1:2 = inet.coconstruct f64 %0#0 f64, f64
// CHECK-NEXT:  %2:2 = inet.cup f64, f64
// CHECK-NEXT:  %3:2 = inet.cup f64, f64
// CHECK-NEXT:  %4 = inet.construct f64 %2#0 f64 %3#0 f64
// CHECK-NEXT:  inet.cap f64 %4 f64 %0#1
// CHECK-NEXT:  return %1#0, %1#1, %2#1, %3#1 : f64, f64, f64, f64
func.func @cup_coerase_coconstruct_commutation(%arg0: f64) -> () {
  inet.inet {
    ^bb1:
    %0 = inet.erase f64
    inet.erase2 ^bb4 %0 f64
    ^bb2(%arg2: f64):
    inet.erase2 ^bb3 %arg2 f64
    ^bb3(%arg3: f64):
    inet.coerase f64 %arg3
    inet.erase2 ^bb4 %arg0 f64
    ^bb4(%arg4: f64):
    inet.coerase f64 %arg4
    inet.erase2 ^bb4 %arg0 f64
  }, {
    ^bb6:
    %0 = inet.erase f64
    inet.erase2 ^bb5 %0 f64
    ^bb5(%arg1: f64):
    inet.erase2 ^bb5 %arg1 f64
  }
  return
}
