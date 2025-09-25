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
    inet.cup2 ^bb4 f64 %arg0 f64 %arg0
    ^bb2(%arg2: f64):
    inet.erase2 ^bb3 %arg2 f64
    ^bb3(%arg3: f64):
    inet.coerase2 {
      ^bb8(%pepe: f64):
      inet.erase2 ^bb7 %0 f64
      // inet.cup2 ^bb7 f64 %arg0 f64 %arg0
      // inet.cap2 f64 %0 f64 %0
      ^bb7(%arg7: f64):
      // inet.cap2 f64 %arg7 f64 %arg7
      inet.erase2 ^bb7 %0 f64
    }
    ^bb4(%arg4: f64, %arg11: f64):
    inet.coerase f64 %arg4
    inet.erase2 ^bb3 %arg0 f64
    ^bb42:
    inet.erase2 ^bb3 %arg0 f64
  }
  return
}
