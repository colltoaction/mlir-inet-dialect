// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s


// CHECK-LABEL: @inet_franchu_example
// CHECK-NEXT:  %0:2 = inet.cup f64, f64
// CHECK-NEXT:  %1:2 = inet.coconstruct f64 %0#0 f64, f64
// CHECK-NEXT:  %2:2 = inet.coconstruct f64 %0#1 f64, f64
// CHECK-NEXT:  %3 = inet.duplicate f64 %1#1 f64 %2#1 f64
// CHECK-NEXT:  inet.cap f64 %1#0 f64 %3
// CHECK-NEXT:  return %2#0 : f64
func.func @inet_franchu_example() -> (f64) {
  %0:2 = inet.cup f64, f64
  %1 = inet.duplicate f64 %0#0 f64 %0#1 f64
  %2:2 = inet.coconstruct f64 %1 f64, f64
  %3:2 = inet.coduplicate f64 %2#0 f64, f64
  inet.cap f64 %3#1 f64 %2#1
  inet.inet {
    %4:2 = inet.cup2 ^bbduplicate f64 f64
  ^bbduplicate:
    %6 = inet.coduplicate2 ^bbco2 %4#0: f64 %4#1: f64 f64
  ^bbco2(%7: f64, %8: f64):
    %10:2 = inet.construct2 ^bb3 %6: f64 f64, f64
  ^bb3(%p1: f64):
    %11:2 = inet.construct2 ^bb3 %10#0: f64 f64, f64
  // ^bb4:
  //   inet.cap2 %11#1: f64 %10#1: f64
  // ^bbcoduplicate(%p2: f64):
  //   inet.cap2 f64 %1 f64 %1
  // ^bbcap(%l2: f64, %r2: f64):
  //   inet.cap2 f64 %1 f64 %1
  }
  return %3#0 : f64
}