// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

func.func @inet_franchu_example(%arg0: f64) -> (f64) {
  inet.inet {
  ^bbentry(%arg1: f64):
    %4:2 = inet.cup2 ^bbduplicate f64 f64
  ^bbduplicate:
    %6 = inet.coduplicate2 ^bbco2 %4#0 : f64 %4#1 : f64 f64
  ^bbco2(%7: f64, %8: f64):
    %10:2 = inet.construct2 ^bb3 %6 : f64 f64, f64
  ^bb3(%p1: f64):
    %11:2 = inet.duplicate2 ^bb4 %10#0 : f64 f64, f64
  ^bb4(%9: f64):
    inet.cap2 %11#1 : f64 %10#1 : f64
  }
  return %arg0 : f64
}

func.func @inet_franchu_example_dual(%arg0: f64) -> (f64) {
  inet.inet {
  ^bbentry(%arg1: f64):
    %4:2 = inet.cup2 ^bbduplicate f64 f64
  ^bbduplicate:
    %6 = inet.coduplicate2 ^bbco2 %4#1 : f64 %arg1 : f64 f64
  ^bbco2(%7: f64, %8: f64):
    %10 = inet.coconstruct2 ^bb3 %4#1 : f64 %6 : f64 f64
  ^bb3(%12: f64, %13: f64):
    %11:2 = inet.duplicate2 ^bb4 %10 : f64 f64, f64
  ^bb4(%9: f64):
    inet.cap2 %11#0 : f64 %11#1 : f64
  }
  return %arg0 : f64
}