// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

inet.inet @inet_franchu_example () -> (f64) {
  %4:2 = inet.cup f64, f64
  %6 = inet.coduplicate %4#0, %4#1 : f64, f64 -> f64
  %10:2 = inet.construct %6 : f64 -> f64, f64
  %11:2 = inet.duplicate %10#0 : f64 -> f64, f64
  inet.cap %11#1, %10#1 : f64, f64
  inet.return %11#0 : f64
}

inet.inet @inet_franchu_example_dual (f64) -> () {
^bbentry(%arg1: f64):
  %4:2 = inet.cup f64, f64
  %6 = inet.coduplicate %4#1, %arg1 : f64, f64 -> f64
  %10 = inet.coconstruct %4#0, %6 : f64, f64 -> f64
  %11:2 = inet.duplicate %10 : f64 -> f64, f64
  inet.cap %11#0, %11#1 : f64, f64
  inet.return
}