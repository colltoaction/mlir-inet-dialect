; XFAIL: *
; REQUIRES: asserts
; RUN: llc -mtriple=r600 -mcpu=cypress < %s

define amdgpu_kernel void @inf_loop_irreducible_cfg() nounwind {
entry:
  br label %block

block:
  br label %block
}
