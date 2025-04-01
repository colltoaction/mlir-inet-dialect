set(LLVM_TARGET_DEFINITIONS InetCanonicalization.td)
mlir_tablegen(InetCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRInetCanonicalizationIncGen)

add_mlir_dialect_library(MLIRInetDialect
  Inet.cpp
  InetCanonicalization.td

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Inet

  DEPENDS
  MLIRInetIncGen
  MLIRInetCanonicalizationIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  )
