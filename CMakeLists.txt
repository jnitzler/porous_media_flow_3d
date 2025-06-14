CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

FIND_PACKAGE(deal.II 9.5.0
  HINTS ${DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT("Darcy Problem")

function(setup_executable target)
  add_executable(${target} ${target}.cc)
  target_include_directories(${target} PRIVATE .)
  target_include_directories(${target} PRIVATE external)
  deal_ii_setup_target(${target})
endfunction()

setup_executable("darcy")
setup_executable("darcy_adjoint")