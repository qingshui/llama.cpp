# Find NCCL library
#
# This module defines:
#  NCCL_FOUND - System has NCCL
#  NCCL_INCLUDE_DIRS - The NCCL include directories
#  NCCL_LIBRARIES - The libraries needed to use NCCL
#  NCCL::NCCL - Imported target

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS
        /home/work/nccl/nccl2.20.3_cuda12.3/include
        /usr/include
        /usr/local/include
        /opt/nccl/include
    PATH_SUFFIXES nccl
)

find_library(NCCL_LIBRARY
    NAMES nccl libnccl
    PATHS
        /home/work/nccl/nccl2.20.3_cuda12.3/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/lib
        /opt/nccl/lib
    PATH_SUFFIXES nccl
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
)

if(NCCL_FOUND)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})

    if(NOT TARGET NCCL::NCCL)
        add_library(NCCL::NCCL UNKNOWN IMPORTED)
        set_target_properties(NCCL::NCCL PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
