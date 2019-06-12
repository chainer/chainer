# - Try to find cuDNN
#
# The following variables are optionally searched for defaults
#  CUDNN_ROOT_DIR:            Base directory where all cuDNN components are found
#
# The following are set after configuration is done:
#  CUDNN_FOUND
#  CUDNN_INCLUDE_DIRS
#  CUDNN_LIBRARIES
#  CUDNN_LIBRARY_DIRS
#
# Borrowed from https://github.com/pytorch/pytorch/blob/93f8d98027f29fc8190658fd52c2d5284e51875f/cmake/Modules/FindCuDNN.cmake,
# and some modifications are appiled.

include(FindPackageHandleStandardArgs)

# Determine CUDNN_ROOT_DIR
if(NOT DEFINED CUDNN_ROOT_DIR)
  if(DEFINED ENV{CUDNN_ROOT_DIR})
    set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR})
  else()
    set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
  endif()
endif()

# Determine CUDNN_INCLUDE_DIR
if(NOT DEFINED CUDNN_INCLUDE_DIR)
  if(DEFINED ENV{CUDNN_INCLUDE_DIR})
    set(CUDNN_INCLUDE_DIR $ENV{CUDNN_INCLUDE_DIR})
  else()
    find_path(CUDNN_INCLUDE_DIR cudnn.h
      HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES cuda/include include)
  endif()
endif()

# Determine CUDNN_LIB_DIR
if(NOT DEFINED CUDNN_LIB_DIR)
  if(DEFINED ENV{CUDNN_LIB_DIR})
    set(CUDNN_LIB_DIR $ENV{CUDNN_LIB_DIR})
  endif()
endif()

# Determine CUDNN_LIBNAME
if(NOT DEFINED CUDNN_LIBNAME)
  if(DEFINED ENV{CUDNN_LIBNAME})
    # libname from envvar
    set(CUDNN_LIBNAME $ENV{CUDNN_LIBNAME})
  elseif(DEFINED ENV{USE_STATIC_CUDNN})
    # Static library
    MESSAGE(STATUS "USE_STATIC_CUDNN detected. Linking against static CUDNN library")
    set(CUDNN_LIBNAME "libcudnn_static.a")
  else()
    # Dynamic library
    set(CUDNN_LIBNAME "cudnn")
  endif()
endif()

# Determine CUDNN_LIBRARY
if(NOT DEFINED CUDNN_LIBRARY)
  if(DEFINED ENV{CUDNN_LIBRARY})
    SET(CUDNN_LIBRARY $ENV{CUDNN_LIBRARY})
  else()
    find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
      HINTS ${CUDNN_LIB_DIR} ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
      PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
  endif()
endif()

# Misc.
if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
  # Find cudnn.h and determine CUDNN_VERSION
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_HEADER_CONTENTS)
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
                 CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
                 CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
                 CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
                 CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
                 CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
                 CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
  # Assemble cuDNN version
  if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "?")
  else()
    set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  endif()

  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  message(STATUS "Found cuDNN: v${CUDNN_VERSION}  (include: ${CUDNN_INCLUDE_DIR}, library: ${CUDNN_LIBRARY})")
  mark_as_advanced(CUDNN_ROOT_DIR CUDNN_LIBRARY CUDNN_INCLUDE_DIR)
endif()

# Find and load cuDNN settings
find_package_handle_standard_args(
    CuDNN
    VERSION_VAR CUDNN_VERSION
    REQUIRED_VARS CUDNN_INCLUDE_DIR CUDNN_LIBRARY
    FAIL_MESSAGE "Failed to find cuDNN in path: ${CUDNN_ROOT_DIR} (Did you set CUDNN_ROOT_DIR properly?)"
)
