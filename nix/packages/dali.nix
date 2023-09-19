{
  cudaPackages,
  fetchFromGitHub,
  lib,
  stdenv,
  symlinkJoin,
  # nativeBuildInputs
  cmake,
  ninja,
  pkg-config,
  python3,
  python3Packages,
  llvmPackages_16,
  # buildInputs
  libnvjpeg_2k,
  libsndfile,
  libvorbis,
  libogg,
  flac,
  libopus,
  ffmpeg_6-headless,
  opencv,
  openjpeg,
  libtiff,
  zstd,
  libjpeg, # libjpeg-turbo
  lmdb,
  protobuf,
  zlib,
  libtar,
  cfitsio,
}: let
  # TODO(@connorbaker): To my EXTREME irritation, it seems that DALI always looks for headers under the root of the
  # CUDA toolkit, which we set as NVCC. It can't find things there and has a bad time.
  cuda-redist = symlinkJoin {
    name = "cuda-redist";
    paths = with cudaPackages; [
      cuda_nvcc
      cuda_cudart
      libnvjpeg.dev
      libnvjpeg.static
      cuda_nvml_dev.lib
      cuda_nvml_dev.dev
      libcublas.dev
      libcufft.dev
      libcufft.static
      libcufile.dev
      libcufile.static
      libcurand.dev
      libnpp.dev
      libnpp.static
      libnvjpeg_2k.dev
      libnvjpeg_2k.static
    ];
  };
in
  cudaPackages.backendStdenv.mkDerivation (finalAttrs: {
    pname = "dali";
    version = "1.29.0";
    strictDeps = true;

    src = fetchFromGitHub {
      owner = "NVIDIA";
      repo = "DALI";
      rev = "v${finalAttrs.version}";
      hash = "sha256-nIiRxIl7syg8ZTyXFTGjidtvCClpup8y670UZm09CyY=";
      fetchSubmodules = true;
    };

    # TODO(@connorbaker): Oh my god why does DALI need the Clang module for Python?
    nativeBuildInputs = [
      cmake
      ninja
      python3
      pkg-config
      cuda-redist
    ];

    buildInputs = [
      llvmPackages_16.libclang
      cuda-redist
      libsndfile
      libvorbis
      protobuf
      libogg
      flac
      libopus
      ffmpeg_6-headless
      opencv
      openjpeg
      libtiff
      zstd
      libjpeg # libjpeg-turbo
      lmdb
      zlib
      libtar
      cfitsio
    ];

    cmakeFlags = [
      "-DBUILD_TEST:BOOL=OFF"
      "-DBUILD_BENCHMARK:BOOL=OFF"
      "-DBUILD_NVJPEG2K:BOOL=ON"
      "-DBUILD_CUFILE:BOOL=ON"
      "-DBUILD_PYTHON:BOOL=ON"
      "-DPYTHON_VERSIONS:STRING=${python3.version}"
      # TODO(@connorbaker): They don't support building with anything using a minor version after 35
      "-DCUDA_TARGET_ARCHS:STRING=80"
      # CMake has trouble finding the protobuf libraries.
      # Something to do with this?
      # https://github.com/NVIDIA/DALI/blob/9b360adba8458f8cd0ec669877be22ae0a768181/cmake/Dependencies.cmake#L118-L121
      "-DProtobuf_LIBRARIES:PATH=${lib.getLib protobuf}/lib"
    ];

    meta = with lib; {
      description = "NVIDIA Data Loading Library";
      homepage = "https://github.com/NVIDIA/DALI";
      license = licenses.asl20;
      platforms = platforms.linux;
      maintainers = with maintainers; [connorbaker];
    };
  })
