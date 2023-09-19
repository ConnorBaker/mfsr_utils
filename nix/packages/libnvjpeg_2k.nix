# TODO(@connorbaker): Manifests are at https://developer.download.nvidia.com/compute/nvjpeg2k/redist/
{
  cudaPackages,
  lib,
  lndir,
  fetchurl,
  stdenv,
  autoPatchelfHook,
}: let
  inherit (lib.strings) optionalString;
  inherit (lib.lists) optionals;
  inherit (lib.meta) getExe;
  hasBin = false;
  hasLib = true;
  hasDev = true;
  hasDoc = false;
  hasStatic = true;
  hasSample = false;
in
  cudaPackages.backendStdenv.mkDerivation {
    pname = "libnvjpeg_2k";
    version = "0.7.0";
    strictDeps = true;

    outputs =
      ["out"]
      ++ optionals hasBin ["bin"]
      ++ optionals hasLib ["lib"]
      ++ optionals hasStatic ["static"]
      ++ optionals hasDev ["dev"]
      ++ optionals hasDoc ["doc"]
      ++ optionals hasSample ["sample"];

    src = fetchurl {
      url = "https://developer.download.nvidia.com/compute/nvjpeg2k/redist/libnvjpeg_2k/linux-x86_64/libnvjpeg_2k-linux-x86_64-0.7.0.31-archive.tar.xz";
      hash = "sha256-LK1mIYhOT98XUwS0Xtdmu2WTagwodp+6n4z3XDD2DXU=";
    };

    # TODO(@connorbaker): lib contains a directory for CUDA 11 and 12, titled "11" and "12".
    prePatch = ''
      mv lib/${cudaPackages.cudaMajorVersion} new_lib
      rm -rf lib
      mv new_lib lib
    '';

    dontBuild = true;

    nativeBuildInputs = [
      autoPatchelfHook
      cudaPackages.autoAddOpenGLRunpathHook
      cudaPackages.markForCudatoolkitRootHook
    ];

    buildInputs = [
      # autoPatchelfHook will search for a libstdc++ and we're giving it
      # one that is compatible with the rest of nixpkgs, even when
      # nvcc forces us to use an older gcc
      # NB: We don't actually know if this is the right thing to do
      stdenv.cc.cc.lib
    ];

    # Picked up by autoPatchelf
    # Needed e.g. for libnvrtc to locate (dlopen) libnvrtc-builtins
    appendRunpaths = [
      "$ORIGIN"
    ];

    installPhase =
      # Pre-install hook
      ''
        runHook preInstall
      ''
      # doc and dev have special output handling. Other outputs need to be moved to their own
      # output.
      # Note that moveToOutput operates on all outputs:
      # https://github.com/NixOS/nixpkgs/blob/2920b6fc16a9ed5d51429e94238b28306ceda79e/pkgs/build-support/setup-hooks/multiple-outputs.sh#L105-L107
      + ''
        mkdir -p "$out"
        rm LICENSE
        mv * "$out"
      ''
      # Handle bin, which defaults to out
      + optionalString hasBin ''
        moveToOutput "bin" "$bin"
      ''
      # Handle lib, which defaults to out
      + optionalString hasLib ''
        moveToOutput "lib" "$lib"
      ''
      # Handle static libs, which isn't handled by the setup hook
      + optionalString hasStatic ''
        moveToOutput "**/*.a" "$static"
      ''
      # Handle samples, which isn't handled by the setup hook
      + optionalString hasSample ''
        moveToOutput "samples" "$sample"
      ''
      # Post-install hook
      + ''
        runHook postInstall
      '';

    # The out output leverages the same functionality which backs the `symlinkJoin` function in
    # Nixpkgs:
    # https://github.com/NixOS/nixpkgs/blob/d8b2a92df48f9b08d68b0132ce7adfbdbc1fbfac/pkgs/build-support/trivial-builders/default.nix#L510
    #
    # That should allow us to emulate "fat" default outputs without having to actually create them.
    #
    # It is important that this run after the autoPatchelfHook, otherwise the symlinks in out will reference libraries in lib, creating a circular dependency.
    postPhases = ["postPatchelf"];
    # For each output, create a symlink to it in the out output.
    # NOTE: We must recreate the out output here, because the setup hook will have deleted it
    # if it was empty.
    # NOTE: Do not use optionalString based on whether `outputs` contains only `out` -- phases
    # which are empty strings are skipped/unset and result in errors of the form "command not
    # found: <customPhaseName>".
    postPatchelf = ''
      mkdir -p "$out"
      for output in $outputs; do
        if [ "$output" = "out" ]; then
          continue
        fi
        ${getExe lndir} "''${!output}" "$out"
      done
    '';

    # Make the CUDA-patched stdenv available
    passthru.stdenv = cudaPackages.backendStdenv;

    # Setting propagatedBuildInputs to false will prevent outputs known to the multiple-outputs
    # from depending on `out` by default.
    # https://github.com/NixOS/nixpkgs/blob/2920b6fc16a9ed5d51429e94238b28306ceda79e/pkgs/build-support/setup-hooks/multiple-outputs.sh#L196
    # Indeed, we want to do the opposite -- fat "out" outputs that contain all the other outputs.
    propagatedBuildOutputs = false;

    # By default, if the dev output exists it just uses that.
    # However, because we disabled propagatedBuildOutputs, dev doesn't contain libraries or
    # anything of the sort. To remedy this, we set outputSpecified to true, and use
    # outputsToInstall, which tells Nix which outputs to use when the package name is used
    # unqualified (that is, without an explicit output).
    outputSpecified = true;

    meta = {
      description = "NVIDIA JPEG 2000 library";
      platforms = lib.platforms.linux;
      license = lib.licenses.unfree;
      maintainers = lib.teams.cuda.members;
      # Force the use of the default, fat output by default (even though `dev` exists, which
      # causes Nix to prefer that output over the others if outputSpecified isn't set).
      outputsToInstall = ["out"];
    };
  }
