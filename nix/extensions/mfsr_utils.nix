final: prev: let
  inherit (prev.lib) attrsets;
  python-overlay = python-final: python-prev: {
    # TODO: torchvision doesn't enforce a CC/CXX compatible with NVCC.
    torchvision = python-prev.torchvision.overrideAttrs (old: {
      preConfigure = ''
        export CC=${python-prev.torch.cudaPackages.cudatoolkit.cc}/bin/gcc
        export CXX=${python-prev.torch.cudaPackages.cudatoolkit.cc}/bin/g++
        export CUDAHOSTCXX=${python-prev.torch.cudaPackages.cudatoolkit.cc}/bin/g++
      '';
    });
    mfsr_utils = python-final.callPackage ../default.nix {};
  };
  new.python310.pkgs = prev.python310.pkgs.overrideScope python-overlay;
in
  attrsets.recursiveUpdate prev new
