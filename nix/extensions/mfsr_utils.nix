final: prev: {
  python3Packages = prev.python3Packages.overrideScope (python-final: python-prev: {
    rawpy = python-final.callPackage ../rawpy.nix {};
    mfsr_utils = python-final.callPackage ../mfsr_utils.nix {};
  });
}
