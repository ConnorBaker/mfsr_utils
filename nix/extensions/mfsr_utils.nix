final: prev: {
  python3Packages = prev.python3Packages.overrideScope (python-final: python-prev: {
    mfsr_utils = python-final.callPackage ../default.nix {};
  });
}
