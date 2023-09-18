{
  perSystem = {
    config,
    inputs',
    pkgs,
    ...
  }: {
    packages = {
      mfsr_utils = pkgs.python3Packages.callPackage ./mfsr_utils.nix {
        inherit (config.packages) rawpy;
        inherit (inputs'.hypothesis_torch_utils.packages) hypothesis_torch_utils;
      };
      rawpy = pkgs.python3Packages.callPackage ./rawpy.nix {};
    };
  };
}
