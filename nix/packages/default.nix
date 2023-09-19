{
  perSystem = {
    config,
    inputs',
    pkgs,
    ...
  }: {
    packages = {
      libnvjpeg_2k = pkgs.callPackage ./libnvjpeg_2k.nix {};
      dali = pkgs.callPackage ./dali.nix {
        inherit (config.packages) libnvjpeg_2k;
      };
      mfsr_utils = pkgs.python3Packages.callPackage ./mfsr_utils.nix {
        inherit (config.packages) rawpy;
        inherit (inputs'.hypothesis_torch_utils.packages) hypothesis_torch_utils;
      };
      rawpy = pkgs.python3Packages.callPackage ./rawpy.nix {};
    };
  };
}
