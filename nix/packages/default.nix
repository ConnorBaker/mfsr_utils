{
  perSystem = {pkgs, ...}: {
    legacyPackages = pkgs;
    packages = {
      inherit (pkgs) libnvjpeg_2k;
      inherit (pkgs.python3Packages) mfsr_utils rawpy;
      # dali = pkgs.callPackage ./dali.nix {
      #   inherit (config.packages) libnvjpeg_2k;
      # };
    };
  };
}
