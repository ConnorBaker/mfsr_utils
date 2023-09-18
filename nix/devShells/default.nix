{
  perSystem = {
    config,
    pkgs,
    ...
  }: {
    devShells = {
      mfsr_utils = pkgs.mkShell {
        inputsFrom = [config.packages.mfsr_utils];
        packages = with config.packages.mfsr_utils.optional-dependencies; [dev test];
      };
      default = pkgs.mkShell {
        packages = [config.packages.mfsr_utils pkgs.python3Packages.jupyterlab];
      };
    };
  };
}
