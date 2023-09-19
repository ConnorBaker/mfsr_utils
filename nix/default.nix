{
  inputs,
  self,
  ...
}: {
  imports = [
    ./devShells
    ./overlays
    ./packages
  ];
  perSystem = {system, ...}: {
    _module.args.pkgs = import inputs.nixpkgs {
      inherit system;
      overlays = [
        inputs.hypothesis_torch_utils.overlays.default
        self.overlays.default
      ];
    };
  };
}
