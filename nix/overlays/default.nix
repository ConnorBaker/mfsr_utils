{
  inputs,
  lib,
  ...
}: {
  flake.overlays.default = lib.composeManyExtensions [
    inputs.hypothesis_torch_utils.overlays.default
    (final: prev: {
      libnvjpeg_2k = final.callPackage ../packages/libnvjpeg_2k.nix {};
      pythonPackagesExtensions =
        prev.pythonPackagesExtensions
        ++ [
          (
            pythonFinal: _: {
              mfsr_utils = pythonFinal.callPackage ../packages/mfsr_utils.nix {};
              rawpy = pythonFinal.callPackage ../packages/rawpy.nix {};
            }
          )
        ];
    })
  ];
}
