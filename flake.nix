{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/refs/pull/218265/head";
    nixGL = {
      url = "github:guibou/nixGL";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    hypothesis_torch_utils.url = "github:ConnorBaker/hypothesis_torch_utils";
  };

  nixConfig = {
    # Add the CUDA maintainer's cache and my cache to the binary cache list.
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = {
    self,
    nixpkgs,
    nixGL,
    hypothesis_torch_utils,
  }: let
    system = "x86_64-linux";
    nvidiaDriver = {
      version = "530.30.02";
      sha256 = "sha256-R/3bvXoiumYZI9vObn9R7sVN9oBQxAbMBJDDv77eeWM=";
    };

    overlay = nixpkgs.lib.composeManyExtensions [
      hypothesis_torch_utils.overlays.default
      (import ./nix/extensions/mfsr_utils.nix)
    ];
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaCapabilities = ["8.6"];
        cudaForwardCompat = true;
      };
      overlays = [
        (final: prev: {
          python3 = prev.python310;
          python3Packages = prev.python310Packages;
          cudaPackages = prev.cudaPackages_11_8;
        })
        # Magma fix for CUDA 11.8
        (final: prev: {
          magma = prev.magma.overrideAttrs (oldAttrs: {
            buildInputs = oldAttrs.buildInputs ++ [final.cudaPackages.cuda_profiler_api];
          });
        })
        # Fix for OpenCV 4.5.4
        (final: prev: {
          opencv4 = prev.opencv4.override {
            stdenv = prev.gcc11Stdenv;
          };
        })
        # Fix for TorchVision not using the correct GCC
        (final: prev: {
          python3Packages = prev.python3Packages.overrideScope (python-final: python-prev: {
            torchvision = python-prev.torchvision.overrideAttrs (oldAttrs: {
              preConfigure = ''
                export CC=${python-prev.torch.cudaPackages.cudatoolkit.cc}/bin/cc
                export CXX=${python-prev.torch.cudaPackages.cudatoolkit.cc}/bin/c++
                export CUDAHOSTCXX=${python-prev.torch.cudaPackages.cudatoolkit.cc}/bin/c++
              '';
            });
          });
        })
        overlay
        nixGL.overlays.default
      ];
    };

    inherit (pkgs) python3 python3Packages nixgl;
    inherit (python3Packages) mfsr_utils;
    inherit (nixgl.nvidiaPackages nvidiaDriver) nixGLNvidia;
  in {
    overlays.default = overlay;
    devShells.${system}.default = pkgs.mkShell {
      packages =
        [
          python3
          mfsr_utils.propagatedBuildInputs
          nixGLNvidia
        ]
        ++ (
          with mfsr_utils.passthru.optional-dependencies;
            lint ++ typecheck ++ test
        );

      # Make an alias for python so it's wrapped with nixGLNvidia.
      shellHook = ''
        alias python3="${nixGLNvidia.name} python3"
        alias python="${nixGLNvidia.name} python3"
      '';
    };
    formatter.${system} = pkgs.alejandra;
  };
}
