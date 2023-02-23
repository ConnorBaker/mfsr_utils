{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs";
  inputs.hypothesis_torch_utils.url = "github:ConnorBaker/hypothesis_torch_utils";

  nixConfig = {
    # Add the CUDA maintainer's cache and my cache to the binary cache list.
    extra-substituters = [
      "https://cache.nixos.org/"
      "https://nix-community.cachix.org"
      "https://cuda-maintainers.cachix.org"
      "https://tiny-cuda-nn.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "tiny-cuda-nn.cachix.org-1:XFknw1My0ep3nR307ytC07jgZmEo/O0bg9PdqYZg29Q="
    ];
  };

  outputs = {
    self,
    nixpkgs,
    hypothesis_torch_utils
  }: let
    system = "x86_64-linux";
    overlay = import ./nix/extensions/mfsr_utils.nix;
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaCapabilities = ["8.0"];
        cudaForwardCompat = true;
      };
      overlays = [
        hypothesis_torch_utils.overlays.default
        overlay
      ];
    };

    inherit (pkgs.cudaPackages) cudatoolkit;
    inherit (pkgs) python310;
    inherit (python310.pkgs) mfsr_utils;
  in {
    overlays.default = overlay;
    packages.${system}.default = pkgs.mkShell {
      buildInputs = [
        cudatoolkit
        # TODO: Can't use python3.withPackages because it doesn't work with overlays?
        python310
        mfsr_utils
      ];

      shellHook = ''
        export CUDA_HOME=${cudatoolkit}
        export PATH=$CUDA_HOME/bin:$PATH
        if [ ! -f /run/opengl-driver/lib/libcuda.so.1 ]; then
          echo
          echo "Could not find /run/opengl-driver/lib/libcuda.so.1."
          echo
          echo "You have at least three options:"
          echo
          echo "1. Use nixGL (https://github.com/guibou/nixGL)"
          echo "2. Add libcuda.so.1 to your LD_PRELOAD environment variable."
          echo "3. Symlink libcuda.so.1 to /run/opengl-driver/lib/libcuda.so.1."
          echo
          echo "   This is the easiest option, but it requires root."
          echo "   You can do this by running:"
          echo
          echo "   sudo mkdir -p /run/opengl-driver/lib"
          echo "   sudo ln -s /usr/lib64/libcuda.so.1 /run/opengl-driver/lib/libcuda.so.1"
          echo
          echo "Continuing to the shell, but be aware that CUDA might not work."
          echo
        fi
      '';
    };
    formatter.${system} = pkgs.alejandra;
  };
}
