# mfsr_utils

Utility functions and data for Multi-Frame Super-Resolution (MFSR).

Contains pipelines for producing RAW bursts from a single RAW image.

Also contains a number of datasets for training and evaluating MFSR models.

## Usage

> **INFO**
>
> The following assumes a NixOS-based host. If that's not the case, most of the commands run inside the development shell will need to be wrapped with NixGL or a similar tool.

Ensure your Nixpkgs configuration is set up for the CUDA capabilities you wish to target. For example, I have a 4090 (capability 8.9):

```console
$ cat ~/.config/nixpkgs/config.nix 
{
  allowUnfree = true;
  cudaSupport = true;
  cudaCapabilities = ["8.9"];
}
```

Open a development shell with

```bash
nix develop --impure .
```

In the shell, you can use VS Code and a Jupyter managed by Nix. To do so, start the Jupyter server by running

```bash
jupyter server
```

Then, using the URL (with token) from the output of the previous command, connect to the server through VS Code's Jupyter extension by choosing the option to connect to an existing server.

Thank you to the authors involved in the following work, from which this code is derived:

- <http://timothybrooks.com/tech/unprocessing>
- <https://github.com/Algolzw/BSRT>

## TODO

- \[ \] Move to DataPipes from the current Dataset implementation
  - Tracked in [pipelines: Research `torchvision` offerings](https://github.com/ConnorBaker/mfsr_utils/issues/3)
- \[x\] Refactor to remove OpenCV usage from `single2lrburst`
  - Tracked in [single2lrburst: Refactor to remove OpenCV usage](https://github.com/ConnorBaker/mfsr_utils/issues/10)
- \[x\] Refactor to remove dependency on PyTorch Lightning
- \[ \] Move to NVIDIA's DALI for data loading
