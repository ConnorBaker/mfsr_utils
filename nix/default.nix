{
  pkgs,
  stdenv,
  lib,
  autoPatchelfHook,
  python,
  fetchFromGitHub,
  buildPythonPackage,
  # Propagated build inputs
  # main deps
  torch,
  torchvision,
  filelock,
  opencv4,
  # lint
  black,
  flake8,
  isort,
  # typecheck
  pyright,
  mypy,
  # test
  pytest,
  pytest-xdist,
  hypothesis,
  hypothesis_torch_utils,
}:
buildPythonPackage {
  pname = "mfsr_utils";
  version = "1.7";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "ConnorBaker";
    repo = "mfsr_utils";
    rev = "b99224431d551d0dd41f51faa7f795598ddc5013";
    hash = "sha256-YUpJ7Hy37UjTBEIu+cTrJNc+22c1PrtPms/TRKqfAq4=";
  };

  propagatedBuildInputs = [
    torch
    torchvision
    filelock
    opencv4
  ];

  # NOTE: We cannot use pythonImportsCheck for this module because it requires CUDA to be
  #   available at the time of import.
  doCheck = false;

  passthru.optional-dependencies = {
    lint = [
      black
      flake8
      isort
    ];
    typecheck = [pyright mypy];
    test = [
      pytest
      pytest-xdist
      hypothesis
      hypothesis_torch_utils
    ];
  };

  pythonImportsCheck = ["mfsr_utils"];
}
