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
  version = "0.1.0";
  format = "pyproject";

  src = ../.;

  propagatedBuildInputs = [
    torch
    torchvision
    filelock
    opencv4
  ];

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
