{ pkgs
, stdenv
, lib
, autoPatchelfHook
, python
, fetchFromGitHub
, buildPythonPackage
, # Propagated build inputs
  # main deps
  torch
, torchvision
, filelock
, opencv
, # lint
  black
, flake8
, isort
, # typecheck
  pyright
, mypy
, # test
  pytest
, pytest-xdist
, hypothesis
, hypothesis_torch_utils
,
}:


buildPythonPackage {
  pname = "mfsr_utils";
  version = "1.7";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "ConnorBaker";
    repo = "mfsr_utils";
    rev = "1bb6ea5f66172112ff4e98a320888aaf753ea745";
    hash = "";
  };

  propagatedBuildInputs = [
    torch
    torchvision
    filelock
    opencv
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
    typecheck = [ pyright mypy ];
    test = [
      pytest
      pytest-xdist
      hypothesis
      hypothesis_torch_utils
    ];
  };

  pythonImportsCheck = [ "mfsr_utils" ];
}
