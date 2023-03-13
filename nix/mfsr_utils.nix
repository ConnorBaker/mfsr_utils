{
  buildPythonPackage,
  lib,
  # buildInputs
  opencv4,
  # propagatedBuildInputs
  filelock,
  torch,
  torchvision,
  # optional-dependencies.lint
  black,
  flake8,
  isort,
  # optional-dependencies.typecheck
  pyright,
  mypy,
  # optional-dependencies.test
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

  buildInputs = [
    opencv4
  ];

  propagatedBuildInputs = [
    filelock
    torch
    torchvision
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

  meta = with lib; {
    description = "Utilities for multi-frame super-resolution";
    homepage = "https://github.com/ConnorBaker/mfsr_utils";
    license = licenses.mit;
    maintainers = with maintainers; [connorbaker];
  };
}
