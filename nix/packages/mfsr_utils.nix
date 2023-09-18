{
  buildPythonPackage,
  lib,
  # propagatedBuildInputs
  filelock,
  rawpy,
  torch,
  torchvision,
  # optional-dependencies.dev
  black,
  ruff,
  pyright,
  mypy,
  # checkInputs
  pytest,
  pytest-xdist,
  hypothesis,
  hypothesis_torch_utils,
}: let
  attrs = {
    pname = "mfsr_utils";
    version = "0.1.0";
    format = "flit";

    src = lib.sources.sourceByRegex ../.. [
      "${attrs.pname}(:?/.*)?"
      "tests(:?/.*)?"
      "pyproject.toml"
    ];

    doCheck = false;

    checkInputs = attrs.passthru.optional-dependencies.test;

    propagatedBuildInputs = [
      filelock
      rawpy
      torch
      torchvision
    ];

    passthru.optional-dependencies = {
      dev = [
        # Linters/formatters
        black
        ruff
        # Type checkers
        pyright
        mypy
      ];
      test = [
        pytest
        pytest-xdist
        hypothesis
        hypothesis_torch_utils
      ];
    };

    pythonImportsCheck = [attrs.pname];

    meta = with lib; {
      description = "Utilities for multi-frame super-resolution";
      homepage = "https://github.com/ConnorBaker/mfsr_utils";
      license = licenses.mit;
      maintainers = with maintainers; [connorbaker];
    };
  };
in
  buildPythonPackage attrs
