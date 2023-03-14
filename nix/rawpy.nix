{
  buildPythonPackage,
  fetchFromGitHub,
  lib,
  stdenv,
  # nativeBuildInputs
  cython,
  pkg-config,
  setuptools,
  wheel,
  # propagatedBuildInputs
  libraw,
  # propagatedBuildInputs and optional-dependencies.dev
  numpy,
  # optional-dependencies.dev
  imageio,
  pytest,
  scikitimage,
  sphinx,
  sphinx_rtd_theme,
}: let
  pname = "rawpy";
  version = "0.18.0";
in
  buildPythonPackage {
    inherit pname version;
    format = "setuptools";

    src = fetchFromGitHub {
      owner = "letmaik";
      repo = pname;
      rev = "v${version}";
      hash = "sha256-vSDzp/ttRloz3E3y2X87VMfQ+Wh+r4SAE5mXlDFmRqE=";
    };

    nativeBuildInputs = [
      cython
      pkg-config
      setuptools
      wheel
    ];

    propagatedBuildInputs = [
      libraw.dev
      numpy
    ];

    doCheck = false;

    passthru.optional-dependencies = {
      dev = [
        cython
        imageio
        numpy
        pytest
        scikitimage
        sphinx
        sphinx_rtd_theme
        wheel
      ];
    };

    pythonImportsCheck = ["rawpy"];

    meta = with lib; {
      description = "RAW image processing for Python, a wrapper for libraw";
      homepage = "https://github.com/letmaik/rawpy";
      license = licenses.mit;
      maintainers = with maintainers; [connorbaker];
      # FIXME: macOS support exists but I have not tested it
      platforms = platforms.linux;
    };
  }
