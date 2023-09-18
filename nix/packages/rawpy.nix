{
  buildPythonPackage,
  fetchFromGitHub,
  lib,
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
  attrs = {
    pname = "rawpy";
    version = "0.18.1";
    format = "setuptools";

    src = fetchFromGitHub {
      owner = "letmaik";
      repo = attrs.pname;
      rev = "v${attrs.version}";
      hash = "sha256-ErQSVtv+pxKIgqCPrh74PbuWv4BKwqhLlBxtljmTCFM=";
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

    passthru.optional-dependencies.dev = [
      cython
      imageio
      numpy
      pytest
      scikitimage
      sphinx
      sphinx_rtd_theme
      wheel
    ];

    pythonImportsCheck = [attrs.pname];

    meta = with lib; {
      description = "RAW image processing for Python, a wrapper for libraw";
      homepage = "https://github.com/letmaik/rawpy";
      license = licenses.mit;
      maintainers = with maintainers; [connorbaker];
      # FIXME: macOS support exists but I have not tested it
      platforms = platforms.linux;
    };
  };
in
  buildPythonPackage attrs
