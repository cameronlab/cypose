{ pkgs ? import <unstable> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gnumake
    pkgs.cmake
    pkgs.clang
    pkgs.python310
    pkgs.python310Packages.lxml
    pkgs.virtualenv
    pkgs.darwin.apple_sdk.frameworks.Accelerate
    pkgs.darwin.apple_sdk.frameworks.Foundation
    pkgs.darwin.apple_sdk.frameworks.Metal
    pkgs.darwin.apple_sdk.frameworks.MetalKit
    pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders
    pkgs.darwin.apple_sdk.frameworks.AVFoundation
    pkgs.xsimd
    pkgs.cairo
    pkgs.git-cliff
  ];
  shellHook = ''
    # Update library paths
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.cairo}/lib/
    export CFLAGS='-std=c++11'
    if [ ! -d ".venv" ]; then
      python -m virtualenv .venv
    fi
  '';
}
