{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gnumake
    pkgs.cmake
    pkgs.clang
    pkgs.python310
    pkgs.virtualenv
    pkgs.darwin.apple_sdk.frameworks.Accelerate
    pkgs.darwin.apple_sdk.frameworks.Foundation
    pkgs.darwin.apple_sdk.frameworks.Metal
    pkgs.darwin.apple_sdk.frameworks.MetalKit
    pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders
    pkgs.darwin.apple_sdk.frameworks.AVFoundation
    pkgs.xsimd
  ];
}
