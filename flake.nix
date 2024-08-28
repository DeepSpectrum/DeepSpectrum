{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
      pkgs = import nixpkgs {
        inherit system;
      };
      poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix {inherit pkgs;};
      overrides = poetry2nix.overrides.withDefaults (self: super: {
        fsspec = super.fsspec.overridePythonAttrs (old: {
          buildInputs = old.buildInputs or [] ++ [super.hatchling super.hatch-vcs];
        });
        lazy-loader = super.lazy-loader.overridePythonAttrs (old: {
          buildInputs = old.buildInputs or [] ++ [super.setuptools];
        });
        namex = super.namex.overridePythonAttrs (old: {
          buildInputs = old.buildInputs or [] ++ [super.setuptools];
        });
        bump-my-version = super.bump-my-version.overridePythonAttrs (old: {
          buildInputs = old.buildInputs or [] ++ [super.setuptools];
        });
        coveralls = super.coveralls.overridePythonAttrs (old: {
          buildInputs = old.buildInputs or [] ++ [super.poetry-core];
        });
        optree = super.optree.overridePythonAttrs (old: {
          buildInputs = old.buildInputs or [] ++ [super.setuptools];
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ [super.cmake super.pybind11];
          dontUseCmakeConfigure = true;
        });
        opencv-python = super.opencv-python.override {
          preferWheel = true;
        };
      });
      projectDir = self;
      python = pkgs.python311;
      src = self;
    in {
      packages = {
        deepspectrum = poetry2nix.mkPoetryApplication {
          inherit src python projectDir overrides;
          preferWheels = true;
        };
        default = self.packages.${system}.deepspectrum;
      };

      devShells.default = pkgs.mkShell {
        packages = [
          (poetry2nix.mkPoetryEnv {
            inherit python projectDir overrides;
            preferWheels = true;
            editablePackageSources = {
              deepspectrum = builtins.getEnv "PWD" + "/src";
            };
            groups = ["dev" "test"];
            checkGroups = ["dev" "test"];
          })
          pkgs.poetry
        ];
      };
    });
}
