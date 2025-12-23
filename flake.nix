{
  description = "Pacman.c development environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        pacman = pkgs.stdenv.mkDerivation {
          pname = "pacman";
          version = "1.0.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
          ];

          buildInputs = with pkgs; [
            # Audio
            alsa-lib

            # OpenGL
            libGL
            libGLU
            mesa

            # X11 libraries
            xorg.libX11
            xorg.libXext
            xorg.libXi
            xorg.libXcursor
            xorg.libXrandr
            xorg.libXinerama
            xorg.libXxf86vm

            # Additional dependencies
            libxkbcommon
          ];

          cmakeFlags = [ ];

          installPhase = ''
            mkdir -p $out/bin
            cp pacman $out/bin/
          '';
        };
      in
      {
        packages = {
          default = pacman;
          pacman = pacman;
        };

        apps = {
          default = {
            type = "app";
            program = "${pacman}/bin/pacman";
          };
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ pacman ];

          # Add extra dev tools not needed for building
          nativeBuildInputs = with pkgs; [
            gdb

            python3
            poetry
            python3Packages.pip
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath pacman.buildInputs}:$LD_LIBRARY_PATH"
          '';
        };
      }
    );
}
