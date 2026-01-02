# pacman.c 

A Pacman clone written in C99 with minimal dependencies for Windows, macOS, Linux and WASM.

[WASM version](https://floooh.github.io/pacman.c/pacman.html)

For implementation details see comments in the pacman.c source file (I've tried
to structure the source code so that it can be read from top to bottom).

Related projects:

- Zig version: https://github.com/floooh/pacman.zig

## HTTP API for AI Training

This fork includes an **optional HTTP API** that allows programmatic control of the game, perfect for AI training and automated gameplay.

**Quick Start:**

```bash
# Initialize submodules (for Mongoose HTTP library)
git submodule update --init --recursive

cmake -B build -DPACMAN_ENABLE_API=ON  
cmake --build build

./build/pacman --api
```

## Clone, Build and Run (Linux, macOS, Windows)

On the command line:

```
git clone https://github.com/floooh/pacman.c
cd pacman.c
mkdir build
cd build
cmake ..
cmake --build .
```

> NOTE: on Linux you'll need to install the OpenGL, X11 and ALSA development packages (e.g. mesa-common-dev, libx11-dev and libasound2-dev).

On Mac and Linux this will create an executable called 'pacman'
in the build directory:

```
./pacman
```

with nix:
```bash
nix run github:floooh/pacman.c
```

On Windows, the executable is in a subdirectory:

```
Debug/pacman.exe
```

## Build and Run WASM/HTML version via Emscripten

> NOTE: You'll run into various problems running the Emscripten SDK tools on Windows, might be better to run this stuff in WSL.

Setup the emscripten SDK as described here:

https://emscripten.org/docs/getting_started/downloads.html#installation-instructions

Don't forget to run ```source ./emsdk_env.sh``` after activating the SDK.

And then in the pacman.c directory:

```
mkdir build
cd build
emcmake cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=MinSizeRel ..
cmake --build .
```

To run the compilation result in the system web browser:

```
> emrun pacman.html
```

## IDE Support

On Windows with Visual Studio cmake will automatically create a **Visual Studio** solution file which can be opened with ```cmake --open .```:
```
cd build
cmake ..
cmake --open .
```

On macOS, the cmake **Xcode** generator can be used to create an
Xcode project which can be opened with ```cmake --open .```
```
cd build
cmake -GXcode ..
cmake --open .
```

On all platforms with **Visual Studio Code** and the Microsoft C/C++ and
CMake Tools extensions, simply open VSCode in the root directory of the
project. The CMake Tools extension will detect the CMakeLists.txt file and
take over from there:
```
cd pacman.c
code .
```


## QLearning after 5000 episodes
```
Window          Avg Score   Avg Dots  Max Score   Max Dots
------------------------------------------------------------
1-500               699.1       63.0       2160        127
501-1000            802.6       72.6       2260        165
1001-1500           825.8       74.0       2050        153
1501-2000           888.4       77.7       2290        190
2001-2500           936.8       80.9       3110        212
2501-3000           981.3       84.0       2620        190
3001-3500          1114.7       94.0       3500        204
3501-4000          1147.8       95.3       3380        218
4001-4500          1271.2      103.1       3420        196
4501-5000          1238.3      102.0       3420        214

=== BEST PERFORMANCES ===
Top 10 by dots eaten:
  Ep 3969: 218 dots, score 3340
  Ep 4759: 214 dots, score 2500
  Ep 2385: 212 dots, score 3040
  Ep 3386: 204 dots, score 2200
  Ep 4798: 203 dots, score 2690
  Ep 4943: 202 dots, score 2380
  Ep 3360: 200 dots, score 2160
  Ep 3111: 197 dots, score 2330
  Ep 4735: 197 dots, score 2290
  Ep 4204: 196 dots, score 2320

Episodes with 200+ dots: 7
Episodes with 180+ dots: 28
```
