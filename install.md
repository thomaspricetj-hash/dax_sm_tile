install.md — SyntheticMind DAX SM‑Tile Installation \& Build Guide

1\. Requirements

Rust Toolchain

Rust 1.74+



cargo installed



rustup installed



CPU Features

The SM‑Tile automatically detects and uses:



AVX2



AVX‑512F



Scalar fallback if SIMD unavailable



Recommended Hardware

AMD Ryzen / EPYC



Intel Xeon / Core with AVX2 or AVX‑512



16GB+ RAM



Multi‑core CPU for parallel tile swarms



2\. Clone the Repository

bash

git clone https://github.com/thomaspricetj-hash/dax_sm_tile.git

cd syntheticmind-dax-sm-tile

3\. Build Instructions

Standard Build

bash

cargo build --release

This produces the optimized SM‑Tile engine with:



Generic delta‑state compute



Physics integrator



Algebra transform engine



Parallel tile swarm execution



AVX2 / AVX‑512 acceleration



Debug Build

bash

cargo build

4\. Running Tests

The project includes full performance tests for:



Generic delta‑state core



Physics integrator



Algebra transform engine



SIMD acceleration



Tile swarm parallelism



Run all tests:



bash

cargo test

Run with performance output:



bash

cargo test -- --nocapture

5\. Using Execution Modes

The SM‑Tile supports three compute modes inside BlockDeltaCore:



Generic Mode (default)

Delta‑state compute:



rust

core.set\_mode\_generic();

Physics Mode

Position/velocity/acceleration integration:



rust

core.set\_mode\_physics();

Physics uses the following packed layout inside Overlay SRAM:



Code

\[pos.x, pos.y, pos.z,

&#x20;vel.x, vel.y, vel.z,

&#x20;acc.x, acc.y, acc.z,

&#x20;inv\_mass]

Algebra Mode

Affine transform:



Code

y = a \* x + b

Where:



a = delta\_value



b = base\_reg



Enable:



rust

core.set\_mode\_algebra();

6\. Running a Tile Swarm

Example:



rust

let mut swarm = TileSwarm::new(32, 16384);

swarm.step\_fast\_swarm();

Swarm supports:



Generic delta compute



Physics integration



Algebra transforms



Parallel execution across all tiles



7\. SIMD Acceleration

The engine automatically selects the best SIMD path:



AVX‑512 → 16‑wide float ops



AVX2 → 8‑wide float ops



Scalar fallback → portable



No configuration required.



8\. Overlay SRAM Layout

Overlay SRAM is used for:



delta‑state buffers



physics bodies



algebra vectors



region routing



tile‑local compute



Physics mode uses a fixed stride of 10 floats per body.



9\. Optional Features

Enable Logging

bash

export RUST\_LOG=info

Enable Debug Assertions

bash

cargo build

10\. Project Structure

Code

src/

&#x20;├── sm\_tile/

&#x20;│    ├── delta\_core.rs      # Generic + Physics + Algebra modes

&#x20;│    ├── overlay\_sram.rs    # Tile-local memory

&#x20;│    ├── tile.rs            # Single tile

&#x20;│    └── swarm.rs           # Multi-tile swarm

&#x20;├── tests/

&#x20;│    └── perf\_tests.rs      # Performance + SIMD tests

&#x20;└── lib.rs

11\. License

You granted AMD full rights to evaluate and integrate the design.

Your repo should include:



Code

AMD\_LICENSE.md

LICENSE

README.md

install.md

12\. Build Troubleshooting

AVX‑512 not detected

Your CPU may not support AVX‑512.

Engine will automatically fall back to AVX2 or scalar.



Slow performance

Ensure you are running:



bash

cargo build --release

Rayon thread count

You can control parallelism:



bash

export RAYON\_NUM\_THREADS=32

