// tests/perf_core.rs
// Upgraded to match the new delta_core / SM tile design.
// Tuned for more streaming, more cache, and heavier load to expose the new fast paths.

use std::time::{Duration, Instant};
use std::thread;

use syntheticmind_dax::sm_tile::{
    DaxSmTile,
    NormalCore,
    DeltaCore,
    DeltaCommand,
    OverlaySram,

    // micro‑fiber system (publicly re‑exported)
    BlockDeltaCore,
    MicroFiber,
    MicroFiberScheduler,
    MicroFiberMode,
    MicroFiberState,
    CoreBridge,
    perf_flags,
};

fn fmt(d: Duration) -> String {
    format!("{:.6} ms", d.as_secs_f64() * 1000.0)
}

#[test]
fn perf_compare_tile_vs_scalar_and_gpu_style_cores() {
    println!("=== SyntheticMind DAX Tile Performance Test (Streaming + More Cache) ===");

    // push scalar cores hard, then compare against streaming SM tile
    let iterations = 100_000;

    // enable perf flags for streaming behavior
    perf_flags::set_prefetch_enabled(true);
    perf_flags::set_non_temporal_stores(true);
    perf_flags::set_adaptive_nt_enabled(true);
    perf_flags::set_skip_scratchpad_for_small(true);
    perf_flags::set_zero_cost_mode(false);

    // --- Scalar NormalCore baseline ---
    let mut normal = NormalCore::new();
    let start_normal = Instant::now();
    for _ in 0..iterations {
        normal.apply(1.0);
    }
    let dur_normal = start_normal.elapsed();

    // --- Scalar DeltaCore baseline ---
    let mut delta = DeltaCore::new(0);
    let mut sram_scalar = OverlaySram::new(1024);
    let cmd = DeltaCommand { region_id: 0, delta_value: 1.0 };

    let start_delta = Instant::now();
    for _ in 0..iterations {
        delta.apply_delta(&cmd, &mut sram_scalar);
    }
    let dur_delta = start_delta.elapsed();

    // --- SM Tile fast-path test (streaming, larger SRAM, more cores) ---
    let sram_size = 8_192;
    let core_count = 8;
    let mut tile = DaxSmTile::new(core_count, sram_size);

    for r in 0..core_count {
        tile.region_table.add_region(r);
    }

    // warm-up to let predictors / NT heuristics settle
    for _ in 0..500 {
        tile.step_fast();
    }

    let start_tile = Instant::now();
    for _ in 0..iterations {
        tile.step_fast();
    }
    let dur_tile = start_tile.elapsed();

    let normal_work_per_iter = 1.0_f64;
    let delta_work_per_iter = 1.0_f64;
    let tile_work_per_iter = (core_count as f64) * (sram_size as f64);

    let normal_ops_per_sec = normal_work_per_iter * iterations as f64 / dur_normal.as_secs_f64();
    let delta_ops_per_sec = delta_work_per_iter * iterations as f64 / dur_delta.as_secs_f64();
    let tile_ops_per_sec = tile_work_per_iter * iterations as f64 / dur_tile.as_secs_f64();

    let theoretical_gpu_ops_per_sec = 1.024e12_f64;

    println!("Iterations: {}", iterations);
    println!("NormalCore total time: {}", fmt(dur_normal));
    println!("DeltaCore total time:  {}", fmt(dur_delta));
    println!("Tile (fast) total time: {}", fmt(dur_tile));

    println!(
        "NormalCore per-iteration: {:.9} µs",
        dur_normal.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
    println!(
        "DeltaCore per-iteration:  {:.9} µs",
        dur_delta.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
    println!(
        "Tile (fast) per-iteration: {:.9} µs",
        dur_tile.as_secs_f64() * 1_000_000.0 / iterations as f64
    );

    println!("--- Scalar throughput ---");
    println!("NormalCore scalar ops/sec: {:.2}", iterations as f64 / dur_normal.as_secs_f64());
    println!("DeltaCore  scalar ops/sec: {:.2}", iterations as f64 / dur_delta.as_secs_f64());
    println!("Tile (fast) scalar ops/sec: {:.2}", iterations as f64 / dur_tile.as_secs_f64());

    println!("--- Effective throughput (streaming tile) ---");
    println!("NormalCore effective ops/sec: {:.2}", normal_ops_per_sec);
    println!("DeltaCore  effective ops/sec: {:.2}", delta_ops_per_sec);
    println!("Tile (fast) effective ops/sec: {:.2}", tile_ops_per_sec);

    println!("--- Theoretical GPU-style core ---");
    println!(
        "Theoretical GPU core ops/sec (1 GHz, 1024 ops/cycle): {:.2}",
        theoretical_gpu_ops_per_sec
    );

    assert_eq!(tile.overlay_sram.data.len(), sram_size);
    assert_eq!(tile.block_delta_cores.len(), core_count);
}

#[test]
fn perf_micro_fiber_scheduler_vs_tile_fast() {
    println!("=== SyntheticMind Micro-Fiber Scheduler Performance Test (Streaming + More Cache) ===");

    let iterations = 50_000;
    let sram_size = 16_384;
    let core_count = 8;

    perf_flags::set_prefetch_enabled(true);
    perf_flags::set_non_temporal_stores(true);
    perf_flags::set_adaptive_nt_enabled(true);
    perf_flags::set_skip_scratchpad_for_small(true);
    perf_flags::set_zero_cost_mode(false);

    let mut tile = DaxSmTile::new(core_count, sram_size);
    for r in 0..core_count {
        tile.region_table.add_region(r);
    }

    let mut sram = OverlaySram::new(sram_size);

    let mut cores: Vec<BlockDeltaCore> =
        (0..core_count).map(|i| BlockDeltaCore::new(i)).collect();

    let mut sched = MicroFiberScheduler::new(&mut cores, 0);

    // more fibers, still contiguous-ish to favor streaming
    for i in 0..128 {
        let mode = match i % 3 {
            0 => MicroFiberMode::Physics,
            1 => MicroFiberMode::Generic,
            _ => MicroFiberMode::Hybrid,
        };

        let fiber = MicroFiber::new(
            i,
            mode,
            (i * 64) % sram_size,
            128,
            0.05,
            if i % 10 == 0 { 1 } else { 0 },
        );

        sched.add_fiber(fiber);
    }

    sched.build_groups(16);

    let bridge = CoreBridge {
        tile_id: 0,
        amd_core_id: 0,
        weight: 1.0,
        amd_callback: Some(|chunk, dt| {
            for x in chunk.iter_mut().take(16) {
                *x += dt * 0.5;
            }
        }),
    };

    // warm-up scheduler to let predictors / NT heuristics adapt
    for _ in 0..200 {
        sched.run_all(&mut sram, Some(&bridge));
    }

    let start_sched = Instant::now();
    for _ in 0..iterations {
        sched.run_all(&mut sram, Some(&bridge));
    }
    let dur_sched = start_sched.elapsed();

    let start_tile = Instant::now();
    for _ in 0..iterations {
        tile.step_fast();
    }
    let dur_tile = start_tile.elapsed();

    println!("Micro-Fiber Scheduler total time: {}", fmt(dur_sched));
    println!("Tile (fast) total time:           {}", fmt(dur_tile));

    println!(
        "Micro-Fiber per-iteration: {:.9} µs",
        dur_sched.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
    println!(
        "Tile (fast) per-iteration: {:.9} µs",
        dur_tile.as_secs_f64() * 1_000_000.0 / iterations as f64
    );

    println!("--- Fiber-Level Perf Counters ---");
    println!("Physics steps: {}", sched.perf.physics_steps);
    println!("Generic steps: {}", sched.perf.generic_steps);
    println!("Hybrid steps:  {}", sched.perf.hybrid_steps);
    println!("Total elements touched: {}", sched.perf.total_elements_touched);

    assert!(sched.perf.total_elements_touched > 0);
    assert!(sched.perf.physics_steps > 0);
    assert!(sched.perf.generic_steps > 0);
    assert!(sched.perf.hybrid_steps > 0);
}

#[test]
fn test_micro_fiber_priority_ordering() {
    println!("=== Micro-Fiber Priority Ordering Test ===");

    let mut cores: Vec<BlockDeltaCore> =
        (0..2).map(|i| BlockDeltaCore::new(i)).collect();

    let mut sched = MicroFiberScheduler::new(&mut cores, 0);

    sched.add_fiber(MicroFiber::new(0, MicroFiberMode::Hybrid, 0, 64, 0.1, 0));
    sched.add_fiber(MicroFiber::new(1, MicroFiberMode::Generic, 64, 64, 0.1, 0));
    sched.add_fiber(MicroFiber::new(2, MicroFiberMode::Physics, 128, 64, 0.1, 0));

    sched.build_groups(4);

    let first_group = &sched.groups[0];
    let first_fiber_idx = first_group.fibers[0];
    let first_fiber = &sched.fibers[first_fiber_idx];

    println!("First fiber mode: {:?}", first_fiber.mode);
    assert!(matches!(first_fiber.mode, MicroFiberMode::Physics));
}

#[test]
fn test_micro_fiber_migration() {
    println!("=== Micro-Fiber Migration Test ===");

    let mut cores: Vec<BlockDeltaCore> =
        (0..2).map(|i| BlockDeltaCore::new(i)).collect();

    let mut sched = MicroFiberScheduler::new(&mut cores, 0);

    // fiber lives on tile 1, scheduler on tile 0
    let f = MicroFiber::new(0, MicroFiberMode::Generic, 0, 64, 0.1, 1);
    sched.add_fiber(f);

    sched.migrate_fibers_between_tiles(0);

    println!("Migration state: {:?}", sched.fibers[0].state);
    assert!(matches!(sched.fibers[0].state, MicroFiberState::Completed));
}

// -----------------------------------------------------------------------------
// Stress test: run multiple configurations and heavier iteration counts to
// observe how the design behaves under load with more cache and streaming.
// -----------------------------------------------------------------------------
#[test]
fn stress_perf_under_load() {
    println!("=== Stress Test: Design Under Load (Streaming + More Cache) ===");

    perf_flags::set_prefetch_enabled(true);
    perf_flags::set_non_temporal_stores(true);
    perf_flags::set_adaptive_nt_enabled(true);
    perf_flags::set_skip_scratchpad_for_small(true);
    perf_flags::set_zero_cost_mode(false);

    // Configurations to try: (core_count, sram_size, iterations)
    let configs = vec![
        (2usize, 4_096usize, 50_000usize),
        (4usize, 8_192usize, 50_000usize),
        (8usize, 16_384usize, 25_000usize),
        (16usize, 32_768usize, 12_000usize),
    ];

    // We'll run each config twice: tile.step_fast and scheduler.run_all
    for (core_count, sram_size, iterations) in configs {
        println!(
            "\nConfig: cores={} sram={} iterations={}",
            core_count, sram_size, iterations
        );

        // Prepare tile
        let mut tile = DaxSmTile::new(core_count, sram_size);
        for r in 0..core_count {
            tile.region_table.add_region(r);
        }

        // Warm-up
        for _ in 0..500 {
            tile.step_fast();
        }

        // Measure tile.step_fast
        let start_tile = Instant::now();
        for _ in 0..iterations {
            tile.step_fast();
        }
        let dur_tile = start_tile.elapsed();

        println!(
            "Tile.step_fast: total={} per-iter={:.3} µs",
            fmt(dur_tile),
            dur_tile.as_secs_f64() * 1_000_000.0 / iterations as f64
        );

        // Prepare scheduler + cores + sram
        let mut cores: Vec<BlockDeltaCore> =
            (0..core_count).map(|i| BlockDeltaCore::new(i)).collect();

        let mut sched = MicroFiberScheduler::new(&mut cores, 0);
        // populate a larger set of fibers to increase scheduling pressure
        let fiber_count = 256.min(sram_size / 16);
        for i in 0..fiber_count {
            let mode = match i % 3 {
                0 => MicroFiberMode::Physics,
                1 => MicroFiberMode::Generic,
                _ => MicroFiberMode::Hybrid,
            };
            let start = (i * 16) % sram_size;
            let len = 64.min(sram_size - start);
            let tile_id = if i % 10 == 0 { 1 } else { 0 };
            sched.add_fiber(MicroFiber::new(i, mode, start, len, 0.05, tile_id));
        }
        sched.build_groups(32);

        let mut sram = OverlaySram::new(sram_size);

        let bridge = CoreBridge {
            tile_id: 0,
            amd_core_id: 0,
            weight: 1.0,
            amd_callback: Some(|chunk, dt| {
                // small callback to simulate hybrid work
                for x in chunk.iter_mut().take(8) {
                    *x += dt * 0.25;
                }
            }),
        };

        // Warm-up scheduler
        for _ in 0..200 {
            sched.run_all(&mut sram, Some(&bridge));
        }

        // Measure scheduler.run_all
        let start_sched = Instant::now();
        for _ in 0..iterations {
            sched.run_all(&mut sram, Some(&bridge));
        }
        let dur_sched = start_sched.elapsed();

        println!(
            "Scheduler.run_all: total={} per-iter={:.3} µs",
            fmt(dur_sched),
            dur_sched.as_secs_f64() * 1_000_000.0 / iterations as f64
        );

        // Quick concurrency probe: spawn a few threads each running a tile loop
        // to simulate multiple tiles running concurrently (non-shared state).
        let thread_tiles = 4usize.min(core_count);
        let thread_iters = iterations / 10;
        let mut handles = Vec::new();
        for _t in 0..thread_tiles {
            let mut local_tile = DaxSmTile::new(core_count, sram_size / 2);
            for r in 0..core_count {
                local_tile.region_table.add_region(r);
            }
            handles.push(thread::spawn(move || {
                let st = Instant::now();
                for _ in 0..thread_iters {
                    local_tile.step_fast();
                }
                st.elapsed()
            }));
        }

        let mut thread_total = Duration::ZERO;
        for h in handles {
            if let Ok(d) = h.join() {
                thread_total += d;
            }
        }

        println!(
            "Concurrent tiles ({} threads) total={} per-thread-iter={:.3} µs",
            thread_tiles,
            fmt(thread_total),
            thread_total.as_secs_f64() * 1_000_000.0 / thread_iters as f64
        );
    }

    println!("\n=== Stress test complete ===");
}

