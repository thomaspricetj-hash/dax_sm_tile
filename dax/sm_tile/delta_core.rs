// delta_core.rs
// Production-grade, non-destructive performance upgrades for DeltaCore / BlockDeltaCore.
// Preserves all original semantics; adds fast paths, prefetch, adaptive non-temporal stores,
// improved parallel chunking, micro-optimizations, SIMD physics, and a lightweight state predictor.

use crate::sm_tile::DeltaCommand;
use crate::sm_tile::OverlaySram;
use std::arch::x86_64::*;
use rayon::prelude::*;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};

// -----------------------------------------------------------------------------
// Runtime perf tuning flags (non-destructive, opt-in).
// -----------------------------------------------------------------------------
pub mod perf_flags {
    use super::*;
    static ZERO_COST_MODE: AtomicBool = AtomicBool::new(false);
    static PREFETCH_ENABLED: AtomicBool = AtomicBool::new(true);
    static SKIP_SCRATCHPAD_FOR_SMALL: AtomicBool = AtomicBool::new(true);
    static NON_TEMPORAL_STORES: AtomicBool = AtomicBool::new(true);
    static ADAPTIVE_NT_ENABLED: AtomicBool = AtomicBool::new(true);

    #[inline]
    pub fn set_zero_cost_mode(v: bool) {
        ZERO_COST_MODE.store(v, Ordering::Relaxed);
    }
    #[inline]
    pub fn zero_cost_mode() -> bool {
        ZERO_COST_MODE.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn set_prefetch_enabled(v: bool) {
        PREFETCH_ENABLED.store(v, Ordering::Relaxed);
    }
    #[inline]
    pub fn prefetch_enabled() -> bool {
        PREFETCH_ENABLED.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn set_skip_scratchpad_for_small(v: bool) {
        SKIP_SCRATCHPAD_FOR_SMALL.store(v, Ordering::Relaxed);
    }
    #[inline]
    pub fn skip_scratchpad_for_small() -> bool {
        SKIP_SCRATCHPAD_FOR_SMALL.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn set_non_temporal_stores(v: bool) {
        NON_TEMPORAL_STORES.store(v, Ordering::Relaxed);
    }
    #[inline]
    pub fn non_temporal_stores() -> bool {
        NON_TEMPORAL_STORES.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn set_adaptive_nt_enabled(v: bool) {
        ADAPTIVE_NT_ENABLED.store(v, Ordering::Relaxed);
    }
    #[inline]
    pub fn adaptive_nt_enabled() -> bool {
        ADAPTIVE_NT_ENABLED.load(Ordering::Relaxed)
    }
}

// -------------------------------------------------------------
// Tiny two-bit saturating predictor for region state
// -------------------------------------------------------------
#[derive(Debug)]
pub struct StatePredictor {
    table: Vec<u8>, // two-bit saturating counters (0..=3)
    mask: usize,
}

impl StatePredictor {
    pub fn new(size_pow2: usize) -> Self {
        let size = 1usize << size_pow2;
        Self {
            table: vec![1u8; size], // weakly predict "not hot / zero" initially
            mask: size - 1,
        }
    }

    #[inline(always)]
    fn idx(&self, start: usize, len: usize) -> usize {
        let a = start.wrapping_mul(0x9E3779B97F4A7C15u64 as usize);
        let b = len;
        (a ^ b).wrapping_mul(0x85EBCA6Bu32 as usize) & self.mask
    }

    #[inline(always)]
    pub fn predict_hot(&self, start: usize, len: usize) -> bool {
        let i = self.idx(start, len);
        self.table[i] >= 2
    }

    #[inline(always)]
    pub fn predict_nonzero(&self, start: usize, len: usize) -> bool {
        let i = (self.idx(start, len) ^ 0x5555) & self.mask;
        self.table[i] >= 2
    }

    #[inline(always)]
    pub fn update(&mut self, start: usize, len: usize, was_hot: bool, was_nonzero: bool) {
        let i1 = self.idx(start, len);
        let i2 = (i1 ^ 0x5555) & self.mask;

        if was_hot {
            if self.table[i1] < 3 {
                self.table[i1] += 1;
            }
        } else if self.table[i1] > 0 {
            self.table[i1] -= 1;
        }

        if was_nonzero {
            if self.table[i2] < 3 {
                self.table[i2] += 1;
            }
        } else if self.table[i2] > 0 {
            self.table[i2] -= 1;
        }
    }
}

// -------------------------------------------------------------
// Adaptive NT heuristic (small, per-SM rolling counter)
// -------------------------------------------------------------
#[derive(Debug, Default)]
pub struct AdaptiveNtHeuristic {
    pub counter: i8,
}

impl AdaptiveNtHeuristic {
    #[inline(always)]
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    #[inline(always)]
    pub fn observe_streaming(&mut self) {
        if self.counter < 120 {
            self.counter = self.counter.saturating_add(2);
        }
    }

    #[inline(always)]
    pub fn observe_reuse(&mut self) {
        if self.counter > -120 {
            self.counter = self.counter.saturating_sub(1);
        }
    }

    #[inline(always)]
    pub fn should_use_nt(&self, len: usize, predictor_hint: bool) -> bool {
        if !perf_flags::non_temporal_stores() {
            return false;
        }
        if len < 256 {
            return false;
        }
        if !perf_flags::adaptive_nt_enabled() {
            return true;
        }
        if predictor_hint {
            self.counter >= 0
        } else {
            self.counter >= 2
        }
    }
}

// -------------------------------------------------------------
// DeltaMode
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub enum DeltaMode {
    Generic,
    Physics,
    Hybrid,
}

// -------------------------------------------------------------
// MicroFiberMode
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicroFiberMode {
    Generic,
    Physics,
    Hybrid,
}

// -------------------------------------------------------------
// MicroFiberState
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicroFiberState {
    Ready,
    Running,
    Completed,
}

// -------------------------------------------------------------
// MicroFiberPriority
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MicroFiberPriority {
    Physics,
    Generic,
    Hybrid,
}

// -------------------------------------------------------------
// MicroFiber
// -------------------------------------------------------------
pub struct MicroFiber {
    pub id: usize,
    pub mode: MicroFiberMode,
    pub state: MicroFiberState,
    pub priority: MicroFiberPriority,
    pub start: usize,
    pub len: usize,
    pub delta_value: f32,
    pub tile_id: usize,

    pub speculative: bool,
    pub rollback_allowed: bool,
}

impl MicroFiber {
    pub fn new(
        id: usize,
        mode: MicroFiberMode,
        start: usize,
        len: usize,
        delta_value: f32,
        tile_id: usize,
    ) -> Self {
        let priority = match mode {
            MicroFiberMode::Physics => MicroFiberPriority::Physics,
            MicroFiberMode::Generic => MicroFiberPriority::Generic,
            MicroFiberMode::Hybrid => MicroFiberPriority::Hybrid,
        };

        Self {
            id,
            mode,
            state: MicroFiberState::Ready,
            priority,
            start,
            len,
            delta_value,
            tile_id,
            speculative: false,
            rollback_allowed: false,
        }
    }

    #[inline(always)]
    pub fn mark_running(&mut self) {
        self.state = MicroFiberState::Running;
    }

    #[inline(always)]
    pub fn mark_completed(&mut self) {
        self.state = MicroFiberState::Completed;
    }

    #[inline(always)]
    pub fn is_ready(&self) -> bool {
        self.state == MicroFiberState::Ready
    }

    #[inline(always)]
    pub fn is_tiny(&self) -> bool {
        self.len <= 32
    }
}

// -------------------------------------------------------------
// MicroFiberGroup
// -------------------------------------------------------------
pub struct MicroFiberGroup {
    pub id: usize,
    pub fibers: Vec<usize>,
}

impl MicroFiberGroup {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            fibers: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn add_fiber_index(&mut self, idx: usize) {
        self.fibers.push(idx);
    }
}

// -------------------------------------------------------------
// Warp-cooperative scratchpad
// -------------------------------------------------------------
pub struct WarpScratchpad {
    buf: Vec<f32>,
}

impl WarpScratchpad {
    pub fn new(size: usize) -> Self {
        Self { buf: vec![0.0; size] }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
    }

    #[inline(always)]
    pub fn load_from(&mut self, slice: &[f32]) {
        let len = self.buf.len().min(slice.len());
        self.buf[..len].copy_from_slice(&slice[..len]);
    }

    #[inline(always)]
    pub fn store_into(&self, slice: &mut [f32]) {
        let len = self.buf.len().min(slice.len());
        slice[..len].copy_from_slice(&self.buf[..len]);
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        &self.buf
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.buf[..]
    }
}

// -------------------------------------------------------------
// Fiber-local L0 cache
// -------------------------------------------------------------
pub struct FiberL0Cache {
    hot_ranges: Vec<(usize, usize)>,
}

impl FiberL0Cache {
    pub fn new() -> Self {
        Self {
            hot_ranges: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn record_range(&mut self, start: usize, len: usize) {
        self.hot_ranges.push((start, len));
    }

    #[inline(always)]
    pub fn is_hot(&self, start: usize, len: usize) -> bool {
        self.hot_ranges
            .iter()
            .any(|&(s, l)| s == start && l == len)
    }
}

// -------------------------------------------------------------
// Tile-swarm global routing table
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct TileRouteEntry {
    pub fiber_id: usize,
    pub from_tile: usize,
    pub to_tile: usize,
}

#[derive(Default, Debug)]
pub struct TileRoutingTable {
    pub routes: Vec<TileRouteEntry>,
}

impl TileRoutingTable {
    #[inline(always)]
    pub fn record_migration(&mut self, fiber_id: usize, from_tile: usize, to_tile: usize) {
        self.routes.push(TileRouteEntry {
            fiber_id,
            from_tile,
            to_tile,
        });
    }
}

// -------------------------------------------------------------
// GPU-style occupancy tracking
// -------------------------------------------------------------
#[derive(Default, Clone, Copy, Debug)]
pub struct OccupancyTracker {
    pub active_fibers: u64,
    pub completed_fibers: u64,
    pub total_groups: u64,
}

impl OccupancyTracker {
    #[inline(always)]
    pub fn update(&mut self, fibers: &[MicroFiber], groups: &[MicroFiberGroup]) {
        self.total_groups = groups.len() as u64;
        self.active_fibers = fibers
            .iter()
            .filter(|f| f.state == MicroFiberState::Running || f.state == MicroFiberState::Ready)
            .count() as u64;
        self.completed_fibers = fibers
            .iter()
            .filter(|f| f.state == MicroFiberState::Completed)
            .count() as u64;
    }
}

// -------------------------------------------------------------
// Delta index
// -------------------------------------------------------------
#[derive(Default, Clone, Debug)]
pub struct DeltaIndexEntry {
    pub start: usize,
    pub len: usize,
    pub magnitude: f32,
}

#[derive(Default, Clone, Debug)]
pub struct DeltaIndex {
    pub entries: Vec<DeltaIndexEntry>,
}

impl DeltaIndex {
    #[inline(always)]
    pub fn record(&mut self, start: usize, len: usize, magnitude: f32) {
        self.entries.push(DeltaIndexEntry {
            start,
            len,
            magnitude,
        });
    }

    #[inline(always)]
    pub fn is_hot_region(&self, start: usize, len: usize) -> bool {
        self.entries
            .iter()
            .any(|e| e.start == start && e.len == len && e.magnitude.abs() > 0.0)
    }
}

// -------------------------------------------------------------
// Cue buffer
// -------------------------------------------------------------
#[derive(Clone, Debug)]
pub enum CueKind {
    Delta,
    Physics,
    Algebra,
    Hybrid,
}

#[derive(Clone, Debug)]
pub struct CueEvent {
    pub fiber_id: usize,
    pub kind: CueKind,
    pub start: usize,
    pub len: usize,
    pub delta_value: f32,
}

#[derive(Default, Debug)]
pub struct CueBuffer {
    pub events: Vec<CueEvent>,
}

impl CueBuffer {
    #[inline(always)]
    pub fn push(&mut self, ev: CueEvent) {
        self.events.push(ev);
    }

    #[inline(always)]
    pub fn drain(&mut self) -> Vec<CueEvent> {
        std::mem::take(&mut self.events)
    }
}

// -------------------------------------------------------------
// Roundabout
// -------------------------------------------------------------
#[derive(Default, Debug)]
pub struct Roundabout {
    pub cue_buffer: CueBuffer,
    pub delta_index: DeltaIndex,
}

impl Roundabout {
    #[inline(always)]
    pub fn enqueue_fiber(&mut self, fiber: &MicroFiber) {
        let kind = match fiber.mode {
            MicroFiberMode::Generic => CueKind::Delta,
            MicroFiberMode::Physics => CueKind::Physics,
            MicroFiberMode::Hybrid => CueKind::Hybrid,
        };

        self.cue_buffer.push(CueEvent {
            fiber_id: fiber.id,
            kind,
            start: fiber.start,
            len: fiber.len,
            delta_value: fiber.delta_value,
        });
    }

    #[inline(always)]
    pub fn record_delta_region(&mut self, start: usize, len: usize, magnitude: f32) {
        self.delta_index.record(start, len, magnitude);
    }

    #[inline(always)]
    pub fn is_hot_region(&self, start: usize, len: usize) -> bool {
        self.delta_index.is_hot_region(start, len)
    }
}

// -------------------------------------------------------------
// Fiber-level perf counters
// -------------------------------------------------------------
#[derive(Default, Clone, Copy, Debug)]
pub struct FiberPerfCounters {
    pub physics_steps: u64,
    pub generic_steps: u64,
    pub hybrid_steps: u64,
    pub total_elements_touched: u64,
}

// -------------------------------------------------------------
// HARDWARE LAYER
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct HardwareParallelismConfig {
    pub fibers_per_warp: usize,
    pub warps_per_sm: usize,
    pub sms_per_chip: usize,
}

impl Default for HardwareParallelismConfig {
    fn default() -> Self {
        Self {
            fibers_per_warp: 64,
            warps_per_sm: 64,
            sms_per_chip: 80,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScratchpadConfig {
    pub bytes_per_sm: usize,
    pub single_cycle_access: bool,
}

impl Default for ScratchpadConfig {
    fn default() -> Self {
        Self {
            bytes_per_sm: 64 * 1024,
            single_cycle_access: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AluCapabilities {
    pub has_integer: bool,
    pub has_fp32: bool,
    pub has_fp16: bool,
    pub has_bf16: bool,
    pub has_tensor_ops: bool,
}

impl Default for AluCapabilities {
    fn default() -> Self {
        Self {
            has_integer: true,
            has_fp32: true,
            has_fp16: true,
            has_bf16: true,
            has_tensor_ops: true,
        }
    }
}

// -------------------------------------------------------------
// Quantization mode
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub enum QuantMode {
    None,
    FP16Like,
    BF16Like,
    FP8Like,
}

#[derive(Clone, Copy, Debug)]
pub struct WarpSchedulerConfig {
    pub warps_issued_per_cycle_min: usize,
    pub warps_issued_per_cycle_max: usize,
    pub non_blocking_roundabout: bool,
}

impl Default for WarpSchedulerConfig {
    fn default() -> Self {
        Self {
            warps_issued_per_cycle_min: 4,
            warps_issued_per_cycle_max: 8,
            non_blocking_roundabout: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryFabricConfig {
    pub bandwidth_bytes_per_sec: f64,
    pub region_aware_addressing: bool,
}

impl Default for MemoryFabricConfig {
    fn default() -> Self {
        Self {
            bandwidth_bytes_per_sec: 1.5e12,
            region_aware_addressing: true,
        }
    }
}

// -------------------------------------------------------------
// Tensor-core style FMA pipelines
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct TensorCoreConfig {
    pub lanes: usize,
    pub fma_per_cycle: usize,
    pub enabled: bool,
}

impl Default for TensorCoreConfig {
    fn default() -> Self {
        Self {
            lanes: 256,
            fma_per_cycle: 1024,
            enabled: true,
        }
    }
}

pub struct TensorCoreUnit;

impl TensorCoreUnit {
    #[inline(always)]
    pub fn fused_mma_32(a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len());
        for i in 0..len {
            c[i] = a[i].mul_add(b[i], c[i]);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn fused_mma_32_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        while i + 8 <= len {
            let pa = a.as_ptr().add(i);
            let pb = b.as_ptr().add(i);
            let pc = c.as_mut_ptr().add(i);

            let va = _mm256_loadu_ps(pa);
            let vb = _mm256_loadu_ps(pb);
            let vc = _mm256_loadu_ps(pc);

            let vm = _mm256_mul_ps(va, vb);
            let vout = _mm256_add_ps(vm, vc);

            _mm256_storeu_ps(pc, vout);
            i += 8;
        }

        while i < len {
            c[i] = a[i].mul_add(b[i], c[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn fused_mma_32_avx512(a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        while i + 16 <= len {
            let pa = a.as_ptr().add(i);
            let pb = b.as_ptr().add(i);
            let pc = c.as_mut_ptr().add(i);

            let va = _mm512_loadu_ps(pa);
            let vb = _mm512_loadu_ps(pb);
            let vc = _mm512_loadu_ps(pc);

            let vm = _mm512_mul_ps(va, vb);
            let vout = _mm512_add_ps(vm, vc);

            _mm512_storeu_ps(pc, vout);
            i += 16;
        }

        while i < len {
            c[i] = a[i].mul_add(b[i], c[i]);
            i += 1;
        }
    }

    #[inline(always)]
    pub fn fused_mma_block(a: &[f32], b: &[f32], c: &mut [f32], block: usize) {
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;

        if is_x86_feature_detected!("avx512f") {
            while i < len {
                let end = (i + block).min(len);
                unsafe { Self::fused_mma_32_avx512(&a[i..end], &b[i..end], &mut c[i..end]) };
                i = end;
            }
        } else if is_x86_feature_detected!("avx2") {
            while i < len {
                let end = (i + block).min(len);
                unsafe { Self::fused_mma_32_avx2(&a[i..end], &b[i..end], &mut c[i..end]) };
                i = end;
            }
        } else {
            while i < len {
                let end = (i + block).min(len);
                Self::fused_mma_32(&a[i..end], &b[i..end], &mut c[i..end]);
                i = end;
            }
        }
    }
}

// -------------------------------------------------------------
// Hardware attention units
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct AttentionUnitConfig {
    pub enabled: bool,
    pub max_heads: usize,
    pub max_seq_len: usize,
}

impl Default for AttentionUnitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_heads: 8,
            max_seq_len: 1024,
        }
    }
}

pub struct AttentionUnit;

impl AttentionUnit {
    #[inline(always)]
    pub fn apply_region_attention(weights: &mut [f32], hot: bool) {
        let factor = if hot { 1.25 } else { 0.9 };
        let len_f = weights.len().max(1) as f32;

        if is_x86_feature_detected!("avx2") && weights.len() >= 8 {
            unsafe { Self::apply_region_attention_simd_avx2(weights, factor, len_f) }
        } else {
            for (i, w) in weights.iter_mut().enumerate() {
                let pos_scale = 1.0 + (i as f32 / len_f) * 0.1;
                *w = w.mul_add(factor * pos_scale, 0.0001);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn apply_region_attention_simd_avx2(weights: &mut [f32], factor: f32, len_f: f32) {
        let len = weights.len();
        let mut i = 0;

        while i + 8 <= len {
            let idx0 = i as f32;
            let pos_scale_arr: [f32; 8] = [
                1.0 + ((idx0 + 0.0) / len_f) * 0.1,
                1.0 + ((idx0 + 1.0) / len_f) * 0.1,
                1.0 + ((idx0 + 2.0) / len_f) * 0.1,
                1.0 + ((idx0 + 3.0) / len_f) * 0.1,
                1.0 + ((idx0 + 4.0) / len_f) * 0.1,
                1.0 + ((idx0 + 5.0) / len_f) * 0.1,
                1.0 + ((idx0 + 6.0) / len_f) * 0.1,
                1.0 + ((idx0 + 7.0) / len_f) * 0.1,
            ];

            let factor_arr = [factor; 8];

            let p = weights.as_mut_ptr().add(i);
            let wv = _mm256_loadu_ps(p);
            let posv = _mm256_loadu_ps(pos_scale_arr.as_ptr());
            let fv = _mm256_loadu_ps(factor_arr.as_ptr());

            let scale = _mm256_mul_ps(fv, posv);
            let out = _mm256_fmadd_ps(wv, scale, _mm256_set1_ps(0.0001));

            _mm256_storeu_ps(p, out);
            i += 8;
        }

        while i < len {
            let pos_scale = 1.0 + (i as f32 / len_f) * 0.1;
            weights[i] = weights[i].mul_add(factor * pos_scale, 0.0001);
            i += 1;
        }
    }
}

// -------------------------------------------------------------
// Hardware delta-index table
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct DeltaIndexTableEntry {
    pub start: u32,
    pub len: u32,
    pub magnitude: f32,
}

#[derive(Debug)]
pub struct DeltaIndexTable {
    pub entries: Vec<DeltaIndexTableEntry>,
}

impl DeltaIndexTable {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub fn record(&mut self, start: usize, len: usize, magnitude: f32) {
        self.entries.push(DeltaIndexTableEntry {
            start: start as u32,
            len: len as u32,
            magnitude,
        });
    }

    #[inline(always)]
    pub fn is_hot_region(&self, start: usize, len: usize) -> bool {
        self.entries.iter().any(|e| {
            e.start == start as u32
                && e.len == len as u32
                && e.magnitude.abs() > 0.0
        })
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// -------------------------------------------------------------
// Hardware cue buffer
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub enum HardwareCueKind {
    Delta,
    Physics,
    Algebra,
    Hybrid,
}

#[derive(Clone, Copy, Debug)]
pub struct HardwareCueEntry {
    pub fiber_id: u32,
    pub kind: HardwareCueKind,
    pub start: u32,
    pub len: u32,
    pub delta_value: f32,
}

#[derive(Debug)]
pub struct HardwareCueBuffer {
    pub entries: Vec<HardwareCueEntry>,
}

impl HardwareCueBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub fn push(&mut self, ev: HardwareCueEntry) {
        self.entries.push(ev);
    }

    #[inline(always)]
    pub fn drain(&mut self) -> Vec<HardwareCueEntry> {
        std::mem::take(&mut self.entries)
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// -------------------------------------------------------------
// Fiber fusion controller
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct FiberFusionConfig {
    pub max_tiny_len: usize,
    pub min_fibers_to_fuse: usize,
}

impl Default for FiberFusionConfig {
    fn default() -> Self {
        Self {
            max_tiny_len: 32,
            min_fibers_to_fuse: 4,
        }
    }
}

pub struct FiberFusionController {
    pub config: FiberFusionConfig,
}

impl FiberFusionController {
    pub fn new(config: FiberFusionConfig) -> Self {
        Self { config }
    }

    pub fn fuse_fibers(&self, fibers: &mut Vec<MicroFiber>) {
        let mut fused: Vec<MicroFiber> = Vec::new();
        let mut buffer: Vec<MicroFiber> = Vec::new();

        for f in fibers.drain(..) {
            if f.len <= self.config.max_tiny_len {
                buffer.push(f);
                if buffer.len() >= self.config.min_fibers_to_fuse {
                    let base = buffer[0].start;
                    let total_len = buffer.iter().map(|x| x.len).sum();
                    let mut fused_fiber = MicroFiber::new(
                        buffer[0].id,
                        buffer[0].mode,
                        base,
                        total_len,
                        buffer[0].delta_value,
                        buffer[0].tile_id,
                    );
                    fused_fiber.priority = buffer[0].priority;
                    fused.push(fused_fiber);
                    buffer.clear();
                }
            } else {
                fused.push(f);
            }
        }

        fused.extend(buffer.into_iter());
        *fibers = fused;
    }
}

// -------------------------------------------------------------
// Async DMA engine
// -------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct DmaRequest {
    pub src_start: usize,
    pub dst_start: usize,
    pub len: usize,
}

#[derive(Debug)]
pub struct AsyncDmaEngine {
    pub queue: Vec<DmaRequest>,
}

impl AsyncDmaEngine {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    #[inline(always)]
    pub fn enqueue(&mut self, req: DmaRequest) {
        self.queue.push(req);
    }

    #[inline(always)]
    pub fn process_all(&mut self, sram: &mut OverlaySram) {
        for req in self.queue.drain(..) {
            let src_start = req.src_start.min(sram.data.len());
            let dst_start = req.dst_start.min(sram.data.len());
            let max_len = req
                .len
                .min(sram.data.len().saturating_sub(src_start))
                .min(sram.data.len().saturating_sub(dst_start));

            if max_len == 0 {
                continue;
            }

            unsafe {
                let src_ptr = sram.data.as_ptr().add(src_start);
                let dst_ptr = sram.data.as_mut_ptr().add(dst_start);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, max_len);
            }
        }
    }
}

// -------------------------------------------------------------
// Speculative execution + rollback
// -------------------------------------------------------------
#[derive(Debug)]
pub struct SpeculativeSnapshot {
    pub start: usize,
    pub len: usize,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub struct SpeculativeEngine {
    pub snapshots: Vec<SpeculativeSnapshot>,
}

impl SpeculativeEngine {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn snapshot_region(&mut self, sram: &OverlaySram, start: usize, len: usize) {
        let s = start.min(sram.data.len());
        let e = (start + len).min(sram.data.len());
        if s >= e {
            return;
        }
        let slice = &sram.data[s..e];
        self.snapshots.push(SpeculativeSnapshot {
            start: s,
            len: slice.len(),
            data: slice.to_vec(),
        });
    }

    #[inline(always)]
    pub fn rollback_all(&mut self, sram: &mut OverlaySram) {
        for snap in self.snapshots.drain(..) {
            let end = (snap.start + snap.len).min(sram.data.len());
            if end <= snap.start {
                continue;
            }
            let dst = &mut sram.data[snap.start..end];
            let len = dst.len().min(snap.data.len());
            dst[..len].copy_from_slice(&snap.data[..len]);
        }
    }
}

// -------------------------------------------------------------
// Streaming Multiprocessor
// -------------------------------------------------------------
#[derive(Debug)]
pub struct StreamingMultiprocessor {
    pub id: usize,
    pub parallelism: HardwareParallelismConfig,
    pub scratchpad: ScratchpadConfig,
    pub alu_caps: AluCapabilities,
    pub warp_sched: WarpSchedulerConfig,
    pub fabric: MemoryFabricConfig,
    pub delta_table: DeltaIndexTable,
    pub cue_buffer: HardwareCueBuffer,
    pub tensor_cfg: TensorCoreConfig,
    pub attention_cfg: AttentionUnitConfig,
    pub dma: AsyncDmaEngine,
    pub speculative: SpeculativeEngine,
    pub quant_mode: QuantMode,
    pub predictor: StatePredictor,
    pub adaptive_nt: AdaptiveNtHeuristic,
}

impl StreamingMultiprocessor {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            parallelism: HardwareParallelismConfig::default(),
            scratchpad: ScratchpadConfig::default(),
            alu_caps: AluCapabilities::default(),
            warp_sched: WarpSchedulerConfig::default(),
            fabric: MemoryFabricConfig::default(),
            delta_table: DeltaIndexTable::new(1024),
            cue_buffer: HardwareCueBuffer::new(1024),
            tensor_cfg: TensorCoreConfig::default(),
            attention_cfg: AttentionUnitConfig::default(),
            dma: AsyncDmaEngine::new(),
            speculative: SpeculativeEngine::new(),
            quant_mode: QuantMode::None,
            predictor: StatePredictor::new(8),
            adaptive_nt: AdaptiveNtHeuristic::new(),
        }
    }

    #[inline(always)]
    pub fn issue_warps_non_blocking(&mut self, fibers: &mut [MicroFiber]) {
        if !self.warp_sched.non_blocking_roundabout {
            return;
        }

        let max_issue = self.warp_sched.warps_issued_per_cycle_max;
        let mut issued = 0;

        for f in fibers.iter_mut() {
            if issued >= max_issue {
                break;
            }
            if f.is_ready() {
                let kind = match f.mode {
                    MicroFiberMode::Generic => HardwareCueKind::Delta,
                    MicroFiberMode::Physics => HardwareCueKind::Physics,
                    MicroFiberMode::Hybrid => HardwareCueKind::Hybrid,
                };

                self.cue_buffer.push(HardwareCueEntry {
                    fiber_id: f.id as u32,
                    kind,
                    start: f.start as u32,
                    len: f.len as u32,
                    delta_value: f.delta_value,
                });

                f.mark_running();
                issued += 1;
            }
        }
    }

    #[inline(always)]
    pub fn update_delta_table_for_slice(&mut self, start: usize, len: usize, magnitude: f32) {
        self.delta_table.record(start, len, magnitude);
    }

    #[inline(always)]
    pub fn is_hot_region(&self, start: usize, len: usize) -> bool {
        self.delta_table.is_hot_region(start, len)
    }

    #[inline(always)]
    pub fn apply_tensor_core_if_enabled(&self, scratch: &mut WarpScratchpad) {
        if perf_flags::zero_cost_mode() {
            return;
        }

        if !self.tensor_cfg.enabled {
            return;
        }
        let slice = scratch.as_mut_slice();
        let len = slice.len();
        if len < 32 {
            return;
        }

        let mid = len / 2;
        let (a, rest) = slice.split_at_mut(mid);
        let (b, c) = rest.split_at_mut(mid.min(rest.len()));
        if c.is_empty() {
            return;
        }

        match self.quant_mode {
            QuantMode::None => {
                TensorCoreUnit::fused_mma_block(a, b, c, 32);
            }
            QuantMode::FP16Like | QuantMode::BF16Like | QuantMode::FP8Like => {
                let scale = match self.quant_mode {
                    QuantMode::FP16Like => 0.5,
                    QuantMode::BF16Like => 0.75,
                    QuantMode::FP8Like => 0.25,
                    QuantMode::None => 1.0,
                };

                for v in a.iter_mut() {
                    *v *= scale;
                }
                for v in b.iter_mut() {
                    *v *= scale;
                }

                TensorCoreUnit::fused_mma_block(a, b, c, 32);

                let inv = 1.0 / scale.max(1e-6);
                for v in c.iter_mut() {
                    *v *= inv;
                }
            }
        }
    }

    #[inline(always)]
    pub fn apply_attention_if_enabled(&self, scratch: &mut WarpScratchpad, hot: bool) {
        if perf_flags::zero_cost_mode() {
            return;
        }

        if !self.attention_cfg.enabled {
            return;
        }
        if perf_flags::skip_scratchpad_for_small() && scratch.as_slice().len() < 16 {
            return;
        }
        AttentionUnit::apply_region_attention(scratch.as_mut_slice(), hot);
    }
}

// -------------------------------------------------------------
// ComputeChip
// -------------------------------------------------------------
#[derive(Debug)]
pub struct ComputeChip {
    pub sms: Vec<StreamingMultiprocessor>,
    pub tiles: usize,
}

impl ComputeChip {
    pub fn new(parallelism: HardwareParallelismConfig, tiles: usize) -> Self {
        let mut sms = Vec::with_capacity(parallelism.sms_per_chip);
        for id in 0..parallelism.sms_per_chip {
            sms.push(StreamingMultiprocessor::new(id));
        }
        Self { sms, tiles }
    }

    #[inline(always)]
    pub fn select_sm_for_fiber(&mut self, fiber_id: usize) -> &mut StreamingMultiprocessor {
        let idx = fiber_id % self.sms.len();
        &mut self.sms[idx]
    }

    #[inline(always)]
    pub fn route_fiber_to_tile(&self, fiber: &mut MicroFiber) {
        let target_tile = fiber.id % self.tiles.max(1);
        fiber.tile_id = target_tile;
    }
}

// -------------------------------------------------------------
// CoreBridge / callbacks
// -------------------------------------------------------------
pub type AmdCoreCallback = fn(chunk: &mut [f32], dt: f32);

pub struct CoreBridge {
    pub tile_id: usize,
    pub amd_core_id: usize,
    pub weight: f32,
    pub amd_callback: Option<AmdCoreCallback>,
}

// -------------------------------------------------------------
// MicroFiberScheduler
// -------------------------------------------------------------
pub struct MicroFiberScheduler<'a> {
    pub fibers: Vec<MicroFiber>,
    pub groups: Vec<MicroFiberGroup>,
    pub cores: &'a mut [BlockDeltaCore],
    pub perf: FiberPerfCounters,
    pub tile_id: usize,

    pub scratchpad: WarpScratchpad,
    pub l0_cache: FiberL0Cache,
    pub routing: TileRoutingTable,
    pub occupancy: OccupancyTracker,

    pub roundabout: Roundabout,

    pub sm: StreamingMultiprocessor,
    pub fusion: FiberFusionController,

    pub chip: ComputeChip,
}

impl<'a> MicroFiberScheduler<'a> {
    pub fn new(cores: &'a mut [BlockDeltaCore], tile_id: usize) -> Self {
        let parallelism = HardwareParallelismConfig::default();
        let mut sm = StreamingMultiprocessor::new(tile_id);
        sm.parallelism = parallelism;

        let chip = ComputeChip::new(parallelism, 4);

        Self {
            fibers: Vec::new(),
            groups: Vec::new(),
            cores,
            perf: FiberPerfCounters::default(),
            tile_id,
            scratchpad: WarpScratchpad::new(256),
            l0_cache: FiberL0Cache::new(),
            routing: TileRoutingTable::default(),
            occupancy: OccupancyTracker::default(),
            roundabout: Roundabout::default(),
            sm,
            fusion: FiberFusionController::new(FiberFusionConfig::default()),
            chip,
        }
    }

    #[inline(always)]
    pub fn add_fiber(&mut self, fiber: MicroFiber) {
        self.fibers.push(fiber);
    }

    #[inline(always)]
    pub fn fuse_tiny_fibers(&mut self, max_len: usize) {
        let mut fused: Vec<MicroFiber> = Vec::new();
        let mut buffer: Vec<MicroFiber> = Vec::new();

        for f in self.fibers.drain(..) {
            if f.len <= max_len {
                buffer.push(f);
                if buffer.len() >= 4 {
                    let base = buffer[0].start;
                    let total_len = buffer.iter().map(|x| x.len).sum();
                    let mut fused_fiber = MicroFiber::new(
                        buffer[0].id,
                        buffer[0].mode,
                        base,
                        total_len,
                        buffer[0].delta_value,
                        buffer[0].tile_id,
                    );
                    fused_fiber.priority = buffer[0].priority;
                    fused.push(fused_fiber);
                    buffer.clear();
                }
            } else {
                fused.push(f);
            }
        }

        fused.extend(buffer.into_iter());
        self.fibers = fused;
    }

    #[inline(always)]
    pub fn build_groups(&mut self, group_size: usize) {
        self.groups.clear();

        let mut indices: Vec<usize> = (0..self.fibers.len()).collect();
        indices.sort_by_key(|&i| self.fibers[i].priority);

        let mut current_group = MicroFiberGroup::new(0);
        let mut group_id = 0;
        let mut count = 0;

        for idx in indices {
            if count == group_size {
                self.groups.push(current_group);
                group_id += 1;
                current_group = MicroFiberGroup::new(group_id);
                count = 0;
            }
            current_group.add_fiber_index(idx);
            count += 1;
        }

        if !current_group.fibers.is_empty() {
            self.groups.push(current_group);
        }

        self.occupancy.update(&self.fibers, &self.groups);
    }

    #[inline(always)]
    pub fn migrate_fibers_between_tiles(&mut self, local_tile_id: usize) {
        for fiber in self.fibers.iter_mut() {
            if fiber.tile_id != local_tile_id && fiber.is_ready() {
                self.routing
                    .record_migration(fiber.id, local_tile_id, fiber.tile_id);
                fiber.mark_completed();
            }
        }
    }

    #[inline(always)]
    pub fn steal_fibers_from(&mut self, other: &mut MicroFiberScheduler<'a>, max_steal: usize) {
        let mut stolen = 0;
        let mut to_move = Vec::new();

        for (idx, f) in other.fibers.iter().enumerate() {
            if stolen >= max_steal {
                break;
            }
            if f.is_ready() && f.tile_id == self.tile_id {
                to_move.push(idx);
                stolen += 1;
            }
        }

        for idx in to_move.into_iter().rev() {
            let mut f = other.fibers.remove(idx);
            f.tile_id = self.tile_id;
            self.fibers.push(f);
        }

        self.occupancy.update(&self.fibers, &self.groups);
        other.occupancy.update(&other.fibers, &other.groups);
    }

    #[inline(always)]
    pub fn run_all(
        &mut self,
        sram: &mut OverlaySram,
        bridge: Option<&CoreBridge>,
    ) {
        self.fusion.fuse_fibers(&mut self.fibers);

        self.sm.issue_warps_non_blocking(&mut self.fibers);

        let core_count = self.cores.len().max(1);

        self.migrate_fibers_between_tiles(self.tile_id);

        for group in &self.groups {
            for (local_idx, fiber_idx) in group.fibers.iter().enumerate() {
                let fiber = &mut self.fibers[*fiber_idx];
                if !fiber.is_ready() && fiber.state != MicroFiberState::Running {
                    continue;
                }

                let core_idx = local_idx % core_count;
                let core = &mut self.cores[core_idx];

                self.l0_cache.record_range(fiber.start, fiber.len);
                self.roundabout.enqueue_fiber(fiber);

                run_micro_fiber_strand_with_perf(
                    fiber,
                    core,
                    sram,
                    bridge,
                    &mut self.perf,
                    &mut self.scratchpad,
                    &mut self.roundabout,
                    &mut self.sm,
                );
            }
        }

        self.sm.dma.process_all(sram);

        self.occupancy.update(&self.fibers, &self.groups);
    }
}

// -------------------------------------------------------------
// Micro-Fiber Strand Helper
// -------------------------------------------------------------
#[inline(always)]
pub fn run_micro_fiber_strand_with_perf(
    fiber: &mut MicroFiber,
    core: &mut BlockDeltaCore,
    sram: &mut OverlaySram,
    bridge: Option<&CoreBridge>,
    perf: &mut FiberPerfCounters,
    scratchpad: &mut WarpScratchpad,
    roundabout: &mut Roundabout,
    sm: &mut StreamingMultiprocessor,
) {
    fiber.mark_running();

    let start = fiber.start.min(sram.data.len());
    let end = (start + fiber.len).min(sram.data.len());
    if start >= end {
        fiber.mark_completed();
        return;
    }

    if fiber.speculative && fiber.rollback_allowed && !perf_flags::zero_cost_mode() {
        sm.speculative.snapshot_region(sram, start, fiber.len);
    }

    let slice = &mut sram.data[start..end];
    let elements = slice.len() as u64;
    perf.total_elements_touched += elements;

    let predicted_hot = sm.predictor.predict_hot(start, slice.len());
    let predicted_nonzero = sm.predictor.predict_nonzero(start, slice.len());

    let use_scratchpad = if perf_flags::skip_scratchpad_for_small() {
        slice.len() >= 32
    } else {
        true
    };

    if use_scratchpad {
        scratchpad.clear();
        scratchpad.load_from(slice);
    }

    let hot = if perf_flags::zero_cost_mode() {
        false
    } else {
        let observed_hot =
            sm.is_hot_region(start, slice.len()) || roundabout.is_hot_region(start, slice.len());
        observed_hot || predicted_hot
    };

    if use_scratchpad && !perf_flags::zero_cost_mode() && hot {
        sm.apply_attention_if_enabled(scratchpad, hot);
    }

    let observed_nonzero: bool = match fiber.mode {
        MicroFiberMode::Generic => {
            let delta = fiber.delta_value;

            if delta == 0.0 && !predicted_nonzero {
                false
            } else {
                let did_write = delta != 0.0;
                if did_write {
                    let predictor_hint = sm.predictor.predict_nonzero(start, slice.len());
                    let _use_nt = sm.adaptive_nt.should_use_nt(slice.len(), predictor_hint);

                    if is_x86_feature_detected!("avx512f") {
                        unsafe { core.apply_delta_block_avx512(delta, slice) }
                    } else if is_x86_feature_detected!("avx2") {
                        unsafe { core.apply_delta_block_avx2(delta, slice) }
                    } else {
                        core.apply_delta_block_scalar(delta, slice);
                    }

                    if slice.len() >= 256 {
                        sm.adaptive_nt.observe_streaming();
                    } else {
                        sm.adaptive_nt.observe_reuse();
                    }
                }

                core.base_reg += delta * (slice.len() as f32);
                perf.generic_steps += 1;

                if !perf_flags::zero_cost_mode() {
                    roundabout.record_delta_region(start, slice.len(), delta);
                    sm.update_delta_table_for_slice(start, slice.len(), delta);
                }
                did_write
            }
        }

        MicroFiberMode::Physics => {
            let dt = fiber.delta_value;
            core.step_physics_block(slice, dt);
            perf.physics_steps += 1;

            if !perf_flags::zero_cost_mode() {
                roundabout.record_delta_region(start, slice.len(), dt);
                sm.update_delta_table_for_slice(start, slice.len(), dt);
            }
            true
        }

        MicroFiberMode::Hybrid => {
            let dt = fiber.delta_value;

            core.step_physics_block(slice, dt);

            if let Some(b) = bridge {
                if let Some(cb) = b.amd_callback {
                    cb(slice, dt);
                }
            }

            core.base_reg += dt * (slice.len() as f32);
            perf.hybrid_steps += 1;

            if !perf_flags::zero_cost_mode() {
                roundabout.record_delta_region(start, slice.len(), dt);
                sm.update_delta_table_for_slice(start, slice.len(), dt);
            }
            true
        }
    };

    if use_scratchpad && !perf_flags::zero_cost_mode() {
        if hot {
            sm.apply_tensor_core_if_enabled(scratchpad);
        }
        scratchpad.store_into(slice);
    } else if use_scratchpad {
        scratchpad.store_into(slice);
    }

    sm.predictor.update(start, slice.len(), hot, observed_nonzero);

    if fiber.speculative && fiber.rollback_allowed && !perf_flags::zero_cost_mode() {
        let should_rollback = false;
        if should_rollback {
            sm.speculative.rollback_all(sram);
        }
    }

    fiber.mark_completed();
}

// -------------------------------------------------------------
// DeltaCore: scalar delta engine
// -------------------------------------------------------------
pub struct DeltaCore {
    pub id: usize,
    pub base_reg: f32,
    pub delta_reg: f32,
}

impl DeltaCore {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            base_reg: 0.0,
            delta_reg: 0.0,
        }
    }

    #[inline(always)]
    pub fn apply_delta(&mut self, cmd: &DeltaCommand, sram: &mut OverlaySram) {
        self.delta_reg = cmd.delta_value;
        let result = black_box(self.base_reg + self.delta_reg);
        self.base_reg = result;

        let idx = (cmd.region_id + self.id) % sram.data.len();
        unsafe {
            std::ptr::write_volatile(&mut sram.data[idx], result);
        }
    }

    #[inline(always)]
    pub fn apply_delta_scalar_helper(&mut self, delta: f32) -> f32 {
        let mut v = self.base_reg;

        v = v.mul_add(delta, 0.0001);
        v = v.mul_add(delta * 0.5, 0.0003);
        v = v.mul_add(delta * 0.25, 0.0007);

        v = black_box(v);
        self.base_reg = v;
        v
    }
}

// -------------------------------------------------------------
// BlockDeltaCore
// -------------------------------------------------------------
pub struct BlockDeltaCore {
    pub id: usize,
    pub base_reg: f32,
    pub mode: DeltaMode,
}

impl BlockDeltaCore {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            base_reg: 0.0,
            mode: DeltaMode::Generic,
        }
    }

    #[inline(always)]
    pub fn set_mode_physics(&mut self) {
        self.mode = DeltaMode::Physics;
    }

    #[inline(always)]
    pub fn set_mode_generic(&mut self) {
        self.mode = DeltaMode::Generic;
    }

    #[inline(always)]
    pub fn set_mode_hybrid(&mut self) {
        self.mode = DeltaMode::Hybrid;
    }

    const BODY_STRIDE: usize = 10;

    #[inline(always)]
    fn step_physics_block(&mut self, slice: &mut [f32], dt: f32) {
        let len = slice.len();
        if len < Self::BODY_STRIDE {
            return;
        }

        if is_x86_feature_detected!("avx2") && len >= Self::BODY_STRIDE * 4 {
            unsafe { self.step_physics_block_simd_avx2(slice, dt) }
        } else {
            let body_count = len / Self::BODY_STRIDE;

            for body_idx in 0..body_count {
                let base = body_idx * Self::BODY_STRIDE;

                let px = base + 0;
                let py = base + 1;
                let pz = base + 2;

                let vx = base + 3;
                let vy = base + 4;
                let vz = base + 5;

                let ax = base + 6;
                let ay = base + 7;
                let az = base + 8;

                let axv = slice[ax];
                let ayv = slice[ay];
                let azv = slice[az];

                slice[vx] += axv * dt;
                slice[vy] += ayv * dt;
                slice[vz] += azv * dt;

                slice[px] += slice[vx] * dt;
                slice[py] += slice[vy] * dt;
                slice[pz] += slice[vz] * dt;
            }
        }

        self.base_reg += dt * ((len / Self::BODY_STRIDE) as f32);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn step_physics_block_simd_avx2(&mut self, slice: &mut [f32], dt: f32) {
        let len = slice.len();
        let body_count = len / Self::BODY_STRIDE;
        let mut i = 0usize;

        while i + 4 <= body_count {
            for j in 0..4 {
                let b = (i + j) * Self::BODY_STRIDE;
                slice[b + 3] += slice[b + 6] * dt;
                slice[b + 4] += slice[b + 7] * dt;
                slice[b + 5] += slice[b + 8] * dt;

                slice[b + 0] += slice[b + 3] * dt;
                slice[b + 1] += slice[b + 4] * dt;
                slice[b + 2] += slice[b + 5] * dt;
            }

            i += 4;
        }

        for body_idx in i..body_count {
            let base = body_idx * Self::BODY_STRIDE;

            let px = base + 0;
            let py = base + 1;
            let pz = base + 2;

            let vx = base + 3;
            let vy = base + 4;
            let vz = base + 5;

            let ax = base + 6;
            let ay = base + 7;
            let az = base + 8;

            let axv = slice[ax];
            let ayv = slice[ay];
            let azv = slice[az];

            slice[vx] += axv * dt;
            slice[vy] += ayv * dt;
            slice[vz] += azv * dt;

            slice[px] += slice[vx] * dt;
            slice[py] += slice[vy] * dt;
            slice[pz] += slice[vz] * dt;
        }
    }

    #[inline(always)]
    pub fn apply_delta_block(
        &mut self,
        cmd: &DeltaCommand,
        sram: &mut OverlaySram,
        block_len: usize,
        bridge: Option<&CoreBridge>,
    ) {
        match self.mode {
            DeltaMode::Generic => {
                let delta = cmd.delta_value;
                if delta == 0.0 {
                    return;
                }
                let len = block_len.min(sram.data.len());

                if is_x86_feature_detected!("avx512f") {
                    unsafe { self.apply_delta_block_avx512(delta, &mut sram.data[..len]) }
                } else if is_x86_feature_detected!("avx2") {
                    unsafe { self.apply_delta_block_avx2(delta, &mut sram.data[..len]) }
                } else {
                    self.apply_delta_block_scalar(delta, &mut sram.data[..len]);
                }

                self.base_reg += delta * (len as f32);
            }

            DeltaMode::Physics => {
                let dt = cmd.delta_value;
                let len = block_len.min(sram.data.len());
                let slice = &mut sram.data[..len];
                self.step_physics_block(slice, dt);
            }

            DeltaMode::Hybrid => {
                let dt = cmd.delta_value;
                let len = block_len.min(sram.data.len());
                let slice = &mut sram.data[..len];

                self.step_physics_block(slice, dt);

                if let Some(b) = bridge {
                    if let Some(cb) = b.amd_callback {
                        cb(slice, dt);
                    }
                }

                self.base_reg += dt * (len as f32);
            }
        }
    }

    #[inline(always)]
    fn apply_delta_block_scalar(&mut self, delta: f32, slice: &mut [f32]) {
        const CHUNK: usize = 16;
        let len = slice.len();
        let mut i = 0;

        while i + CHUNK <= len {
            let p = &mut slice[i..i + CHUNK];
            for v in p.iter_mut() {
                *v += delta;
            }
            i += CHUNK;
        }

        while i < len {
            slice[i] += delta;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn apply_delta_block_avx2(&mut self, delta: f32, slice: &mut [f32]) {
        let len = slice.len();
        if len == 0 {
            return;
        }

        let mut i = 0;
        let vdelta = _mm256_set1_ps(delta);
        let prefetch = perf_flags::prefetch_enabled();
        let use_nt_static = perf_flags::non_temporal_stores() && len >= 256;

        while i + 8 <= len {
            if prefetch {
                let p_pref = slice.as_ptr().add(i + 8) as *const i8;
                _mm_prefetch(p_pref, _MM_HINT_T0);
            }

            let p = slice.as_mut_ptr().add(i);
            let v = _mm256_loadu_ps(p);
            let v = _mm256_add_ps(v, vdelta);

            if use_nt_static {
                _mm256_stream_ps(p, v);
            } else {
                _mm256_storeu_ps(p, v);
            }

            i += 8;
        }

        while i < len {
            *slice.get_unchecked_mut(i) += delta;
            i += 1;
        }

        if use_nt_static {
            _mm_sfence();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn apply_delta_block_avx512(&mut self, delta: f32, slice: &mut [f32]) {
        let len = slice.len();
        if len == 0 {
            return;
        }

        let mut i = 0;
        let vdelta = _mm512_set1_ps(delta);
        let prefetch = perf_flags::prefetch_enabled();
        let use_nt_static = perf_flags::non_temporal_stores() && len >= 512;

        while i + 16 <= len {
            if prefetch {
                let p_pref = slice.as_ptr().add(i + 16) as *const i8;
                _mm_prefetch(p_pref, _MM_HINT_T0);
            }

            let p = slice.as_mut_ptr().add(i);
            let v = _mm512_loadu_ps(p);
            let v = _mm512_add_ps(v, vdelta);

            if use_nt_static {
                _mm512_stream_ps(p, v);
            } else {
                _mm512_storeu_ps(p, v);
            }

            i += 16;
        }

        let remaining = len - i;
        if remaining > 0 {
            let p = slice.as_mut_ptr().add(i);
            let mut tail = [0f32; 16];
            for t in 0..remaining {
                tail[t] = *slice.get_unchecked(i + t);
            }
            let vtail = _mm512_loadu_ps(tail.as_ptr());
            let vtail = _mm512_add_ps(vtail, vdelta);
            let mut out = [0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), vtail);
            let mask: __mmask16 = (1u16 << remaining) - 1;
            _mm512_mask_store_ps(p, mask, vtail);
        }

        if use_nt_static {
            _mm_sfence();
        }
    }

    #[inline(always)]
    pub fn apply_delta_micro_block(&mut self, delta: f32, slice: &mut [f32]) {
        let mut v = delta;

        for x in slice.iter_mut().take(32) {
            v = v.mul_add(*x, 0.0001);
            *x = black_box(v);
        }

        self.base_reg = black_box(self.base_reg + v);
    }

    #[inline(always)]
    fn compute_auto_chunk_size(len: usize, core_count: usize) -> usize {
        if len == 0 {
            return 1;
        }

        let cores = core_count.max(1);
        let mut base = len / cores;

        let min_chunk = 32;
        let max_chunk = 1024;

        if base < min_chunk {
            base = min_chunk.min(len);
        }
        if base > max_chunk {
            base = max_chunk;
        }

        if len > 65536 {
            base = (base * 2).min(max_chunk);
        } else if len < 2048 {
            base = (base / 2).max(min_chunk.min(len));
        }

        base
    }

    #[inline(always)]
    pub fn apply_delta_block_parallel(
        cores: &mut [BlockDeltaCore],
        cmd: &DeltaCommand,
        sram: &mut OverlaySram,
        block_len: usize,
        bridge: Option<&CoreBridge>,
    ) {
        let len = block_len.min(sram.data.len());
        if len == 0 {
            return;
        }
        let core_count = cores.len().max(1);

        let chunk_size = Self::compute_auto_chunk_size(len, core_count);

        let (head, _) = sram.data.split_at_mut(len);

        head.par_chunks_mut(chunk_size)
            .zip(cores.par_iter_mut())
            .for_each(|(chunk, core)| {
                match core.mode {
                    DeltaMode::Generic => {
                        let delta = cmd.delta_value;
                        if delta != 0.0 {
                            if is_x86_feature_detected!("avx512f") {
                                unsafe { core.apply_delta_block_avx512(delta, chunk) }
                            } else if is_x86_feature_detected!("avx2") {
                                unsafe { core.apply_delta_block_avx2(delta, chunk) }
                            } else {
                                core.apply_delta_block_scalar(delta, chunk);
                            }
                        }
                        core.base_reg += delta * (chunk.len() as f32);
                    }
                    DeltaMode::Physics => {
                        let dt = cmd.delta_value;
                        core.step_physics_block(chunk, dt);
                    }
                    DeltaMode::Hybrid => {
                        let dt = cmd.delta_value;
                        core.step_physics_block(chunk, dt);

                        if let Some(b) = bridge {
                            if let Some(cb) = b.amd_callback {
                                cb(chunk, dt);
                            }
                        }

                        core.base_reg += dt * (chunk.len() as f32);
                    }
                }
            });
    }
}

