// overlay_chain.rs
// High-performance Overlay / OverlayChain tuned for max speed.
// - Minimal metadata
// - SIMD-accelerated blend and delta application
// - Tiny overlay fusion to reduce overhead on many small overlays

use std::arch::x86_64::*;

// -------------------------------------------------------------
// Overlay
// -------------------------------------------------------------
#[derive(Clone)]
pub struct Overlay {
    pub values: Vec<f32>,
}

impl Overlay {
    #[inline(always)]
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// -------------------------------------------------------------
// OverlayChain
// -------------------------------------------------------------
pub struct OverlayChain {
    chain: Vec<Overlay>,
}

impl OverlayChain {
    #[inline(always)]
    pub fn new() -> Self {
        Self { chain: vec![] }
    }

    // Original semantics
    #[inline(always)]
    pub fn push_overlay(&mut self, values: Vec<f32>) {
        self.chain.push(Overlay::new(values));
    }

    #[inline(always)]
    pub fn fetch_overlays(&self) -> Vec<Overlay> {
        self.chain.clone()
    }

    // ---------------------------------------------------------
    // Performance-oriented helpers
    // ---------------------------------------------------------

    /// Fuse tiny overlays into larger ones to reduce per-overlay overhead.
    #[inline(always)]
    pub fn fuse_tiny_overlays(&mut self, max_tiny_len: usize, min_to_fuse: usize) {
        if self.chain.is_empty() {
            return;
        }

        let mut fused: Vec<Overlay> = Vec::new();
        let mut buffer: Vec<Overlay> = Vec::new();

        for ov in self.chain.drain(..) {
            if ov.len() <= max_tiny_len {
                buffer.push(ov);
                if buffer.len() >= min_to_fuse {
                    let total_len: usize =
                        buffer.iter().map(|x| x.values.len()).sum();
                    let mut fused_values = Vec::with_capacity(total_len);

                    for b in &buffer {
                        fused_values.extend_from_slice(&b.values);
                    }

                    fused.push(Overlay::new(fused_values));
                    buffer.clear();
                }
            } else {
                fused.push(ov);
            }
        }

        fused.extend(buffer.into_iter());
        self.chain = fused;
    }

    /// Blend all overlays into `out` (sum), using SIMD where available.
    #[inline(always)]
    pub fn blend_all_into(&self, out: &mut [f32]) {
        if self.chain.is_empty() || out.is_empty() {
            return;
        }

        let len = out.len();
        out.fill(0.0);

        for ov in &self.chain {
            let ov_len = ov.values.len().min(len);
            if ov_len == 0 {
                continue;
            }

            let src = &ov.values[..ov_len];

            if is_x86_feature_detected!("avx512f") && ov_len >= 16 {
                unsafe {
                    Self::blend_overlay_avx512(out, src);
                }
            } else if is_x86_feature_detected!("avx2") && ov_len >= 8 {
                unsafe {
                    Self::blend_overlay_avx2(out, src);
                }
            } else {
                for i in 0..ov_len {
                    out[i] += src[i];
                }
            }
        }
    }

    /// Apply a scalar delta to all overlays (in-place), SIMD-accelerated.
    #[inline(always)]
    pub fn apply_delta_all(&mut self, delta: f32) {
        if delta == 0.0 {
            return;
        }

        for ov in &mut self.chain {
            let slice = &mut ov.values;
            let len = slice.len();
            if len == 0 {
                continue;
            }

            if is_x86_feature_detected!("avx512f") && len >= 16 {
                unsafe {
                    Self::apply_delta_avx512(slice, delta);
                }
            } else if is_x86_feature_detected!("avx2") && len >= 8 {
                unsafe {
                    Self::apply_delta_avx2(slice, delta);
                }
            } else {
                for v in slice.iter_mut() {
                    *v += delta;
                }
            }
        }
    }

    /// Apply a scalar delta to a single overlay region.
    #[inline(always)]
    pub fn apply_delta_to_overlay(
        &mut self,
        overlay_idx: usize,
        start: usize,
        len: usize,
        delta: f32,
    ) {
        if overlay_idx >= self.chain.len() || delta == 0.0 {
            return;
        }

        let ov = &mut self.chain[overlay_idx];
        let ov_len = ov.values.len();
        if ov_len == 0 {
            return;
        }

        let s = start.min(ov_len);
        let e = (start + len).min(ov_len);
        if s >= e {
            return;
        }

        let slice = &mut ov.values[s..e];
        let region_len = slice.len();

        if is_x86_feature_detected!("avx512f") && region_len >= 16 {
            unsafe {
                Self::apply_delta_avx512(slice, delta);
            }
        } else if is_x86_feature_detected!("avx2") && region_len >= 8 {
            unsafe {
                Self::apply_delta_avx2(slice, delta);
            }
        } else {
            for v in slice.iter_mut() {
                *v += delta;
            }
        }
    }

    // ---------------------------------------------------------
    // SIMD helpers
    // ---------------------------------------------------------
    #[target_feature(enable = "avx2")]
    unsafe fn blend_overlay_avx2(out: &mut [f32], src: &[f32]) {
        let len = out.len().min(src.len());
        let mut i = 0;

        while i + 8 <= len {
            let po = out.as_mut_ptr().add(i);
            let ps = src.as_ptr().add(i);

            let vo = _mm256_loadu_ps(po);
            let vs = _mm256_loadu_ps(ps);
            let vout = _mm256_add_ps(vo, vs);

            _mm256_storeu_ps(po, vout);
            i += 8;
        }

        while i < len {
            out[i] += src[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn blend_overlay_avx512(out: &mut [f32], src: &[f32]) {
        let len = out.len().min(src.len());
        let mut i = 0;

        while i + 16 <= len {
            let po = out.as_mut_ptr().add(i);
            let ps = src.as_ptr().add(i);

            let vo = _mm512_loadu_ps(po);
            let vs = _mm512_loadu_ps(ps);
            let vout = _mm512_add_ps(vo, vs);

            _mm512_storeu_ps(po, vout);
            i += 16;
        }

        while i < len {
            out[i] += src[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn apply_delta_avx2(slice: &mut [f32], delta: f32) {
        let len = slice.len();
        if len == 0 {
            return;
        }

        let mut i = 0;
        let vdelta = _mm256_set1_ps(delta);

        while i + 8 <= len {
            let p = slice.as_mut_ptr().add(i);
            let v = _mm256_loadu_ps(p);
            let v = _mm256_add_ps(v, vdelta);
            _mm256_storeu_ps(p, v);
            i += 8;
        }

        while i < len {
            *slice.get_unchecked_mut(i) += delta;
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn apply_delta_avx512(slice: &mut [f32], delta: f32) {
        let len = slice.len();
        if len == 0 {
            return;
        }

        let mut i = 0;
        let vdelta = _mm512_set1_ps(delta);

        while i + 16 <= len {
            let p = slice.as_mut_ptr().add(i);
            let v = _mm512_loadu_ps(p);
            let v = _mm512_add_ps(v, vdelta);
            _mm512_storeu_ps(p, v);
            i += 16;
        }

        while i < len {
            *slice.get_unchecked_mut(i) += delta;
            i += 1;
        }
    }
}


