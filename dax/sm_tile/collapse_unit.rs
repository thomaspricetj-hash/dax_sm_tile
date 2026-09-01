use super::overlay_chain::OverlayChain;
use super::overlay_sram::OverlaySram;
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_andnot_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps,
};

/// High‑performance collapse unit:
/// - Normalizes SRAM data with AVX2 when available
/// - Pushes overlays into the chain
/// - Fuses tiny overlays to reduce overhead
/// - Can collapse directly into an output buffer for fast readout
pub struct CollapseUnit {
    tiny_threshold: usize,
    min_to_fuse: usize,
}

impl CollapseUnit {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            tiny_threshold: 32,
            min_to_fuse: 4,
        }
    }

    #[inline(always)]
    pub fn collapse(&self, sram: &mut OverlaySram, chain: &mut OverlayChain) {
        if sram.data.is_empty() {
            return;
        }

        self.normalize_sram_in_place(&mut sram.data);
        chain.push_overlay(sram.data.clone());
        chain.fuse_tiny_overlays(self.tiny_threshold, self.min_to_fuse);
    }

    #[inline(always)]
    pub fn collapse_into(&self, sram: &mut OverlaySram, chain: &mut OverlayChain, out: &mut [f32]) {
        if sram.data.is_empty() || out.is_empty() {
            return;
        }

        self.normalize_sram_in_place(&mut sram.data);
        chain.push_overlay(sram.data.clone());
        chain.fuse_tiny_overlays(self.tiny_threshold, self.min_to_fuse);
        chain.blend_all_into(out);
    }

    #[inline(always)]
    fn normalize_sram_in_place(&self, data: &mut [f32]) {
        if data.is_empty() {
            return;
        }

        let norm = unsafe { Self::fast_l1_norm_avx(data) };
        if norm <= 0.0 {
            return;
        }

        let scale = 1.0 / norm;
        unsafe { Self::scale_in_place_avx(data, scale) };
    }

    #[inline(always)]
    unsafe fn fast_l1_norm_avx(slice: &[f32]) -> f32 {
        if is_x86_feature_detected!("avx2") && slice.len() >= 8 {
            let mut sum = _mm256_setzero_ps();
            let mut i = 0;

            while i + 8 <= slice.len() {
                let p = slice.as_ptr().add(i);
                let v = _mm256_loadu_ps(p);
                let mask = _mm256_set1_ps(-0.0);
                let abs = _mm256_andnot_ps(mask, v);
                sum = _mm256_add_ps(sum, abs);
                i += 8;
            }

            let mut tmp = [0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), sum);

            let mut total = tmp.iter().sum::<f32>();
            while i < slice.len() {
                total += slice.get_unchecked(i).abs();
                i += 1;
            }

            total
        } else {
            slice.iter().map(|x| x.abs()).sum()
        }
    }

    #[inline(always)]
    unsafe fn scale_in_place_avx(slice: &mut [f32], scale: f32) {
        if is_x86_feature_detected!("avx2") && slice.len() >= 8 {
            let vscale = _mm256_set1_ps(scale);
            let mut i = 0;

            while i + 8 <= slice.len() {
                let p = slice.as_mut_ptr().add(i);
                let v = _mm256_loadu_ps(p);
                let v = _mm256_mul_ps(v, vscale);
                _mm256_storeu_ps(p, v);
                i += 8;
            }

            while i < slice.len() {
                *slice.get_unchecked_mut(i) *= scale;
                i += 1;
            }
        } else {
            for v in slice.iter_mut() {
                *v *= scale;
            }
        }
    }
}
