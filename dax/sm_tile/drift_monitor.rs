use super::OverlayChain;
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_andnot_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};

/// High‑performance semantic drift monitor:
/// - Tracks drift score over time
/// - Uses SIMD to compute overlay magnitude
/// - Maintains rolling window of drift
/// - Exposes a simple `is_drifting()` check
pub struct DriftMonitor {
    /// Rolling drift scores
    history: Vec<f32>,
    /// Max history length
    max_history: usize,
    /// Threshold above which we consider drift significant
    drift_threshold: f32,
}

impl DriftMonitor {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(64),
            max_history: 64,
            drift_threshold: 0.05,
        }
    }

    /// Check current overlays for drift and record a drift score.
    #[inline(always)]
    pub fn check(&mut self, chain: &OverlayChain) {
        let overlays = chain.fetch_overlays();
        if overlays.is_empty() {
            self.record_drift(0.0);
            return;
        }

        // Compute aggregate magnitude across overlays
        let mut total = 0.0;
        for ov in &overlays {
            if ov.values.is_empty() {
                continue;
            }
            let mag = unsafe { Self::fast_magnitude_avx(ov.values.as_slice()) };
            total += mag;
        }

        // Normalize by number of overlays to keep scale stable
        let drift_score = total / (overlays.len() as f32).max(1.0);
        self.record_drift(drift_score);
    }

    /// Returns true if recent drift exceeds threshold.
    #[inline(always)]
    pub fn is_drifting(&self) -> bool {
        if self.history.is_empty() {
            return false;
        }

        // Simple moving average over recent window
        let window = self.history.len().min(16);
        let start = self.history.len().saturating_sub(window);
        let avg = self.history[start..]
            .iter()
            .copied()
            .sum::<f32>()
            / (window as f32);

        avg > self.drift_threshold
    }

    #[inline(always)]
    fn record_drift(&mut self, score: f32) {
        if self.history.len() == self.max_history {
            // Drop oldest to keep window bounded
            self.history.remove(0);
        }
        self.history.push(score);
    }

    #[inline(always)]
    unsafe fn fast_magnitude_avx(slice: &[f32]) -> f32 {
        if is_x86_feature_detected!("avx2") && slice.len() >= 8 {
            let mut sum = _mm256_setzero_ps();
            let mut i = 0;

            while i + 8 <= slice.len() {
                let p = slice.as_ptr().add(i);
                let v = _mm256_loadu_ps(p);
                let abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), v);
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

            return total;
        }

        slice.iter().map(|x| x.abs()).sum()
    }
}
