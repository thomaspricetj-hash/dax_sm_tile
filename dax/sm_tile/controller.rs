use crate::sm_tile::{Region, Overlay};
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_andnot_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};

#[derive(Clone, Default)]
pub struct DeltaCommand {
    pub region_id: usize,
    pub delta_value: f32,
}

/// High‑performance controller for SM‑Tile.
/// Generates warp‑friendly, cache‑friendly delta commands.
pub struct DaxController {
    hot_flags: Vec<u8>,
    last_delta: Vec<f32>,
    warp_size: usize,
    tiny_threshold: usize,
}

impl DaxController {
    pub fn new() -> Self {
        Self {
            hot_flags: vec![],
            last_delta: vec![],
            warp_size: 8,
            tiny_threshold: 32,
        }
    }

    #[inline(always)]
    fn ensure_capacity(&mut self, max_id: usize) {
        if self.hot_flags.len() <= max_id {
            self.hot_flags.resize(max_id + 1, 0);
            self.last_delta.resize(max_id + 1, 0.0);
        }
    }

    /// Use overlays to update hotness, but ignore tiny overlays
    /// below `tiny_threshold` to avoid noise.
    #[inline(always)]
    fn update_hotness(&mut self, overlays: &[Overlay]) {
        for ov in overlays {
            let len = ov.values.len();
            if len == 0 {
                continue;
            }

            // Ignore very small overlays: they are treated as noise
            if len < self.tiny_threshold {
                continue;
            }

            let magnitude = unsafe { Self::fast_magnitude_avx(ov.values.as_slice()) };

            let hot = magnitude > 0.001;
            let region_id = (len ^ (magnitude as usize)) & 0xFFFF;

            self.ensure_capacity(region_id);

            let flag = &mut self.hot_flags[region_id];
            if hot {
                *flag = (*flag + 1).min(3);
            } else {
                *flag = flag.saturating_sub(1);
            }
        }
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

    pub fn schedule(&mut self, regions: &[Region], overlays: &[Overlay]) -> Vec<DeltaCommand> {
        if regions.is_empty() {
            return vec![];
        }

        self.update_hotness(overlays);

        let mut sorted: Vec<&Region> = regions.iter().collect();
        sorted.sort_unstable_by(|a, b| {
            let hot_a = self.hot_flags.get(a.id).copied().unwrap_or(0);
            let hot_b = self.hot_flags.get(b.id).copied().unwrap_or(0);

            hot_b.cmp(&hot_a).then_with(|| a.id.cmp(&b.id))
        });

        let mut commands = Vec::with_capacity(sorted.len());
        let mut idx = 0;

        while idx < sorted.len() {
            let end = (idx + self.warp_size).min(sorted.len());
            let warp = &sorted[idx..end];

            let warp_delta = Self::compute_warp_delta(warp, self.tiny_threshold);

            for r in warp {
                self.ensure_capacity(r.id);

                let last = self.last_delta[r.id];
                let delta = if (warp_delta - last).abs() < 0.00001 {
                    0.0
                } else {
                    warp_delta
                };

                self.last_delta[r.id] = delta;

                commands.push(DeltaCommand {
                    region_id: r.id,
                    delta_value: delta,
                });
            }

            idx = end;
        }

        commands
    }

    #[inline(always)]
    fn compute_warp_delta(warp: &[&Region], tiny_threshold: usize) -> f32 {
        let mut total = 0.0;

        for r in warp {
            let id_factor = ((r.id as f32) * 0.0001).sin().abs();
            // Bias delta slightly by tiny_threshold to keep it meaningful
            let scale = 1.0 + (tiny_threshold as f32) * 0.00001;
            total += id_factor * scale;
        }

        total * 0.001
    }
}
