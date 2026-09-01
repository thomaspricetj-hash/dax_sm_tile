use crate::sm_tile::OverlaySram;


/// Overlay applied to attention scores/weights.
#[derive(Clone)]
pub struct AttentionOverlay {
    pub layer_id: usize,
    pub head_id: usize,
    pub scores_delta: Vec<f32>,
}

pub struct AttentionHook;

impl AttentionHook {
    pub fn new() -> Self {
        Self
    }

    /// Derive attention overlays from DAX overlay SRAM.
    /// You’ll later align this with your actual attention tensor layout.
    pub fn sram_to_attention_overlay(
        &self,
        layer_id: usize,
        head_id: usize,
        sram: &OverlaySram,
    ) -> AttentionOverlay {
        AttentionOverlay {
            layer_id,
            head_id,
            scores_delta: sram.data.clone(),
        }
    }

    /// Apply attention overlay to raw attention scores.
    pub fn apply_overlay(
        &self,
        scores: &mut [f32],
        overlay: &AttentionOverlay,
    ) {
        let delta = &overlay.scores_delta;
        let len = scores.len().min(delta.len());

        for i in 0..len {
            scores[i] += delta[i];
        }
    }
}
