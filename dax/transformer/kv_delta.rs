use crate::sm_tile::{OverlayChain, Overlay};

/// Represents a single KV-cache delta update for a head/layer.
#[derive(Clone)]
pub struct KvDeltaUpdate {
    pub layer_id: usize,
    pub head_id: usize,
    pub key_delta: Vec<f32>,
    pub value_delta: Vec<f32>,
}

pub struct KvDeltaEngine;

impl KvDeltaEngine {
    pub fn new() -> Self {
        Self
    }

    /// Convert overlay chain into KV-cache deltas.
    /// This is where your delta-state semantics live.
    pub fn overlays_to_kv_deltas(&self, chain: &OverlayChain) -> Vec<KvDeltaUpdate> {
        let overlays: Vec<Overlay> = chain.fetch_overlays();

        overlays
            .iter()
            .enumerate()
            .map(|(idx, ov)| KvDeltaUpdate {
                layer_id: idx,          // TODO: real mapping
                head_id: 0,             // TODO: per-head routing
                key_delta: ov.values.clone(),
                value_delta: ov.values.clone(),
            })
            .collect()
    }
}
