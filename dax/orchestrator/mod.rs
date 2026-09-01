// Global DAX Orchestrator

pub mod scheduler;
pub mod collapse_control;
pub mod commands;

pub use scheduler::SemanticScheduler;
pub use collapse_control::CollapseControl;
pub use commands::{DeltaCommandExt, CollapseCommand};

use crate::sm_tile::DaxSmTile;
use crate::transformer::{KvDeltaEngine, AttentionHook};

/// The global orchestrator that coordinates SM tiles + transformer hooks.
pub struct GlobalDaxOrchestrator {
    pub scheduler: SemanticScheduler,
    pub collapse_control: CollapseControl,
    pub kv_engine: KvDeltaEngine,
    pub att_hook: AttentionHook,
}

impl GlobalDaxOrchestrator {
    pub fn new() -> Self {
        Self {
            scheduler: SemanticScheduler::new(),
            collapse_control: CollapseControl::new(),
            kv_engine: KvDeltaEngine::new(),
            att_hook: AttentionHook::new(),
        }
    }

    /// Execute one full DAX → Transformer cycle.
    pub fn step(
        &mut self,
        tile: &mut DaxSmTile,
        layer_id: usize,
        head_id: usize,
        kv_keys: &mut [f32],
        kv_values: &mut [f32],
        att_scores: &mut [f32],
    ) {
        // 1. Run tile compute
        tile.step();

        // 2. Scheduler may modify tile state or issue commands
        self.scheduler.apply(tile);

        // 3. Collapse control may trigger additional collapse passes
        self.collapse_control.apply(tile);

        // 4. Convert overlays → KV deltas
        let kv_deltas = self.kv_engine.overlays_to_kv_deltas(&tile.overlay_chain);

        // Apply KV deltas (placeholder routing)
        for delta in kv_deltas {
            let len = kv_keys.len().min(delta.key_delta.len());
            for i in 0..len {
                kv_keys[i] += delta.key_delta[i];
            }

            let len = kv_values.len().min(delta.value_delta.len());
            for i in 0..len {
                kv_values[i] += delta.value_delta[i];
            }
        }

        // 5. Attention overlays
        let overlay = self
            .att_hook
            .sram_to_attention_overlay(layer_id, head_id, &tile.overlay_sram);

        self.att_hook.apply_overlay(att_scores, &overlay);
    }
}
