// DAX GPU ARCHITECTURE — Delta-State Compute Accelerator
// Top-level DAX module

pub mod sm_tile;
pub mod orchestrator;
pub mod fabric;
pub mod transformer;

pub use sm_tile::DaxSmTile;
pub use orchestrator::GlobalDaxOrchestrator;
pub use fabric::{HbmInterface, L2Overlay, RegionOverlayTable};
pub use transformer::{KvDeltaEngine, AttentionHook};

/// High-level DAX context: one SM tile + orchestrator + fabric.
pub struct DaxContext {
    pub tile: DaxSmTile,
    pub orchestrator: GlobalDaxOrchestrator,
    pub hbm: HbmInterface,
    pub l2: L2Overlay,
    pub region_overlays: RegionOverlayTable,
}

impl DaxContext {
    pub fn new(
        core_count: usize,
        sram_size: usize,
        hbm_size: usize,
        l2_capacity: usize,
    ) -> Self {
        Self {
            tile: DaxSmTile::new(core_count, sram_size),
            orchestrator: GlobalDaxOrchestrator::new(),
            hbm: HbmInterface::new(hbm_size),
            l2: L2Overlay::new(l2_capacity),
            region_overlays: RegionOverlayTable::new(),
        }
    }

    /// One full DAX step: fabric ↔ tile ↔ transformer.
    pub fn step(
        &mut self,
        layer_id: usize,
        head_id: usize,
        kv_keys: &mut [f32],
        kv_values: &mut [f32],
        att_scores: &mut [f32],
    ) {
        // Load region from HBM into tile SRAM
        self.hbm.load_region(0, &mut self.tile.overlay_sram);

        // Run orchestrated DAX → transformer cycle
        self.orchestrator.step(
            &mut self.tile,
            layer_id,
            head_id,
            kv_keys,
            kv_values,
            att_scores,
        );

        // Store updated SRAM back to HBM
        self.hbm.store_region(0, &self.tile.overlay_sram);

        // Cache latest overlay in L2 + region overlay table
        if let Some(last) = self.tile.overlay_chain.fetch_overlays().last() {
            self.l2.push(last.clone());
            self.region_overlays.add(
                sm_tile::Region { id: 0 },
                last.clone(),
            );
        }
    }
}
