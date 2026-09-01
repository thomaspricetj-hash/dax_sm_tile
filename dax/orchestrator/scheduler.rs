use crate::sm_tile::DaxSmTile;

/// Semantic scheduler: decides which regions/overlays should be updated,
/// collapsed, or routed based on drift, region activity, etc.
pub struct SemanticScheduler;

impl SemanticScheduler {
    pub fn new() -> Self {
        Self
    }

    pub fn apply(&self, tile: &mut DaxSmTile) {
        // TODO: advanced semantic scheduling
        // For now: no-op
        let _regions = tile.region_table.fetch_regions();
        let _overlays = tile.overlay_chain.fetch_overlays();
    }
}
