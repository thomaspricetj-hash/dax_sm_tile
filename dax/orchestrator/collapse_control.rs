use crate::sm_tile::DaxSmTile;

/// CollapseControl: triggers multi-pass collapse, reversible rules,
/// and overlay chain compaction.
pub struct CollapseControl;

impl CollapseControl {
    pub fn new() -> Self {
        Self
    }

    pub fn apply(&self, tile: &mut DaxSmTile) {
        // TODO: multi-pass collapse logic
        // For now: single collapse pass already done in tile.step()
        let _ = &tile.overlay_chain;
    }
}
