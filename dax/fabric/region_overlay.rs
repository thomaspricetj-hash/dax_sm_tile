use crate::sm_tile::{Region, Overlay};

#[derive(Clone)]
pub struct RegionOverlay {
    pub region: Region,
    pub overlay: Overlay,
}

pub struct RegionOverlayTable {
    entries: Vec<RegionOverlay>,
}

impl RegionOverlayTable {
    pub fn new() -> Self {
        Self { entries: vec![] }
    }

    pub fn add(&mut self, region: Region, overlay: Overlay) {
        self.entries.push(RegionOverlay { region, overlay });
    }

    pub fn get_by_region(&self, region_id: usize) -> Option<&RegionOverlay> {
        self.entries.iter().find(|e| e.region.id == region_id)
    }

    pub fn all(&self) -> &[RegionOverlay] {
        &self.entries
    }
}
