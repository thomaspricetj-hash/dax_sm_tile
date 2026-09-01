use crate::sm_tile::OverlaySram;

/// HBM interface for DAX tiles.
/// In real hardware this would be DMA-like, but here it's a clean abstraction.
pub struct HbmInterface {
    pub hbm: Vec<f32>,
}

impl HbmInterface {
    pub fn new(size: usize) -> Self {
        Self {
            hbm: vec![0.0; size],
        }
    }

    /// Read a region from HBM into SRAM.
    pub fn load_region(&self, region_id: usize, sram: &mut OverlaySram) {
        let hbm_len = self.hbm.len();          // PREVENT aliasing
        let len = sram.data.len().min(hbm_len);

        for i in 0..len {
            let idx = (region_id + i) % hbm_len;
            sram.data[i] = self.hbm[idx];
        }
    }

    /// Write SRAM overlay back into HBM.
    pub fn store_region(&mut self, region_id: usize, sram: &OverlaySram) {
        let hbm_len = self.hbm.len();          // PREVENT aliasing
        let len = sram.data.len().min(hbm_len);

        for i in 0..len {
            let idx = (region_id + i) % hbm_len;
            self.hbm[idx] = sram.data[i];
        }
    }
}
