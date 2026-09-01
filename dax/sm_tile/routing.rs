use super::{Overlay, Region};
use crate::sm_tile::OverlaySram;

#[derive(Default, Clone)]
pub struct RoundaboutRouting {
    pub active_regions: Vec<usize>,
    pub cold_regions: Vec<usize>,
}

pub struct TileRouter {
    pub incoming_rt: RoundaboutRouting,
    pub outgoing_rt: RoundaboutRouting,
    pub chunk_size: usize,
}

impl TileRouter {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            incoming_rt: RoundaboutRouting::default(),
            outgoing_rt: RoundaboutRouting::default(),
            chunk_size,
        }
    }

    pub fn update(
        &mut self,
        regions: &[Region],
        _overlays: &[Overlay],   // Overlay has only `values`, not used here
        sram: &OverlaySram,      // global SRAM with heatmaps
    ) {
        self.incoming_rt.active_regions.clear();
        self.incoming_rt.cold_regions.clear();
        self.outgoing_rt.active_regions.clear();
        self.outgoing_rt.cold_regions.clear();

        let total_len = sram.len();

        for region in regions {
            let region_id = region.id;

            // Region bounds computed from ID + chunk size
            let start = region_id * self.chunk_size;
            let end   = (start + self.chunk_size).min(total_len);

            let mut short_sum = 0.0;
            let mut mid_sum   = 0.0;
            let mut long_sum  = 0.0;

            for i in start..end {
                short_sum += sram.heatmap_short[i];
                mid_sum   += sram.heatmap_mid[i];
                long_sum  += sram.heatmap_long[i];
            }

            // Incoming roundabout: stable importance
            let incoming_active =
                mid_sum  > 1.0 ||
                long_sum > 0.5;

            // Outgoing roundabout: recent activity
            let outgoing_active =
                short_sum > 2.0 ||
                mid_sum   > 1.0;

            if incoming_active {
                self.incoming_rt.active_regions.push(region_id);
            } else {
                self.incoming_rt.cold_regions.push(region_id);
            }

            if outgoing_active {
                self.outgoing_rt.active_regions.push(region_id);
            } else {
                self.outgoing_rt.cold_regions.push(region_id);
            }
        }
    }
}
