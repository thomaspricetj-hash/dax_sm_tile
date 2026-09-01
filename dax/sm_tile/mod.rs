// DAX SM Tile — Delta-State Compute Tile

mod controller;
mod delta_core;
mod overlay_sram;
mod collapse_unit;
mod routing;
mod drift_monitor;
mod region_table;
mod overlay_chain;
mod normal_core;

pub use controller::{DaxController, DeltaCommand};

// --- PUBLIC EXPORTS FROM delta_core (fix for tests) ---
pub use delta_core::{
    DeltaCore,
    BlockDeltaCore,
    CoreBridge,
    AmdCoreCallback,

    // micro-fiber system
    MicroFiber,
    MicroFiberScheduler,
    MicroFiberMode,
    MicroFiberState,
    FiberPerfCounters,
};

pub use overlay_sram::OverlaySram;
pub use collapse_unit::CollapseUnit;
pub use routing::TileRouter;
pub use drift_monitor::DriftMonitor;
pub use region_table::{Region, RegionTable};
pub use overlay_chain::{Overlay, OverlayChain};
pub use normal_core::NormalCore;
pub use crate::sm_tile::delta_core::perf_flags;

/// Skimming unit: decides which regions are worth touching this step.
pub struct SkimUnit;

impl SkimUnit {
    pub fn new() -> Self {
        Self
    }

    pub fn skim(&self, regions: &[Region]) -> Vec<Region> {
        regions.to_vec()
    }
}

/// A group represents a logical execution block over multiple regions.
pub struct Group {
    pub region_indices: Vec<usize>,
}

/// A cluster represents a set of groups that can be processed together.
pub struct Cluster {
    pub groups: Vec<Group>,
}

/// Grouping unit: forms groups (blocks) from regions.
pub struct GroupUnit;

impl GroupUnit {
    pub fn new() -> Self {
        Self
    }

    pub fn form_groups(&self, regions: &[Region]) -> Vec<Group> {
        let mut groups = Vec::new();
        let mut current = Vec::new();

        for (idx, _) in regions.iter().enumerate() {
            current.push(idx);
            if current.len() == 2 {
                groups.push(Group {
                    region_indices: current.clone(),
                });
                current.clear();
            }
        }

        if !current.is_empty() {
            groups.push(Group {
                region_indices: current,
            });
        }

        groups
    }
}

/// Cluster unit: assigns groups into clusters.
pub struct ClusterUnit;

impl ClusterUnit {
    pub fn new() -> Self {
        Self
    }

    pub fn assign_clusters(&self, groups: Vec<Group>) -> Vec<Cluster> {
        vec![Cluster { groups }]
    }
}

/// Roundabout fabric: ring-buffer style memory fabric.
pub struct RoundaboutFabric {
    hbm: Vec<f32>,
}

impl RoundaboutFabric {
    pub fn new(size: usize) -> Self {
        Self {
            hbm: vec![0.0; size],
        }
    }

    pub fn load_slices(&self, _regions: &[Region], sram: &mut OverlaySram) {
        let len = sram.data.len().min(self.hbm.len());
        sram.data[..len].copy_from_slice(&self.hbm[..len]);
    }

    pub fn store_slices(&mut self, _regions: &[Region], sram: &OverlaySram) {
        let len = sram.data.len().min(self.hbm.len());
        self.hbm[..len].copy_from_slice(&sram.data[..len]);
    }
}

/// Tile swarm: multiple tiles sharing a roundabout fabric.
pub struct TileSwarm {
    pub tiles: Vec<DaxSmTile>,
    pub roundabout: RoundaboutFabric,
    pub group_unit: GroupUnit,
    pub cluster_unit: ClusterUnit,
}

impl TileSwarm {
    pub fn new(tile_count: usize, core_count: usize, sram_size: usize) -> Self {
        let tiles = (0..tile_count)
            .map(|_| DaxSmTile::new(core_count, sram_size))
            .collect();

        Self {
            tiles,
            roundabout: RoundaboutFabric::new(sram_size),
            group_unit: GroupUnit::new(),
            cluster_unit: ClusterUnit::new(),
        }
    }

    pub fn step(&mut self) {
        for tile in self.tiles.iter_mut() {
            tile.step();
        }
    }
}

pub struct DaxSmTile {
    pub controller: DaxController,
    pub delta_cores: Vec<DeltaCore>,
    pub block_delta_cores: Vec<BlockDeltaCore>,
    pub overlay_sram: OverlaySram,
    pub collapse_unit: CollapseUnit,
    pub router: TileRouter,
    pub drift_monitor: DriftMonitor,
    pub region_table: RegionTable,
    pub overlay_chain: OverlayChain,

    pub skim_unit: SkimUnit,
    pub recv_fabric: RoundaboutFabric,
    pub send_fabric: RoundaboutFabric,

    pub group_unit: GroupUnit,
    pub cluster_unit: ClusterUnit,

    // fused AMD core bridge
    pub core_bridge: CoreBridge,
}

impl DaxSmTile {
    pub fn new(core_count: usize, sram_size: usize) -> Self {
        fn amd_dense_pass(chunk: &mut [f32], dt: f32) {
            let scale = 1.0 + 0.01 * dt;
            for v in chunk.iter_mut() {
                *v *= scale;
            }
        }

        let mut block_delta_cores: Vec<BlockDeltaCore> =
            (0..core_count).map(BlockDeltaCore::new).collect();

        for core in block_delta_cores.iter_mut() {
            core.set_mode_hybrid();
        }

        Self {
            controller: DaxController::new(),
            delta_cores: (0..core_count).map(DeltaCore::new).collect(),
            block_delta_cores,
            overlay_sram: OverlaySram::new(sram_size),
            collapse_unit: CollapseUnit::new(),
            // FIX: pass chunk_size to TileRouter::new
            router: TileRouter::new(128),
            drift_monitor: DriftMonitor::new(),
            region_table: RegionTable::new(),
            overlay_chain: OverlayChain::new(),

            skim_unit: SkimUnit::new(),
            recv_fabric: RoundaboutFabric::new(sram_size),
            send_fabric: RoundaboutFabric::new(sram_size),

            group_unit: GroupUnit::new(),
            cluster_unit: ClusterUnit::new(),

            core_bridge: CoreBridge {
                tile_id: 0,
                amd_core_id: 0,
                weight: 0.5,
                amd_callback: Some(amd_dense_pass as AmdCoreCallback),
            },
        }
    }

    pub fn step(&mut self) {
        let regions = self.region_table.fetch_regions();
        let groups = self.group_unit.form_groups(&regions);
        let _clusters = self.cluster_unit.assign_clusters(groups);

        let skimmed = self.skim_unit.skim(&regions);

        self.recv_fabric.load_slices(&skimmed, &mut self.overlay_sram);

        let overlays = self.overlay_chain.fetch_overlays();
        let commands = self.controller.schedule(&skimmed, &overlays);

        for (core, cmd) in self.delta_cores.iter_mut().zip(commands.iter()) {
            core.apply_delta(cmd, &mut self.overlay_sram);
        }

        let block_len = 8;
        for (core, cmd) in self.block_delta_cores.iter_mut().zip(commands.iter()) {
            core.apply_delta_block(
                cmd,
                &mut self.overlay_sram,
                block_len,
                Some(&self.core_bridge),
            );
        }

        self.collapse_unit
            .collapse(&mut self.overlay_sram, &mut self.overlay_chain);

        self.drift_monitor.check(&self.overlay_chain);
        // FIX: pass &self.overlay_sram to router.update
        self.router.update(&skimmed, &overlays, &self.overlay_sram);

        self.send_fabric.store_slices(&skimmed, &self.overlay_sram);
    }

    pub fn step_fast(&mut self) {
        let len = self.overlay_sram.data.len();
        let cmd = DeltaCommand {
            region_id: 0,
            delta_value: 1.0,
        };

        BlockDeltaCore::apply_delta_block_parallel(
            &mut self.block_delta_cores,
            &cmd,
            &mut self.overlay_sram,
            len,
            Some(&self.core_bridge),
        );
    }

    pub fn step_micro(&mut self) {
        self.step_fast();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_delta_flow() {
        let mut tile = DaxSmTile::new(32, 16384);

        tile.region_table.add_region(0);
        tile.region_table.add_region(1);

        tile.step();

        assert_eq!(tile.overlay_sram.data.len(), 16384);
        let overlays = tile.overlay_chain.fetch_overlays();
        assert!(!overlays.is_empty());
    }

    #[test]
    fn tile_swarm_runs() {
        let mut swarm = TileSwarm::new(4, 32, 16384);

        if let Some(first) = swarm.tiles.get_mut(0) {
            first.region_table.add_region(0);
            first.region_table.add_region(1);
        }

        swarm.step();

        for tile in swarm.tiles.iter() {
            assert_eq!(tile.overlay_sram.data.len(), 16384);
        }
    }
}
