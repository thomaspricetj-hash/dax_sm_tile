#[derive(Clone)]
pub struct Region {
    pub id: usize,
}

#[derive(Clone)]
pub enum RegionDelta {
    Add(usize),
    Remove(usize),
}

pub struct RegionTable {
    master: Vec<Region>,        // full canonical list
    deltas: Vec<RegionDelta>,   // only changes since last commit
    last_snapshot: Vec<Region>, // previous committed state
}

impl RegionTable {
    pub fn new() -> Self {
        Self {
            master: vec![],
            deltas: vec![],
            last_snapshot: vec![],
        }
    }

    // Record a delta instead of mutating master directly
    pub fn add_region(&mut self, id: usize) {
        self.deltas.push(RegionDelta::Add(id));
    }

    pub fn remove_region(&mut self, id: usize) {
        self.deltas.push(RegionDelta::Remove(id));
    }

    // Apply deltas to master (DAX commit)
    pub fn apply_deltas(&mut self) {
        for d in self.deltas.drain(..) {
            match d {
                RegionDelta::Add(id) => {
                    self.master.push(Region { id });
                }
                RegionDelta::Remove(id) => {
                    self.master.retain(|r| r.id != id);
                }
            }
        }
    }

    // Fetch only the regions that changed since last snapshot
    pub fn fetch_regions(&mut self) -> Vec<Region> {
        let mut changed = Vec::new();

        for r in &self.master {
            if !self.last_snapshot.iter().any(|old| old.id == r.id) {
                changed.push(r.clone());
            }
        }

        self.last_snapshot = self.master.clone();
        changed
    }

    // Fetch full canonical region list
    pub fn fetch_master(&self) -> Vec<Region> {
        self.master.clone()
    }
}
