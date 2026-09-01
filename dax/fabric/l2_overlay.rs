use crate::sm_tile::Overlay;

pub struct L2Overlay {
    cache: Vec<Overlay>,
    capacity: usize,
}

impl L2Overlay {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Vec::new(),
            capacity,
        }
    }

    pub fn push(&mut self, overlay: Overlay) {
        if self.cache.len() >= self.capacity {
            self.cache.remove(0); // simple eviction
        }
        self.cache.push(overlay);
    }

    pub fn get_recent(&self) -> Option<&Overlay> {
        self.cache.last()
    }

    pub fn all(&self) -> &[Overlay] {
        &self.cache
    }
}
