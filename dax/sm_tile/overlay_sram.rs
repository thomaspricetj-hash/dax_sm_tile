pub struct OverlaySram {
    // Legacy unified view used all over the codebase.
    pub data: Vec<f32>,

    // Split buffers: stable incoming state + outgoing speculative/delta state.
    pub incoming: Vec<f32>,
    pub outgoing: Vec<f32>,

    // Multilayered heatmap:
    pub heatmap_short: Vec<f32>,
    pub heatmap_mid: Vec<f32>,
    pub heatmap_long: Vec<f32>,

    // Skimming mask: true = region is active/hot and should be processed.
    // This is chunk-level, not element-level.
    pub active_chunks: Vec<bool>,

    // Chunk size used for skimming (e.g., 64, 128, 256 elements).
    pub chunk_size: usize,
}

impl OverlaySram {
    pub fn new(size: usize) -> Self {
        let base = vec![0.0; size];

        // Default chunk size: 128 elements per chunk.
        let chunk_size = 128;
        let chunk_count = (size + chunk_size - 1) / chunk_size;

        Self {
            data: base.clone(),
            incoming: base.clone(),
            outgoing: vec![0.0; size],

            heatmap_short: vec![0.0; size],
            heatmap_mid: vec![0.0; size],
            heatmap_long: vec![0.0; size],

            active_chunks: vec![true; chunk_count], // start fully active
            chunk_size,
        }
    }

    // -----------------------------
    // Basic read/write (no heatmap updates)
    // -----------------------------

    #[inline(always)]
    pub fn write_outgoing(&mut self, idx: usize, value: f32) {
        if idx < self.outgoing.len() {
            self.outgoing[idx] = value;
        }
    }

    #[inline(always)]
    pub fn read_incoming(&self, idx: usize) -> f32 {
        self.incoming.get(idx).copied().unwrap_or(0.0)
    }

    #[inline(always)]
    pub fn read_outgoing(&self, idx: usize) -> f32 {
        self.outgoing.get(idx).copied().unwrap_or(0.0)
    }

    // -----------------------------
    // Heatmap updates (scheduler/DMA only)
    // -----------------------------

    #[inline(always)]
    pub fn record_read_region(&mut self, start: usize, len: usize) {
        let end = (start + len).min(self.incoming.len());
        for i in start..end {
            self.heatmap_short[i] += 0.5;
            self.heatmap_mid[i] += 0.1;
        }
    }

    #[inline(always)]
    pub fn record_write_region(&mut self, start: usize, len: usize) {
        let end = (start + len).min(self.outgoing.len());
        for i in start..end {
            self.heatmap_short[i] += 1.0;
            self.heatmap_mid[i] += 0.3;
            self.heatmap_long[i] += 0.01;
        }
    }

    #[inline(always)]
    pub fn decay_heatmaps(&mut self) {
        for i in 0..self.heatmap_short.len() {
            self.heatmap_short[i] *= 0.5;
            self.heatmap_mid[i] *= 0.9;
            self.heatmap_long[i] *= 0.99;
        }
    }

    // -----------------------------
    // Skimming: compute active chunks
    // -----------------------------

    pub fn update_active_chunks(&mut self) {
        let chunk_count = self.active_chunks.len();
        let cs = self.chunk_size;

        for chunk_id in 0..chunk_count {
            let start = chunk_id * cs;
            let end = (start + cs).min(self.len());

            // Aggregate heatmap values for this chunk.
            let mut short_sum = 0.0;
            let mut mid_sum = 0.0;
            let mut long_sum = 0.0;

            for i in start..end {
                short_sum += self.heatmap_short[i];
                mid_sum   += self.heatmap_mid[i];
                long_sum  += self.heatmap_long[i];
            }

            // Thresholds — tune these based on perf.
            let active =
                short_sum > 2.0 ||
                mid_sum   > 1.0 ||
                long_sum  > 0.5;

            self.active_chunks[chunk_id] = active;
        }
    }

    // -----------------------------
    // Commit / rollback
    // -----------------------------

    #[inline(always)]
    pub fn commit_outgoing(&mut self) {
        let len = self.incoming.len().min(self.outgoing.len());
        for i in 0..len {
            let v = self.outgoing[i];
            self.incoming[i] = v;
            self.data[i] = v;
        }
    }

    #[inline(always)]
    pub fn rollback_outgoing(&mut self) {
        for v in self.outgoing.iter_mut() {
            *v = 0.0;
        }
    }

    // -----------------------------
    // Legacy unified access
    // -----------------------------

    #[inline(always)]
    pub fn write(&mut self, idx: usize, value: f32) {
        self.write_outgoing(idx, value);
    }

    #[inline(always)]
    pub fn read(&self, idx: usize) -> f32 {
        self.read_incoming(idx)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

