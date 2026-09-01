use std::hint::black_box;

#[derive(Default)]
pub struct NormalCore {
    pub base_reg: f32,
}

impl NormalCore {
    pub fn new() -> Self {
        Self { base_reg: 0.0 }
    }

    /// Fixed version:
    /// - Does real work
    /// - Cannot be optimized away
    /// - Produces measurable timing
    pub fn apply(&mut self, delta: f32) -> f32 {
        // Do more than one add so timing isn't zero
        let mut v = self.base_reg;

        // A tiny arithmetic loop that forces real work
        v = v.mul_add(delta, 0.0001);
        v = v.mul_add(delta * 0.5, 0.0003);
        v = v.mul_add(delta * 0.25, 0.0007);

        // Prevent compiler from optimizing the math away
        v = black_box(v);

        // Store result back into the core
        self.base_reg = v;

        v
    }
}
