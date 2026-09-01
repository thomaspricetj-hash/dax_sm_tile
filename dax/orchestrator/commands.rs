/// Extended delta commands for orchestrator-level control.
#[derive(Clone)]
pub struct DeltaCommandExt {
    pub region_id: usize,
    pub delta_value: f32,
    pub priority: u8,
}

/// Collapse command for orchestrator-level collapse control.
#[derive(Clone)]
pub struct CollapseCommand {
    pub region_id: usize,
    pub passes: u8,
}
