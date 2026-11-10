use super::{Case, endpoint_inputs, motion_inputs, rejection_inputs};
pub fn cases() -> Vec<Case> {
    let mut result = Vec::new();
    motion_inputs::append(&mut result);
    endpoint_inputs::append(&mut result);
    rejection_inputs::append(&mut result);
    result
}
