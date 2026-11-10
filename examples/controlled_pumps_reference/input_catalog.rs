use super::{Case, adoption_inputs, qualification_inputs, switch_inputs};
pub fn cases() -> Vec<Case> {
    let mut output = Vec::new();
    switch_inputs::append(&mut output);
    qualification_inputs::append(&mut output);
    adoption_inputs::append(&mut output);
    output
}
