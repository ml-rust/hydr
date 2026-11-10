use super::{
    Case, command_inputs, extreme_inputs, ordinary_inputs, small_inputs, supplement_inputs,
};
pub fn cases() -> Vec<Case> {
    let mut output = Vec::new();
    ordinary_inputs::append(&mut output);
    command_inputs::append(&mut output);
    small_inputs::append(&mut output);
    extreme_inputs::append(&mut output);
    supplement_inputs::append(&mut output);
    output
}
