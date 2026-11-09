//! Compares matched wet-bed dam fields against an independently executed reference.

#[path = "reference_comparison/mod.rs"]
mod comparison;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    comparison::run()
}
