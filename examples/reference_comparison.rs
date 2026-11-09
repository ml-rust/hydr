//! Compares matched wet-bed and dry-bed dam fields against independently executed references.

#[path = "reference_comparison/mod.rs"]
mod comparison;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    comparison::run()
}
