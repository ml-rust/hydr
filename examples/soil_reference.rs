//! Compare the public soil component against pinned native EPA evidence.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "soil_reference/comparison.rs"]
mod comparison;
#[path = "soil_reference/records.rs"]
mod records;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: cargo run --example soil_reference -- NATIVE.csv")?;
    comparison::compare(&records::read(std::path::Path::new(&path))?)
}

#[cfg(test)]
#[path = "soil_reference/tests.rs"]
mod tests;
