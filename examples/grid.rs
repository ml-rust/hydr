//! Creates a Cartesian grid and prints its SI geometry.

use hydr::grid::CartesianGrid;

fn main() -> Result<(), hydr::error::GridError> {
    let grid = CartesianGrid::new(100, 50, 2.0, 2.0)?;
    println!("Cells: {}", grid.cell_count());
    println!("Cell area: {} m²", grid.cell_area());
    println!("Domain extent: {:?} m", grid.extent());
    Ok(())
}
