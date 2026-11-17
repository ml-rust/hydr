use super::ModelError;

/// Caller-attested projected coordinates and elevation reference.
///
/// Horizontal coordinates and elevations use metres. Construction performs no CRS or datum transformations.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialReference {
    projected_crs: String,
    vertical_datum: String,
    origin: [f64; 2],
}

impl SpatialReference {
    /// Creates reference metadata with the grid's lower-left origin in projected metres.
    ///
    /// The caller attests that both identifiers describe compatible input coordinates and elevations.
    ///
    /// # Errors
    ///
    /// Rejects empty identifiers and nonfinite origin coordinates.
    pub fn new(
        projected_crs: String,
        vertical_datum: String,
        origin: [f64; 2],
    ) -> Result<Self, ModelError> {
        for (field, value) in [
            ("projected_crs", projected_crs.as_str()),
            ("vertical_datum", vertical_datum.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ModelError::invalid(
                    field,
                    None,
                    "identifier is empty: supply the input reference identifier",
                ));
            }
        }
        for (index, value) in origin.iter().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::invalid(
                    "origin",
                    Some(index),
                    format!("coordinate {value} is nonfinite: supply finite projected metres"),
                ));
            }
        }
        Ok(Self {
            projected_crs,
            vertical_datum,
            origin,
        })
    }

    /// Returns the caller-attested projected CRS identifier.
    pub fn projected_crs(&self) -> &str {
        &self.projected_crs
    }

    /// Returns the caller-attested vertical datum identifier.
    pub fn vertical_datum(&self) -> &str {
        &self.vertical_datum
    }

    /// Returns the lower-left origin in projected metres.
    pub fn origin(&self) -> [f64; 2] {
        self.origin
    }
}
