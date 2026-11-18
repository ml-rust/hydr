//! Endpoint cutoffs per structure law and pool attachment admission.

use crate::structures::{DirectionPolicy, StructureElement, StructureLaw};

/// Stage at or below which each endpoint of `element` stops giving water, `[A, B]`.
///
/// Each cutoff starts at the attachment bed. A weir raises both to its crest, a gate to its sill,
/// and an orifice to its crown. A pump raises only its source side to its minimum suction stage.
/// The source is B for `BtoAOnly` and A otherwise.
pub(crate) fn endpoint_cutoffs(element: &StructureElement, beds: [f64; 2]) -> [f64; 2] {
    let mut cuts = beds;
    match &element.law {
        StructureLaw::SuppressedRectangularWeir(p) => {
            cuts[0] = cuts[0].max(p.crest_m());
            cuts[1] = cuts[1].max(p.crest_m());
        }
        StructureLaw::HydrostaticRectangularGate(p) => {
            cuts[0] = cuts[0].max(p.sill_m());
            cuts[1] = cuts[1].max(p.sill_m());
        }
        StructureLaw::SubmergedOrifice(p) => {
            cuts[0] = cuts[0].max(p.crown_m());
            cuts[1] = cuts[1].max(p.crown_m());
        }
        StructureLaw::Pump(p) => {
            let source = if element.direction == DirectionPolicy::BtoAOnly {
                1
            } else {
                0
            };
            cuts[source] = cuts[source].max(p.minimum_suction_stage_m());
        }
    }
    cuts
}

/// The structure datum equals the pool datum and the finite attachment bed sits at or above the
/// pool origin.
pub(crate) fn pool_attachment_admitted(
    structure_datum: &str,
    pool_datum: &str,
    bed_m: f64,
    origin_m: f64,
) -> bool {
    structure_datum == pool_datum && bed_m.is_finite() && bed_m >= origin_m
}

#[cfg(test)]
mod tests {
    use super::pool_attachment_admitted;

    #[test]
    fn attachment_admission_needs_datum_finite_bed_and_origin() {
        assert!(pool_attachment_admitted("d", "d", 1.0, 1.0));
        assert!(!pool_attachment_admitted("d", "e", 1.0, 0.0));
        assert!(!pool_attachment_admitted("d", "d", f64::NAN, 0.0));
        assert!(!pool_attachment_admitted("d", "d", f64::INFINITY, 0.0));
        assert!(!pool_attachment_admitted("d", "d", 0.5, 1.0));
    }
}
