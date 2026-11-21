//! Laws of every link under one operation: the static operation of a manual coordinator, or the
//! interval operation of one attempt.

use super::LinkLaw;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::structures::{
    CulvertClient, StructureElement, StructureOperatingPoint, StructureOperation,
};
use hydr_structures::internal::CouplingStructures;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, IndexingOps, ReduceOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;

/// Link laws, their classes, and their operations, every list in link table order.
pub(in super::super) struct LinkOperations<R: Runtime<DType = DType>> {
    /// Law tensors of every link, or `None` without links.
    structures: Option<CouplingStructures<R>>,
    /// Law class and cutoffs of every link under its operation.
    pub laws: Vec<LinkLaw>,
    /// Operation of every link.
    pub points: Vec<StructureOperatingPoint>,
}

impl<R: Runtime<DType = DType>> LinkOperations<R> {
    /// Laws of `elements` under their declared `points`, on attachment `beds`, on `client`.
    ///
    /// `allowance` is the head allowance `(absolute, relative)` of the coordinator.
    ///
    /// Culvert laws build their certified rows here once.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] for a law the structures crate rejects.
    pub fn declared(
        client: &R::Client,
        elements: &[StructureElement],
        points: Vec<StructureOperatingPoint>,
        beds: &[[f64; 2]],
        allowance: (f64, f64),
    ) -> Result<Self, SurfaceChannelError>
    where
        R::Client: CulvertClient<R>,
    {
        let laws = elements
            .iter()
            .zip(&points)
            .zip(beds)
            .map(|((element, &point), &beds)| LinkLaw::new(element, point, beds, allowance))
            .collect();
        let structures = if elements.is_empty() {
            None
        } else {
            let operations = elements
                .iter()
                .zip(&points)
                .map(|(element, &operating_point)| StructureOperation {
                    structure_id: element.id.clone(),
                    operating_point,
                })
                .collect();
            Some(
                CouplingStructures::with_culverts(client.clone(), elements.to_vec(), operations)
                    .map_err(|error| {
                        SurfaceChannelError::input(format!("link structures: {error}"))
                    })?,
            )
        };
        Ok(Self {
            structures,
            laws,
            points,
        })
    }

    /// The same laws under `points`, one per link in table order.
    ///
    /// Law tensors keep their checked elements and take new operation tensors.
    pub fn operated(
        &self,
        elements: &[StructureElement],
        points: Vec<StructureOperatingPoint>,
        allowance: (f64, f64),
    ) -> Result<Self, SurfaceChannelError>
    where
        R::Client: BinaryOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ReduceOps<R>
            + TypeConversionOps<R>
            + UnaryOps<R>,
    {
        if elements.len() != self.laws.len() || points.len() != self.laws.len() {
            return Err(SurfaceChannelError::Numerical(format!(
                "link operations={}, elements={}, links={}: supply one operation per link",
                points.len(),
                elements.len(),
                self.laws.len()
            )));
        }
        let operations: Vec<StructureOperation> = elements
            .iter()
            .zip(&points)
            .map(|(element, &operating_point)| StructureOperation {
                structure_id: element.id.clone(),
                operating_point,
            })
            .collect();
        let structures = self
            .structures
            .as_ref()
            .map(|structures| structures.with_operations(&operations))
            .transpose()
            .map_err(|error| SurfaceChannelError::Numerical(format!("link operations: {error}")))?;
        let laws = elements
            .iter()
            .zip(&points)
            .zip(&self.laws)
            .map(|((element, &point), law)| LinkLaw::new(element, point, law.bed, allowance))
            .collect();
        Ok(Self {
            structures,
            laws,
            points,
        })
    }

    /// Law tensors of every link. A coordinator with ports only has none.
    pub fn structures(&self) -> Result<&CouplingStructures<R>, SurfaceChannelError> {
        self.structures.as_ref().ok_or_else(|| {
            SurfaceChannelError::Numerical("a coordinator without links has no weir laws".into())
        })
    }

    /// Exact bits of every link operation: a tag, then its value, per link.
    ///
    /// Trial memo keys hold them, so no kept child trial crosses an operation.
    pub fn bits(&self) -> Vec<u64> {
        let mut bits = Vec::with_capacity(2 * self.points.len());
        for point in &self.points {
            let (tag, value) = match *point {
                StructureOperatingPoint::Passive => (0, 0.0),
                StructureOperatingPoint::GateOpening { opening_m } => (1, opening_m),
                StructureOperatingPoint::PumpRunning { speed_ratio } => (2, speed_ratio),
                StructureOperatingPoint::PumpStoppedIsolated => (3, 0.0),
            };
            bits.push(tag);
            bits.push(value.to_bits());
        }
        bits
    }
}
