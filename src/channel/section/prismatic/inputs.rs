use super::topology::{Band, Profile, Topology};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

pub(super) struct BankInputs<R: Runtime<DType = DType>> {
    pub left_station: Tensor<R>,
    pub right_station: Tensor<R>,
    pub low_elevation: Tensor<R>,
    pub high_elevation: Tensor<R>,
    pub vertical: Tensor<R>,
}

pub(super) struct Inputs<R: Runtime<DType = DType>> {
    pub bed: Tensor<R>,
    pub bottom_left: Tensor<R>,
    pub bottom_right: Tensor<R>,
    pub pointed: Tensor<R>,
    pub valid: Tensor<R>,
    pub band_bottom: Tensor<R>,
    pub band_top: Tensor<R>,
    pub left_bank: BankInputs<R>,
    pub right_bank: BankInputs<R>,
}

impl<R: Runtime<DType = DType>> Inputs<R> {
    pub fn new(client: &R::Client, topology: &Topology) -> ArithmeticResult<Self> {
        let rows = topology.profiles.len();
        let scalar_rows = |field: fn(&Profile) -> f64| {
            let values: Vec<_> = topology.profiles.iter().map(field).collect();
            Tensor::from_slice(&values, &[rows], client.device())
        };
        let bands = topology.maximum_bands;
        let entries =
            rows.checked_mul(bands)
                .ok_or_else(|| numr::error::Error::InvalidArgument {
                    arg: "band_table",
                    reason: "Padded band count overflows usize. Split the profile batch.".into(),
                })?;
        let band_rows = |field: fn(&Profile, Option<&Band>) -> f64| {
            let mut values = Vec::with_capacity(entries);
            for index in 0..bands {
                for profile in &topology.profiles {
                    values.push(field(profile, profile.bands.get(index)));
                }
            }
            Tensor::from_slice(&values, &[bands, rows], client.device())
        };
        Ok(Self {
            bed: scalar_rows(|profile| profile.bed_elevation)?,
            bottom_left: scalar_rows(|profile| profile.bottom_left_station)?,
            bottom_right: scalar_rows(|profile| profile.bottom_right_station)?,
            pointed: scalar_rows(|profile| f64::from(u8::from(profile.pointed)))?,
            valid: band_rows(|_, band| f64::from(u8::from(band.is_some())))?,
            band_bottom: band_rows(|profile, band| {
                band.map_or(profile.bed_elevation, |band| band.bottom_elevation)
            })?,
            band_top: band_rows(|profile, band| {
                band.map_or(profile.bed_elevation, |band| band.top_elevation)
            })?,
            left_bank: BankInputs {
                left_station: band_rows(|_, band| {
                    band.map_or(0.0, |band| band.left_bank.left_station)
                })?,
                right_station: band_rows(|_, band| {
                    band.map_or(0.0, |band| band.left_bank.right_station)
                })?,
                low_elevation: band_rows(|_, band| {
                    band.map_or(0.0, |band| band.left_bank.low_elevation)
                })?,
                high_elevation: band_rows(|_, band| {
                    band.map_or(1.0, |band| band.left_bank.high_elevation)
                })?,
                vertical: band_rows(|_, band| {
                    f64::from(u8::from(band.is_none_or(|band| band.left_bank.vertical())))
                })?,
            },
            right_bank: BankInputs {
                left_station: band_rows(|_, band| {
                    band.map_or(0.0, |band| band.right_bank.left_station)
                })?,
                right_station: band_rows(|_, band| {
                    band.map_or(0.0, |band| band.right_bank.right_station)
                })?,
                low_elevation: band_rows(|_, band| {
                    band.map_or(0.0, |band| band.right_bank.low_elevation)
                })?,
                high_elevation: band_rows(|_, band| {
                    band.map_or(1.0, |band| band.right_bank.high_elevation)
                })?,
                vertical: band_rows(|_, band| {
                    f64::from(u8::from(band.is_none_or(|band| band.right_bank.vertical())))
                })?,
            },
        })
    }
}

pub(super) fn column<R: Runtime<DType = DType>>(
    table: &Tensor<R>,
    band: usize,
    rows: usize,
) -> ArithmeticResult<Tensor<R>> {
    Ok(table.narrow(0, band, 1)?.reshape(&[rows])?)
}
