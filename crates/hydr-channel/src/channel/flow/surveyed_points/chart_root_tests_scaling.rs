use crate::channel::flow::surveyed_points::source_tests::support::{Client, Result, hydraulic};
use crate::channel::{
    flow::surveyed::error,
    section::{AreaAnchor, AreaDirection, AreaQuery, HydraulicGeometry},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    for (width_exponent, depth_exponent, area_exponent, gravity_exponent, bound_exponent) in
        [(500, 250, 750, -1074, -411), (-500, -250, -750, 1000, 376)]
    {
        let width = 2.0_f64.powi(width_exponent);
        let depth = 2.0_f64.powi(depth_exponent);
        let area = 2.0_f64.powi(area_exponent);
        let gravity = if gravity_exponent == -1074 {
            f64::from_bits(1)
        } else {
            2.0_f64.powi(gravity_exponent)
        };
        let context = format!(
            "Derivative profile width=2^{width_exponent}, cap_depth=2^{depth_exponent}, scale=2^{area_exponent}, gravity=2^{gravity_exponent}"
        );
        let section = super::support::rectangle::<R>(client, width, depth)
            .map_err(|error| format!("{context}: {error}"))?;
        let geometry = HydraulicGeometry::from_batch(&section);
        hydraulic(error::checked::<R, _>(
            client,
            "chart-scaled derivative without raw range loss",
            |a| {
                let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let bands = geometry.gather(a, &owners)?;
                let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                let half = Tensor::full_scalar(&[1], DType::F64, 0.5, client.device())?;
                let scale =
                    a.expansion_point(&Tensor::from_slice(&[area], &[1], client.device())?)?;
                // Interior support avoids requiring a zero cap distance from independent certificates.
                let parameter = a.bounds(&zero, &half)?;
                let query = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let lower = crate::channel::flow::surveyed_points::chart_root_derivative::lower(
                    a, &bands, &query, gravity,
                )?;
                let lower = lower.item::<f64>()?;
                assert!(lower.is_finite() && lower > 0.0 && lower <= 2.0_f64.powi(bound_exponent));
                Ok(())
            },
        ))
        .map_err(|error| format!("{context}: {error}"))?;
    }
    Ok(())
}
