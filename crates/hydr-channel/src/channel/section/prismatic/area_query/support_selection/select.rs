use super::super::super::hydraulic_geometry::GatheredBands;
use super::super::{chart::AreaQuery, support_assessment::SupportAssessment};
use super::SupportSelection;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
};

impl<R> AreaQuery<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Select rows whose complete area image proves physical support.
    /// Results remain provisional until the enclosing checked call completes.
    pub(crate) fn select_supported(
        self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
    ) -> HydraulicResult<SupportSelection<R>> {
        let assessment = SupportAssessment::assess(a, bands, &self)?;
        let c = a.client();
        let proved = c.mul(
            &c.ge(assessment.area.lower(), &assessment.zero_tensor)?,
            &c.ge(assessment.deficit.lower(), &assessment.zero_tensor)?,
        )?;
        let owner = c.nonzero(&proved)?.flatten()?;
        let query = if owner.numel() == 0 {
            None
        } else {
            let query = self.select(a, &owner)?;
            let bands = bands.select(a, &owner)?;
            let assessment = assessment.select(a, &owner)?;
            Some(assessment.finish(a, &bands, query)?)
        };
        Ok(SupportSelection {
            proved,
            owner,
            query,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::fixtures::AnchorKind;
    use super::super::super::query_fixture::{rectangle_chart, shape, snapshot};
    use super::super::super::test_support as support;
    use crate::channel::{
        flow::surveyed::error::{self, HydraulicError, HydraulicResult},
        section::{CrossSectionError, HydraulicGeometry, PrismaticSectionBatch},
    };
    use numr::{
        dtype::DType,
        ops::BinaryOps,
        runtime::{Runtime, RuntimeClient},
        tensor::Tensor,
    };
    use support::{Client, Result, hydraulic};

    const NONNEGATIVE: &str =
        "area query image is not proved nonnegative. Change the chart or tighten its certificates";
    const CAP: &str = "area query image is not proved below its physical cap. Change the chart or tighten its certificates";

    fn rejected_context<T>(result: HydraulicResult<T>) -> HydraulicResult<String> {
        match result {
            Err(HydraulicError::Geometry(CrossSectionError::InvalidInput { context, .. })) => {
                Ok(context)
            }
            Err(error) => Err(error),
            Ok(_) => Err(HydraulicError::Numerical {
                context: "Unsupported support request unexpectedly returns a query".into(),
            }),
        }
    }

    pub(in super::super) fn run<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        strict_order::<R>(client, section)?;
        repeated_rows::<R>(client, section)?;
        cap_subtract::<R>(client, section)?;
        empty::<R>(client, section)?;
        hard_errors::<R>(client, section)
    }

    fn strict_order<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        for (points, expected, mask) in [
            ([-1.0, 7.0], NONNEGATIVE, [0.0, 0.0]),
            ([7.0, -1.0], NONNEGATIVE, [0.0, 0.0]),
            ([1.0, 7.0], CAP, [1.0, 0.0]),
        ] {
            let charts = points.map(|point| rectangle_chart(AnchorKind::Zero, true, point, point));
            let input = support::input::<R>(client, section, &charts, false, false)?;
            let (context, selection) = hydraulic(error::checked::<R, _>(
                client,
                "strict support rejection order",
                |a| {
                    let bands = geometry.gather(a, &input.owners)?;
                    let raw = support::query(a, &bands, &input)?;
                    let rows = Tensor::from_slice(&[0_i64, 1], &[2], client.device())?;
                    let context = rejected_context(raw.select(a, &rows)?.prove_support(a, &bands))?;
                    Ok((context, raw.select_supported(a, &bands)?))
                },
            ))?;
            assert_eq!(context, expected);
            shape(&selection.proved, &selection.owner, 2);
            assert_eq!(selection.proved.try_to_vec::<f64>()?, mask);
            if mask == [0.0, 0.0] {
                assert_eq!(selection.owner.numel(), 0);
                assert!(selection.query.is_none());
            } else {
                assert_eq!(selection.owner.try_to_vec::<i64>()?, [0]);
                assert!(selection.query.is_some());
            }
        }
        Ok(())
    }

    fn repeated_rows<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        let charts = [1.0, 2.0, -1.0, 7.0]
            .map(|point| rectangle_chart(AnchorKind::Zero, true, point, point));
        let input = support::input::<R>(client, section, &charts, true, false)?;
        let (strict, selection) = hydraulic(error::checked::<R, _>(
            client,
            "repeated support invocation rows",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                let raw = support::query(a, &bands, &input)?;
                let identities =
                    Tensor::from_slice(&[701_i64, 702, 703, 704], &[4], client.device())?;
                client.copy_into(&raw.original_rows, &identities)?;
                let repeated = Tensor::from_slice(&[1_i64, 3, 0, 1, 2], &[5], client.device())?;
                let raw = raw.select(a, &repeated)?;
                let bands = bands.select(a, &repeated)?;
                let passing = Tensor::from_slice(&[0_i64, 2, 3], &[3], client.device())?;
                let expected_bands = bands.select(a, &passing)?;
                let strict = raw.select(a, &passing)?.prove_support(a, &expected_bands)?;
                Ok((strict, raw.select_supported(a, &bands)?))
            },
        ))?;
        shape(&selection.proved, &selection.owner, 5);
        assert_eq!(
            selection.proved.try_to_vec::<f64>()?,
            [1.0, 0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(selection.owner.try_to_vec::<i64>()?, [0, 2, 3]);
        let selected = selection
            .query
            .ok_or("Repeated support selection loses passing rows")?;
        assert_eq!(
            selected.query.original_rows.try_to_vec::<i64>()?,
            [702, 701, 702]
        );
        assert_eq!(selected.query.owners.try_to_vec::<i64>()?, [1, 1, 1]);
        assert_eq!(snapshot(&selected)?, snapshot(&strict)?);
        let before = snapshot(&selected)?;
        client.copy_into(
            &selection.owner,
            &Tensor::zeros(&[3], DType::I64, client.device())?,
        )?;
        client.copy_into(
            &selection.proved,
            &Tensor::zeros(&[5], DType::F64, client.device())?,
        )?;
        assert_eq!(snapshot(&selected)?, before);
        Ok(())
    }

    fn cap_subtract<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        let charts = [
            rectangle_chart(AnchorKind::Cap, false, 1.0, 2.0),
            rectangle_chart(AnchorKind::Cap, false, 7.0, 7.0),
        ];
        let input = support::input::<R>(client, section, &charts, false, false)?;
        let (strict, selection) = hydraulic(error::checked::<R, _>(
            client,
            "cap-deficit partial support",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                let raw = support::query(a, &bands, &input)?;
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let passing_bands = bands.select(a, &rows)?;
                let strict = raw.select(a, &rows)?.prove_support(a, &passing_bands)?;
                Ok((strict, raw.select_supported(a, &bands)?))
            },
        ))?;
        assert_eq!(selection.proved.try_to_vec::<f64>()?, [1.0, 0.0]);
        assert_eq!(selection.owner.try_to_vec::<i64>()?, [0]);
        let selected = selection
            .query
            .ok_or("Cap-deficit selection loses supported row")?;
        assert_eq!(snapshot(&selected)?, snapshot(&strict)?);
        Ok(())
    }

    fn empty<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        let input = support::input::<R>(
            client,
            section,
            &[rectangle_chart(AnchorKind::Zero, true, 1.0, 2.0)],
            false,
            false,
        )?;
        let selection = hydraulic(error::checked::<R, _>(
            client,
            "empty support selection",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                let raw = support::query(a, &bands, &input)?;
                let rows = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
                let bands = bands.select(a, &rows)?;
                raw.select(a, &rows)?.select_supported(a, &bands)
            },
        ))?;
        shape(&selection.proved, &selection.owner, 0);
        assert!(selection.proved.try_to_vec::<f64>()?.is_empty());
        assert!(selection.owner.try_to_vec::<i64>()?.is_empty());
        assert!(selection.query.is_none());
        Ok(())
    }

    fn hard_errors<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        let foreign_section = support::batch::<R>(client)?;
        let foreign_geometry = HydraulicGeometry::from_batch(&foreign_section);
        // The unsupported area cannot turn model or ownership errors into a false mask.
        let input = support::input::<R>(
            client,
            section,
            &[rectangle_chart(AnchorKind::Zero, true, -1.0, -1.0)],
            false,
            false,
        )?;
        let (model, owners) = hydraulic(error::checked::<R, _>(
            client,
            "hard support identity errors",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                let foreign = foreign_geometry.gather(a, &input.owners)?;
                let raw = support::query(a, &bands, &input)?;
                let model = rejected_context(raw.select_supported(a, &foreign))?;
                let different = Tensor::from_slice(&[2_i64], &[1], client.device())?;
                let wrong_owners = geometry.gather(a, &different)?;
                let raw = support::query(a, &bands, &input)?;
                let owners = rejected_context(raw.select_supported(a, &wrong_owners))?;
                Ok((model, owners))
            },
        ))?;
        assert_eq!(
            model,
            "area query inputs use different compiled models. Rebuild inputs from one prismatic batch"
        );
        assert_eq!(
            owners,
            "area query owners differ from the cached geometry rows. Select matching query and table rows"
        );
        let mut chart = rectangle_chart(AnchorKind::Source, true, 1.0, 1.0);
        chart.scale[0] = f64::MAX.to_bits();
        let input = support::input::<R>(client, section, &[chart], false, false)?;
        let (bands, raw) = hydraulic(error::checked::<R, _>(
            client,
            "finite scale before support range error",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                let raw = support::query(a, &bands, &input)?;
                Ok((bands, raw))
            },
        ))?;
        assert!(
            error::checked::<R, _>(client, "hard arithmetic support range", |a| raw
                .select_supported(a, &bands))
            .is_err()
        );
        Ok(())
    }
}
