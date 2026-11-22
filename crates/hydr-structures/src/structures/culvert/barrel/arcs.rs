//! Constructors of the barrels built from tangent circular arcs.
//!
//! Each derived radius follows from tangency of neighbouring arcs. The closed forms keep every
//! difference between positive terms:
//!
//! | Shape | Derived radius |
//! | --- | --- |
//! | Horizontal ellipse | `R = r_c + (x_c^2 + e^2) / (2 e)`, `x_c = B/2 - r_c`, `e = D/2 - r_c` |
//! | Vertical ellipse | `r_c = r + (f^2 + e^2) / (2 f)`, `f = B/2 - r`, `e = D/2 - r` |
//! | Pipe-arch | `r_t = r_c + (x_c^2 + g^2) / (2 g)`, `g = D - y_c - r_c`, `y_c = r_b - h` |
//! | Flat-floor arch | `r = (B^2/4 + D^2) / (2 D)` |
//!
//! The pipe-arch corner centres sit at height `y_c`, with
//! `h^2 = (r_b - B/2) (r_b + B/2 - 2 r_c)`. Its rules on `y_c` hold for the exact section:
//! the host enclosure certifies them, so the kernel enclosures of the same forms agree.

use super::enclosure::Enclosure;
use super::{CulvertBarrel, CulvertBarrelShape, dimension, scales};
use crate::structures::StructureError;

impl CulvertBarrel {
    /// Admits a four-arc horizontal ellipse of span `span_m`, rise `rise_m` and side radius
    /// `side_radius_m`.
    ///
    /// The invert and crown arcs share one radius, derived from tangency. Rejects a span at or
    /// below the rise and a side radius at or above half the rise. Also rejects as the box
    /// constructor does, and when a radius cubed, scaled by 64, overflows F64.
    pub fn horizontal_ellipse(
        span_m: f64,
        rise_m: f64,
        side_radius_m: f64,
    ) -> Result<Self, StructureError> {
        dimension("span_m", "span", span_m)?;
        dimension("rise_m", "rise", rise_m)?;
        dimension("side_radius_m", "side radius", side_radius_m)?;
        if span_m <= rise_m {
            return Err(StructureError::definition(format!(
                "span_m={span_m:?},rise_m={rise_m:?}: a horizontal ellipse needs span above rise. Use vertical_ellipse for a rise above the span"
            )));
        }
        if side_radius_m >= 0.5 * rise_m {
            return Err(StructureError::definition(format!(
                "rise_m={rise_m:?},side_radius_m={side_radius_m:?}: a horizontal ellipse needs a side radius below half the rise. Supply a smaller side radius"
            )));
        }
        let fields = format!("span_m={span_m:?},rise_m={rise_m:?},side_radius_m={side_radius_m:?}");
        scales(span_m, rise_m, &[("side", side_radius_m)], &fields)?;
        let radius = ellipse_radius(span_m, rise_m, side_radius_m);
        scales(
            span_m,
            rise_m,
            &[("invert and crown", radius.upper)],
            &fields,
        )?;
        Ok(Self::new(
            CulvertBarrelShape::HorizontalEllipse,
            span_m,
            rise_m,
            &[side_radius_m],
        ))
    }

    /// Admits a four-arc vertical ellipse of span `span_m`, rise `rise_m` and end radius
    /// `end_radius_m`.
    ///
    /// The two side arcs share one radius, derived from tangency. Rejects a rise at or below
    /// the span and an end radius at or above half the span. Also rejects as the box
    /// constructor does, and when a radius cubed, scaled by 64, overflows F64.
    pub fn vertical_ellipse(
        span_m: f64,
        rise_m: f64,
        end_radius_m: f64,
    ) -> Result<Self, StructureError> {
        dimension("span_m", "span", span_m)?;
        dimension("rise_m", "rise", rise_m)?;
        dimension("end_radius_m", "end radius", end_radius_m)?;
        if rise_m <= span_m {
            return Err(StructureError::definition(format!(
                "span_m={span_m:?},rise_m={rise_m:?}: a vertical ellipse needs rise above span. Use horizontal_ellipse for a span above the rise"
            )));
        }
        if end_radius_m >= 0.5 * span_m {
            return Err(StructureError::definition(format!(
                "span_m={span_m:?},end_radius_m={end_radius_m:?}: a vertical ellipse needs an end radius below half the span. Supply a smaller end radius"
            )));
        }
        let fields = format!("span_m={span_m:?},rise_m={rise_m:?},end_radius_m={end_radius_m:?}");
        scales(span_m, rise_m, &[("end", end_radius_m)], &fields)?;
        // The side radius is the horizontal-ellipse radius of the section turned a quarter turn.
        let radius = ellipse_radius(rise_m, span_m, end_radius_m);
        scales(span_m, rise_m, &[("side", radius.upper)], &fields)?;
        Ok(Self::new(
            CulvertBarrelShape::VerticalEllipse,
            span_m,
            rise_m,
            &[end_radius_m],
        ))
    }

    /// Admits a pipe-arch of span `span_m`, rise `rise_m`, corner radius `corner_radius_m` and
    /// bottom radius `bottom_radius_m`.
    ///
    /// The top radius is derived from tangency. Rejects a corner radius at or above half the
    /// span, a bottom radius at or below half the span, a rise at or below the corner-centre
    /// height plus the corner radius, and a rise above the corner centres of more than half
    /// the span, which would put the widest point on the top arc. Also rejects as the box
    /// constructor does, and when a radius cubed, scaled by 64, overflows F64.
    pub fn pipe_arch(
        span_m: f64,
        rise_m: f64,
        corner_radius_m: f64,
        bottom_radius_m: f64,
    ) -> Result<Self, StructureError> {
        dimension("span_m", "span", span_m)?;
        dimension("rise_m", "rise", rise_m)?;
        dimension("corner_radius_m", "corner radius", corner_radius_m)?;
        dimension("bottom_radius_m", "bottom radius", bottom_radius_m)?;
        if corner_radius_m >= 0.5 * span_m {
            return Err(StructureError::definition(format!(
                "span_m={span_m:?},corner_radius_m={corner_radius_m:?}: a pipe-arch needs a corner radius below half the span. Supply a smaller corner radius"
            )));
        }
        if bottom_radius_m <= 0.5 * span_m {
            return Err(StructureError::definition(format!(
                "span_m={span_m:?},bottom_radius_m={bottom_radius_m:?}: a pipe-arch needs a bottom radius above half the span. Supply a larger bottom radius"
            )));
        }
        let fields = format!(
            "span_m={span_m:?},rise_m={rise_m:?},corner_radius_m={corner_radius_m:?},bottom_radius_m={bottom_radius_m:?}"
        );
        let given = [("corner", corner_radius_m), ("bottom", bottom_radius_m)];
        scales(span_m, rise_m, &given, &fields)?;
        let layout = PipeArch::new(span_m, rise_m, corner_radius_m, bottom_radius_m);
        if layout.gap.lower.is_nan() || layout.gap.lower <= 0.0 {
            let top = Enclosure::exact(corner_radius_m).add(layout.corner_height);
            return Err(StructureError::definition(format!(
                "{fields}: a pipe-arch needs a rise above the corner-centre height plus the corner radius, [{:?}, {:?}] m. Supply a larger rise or a larger bottom radius",
                top.lower, top.upper
            )));
        }
        if layout.reach.lower.is_nan() || layout.reach.lower < 0.0 {
            let above = Enclosure::exact(rise_m).sub(layout.corner_height);
            return Err(StructureError::definition(format!(
                "{fields}: a pipe-arch needs the rise above the corner centres, [{:?}, {:?}] m, at most half the span. Supply a smaller rise or a smaller bottom radius",
                above.lower, above.upper
            )));
        }
        let top = layout.top_radius();
        scales(span_m, rise_m, &[("top", top.upper)], &fields)?;
        Ok(Self::new(
            CulvertBarrelShape::PipeArch,
            span_m,
            rise_m,
            &[corner_radius_m, bottom_radius_m],
        ))
    }

    /// Admits a flat-floor arch of floor chord `span_m` and rise `rise_m`.
    ///
    /// The arc radius is derived. Rejects a rise above the span or below a quarter of the
    /// span. Also rejects as the box constructor does.
    pub fn arch(span_m: f64, rise_m: f64) -> Result<Self, StructureError> {
        dimension("span_m", "span", span_m)?;
        dimension("rise_m", "rise", rise_m)?;
        if rise_m > span_m {
            return Err(StructureError::definition(format!(
                "span_m={span_m:?},rise_m={rise_m:?}: a flat-floor arch needs a rise of at most the span. Supply a rise between a quarter of the span and the span"
            )));
        }
        if rise_m < 0.25 * span_m {
            return Err(StructureError::definition(format!(
                "span_m={span_m:?},rise_m={rise_m:?}: a flat-floor arch needs a rise of at least a quarter of the span. Supply a rise between a quarter of the span and the span"
            )));
        }
        let fields = format!("span_m={span_m:?},rise_m={rise_m:?}");
        scales(span_m, rise_m, &[], &fields)?;
        let radius = arch_radius(span_m, rise_m);
        scales(span_m, rise_m, &[("arc", radius.upper)], &fields)?;
        Ok(Self::new(CulvertBarrelShape::Arch, span_m, rise_m, &[]))
    }
}

/// Encloses `R = r_c + (x_c^2 + e^2) / (2 e)` of a horizontal ellipse.
fn ellipse_radius(span_m: f64, rise_m: f64, side_m: f64) -> Enclosure {
    let half = Enclosure::exact(span_m).mul(Enclosure::exact(0.5));
    let half_rise = Enclosure::exact(rise_m).mul(Enclosure::exact(0.5));
    let side = Enclosure::exact(side_m);
    let offset = half.sub(side);
    let lift = half_rise.sub(side);
    let sum = offset.mul(offset).add(lift.mul(lift));
    let twice = Enclosure::exact(2.0).mul(lift);
    side.add(sum.div(twice))
}

/// Encloses `r = (B^2/4 + D^2) / (2 D)` of a flat-floor arch.
fn arch_radius(span_m: f64, rise_m: f64) -> Enclosure {
    let half = Enclosure::exact(span_m).mul(Enclosure::exact(0.5));
    let rise = Enclosure::exact(rise_m);
    let sum = half.mul(half).add(rise.mul(rise));
    sum.div(Enclosure::exact(2.0).mul(rise))
}

/// Pipe-arch quantities that decide admission.
struct PipeArch {
    offset: Enclosure,
    corner: Enclosure,
    /// `y_c = r_b - h`.
    corner_height: Enclosure,
    /// `g = D - y_c - r_c`, positive when the top arc closes above the corner arcs.
    gap: Enclosure,
    /// `B/2 - (D - y_c)`, nonnegative when the widest point lies on the corner arcs.
    reach: Enclosure,
}

impl PipeArch {
    fn new(span_m: f64, rise_m: f64, corner_m: f64, bottom_m: f64) -> Self {
        let half = Enclosure::exact(span_m).mul(Enclosure::exact(0.5));
        let corner = Enclosure::exact(corner_m);
        let bottom = Enclosure::exact(bottom_m);
        let offset = half.sub(corner);
        let doubled = Enclosure::exact(2.0).mul(corner);
        let lift = bottom.sub(half).mul(bottom.add(half).sub(doubled));
        let rise_to_centre = lift.nonnegative().sqrt();
        let corner_height = bottom.sub(rise_to_centre);
        let above = Enclosure::exact(rise_m).sub(corner_height);
        Self {
            offset,
            corner,
            corner_height,
            gap: above.sub(corner),
            reach: half.sub(above),
        }
    }

    /// `r_t = r_c + (x_c^2 + g^2) / (2 g)`.
    fn top_radius(&self) -> Enclosure {
        let sum = self.offset.mul(self.offset).add(self.gap.mul(self.gap));
        let twice = Enclosure::exact(2.0).mul(self.gap);
        self.corner.add(sum.div(twice))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureInputErrorKind;

    fn message(result: Result<CulvertBarrel, StructureError>) -> Option<String> {
        match result {
            Err(StructureError::InvalidInput {
                kind: StructureInputErrorKind::Definition,
                context,
                element: None,
            }) => Some(context),
            _ => None,
        }
    }

    #[test]
    fn every_rule_rejects_with_its_values_and_action() {
        let cases = [
            (
                CulvertBarrel::horizontal_ellipse(1.2, 1.5, 0.3),
                "span_m=1.2,rise_m=1.5: a horizontal ellipse needs span above rise. Use vertical_ellipse for a rise above the span",
            ),
            (
                CulvertBarrel::horizontal_ellipse(2.0, 1.0, 0.5),
                "rise_m=1.0,side_radius_m=0.5: a horizontal ellipse needs a side radius below half the rise. Supply a smaller side radius",
            ),
            (
                CulvertBarrel::horizontal_ellipse(2.0, 1.0, -0.1),
                "side_radius_m=-0.1: supply a finite positive barrel side radius in metres",
            ),
            (
                CulvertBarrel::vertical_ellipse(1.5, 1.2, 0.3),
                "span_m=1.5,rise_m=1.2: a vertical ellipse needs rise above span. Use horizontal_ellipse for a span above the rise",
            ),
            (
                CulvertBarrel::vertical_ellipse(1.0, 2.0, 0.5),
                "span_m=1.0,end_radius_m=0.5: a vertical ellipse needs an end radius below half the span. Supply a smaller end radius",
            ),
            (
                CulvertBarrel::vertical_ellipse(1.0, 2.0, f64::NAN),
                "end_radius_m=NaN: supply a finite positive barrel end radius in metres",
            ),
            (
                CulvertBarrel::pipe_arch(2.0, 1.5, 1.0, 2.0),
                "span_m=2.0,corner_radius_m=1.0: a pipe-arch needs a corner radius below half the span. Supply a smaller corner radius",
            ),
            (
                CulvertBarrel::pipe_arch(2.0, 1.5, 0.5, 1.0),
                "span_m=2.0,bottom_radius_m=1.0: a pipe-arch needs a bottom radius above half the span. Supply a larger bottom radius",
            ),
            (
                CulvertBarrel::pipe_arch(2.0, 1.0, 0.5, 1.01),
                "span_m=2.0,rise_m=1.0,corner_radius_m=0.5,bottom_radius_m=1.01: a pipe-arch needs a rise above the corner-centre height plus the corner radius",
            ),
            (
                CulvertBarrel::pipe_arch(2.0, 1.8, 0.5, 2.0),
                "span_m=2.0,rise_m=1.8,corner_radius_m=0.5,bottom_radius_m=2.0: a pipe-arch needs the rise above the corner centres",
            ),
            (
                CulvertBarrel::pipe_arch(2.0, 1.5, 0.0, 2.0),
                "corner_radius_m=0.0: supply a finite positive barrel corner radius in metres",
            ),
            (
                CulvertBarrel::pipe_arch(2.0, 1.5, 0.5, f64::INFINITY),
                "bottom_radius_m=inf: supply a finite positive barrel bottom radius in metres",
            ),
            (
                CulvertBarrel::arch(2.0, 2.5),
                "span_m=2.0,rise_m=2.5: a flat-floor arch needs a rise of at most the span. Supply a rise between a quarter of the span and the span",
            ),
            (
                CulvertBarrel::arch(2.0, 0.4),
                "span_m=2.0,rise_m=0.4: a flat-floor arch needs a rise of at least a quarter of the span. Supply a rise between a quarter of the span and the span",
            ),
            (
                CulvertBarrel::arch(-2.0, 0.4),
                "span_m=-2.0: supply a finite positive barrel span in metres",
            ),
        ];
        for (result, expected) in cases {
            let text = message(result);
            assert!(
                text.as_deref().is_some_and(|t| t.contains(expected)),
                "{text:?} lacks {expected:?}"
            );
        }
    }

    #[test]
    fn radius_and_scale_overflow_reject() {
        let cases = [
            (
                CulvertBarrel::pipe_arch(2.0, 1.5, 0.5, 1e110),
                "the bottom radius cubed times 64 overflows F64",
            ),
            (
                CulvertBarrel::horizontal_ellipse(1e100, 1e-100, 4e-101),
                "the invert and crown radius cubed times 64 overflows F64",
            ),
            (
                CulvertBarrel::vertical_ellipse(1e-100, 1e100, 4e-101),
                "the side radius cubed times 64 overflows F64",
            ),
            (
                CulvertBarrel::arch(1e300, 1e300),
                "B D times 64 overflows F64",
            ),
            (
                CulvertBarrel::arch(1e-300, 1e-300),
                "rise times 2^-30 is subnormal",
            ),
        ];
        for (result, expected) in cases {
            let text = message(result);
            assert!(
                text.as_deref().is_some_and(|t| t.contains(expected)),
                "{text:?} lacks {expected:?}"
            );
        }
    }

    #[test]
    fn rule_limits_admit_the_nearest_valid_dimensions() -> Result<(), StructureError> {
        // Exact comparisons: the nearest F64 inside each limit admits.
        CulvertBarrel::horizontal_ellipse(1.0_f64.next_up(), 1.0, 0.25)?;
        CulvertBarrel::horizontal_ellipse(2.0, 1.0, 0.5_f64.next_down())?;
        CulvertBarrel::vertical_ellipse(1.0, 1.0_f64.next_up(), 0.25)?;
        CulvertBarrel::vertical_ellipse(1.0, 2.0, 0.5_f64.next_down())?;
        // A bottom radius just above half the span lifts the corner centres to B/2.
        CulvertBarrel::pipe_arch(2.0, 1.8, 0.5, 1.0_f64.next_up())?;
        CulvertBarrel::arch(2.0, 2.0)?;
        CulvertBarrel::arch(2.0, 0.5)?;
        Ok(())
    }

    #[test]
    fn admitted_arc_barrels_report_their_inputs() -> Result<(), StructureError> {
        let ellipse = CulvertBarrel::horizontal_ellipse(1.524, 0.9652, 0.419_475_92)?;
        assert_eq!(ellipse.shape(), CulvertBarrelShape::HorizontalEllipse);
        assert_eq!((ellipse.span_m(), ellipse.rise_m()), (1.524, 0.9652));
        assert_eq!(ellipse.given_radius_m(), Some(&[0.419_475_92][..]));
        let turned = CulvertBarrel::vertical_ellipse(0.9652, 1.524, 0.419_475_92)?;
        assert_eq!(turned.shape(), CulvertBarrelShape::VerticalEllipse);
        assert_eq!(turned.given_radius_m(), Some(&[0.419_475_92][..]));
        let pipe = CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?;
        assert_eq!(pipe.shape(), CulvertBarrelShape::PipeArch);
        assert_eq!(pipe.given_radius_m(), Some(&[0.4572, 1.938_528][..]));
        let arch = CulvertBarrel::arch(3.048, 1.6002)?;
        assert_eq!(arch.shape(), CulvertBarrelShape::Arch);
        assert_eq!((arch.span_m(), arch.rise_m()), (3.048, 1.6002));
        assert_eq!(arch.given_radius_m(), None);
        for barrel in [ellipse, turned, pipe, arch] {
            assert!(barrel.shape().arced());
        }
        Ok(())
    }

    #[test]
    fn pipe_arch_rules_certify_the_aisi_layout() {
        // AISI Table 1.9 row 1 in metres: corner centres near 21 in above the invert.
        let layout = PipeArch::new(1.8542, 1.397, 0.4572, 1.938_528);
        let inches = 0.0254;
        assert!(layout.corner_height.lower > 20.9 * inches);
        assert!(layout.corner_height.upper < 21.1 * inches);
        assert!(layout.gap.lower > 0.0);
        assert!(layout.reach.lower > 0.0);
        let top = layout.top_radius();
        assert!(top.lower > 36.0 * inches && top.upper < 37.5 * inches);
    }
}
