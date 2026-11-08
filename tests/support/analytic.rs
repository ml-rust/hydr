//! Point samples of independent shallow-water analytical solutions.
//!
//! Riemann relations: <https://www.clawpack.org/riemann_book/html/Shallow_water.html>.
//! Moving-bowl reference: <https://arxiv.org/abs/1110.0288>.
//! These evaluations use no production flux or wave-bound implementation.
//! Inputs require finite coordinates, finite time, and finite positive gravity.
//! Riemann functions accept nonnegative time and select the right state at the initial discontinuity.
//! Samples represent points, rather than finite-volume cell averages.

/// Returns `(h, q)` for unit-depth water released onto a dry bed at `x = 0`.
pub fn ritter(x: f64, t: f64, g: f64) -> (f64, f64) {
    if t <= 0.0 {
        return (if x < 0.0 { 1.0 } else { 0.0 }, 0.0);
    }
    let c = g.sqrt();
    let speed = x / t;
    if speed <= -c {
        (1.0, 0.0)
    } else if speed >= 2.0 * c {
        (0.0, 0.0)
    } else {
        let local_c = (2.0 * c - speed) / 3.0;
        let h = local_c * local_c / g;
        let u = 2.0 * (c + speed) / 3.0;
        (h, h * u)
    }
}

/// Returns `(h, q)` for a stationary dam between depths `1` and `0.25`.
pub fn stoker(x: f64, t: f64, g: f64) -> (f64, f64) {
    const RIGHT_DEPTH: f64 = 0.25;
    if t <= 0.0 {
        return (if x < 0.0 { 1.0 } else { RIGHT_DEPTH }, 0.0);
    }

    // Divide the rarefaction/shock matching equation by sqrt(g).
    let mut lower: f64 = RIGHT_DEPTH;
    let mut upper: f64 = 1.0;
    for _ in 0..256 {
        let depth = 0.5 * (lower + upper);
        let residual = 2.0 * (depth.sqrt() - 1.0)
            + (depth - RIGHT_DEPTH) * (0.5 * (1.0 / depth + 1.0 / RIGHT_DEPTH)).sqrt();
        if residual > 0.0 {
            upper = depth;
        } else {
            lower = depth;
        }
    }
    let star_h = 0.5 * (lower + upper);
    let c_left = g.sqrt();
    let c_star = (g * star_h).sqrt();
    let star_u = 2.0 * (c_left - c_star);
    // Rankine-Hugoniot mass balance gives the shock speed, distinct from star_u.
    let shock_speed = star_h * star_u / (star_h - RIGHT_DEPTH);
    let speed = x / t;
    if speed <= -c_left {
        (1.0, 0.0)
    } else if speed < star_u - c_star {
        let local_c = (2.0 * c_left - speed) / 3.0;
        let h = local_c * local_c / g;
        let u = 2.0 * (c_left + speed) / 3.0;
        (h, h * u)
    } else if speed < shock_speed {
        (star_h, star_h * star_u)
    } else {
        (RIGHT_DEPTH, 0.0)
    }
}

/// Returns `(h, q)` for separating unit-depth states with velocities `-8` and `8`.
/// Gravity above `16` produces a wet central state instead of vacuum.
pub fn vacuum(x: f64, t: f64, g: f64) -> (f64, f64) {
    if t <= 0.0 {
        return (1.0, if x < 0.0 { -8.0 } else { 8.0 });
    }
    let c = g.sqrt();
    let speed = x / t;
    let centre_c = (c - 4.0).max(0.0);
    let left_tail = if c <= 4.0 { -8.0 + 2.0 * c } else { -centre_c };
    let right_tail = -left_tail;
    if speed <= -8.0 - c {
        (1.0, -8.0)
    } else if speed < left_tail {
        let local_c = (-8.0 + 2.0 * c - speed) / 3.0;
        let h = local_c * local_c / g;
        let u = (-8.0 + 2.0 * c + 2.0 * speed) / 3.0;
        (h, h * u)
    } else if speed <= right_tail {
        (centre_c * centre_c / g, 0.0)
    } else if speed < 8.0 + c {
        let local_c = (speed - 8.0 + 2.0 * c) / 3.0;
        let h = local_c * local_c / g;
        let u = (8.0 - 2.0 * c + 2.0 * speed) / 3.0;
        (h, h * u)
    } else {
        (1.0, 8.0)
    }
}

/// Returns `(h, qx, qy)` for a circularly translating pool over `z = 0.1 (x² + y²)`.
/// The pool has central depth `0.1`, radius `1`, and centre `(0.2 cos(omega t), 0.2 sin(omega t))`.
pub fn thacker(x: f64, y: f64, t: f64, g: f64) -> (f64, f64, f64) {
    let omega = (0.2 * g).sqrt();
    let phase = omega * t;
    let centre_x = 0.2 * phase.cos();
    let centre_y = 0.2 * phase.sin();
    let dx = x - centre_x;
    let dy = y - centre_y;
    let h = (0.1 - 0.1 * (dx * dx + dy * dy)).max(0.0);
    let u = -omega * centre_y;
    let v = omega * centre_x;
    (h, h * u, h * v)
}
