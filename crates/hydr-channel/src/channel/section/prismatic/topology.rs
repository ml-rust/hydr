use super::super::{CrossSectionDefinition, CrossSectionError, CrossSectionPoint};
use super::config::check_work;

#[derive(Clone, Copy)]
pub(super) struct BankSegment {
    pub left_station: f64,
    pub right_station: f64,
    pub low_elevation: f64,
    pub high_elevation: f64,
}

impl BankSegment {
    pub fn vertical(&self) -> bool {
        self.left_station == self.right_station
    }
}

pub(super) struct Band {
    pub bottom_elevation: f64,
    pub top_elevation: f64,
    pub left_bank: BankSegment,
    pub right_bank: BankSegment,
}

pub(super) struct Profile {
    pub bed_elevation: f64,
    pub bottom_left_station: f64,
    pub bottom_right_station: f64,
    pub pointed: bool,
    pub bands: Vec<Band>,
}

pub(super) struct Topology {
    pub profiles: Vec<Profile>,
    pub maximum_bands: usize,
}

pub(super) fn classify(
    definitions: &[CrossSectionDefinition],
) -> Result<Topology, CrossSectionError> {
    let profiles = definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| classify_profile(definition).map_err(|error| error.at(index)))
        .collect::<Result<Vec<_>, _>>()?;
    let maximum_bands = profiles
        .iter()
        .map(|profile| profile.bands.len())
        .max()
        .ok_or_else(|| {
            CrossSectionError::input("section count=0: supply at least one surveyed profile")
        })?;
    check_work(profiles.len(), maximum_bands)?;
    Ok(Topology {
        profiles,
        maximum_bands,
    })
}

fn classify_profile(definition: &CrossSectionDefinition) -> Result<Profile, CrossSectionError> {
    let points = &definition.points;
    let cap = definition.maximum_stage_m;
    let first = points
        .iter()
        .position(|point| point.elevation_m < cap)
        .ok_or_else(|| unsupported("no surveyed point lies below the cap"))?;
    let last = points
        .iter()
        .rposition(|point| point.elevation_m < cap)
        .ok_or_else(|| unsupported("no surveyed point lies below the cap"))?;
    let left = first
        .checked_sub(1)
        .ok_or_else(|| unsupported("left bank does not reach the cap"))?;
    let right = last
        .checked_add(1)
        .filter(|&index| index < points.len())
        .ok_or_else(|| unsupported("right bank does not reach the cap"))?;
    let bed = points[first..=last]
        .iter()
        .map(|point| point.elevation_m)
        .fold(f64::INFINITY, f64::min);
    let bottom_first = points
        .iter()
        .position(|point| point.elevation_m == bed)
        .ok_or_else(|| unsupported("the surveyed bottom cannot be identified"))?;
    let bottom_last = points
        .iter()
        .rposition(|point| point.elevation_m == bed)
        .ok_or_else(|| unsupported("the surveyed bottom cannot be identified"))?;
    if points[bottom_first..=bottom_last]
        .iter()
        .any(|point| point.elevation_m != bed)
    {
        return Err(unsupported(
            "an interior ridge separates bottom depressions",
        ));
    }
    let mut left_bank = bank_segments(&points[left..=bottom_first], false)?;
    left_bank.reverse();
    let right_bank = bank_segments(&points[bottom_last..=right], true)?;
    let mut knots = Vec::with_capacity(right - left + 1);
    knots.push(bed);
    for point in &points[left..=right] {
        if point.elevation_m > bed && point.elevation_m < cap {
            knots.push(point.elevation_m);
        }
    }
    knots.push(cap);
    knots.sort_by(f64::total_cmp);
    knots.dedup_by(|a, b| *a == *b);
    let pointed = points[bottom_first].station_m == points[bottom_last].station_m;
    let mut left_index = 0;
    let mut right_index = 0;
    let mut bands = Vec::with_capacity(knots.len().saturating_sub(1));
    for pair in knots.windows(2) {
        let left_segment = select_segment(&left_bank, &mut left_index, pair[0], pair[1])?;
        let right_segment = select_segment(&right_bank, &mut right_index, pair[0], pair[1])?;
        if bands.is_empty() && pointed && left_segment.vertical() && right_segment.vertical() {
            return Err(unsupported("the bottom has no positive wet width above it"));
        }
        bands.push(Band {
            bottom_elevation: pair[0],
            top_elevation: pair[1],
            left_bank: left_segment,
            right_bank: right_segment,
        });
    }
    Ok(Profile {
        bed_elevation: bed,
        bottom_left_station: points[bottom_first].station_m,
        bottom_right_station: points[bottom_last].station_m,
        pointed,
        bands,
    })
}

fn bank_segments(
    points: &[CrossSectionPoint],
    ascending: bool,
) -> Result<Vec<BankSegment>, CrossSectionError> {
    points
        .windows(2)
        .map(|pair| {
            let (a, b) = (pair[0], pair[1]);
            let ordered = if ascending {
                a.elevation_m < b.elevation_m
            } else {
                a.elevation_m > b.elevation_m
            };
            if !ordered {
                return Err(unsupported(
                    "a bank contains a subcap bench or reverses across a ridge",
                ));
            }
            Ok(BankSegment {
                left_station: a.station_m,
                right_station: b.station_m,
                low_elevation: if ascending {
                    a.elevation_m
                } else {
                    b.elevation_m
                },
                high_elevation: if ascending {
                    b.elevation_m
                } else {
                    a.elevation_m
                },
            })
        })
        .collect()
}

fn select_segment(
    segments: &[BankSegment],
    cursor: &mut usize,
    lower: f64,
    upper: f64,
) -> Result<BankSegment, CrossSectionError> {
    while segments
        .get(*cursor)
        .is_some_and(|segment| segment.high_elevation <= lower)
    {
        *cursor = cursor
            .checked_add(1)
            .ok_or_else(|| unsupported("bank segment index overflows usize"))?;
    }
    segments
        .get(*cursor)
        .copied()
        .filter(|segment| segment.low_elevation <= lower && segment.high_elevation >= upper)
        .ok_or_else(|| unsupported("a bank does not span a complete supported height band"))
}

fn unsupported(reason: &str) -> CrossSectionError {
    CrossSectionError::input(format!(
        "prismatic certification: {reason}. Supply one connected monotone-bank profile below its cap"
    ))
}
