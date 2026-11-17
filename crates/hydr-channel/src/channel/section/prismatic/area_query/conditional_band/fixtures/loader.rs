use super::model::FixtureSet;
use std::fmt;

const PANELS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/surveyed-point-panels/panels.tsv"
));
const SAMPLES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/surveyed-point-panels/samples.tsv"
));

#[derive(Debug)]
pub(crate) struct FixtureError {
    pub table: &'static str,
    pub line: usize,
    pub detail: &'static str,
}
impl fmt::Display for FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.table, self.line, self.detail)
    }
}
impl std::error::Error for FixtureError {}

pub(super) struct Row<'a> {
    fields: Vec<&'a str>,
    table: &'static str,
    line: usize,
}
impl<'a> Row<'a> {
    pub fn new(
        text: &'a str,
        table: &'static str,
        line: usize,
        columns: usize,
    ) -> Result<Self, FixtureError> {
        let row = Self {
            fields: text.split('\t').collect(),
            table,
            line,
        };
        row.require(row.fields.len() == columns, "Wrong column count")?;
        Ok(row)
    }
    pub fn error(&self, detail: &'static str) -> FixtureError {
        FixtureError {
            table: self.table,
            line: self.line,
            detail,
        }
    }
    pub fn require(&self, predicate: bool, detail: &'static str) -> Result<(), FixtureError> {
        if predicate {
            Ok(())
        } else {
            Err(self.error(detail))
        }
    }
    pub fn text(&self, index: usize) -> Result<&'a str, FixtureError> {
        self.fields
            .get(index)
            .copied()
            .ok_or_else(|| self.error("Missing column"))
    }
    pub fn usize(&self, index: usize) -> Result<usize, FixtureError> {
        self.text(index)?
            .parse()
            .map_err(|_| self.error("Invalid unsigned integer"))
    }
    pub fn bits(&self, index: usize) -> Result<u64, FixtureError> {
        let text = self.text(index)?;
        self.require(text.len() == 16, "Expected sixteen hexadecimal digits")?;
        let bits = u64::from_str_radix(text, 16).map_err(|_| self.error("Invalid F64 bits"))?;
        self.require(f64::from_bits(bits).is_finite(), "Nonfinite F64 bits")?;
        Ok(bits)
    }
    pub fn boolean(&self, index: usize) -> Result<bool, FixtureError> {
        match self.text(index)? {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(self.error("Invalid boolean")),
        }
    }
}

pub(crate) fn load() -> Result<FixtureSet, FixtureError> {
    let mut set = super::metadata::parse(PANELS)?;
    super::samples::parse(SAMPLES, &mut set)?;
    Ok(set)
}
