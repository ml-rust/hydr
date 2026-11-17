//! Segmented running maximum over a one-dimensional tensor, on the tensor's device.
//!
//! A segment starts at every entry flagged `1`. Each output entry holds the maximum of its
//! segment from the segment start up to that entry, in the scan direction. The scan doubles its
//! reach each pass, so it takes `ceil(log2 n)` passes of whole-tensor operations.

use super::ChannelError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Order in which a segmented scan visits the entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScanDirection {
    /// From the first entry toward the last.
    Forward,
    /// From the last entry toward the first.
    Backward,
}

/// Segmented inclusive running maximum of nonnegative `values`.
///
/// `starts` holds `1` where a segment begins and `0` elsewhere. The first entry in the scan
/// direction always begins a segment. Both tensors are F64 with one dimension of equal length.
pub(super) fn segmented_max<R, C>(
    client: &C,
    values: &Tensor<R>,
    starts: &Tensor<R>,
    direction: ScanDirection,
) -> Result<Tensor<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + ConditionalOps<R> + ShapeOps<R>,
{
    if values.shape().len() != 1 || values.shape() != starts.shape() {
        return Err(ChannelError::numerical(format!(
            "segmented maximum values shape={:?}, starts shape={:?}: pass two equal one-dimensional tensors",
            values.shape(),
            starts.shape()
        )));
    }
    let n = values.numel();
    let mut maximum = values.clone();
    let mut started = starts.clone();
    let mut reach = 1;
    while reach < n {
        let kept = n - reach;
        let pad =
            |value: f64| Tensor::<R>::full_scalar(&[reach], DType::F64, value, client.device());
        // Entries shifted in from outside the tensor hold zero and begin a segment.
        let (shifted_max, shifted_start) = match direction {
            ScanDirection::Forward => (
                client.cat(&[&pad(0.0)?, &maximum.narrow(0, 0, kept)?], 0)?,
                client.cat(&[&pad(1.0)?, &started.narrow(0, 0, kept)?], 0)?,
            ),
            ScanDirection::Backward => (
                client.cat(&[&maximum.narrow(0, reach, kept)?, &pad(0.0)?], 0)?,
                client.cat(&[&started.narrow(0, reach, kept)?, &pad(1.0)?], 0)?,
            ),
        };
        maximum =
            client.where_cond(&started, &maximum, &client.maximum(&maximum, &shifted_max)?)?;
        started = client.maximum(&started, &shifted_start)?;
        reach *= 2;
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn cpu() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    fn scan(
        values: &[f64],
        starts: &[f64],
        direction: ScanDirection,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let client = cpu();
        let tensor =
            |data: &[f64]| Tensor::<CpuRuntime>::from_slice(data, &[data.len()], client.device());
        let result = segmented_max(&client, &tensor(values)?, &tensor(starts)?, direction)?;
        Ok(result.try_to_vec::<f64>()?)
    }

    /// Host reference: one pass that resets at every start.
    fn reference(values: &[f64], starts: &[f64], direction: ScanDirection) -> Vec<f64> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        if direction == ScanDirection::Backward {
            order.reverse();
        }
        let mut out = vec![0.0; values.len()];
        let mut running = 0.0_f64;
        for (position, &index) in order.iter().enumerate() {
            running = if position == 0 || starts[index] == 1.0 {
                values[index]
            } else {
                running.max(values[index])
            };
            out[index] = running;
        }
        out
    }

    #[test]
    fn forward_scan_resets_at_each_start() -> TestResult {
        let values = [3.0, 1.0, 4.0, 0.0, 2.0, 5.0, 1.0];
        let starts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let result = scan(&values, &starts, ScanDirection::Forward)?;
        assert_eq!(result, [3.0, 3.0, 4.0, 0.0, 2.0, 5.0, 1.0]);
        Ok(())
    }

    #[test]
    fn backward_scan_resets_at_each_start() -> TestResult {
        let values = [3.0, 1.0, 4.0, 0.0, 2.0, 5.0, 1.0];
        let starts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let result = scan(&values, &starts, ScanDirection::Backward)?;
        assert_eq!(result, [4.0, 4.0, 4.0, 0.0, 5.0, 5.0, 1.0]);
        Ok(())
    }

    #[test]
    fn scans_match_the_host_reference_across_lengths() -> TestResult {
        for n in 1..=19 {
            let values: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64).collect();
            let starts: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 5 == 2))).collect();
            for direction in [ScanDirection::Forward, ScanDirection::Backward] {
                assert_eq!(
                    scan(&values, &starts, direction)?,
                    reference(&values, &starts, direction),
                    "n={n}, direction={direction:?}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn mismatched_shapes_are_rejected() -> TestResult {
        let client = cpu();
        let values = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0], &[2], client.device())?;
        let starts = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], client.device())?;
        assert!(segmented_max(&client, &values, &starts, ScanDirection::Forward).is_err());
        Ok(())
    }
}
