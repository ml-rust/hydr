//! CPU/CUDA result comparison and WGPU F64 rejection checks.

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use super::Result;

/// Requires bit-identical CPU and CUDA results.
///
/// `Debug` prints each F64 in round-trip form, so equal text means equal bits.
#[cfg(feature = "cuda")]
pub fn same_bits<T: std::fmt::Debug>(cpu: &T, cuda: &T) -> Result {
    let (cpu, cuda) = (format!("{cpu:?}"), format!("{cuda:?}"));
    if cpu != cuda {
        return Err(format!("CUDA result differs from CPU.\nCPU:  {cpu}\nCUDA: {cuda}").into());
    }
    Ok(())
}

/// Splits `Debug` text into numeric literals and the text between them.
///
/// Each token is `(numeric, text)`. Digits inside identifiers stay in text tokens.
#[cfg(feature = "cuda")]
fn debug_tokens(text: &str) -> Vec<(bool, &str)> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let (mut start, mut index) = (0, 0);
    while index < bytes.len() {
        let previous = index.checked_sub(1).map(|i| bytes[i]);
        let bound = previous.is_none_or(|c| !(c.is_ascii_alphanumeric() || b"_-.".contains(&c)));
        let digit = |i: usize| bytes.get(i).is_some_and(u8::is_ascii_digit);
        if bound && (digit(index) || (bytes[index] == b'-' && digit(index + 1))) {
            tokens.push((false, &text[start..index]));
            let begin = index;
            index += 1;
            while index < bytes.len() {
                let c = bytes[index];
                let sign = (c == b'-' || c == b'+') && matches!(bytes[index - 1], b'e' | b'E');
                if !(c.is_ascii_digit() || b".eE".contains(&c) || sign) {
                    break;
                }
                index += 1;
            }
            tokens.push((true, &text[begin..index]));
            start = index;
        } else {
            index += 1;
        }
    }
    tokens.push((false, &text[start..]));
    tokens
}

/// Requires bit-identical per-element states and complete data within a volume tolerance.
///
/// Every differing F64 must satisfy the documented `absolute + relative * scale` gate.
/// Integers, identifiers, and structure must match exactly.
#[cfg(feature = "cuda")]
pub fn same_states_bounded_data<S: std::fmt::Debug, D: std::fmt::Debug>(
    cpu: &[(S, D)],
    cuda: &[(S, D)],
    absolute: f64,
    relative: f64,
) -> Result {
    if cpu.len() != cuda.len() {
        return Err(format!("CPU has {} results, CUDA {}", cpu.len(), cuda.len()).into());
    }
    for ((cpu_states, cpu_data), (cuda_states, cuda_data)) in cpu.iter().zip(cuda) {
        same_bits(cpu_states, cuda_states)?;
        let (cpu_text, cuda_text) = (format!("{cpu_data:?}"), format!("{cuda_data:?}"));
        let (left, right) = (debug_tokens(&cpu_text), debug_tokens(&cuda_text));
        if left.len() != right.len() {
            return same_bits(cpu_data, cuda_data);
        }
        for (index, (&(numeric, a), &(_, b))) in left.iter().zip(&right).enumerate() {
            if a == b {
                continue;
            }
            let context = left[index.saturating_sub(1)].1;
            let (x, y) = match (numeric, a.parse::<f64>(), b.parse::<f64>()) {
                (true, Ok(x), Ok(y)) if a.contains(['.', 'e']) && b.contains(['.', 'e']) => (x, y),
                _ => {
                    return Err(
                        format!("CUDA differs from CPU after `{context}`: {a} vs {b}").into(),
                    );
                }
            };
            let allowance = absolute + relative * x.abs().max(y.abs());
            if (x - y).abs() > allowance {
                return Err(format!(
                    "CUDA differs from CPU after `{context}`: {a} vs {b} exceeds {allowance:e}"
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Returns a WGPU client after checking that numr WGPU still lacks F64 arithmetic.
///
/// A numr release with F64 WGPU fails here. Then replace the WGPU rejection claims.
#[cfg(feature = "wgpu")]
pub fn wgpu_client() -> Result<numr::runtime::wgpu::WgpuClient> {
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    if WgpuRuntime::computes_float_dtype(DType::F64) {
        return Err("numr WGPU computes F64: replace the WGPU rejection evidence".into());
    }
    Ok(WgpuClient::new(WgpuDevice::new(0))?)
}

/// Returns the construction error after checking that it names the missing F64 arithmetic.
///
/// An admitted WGPU construction fails here, because no CPU fallback can run it.
#[cfg(feature = "wgpu")]
pub fn f64_rejection<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    component: &str,
) -> Result<E> {
    let error = result
        .err()
        .ok_or(format!("WGPU admitted unsupported {component} F64"))?;
    let message = error.to_string();
    if !message.contains("F64") || !message.contains("wgpu") {
        return Err(format!("{component} WGPU rejection omits the F64 runtime: {message}").into());
    }
    Ok(error)
}
