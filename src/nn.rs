//! # Lightweight neural-network building blocks
//!
//! Tinygrad keeps `nn` intentionally small: layers are thin wrappers that own
//! parameter tensors and compose regular tensor ops in `__call__`. This module
//! follows the same idea with simple Rust structs plus explicit parameter
//! collection instead of Python-style object graph reflection.

use std::cell::Cell;

use crate::tensor::{cpu, Tensor};

thread_local! {
    static INIT_SEED: Cell<u64> = const { Cell::new(42) };
}

#[allow(clippy::cast_precision_loss)]
fn next_unit_f32() -> f32 {
    INIT_SEED.with(|seed| {
        let next = seed
            .get()
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        seed.set(next);
        (next >> 33) as f32 / (1_u64 << 31) as f32
    })
}

#[allow(clippy::cast_precision_loss)]
fn kaiming_normal(shape: &[usize], fan_in: usize) -> Tensor {
    let numel: usize = shape.iter().product();
    let std = (2.0 / fan_in as f32).sqrt();
    let mut data = Vec::with_capacity(numel);

    while data.len() < numel {
        let u1 = next_unit_f32().max(f32::MIN_POSITIVE);
        let u2 = next_unit_f32();
        let radius = (-2.0 * u1.ln()).sqrt() * std;
        let theta = 2.0 * std::f32::consts::PI * u2;

        data.push(radius * theta.cos());
        if data.len() < numel {
            data.push(radius * theta.sin());
        }
    }

    Tensor::new(&data, shape, cpu())
}

/// Explicit parameter collection for Rust model structs.
///
/// Tinygrad uses reflective object walking in `nn.state.get_parameters`. Rust
/// does not have an equivalent lightweight mechanism, so model and layer types
/// expose their trainable tensors directly through this trait instead.
pub trait Parameters {
    /// Return all trainable parameter handles owned by this value.
    fn parameters(&self) -> Vec<Tensor>;
}

/// A fully connected layer, mirroring tinygrad's `nn.Linear`.
///
/// The weight is stored as `[out_features, in_features]`, like tinygrad, and
/// transposed in [`forward`](Self::forward) to feed ferrograd's `[M, K] @ [K, N]`
/// matrix multiply. The default initialization uses Kaiming normal weights and
/// zero bias, which is a better default than bounded uniform for `ReLU` MLPs.
#[derive(Debug)]
pub struct Linear {
    /// Trainable weight matrix with shape `[out_features, in_features]`.
    pub weight: Tensor,
    /// Optional trainable bias vector with shape `[out_features]`.
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Create a linear layer with bias enabled.
    #[must_use]
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight =
            kaiming_normal(&[out_features, in_features], in_features).with_requires_grad(true);
        let bias = Some(
            Tensor::zeros(&[out_features], cpu(), crate::dtype::DType::F32)
                .with_requires_grad(true),
        );
        Self { weight, bias }
    }

    /// Create a linear layer without a bias term.
    #[must_use]
    pub fn without_bias(in_features: usize, out_features: usize) -> Self {
        let weight =
            kaiming_normal(&[out_features, in_features], in_features).with_requires_grad(true);
        Self { weight, bias: None }
    }

    /// Apply the layer to a rank-2 input of shape `[batch, in_features]`.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let weight = self.weight.permute(&[1, 0]);
        let logits = x.matmul(&weight);
        match &self.bias {
            Some(bias) => logits.add(bias),
            None => logits,
        }
    }
}

impl Parameters for Linear {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        params.extend(self.bias.iter().cloned());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn test_linear_forward_matches_output_shape() {
        // Arrange
        let layer = Linear::new(3, 2);
        let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], cpu());

        // Act
        let output = layer.forward(&input);

        // Assert
        assert_eq!(output.shape(), [2, 2]);
        assert!(output.requires_grad());
    }

    #[test]
    fn test_linear_parameters_include_bias() {
        // Arrange
        let layer = Linear::new(4, 5);

        // Act
        let params = layer.parameters();

        // Assert
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), [5, 4]);
        assert_eq!(params[0].dtype(), DType::F32);
        assert_eq!(params[1].shape(), [5]);
        assert_eq!(params[1].to_vec(), vec![0.0; 5]);
    }

    #[test]
    fn test_linear_without_bias_has_single_parameter() {
        // Arrange
        let layer = Linear::without_bias(4, 5);

        // Act
        let params = layer.parameters();

        // Assert
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].shape(), [5, 4]);
    }
}
