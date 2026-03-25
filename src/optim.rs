//! # Optimizers
//!
//! Training updates are expressed as lazy tensor assignments and then realized
//! together, matching tinygrad's "build more graph, then realize it" model.
//!
//! `Sgd::step` builds a lazy `assign` graph for every parameter (`param =
//! param - lr * grad`), then calls `realize_many` once to execute all updates
//! in a single scheduling pass. This keeps the optimizer out of the execution
//! path — it only constructs graph nodes — and lets the scheduler fuse or
//! reorder kernels freely.

use crate::tensor::Tensor;

/// Stochastic gradient descent with explicit parameter lists.
pub struct Sgd {
    params: Vec<Tensor>,
    lr: f32,
}

impl Sgd {
    /// Create a new SGD optimizer over the given parameters.
    ///
    /// # Panics
    ///
    /// Panics if `params` is empty.
    #[must_use]
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        assert!(!params.is_empty(), "optimizer must have at least one parameter");
        let mut deduped: Vec<Tensor> = Vec::new();
        for param in params {
            // Optimizers own stable parameter handles, so dedup by handle identity
            // instead of graph/value equality.
            if deduped.iter().any(|existing| existing.ptr_eq(&param)) {
                continue;
            }
            deduped.push(param);
        }
        Self { params: deduped, lr }
    }

    /// Clear all stored parameter gradients.
    pub fn zero_grad(&self) {
        for param in &self.params {
            param.clear_grad();
        }
    }

    /// Apply one SGD update step to every parameter.
    ///
    /// # Panics
    ///
    /// Panics if any parameter is missing a gradient.
    pub fn step(&self) {
        let lr = Tensor::scalar(self.lr);
        for param in &self.params {
            let grad = param.grad().expect("optimizer step requires parameter gradients");
            let update = param.detach().sub(&grad.mul(&lr));
            param.assign(&update);
        }
        let refs: Vec<&Tensor> = self.params.iter().collect();
        Tensor::realize_many(&refs);
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;

    use super::*;

    #[test]
    fn test_sgd_step_updates_parameter() {
        let param = Tensor::from_slice(&[3.0], &[1]).with_requires_grad(true);
        let target = Tensor::from_slice(&[2.0], &[1]);
        let loss = param.sub(&target).mul(&param.sub(&target)).sum(&[0]);

        let optim = Sgd::new(vec![param.clone()], 0.1);
        optim.zero_grad();
        loss.backward();
        optim.step();

        let value = param.to_vec()[0];
        assert!(value < 3.0);
        assert_eq!(param.dtype(), DType::F32);
    }
}
