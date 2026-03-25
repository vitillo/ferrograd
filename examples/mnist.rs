//! Train an MLP on MNIST — the full integration test.
//!
//! Model: Linear(784, 128) → `ReLU` → Linear(128, 10)
//! Loss: cross-entropy via log-softmax
//! Optimizer: SGD
//!
//! ```sh
//! cargo run --example mnist --release
//! ```

use ferrograd::dataset::MNISTDataset;
use ferrograd::tensor::Tensor;

// ── Helpers ──────────────────────────────────────────────────────────────

fn one_hot(labels: &[u8], num_classes: usize) -> Tensor {
    let n = labels.len();
    let mut data = vec![0.0_f32; n * num_classes];
    for (i, &label) in labels.iter().enumerate() {
        data[i * num_classes + label as usize] = 1.0;
    }
    Tensor::from_slice(&data, &[n, num_classes])
}

#[allow(clippy::cast_precision_loss)]
fn rand_tensor(shape: &[usize], scale: f32) -> Tensor {
    // Simple deterministic "random" init using a linear congruential generator.
    // Good enough for a demo — real training would use proper RNG.
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(42) };
    }
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|_| {
            SEED.with(|s| {
                let x = s
                    .get()
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                s.set(x);
                // Map to [-scale, scale]
                ((x >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * scale
            })
        })
        .collect();
    Tensor::from_slice(&data, shape)
}

// ── Model ────────────────────────────────────────────────────────────────

fn log_softmax(x: &Tensor) -> Tensor {
    // log_softmax(x) = x - log(sum(exp(x - max(x))))
    // Numerically stable: subtract max before exp.
    let max_x = x.max(&[1]); // [N, 1]
    let shifted = x.sub(&max_x); // [N, 10]
    let exp_shifted = shifted.exp(); // [N, 10]
    let sum_exp = exp_shifted.sum(&[1]); // [N, 1]
    let log_sum = sum_exp.log(); // [N, 1]
    shifted.sub(&log_sum) // [N, 10]
}

fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    // -mean(sum(targets * log_softmax(logits), axis=1))
    let log_probs = log_softmax(logits);
    let per_sample = targets.mul(&log_probs).sum(&[1]); // [N, 1]
    per_sample.neg().sum(&[0]).reshape(&[1]) // scalar
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let lr = 0.01_f32;
    let batch_size = 64;
    let epochs = 5;

    println!("Loading MNIST...");
    let dataset = MNISTDataset::load().unwrap();
    let train_targets = one_hot(&dataset.train_labels, dataset.num_classes);

    println!(
        "Train: {} images, Test: {} images",
        dataset.train_len(),
        dataset.test_len(),
    );

    let mut w1 = rand_tensor(&[784, 128], (2.0 / 784.0_f32).sqrt()).with_requires_grad(true);
    let mut b1 = Tensor::zeros(&[1, 128], ferrograd::dtype::DType::F32).with_requires_grad(true);
    let mut w2 = rand_tensor(&[128, 10], (1.0 / 128.0_f32).sqrt()).with_requires_grad(true);
    let mut b2 = Tensor::zeros(&[1, 10], ferrograd::dtype::DType::F32).with_requires_grad(true);

    let num_batches = dataset.train_len() / batch_size;
    let batch_scale = Tensor::scalar(1.0 / batch_size as f32);

    let max_batches = std::env::var("MAX_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(num_batches);
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0_f32;

        for batch_idx in 0..max_batches.min(num_batches) {
            let start = batch_idx * batch_size;
            let batch_x = dataset.train_images.narrow(0, start, batch_size);
            let batch_t = train_targets.narrow(0, start, batch_size);

            let hidden = batch_x.matmul(&w1).add(&b1).relu();
            let logits = hidden.matmul(&w2).add(&b2);
            let loss = cross_entropy(&logits, &batch_t).mul(&batch_scale);

            let grads = loss.gradient(&[&w1, &b1, &w2, &b2]);
            let lr_t = Tensor::scalar(lr);

            w1 = w1.sub(&grads[0].mul(&lr_t)).realize();
            b1 = b1.sub(&grads[1].mul(&lr_t)).realize();
            w2 = w2.sub(&grads[2].mul(&lr_t)).realize();
            b2 = b2.sub(&grads[3].mul(&lr_t)).realize();

            let loss_val = loss.to_vec()[0];
            epoch_loss += loss_val;

            println!("  epoch {epoch} batch {batch_idx:>4}/{num_batches}  loss={loss_val:.4}");
        }

        let batches_run = max_batches.min(num_batches);
        let avg_loss = epoch_loss / batches_run as f32;

        let test_logits = dataset
            .test_images
            .matmul(&w1)
            .add(&b1)
            .relu()
            .matmul(&w2)
            .add(&b2);
        let test_preds = test_logits.to_vec();
        let mut correct = 0;
        for (i, label) in dataset.test_labels.iter().enumerate() {
            let logits_i = &test_preds[i * 10..(i + 1) * 10];
            let pred = logits_i
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred == *label as usize {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / dataset.test_len() as f32 * 100.0;
        println!("Epoch {epoch}: avg_loss={avg_loss:.4}, test_accuracy={accuracy:.1}%");
    }
}
