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
use ferrograd::nn::{Linear, Parameters};
use ferrograd::optim::Sgd;
use ferrograd::tensor::{cpu, Tensor};

// ── Helpers ──────────────────────────────────────────────────────────────

fn one_hot(labels: &[u8], num_classes: usize) -> Tensor {
    let n = labels.len();
    let mut data = vec![0.0_f32; n * num_classes];
    for (i, &label) in labels.iter().enumerate() {
        data[i * num_classes + label as usize] = 1.0;
    }
    Tensor::new(&data, &[n, num_classes], cpu())
}

// ── Model ────────────────────────────────────────────────────────────────

struct Mlp {
    l1: Linear,
    l2: Linear,
}

impl Mlp {
    fn new() -> Self {
        Self {
            l1: Linear::new(784, 128),
            l2: Linear::new(128, 10),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = self.l1.forward(x).relu();
        self.l2.forward(&hidden)
    }
}

impl Parameters for Mlp {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.l1.parameters();
        params.extend(self.l2.parameters());
        params
    }
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let lr = 0.01_f32;
    let batch_size = 256;
    let epochs = 5;

    println!("Loading MNIST...");
    let dataset = MNISTDataset::load().unwrap();
    let train_targets = one_hot(&dataset.train_labels, dataset.num_classes);

    println!(
        "Train: {} images, Test: {} images",
        dataset.train_len(),
        dataset.test_len(),
    );

    let model = Mlp::new();
    let optim = Sgd::new(model.parameters(), lr);

    let num_batches = dataset.train_len() / batch_size;

    let max_batches = std::env::var("MAX_BATCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(num_batches);
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0_f32;

        for batch_idx in 0..max_batches.min(num_batches) {
            let batch_start = std::time::Instant::now();
            let start = batch_idx * batch_size;
            let batch_x = dataset.train_images.narrow(0, start, batch_size);
            let batch_t = train_targets.narrow(0, start, batch_size);

            let logits = model.forward(&batch_x);
            let loss = logits.cross_entropy(&batch_t);

            optim.zero_grad();
            loss.backward();
            optim.step();

            let loss_val = loss.to_vec()[0];
            epoch_loss += loss_val;
            let batch_ms = batch_start.elapsed().as_secs_f64() * 1000.0;

            println!("  epoch {epoch} batch {batch_idx:>4}/{num_batches}  loss={loss_val:.4}  {batch_ms:.1}ms");
        }

        let batches_run = max_batches.min(num_batches);
        let avg_loss = epoch_loss / batches_run as f32;

        let test_logits = model.forward(&dataset.test_images);
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
