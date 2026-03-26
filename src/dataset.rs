//! # Datasets
//!
//! Pre-packaged datasets for training and evaluation. Each dataset handles
//! downloading, caching, and parsing so examples can focus on model logic.
//!
//! Mirrors the dataset layer in [deers](https://github.com/robertogreco/deers):
//! a thin struct that owns the parsed tensors and exposes them as public fields.

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use crate::tensor::{cpu, Tensor};

/// Errors that can occur while loading a dataset.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// An I/O error (file read, directory creation, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A network download failed.
    #[error("download failed: {0}")]
    Download(String),
}

/// Convenience alias used throughout this module.
pub type Result<T> = std::result::Result<T, DatasetError>;

/// The MNIST handwritten-digit dataset (60 k train + 10 k test images).
///
/// Images are `(N, 784)` tensors with pixel values normalized to `[0, 1]`.
/// Labels are plain `Vec<u8>` with class indices `0..=9` — ferrograd doesn't
/// have an integer tensor type yet, so we keep them on the Rust side and let
/// examples decide how to encode them (one-hot, sparse index, etc.).
#[derive(Debug)]
pub struct MNISTDataset {
    /// Training images shaped `(60000, 784)`, values in `[0, 1]`.
    pub train_images: Tensor,
    /// Training labels, one `u8` per image (`0..=9`).
    pub train_labels: Vec<u8>,
    /// Test images shaped `(10000, 784)`, values in `[0, 1]`.
    pub test_images: Tensor,
    /// Test labels, one `u8` per image (`0..=9`).
    pub test_labels: Vec<u8>,
    /// Number of digit classes (always 10).
    pub num_classes: usize,
}

impl MNISTDataset {
    /// Loads MNIST from IDX files in `data/mnist/`, downloading if needed.
    ///
    /// On first call the four gzipped IDX files are fetched from Google Cloud
    /// Storage, decompressed, and cached locally. Subsequent calls read
    /// straight from disk.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if a download or file I/O operation fails.
    pub fn load() -> Result<Self> {
        let base_url = "https://storage.googleapis.com/cvdf-datasets/mnist";
        let dir = Path::new("data/mnist");

        let files = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ];
        for file in &files {
            let gz_path = dir.join(format!("{file}.gz"));
            let final_path = dir.join(file);
            if !final_path.exists() {
                download_if_missing(&format!("{base_url}/{file}.gz"), &gz_path)?;
                decompress_gz(&gz_path, &final_path)?;
                fs::remove_file(&gz_path).ok();
            }
        }

        let (train_images, _) = parse_images(&dir.join("train-images-idx3-ubyte"))?;
        let train_labels = parse_labels(&dir.join("train-labels-idx1-ubyte"))?;
        let (test_images, _) = parse_images(&dir.join("t10k-images-idx3-ubyte"))?;
        let test_labels = parse_labels(&dir.join("t10k-labels-idx1-ubyte"))?;

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            num_classes: 10,
        })
    }

    /// Number of training samples.
    #[must_use]
    pub fn train_len(&self) -> usize {
        self.train_labels.len()
    }

    /// Number of test samples.
    #[must_use]
    pub fn test_len(&self) -> usize {
        self.test_labels.len()
    }
}

// ── IDX parsing ──────────────────────────────────────────────────────────

/// Parses an IDX image file into a flat `(N, 784)` tensor and returns the
/// image count alongside it.
fn parse_images(path: &Path) -> Result<(Tensor, usize)> {
    let mut file = File::open(path)?;
    let magic = read_u32(&mut file)?;
    assert_eq!(magic, 2051, "bad IDX image magic");

    let count = read_u32(&mut file)? as usize;
    let rows = read_u32(&mut file)? as usize;
    let cols = read_u32(&mut file)? as usize;
    assert_eq!((rows, cols), (28, 28));

    let mut pixels = vec![0u8; count * rows * cols];
    file.read_exact(&mut pixels)?;

    let f32_data: Vec<f32> = pixels.iter().map(|&b| f32::from(b) / 255.0).collect();
    Ok((Tensor::new(&f32_data, &[count, 784], cpu()), count))
}

/// Parses an IDX label file into a `Vec<u8>`.
fn parse_labels(path: &Path) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let magic = read_u32(&mut file)?;
    assert_eq!(magic, 2049, "bad IDX label magic");

    let count = read_u32(&mut file)? as usize;
    let mut labels = vec![0u8; count];
    file.read_exact(&mut labels)?;
    Ok(labels)
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

// ── Download helpers ─────────────────────────────────────────────────────

/// Downloads `url` to `path` if the file doesn't already exist.
///
/// Uses a `.part` suffix during download so a partial file is never mistaken
/// for a cached copy.
fn download_if_missing(url: &str, path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Downloading {url}");
    let part_path = path.with_extension("part");
    let resp = ureq::get(url)
        .call()
        .map_err(|e| DatasetError::Download(e.to_string()))?;
    let mut reader = resp.into_body().into_reader();
    let mut file = File::create(&part_path)?;
    std::io::copy(&mut reader, &mut file)?;
    fs::rename(&part_path, path)?;
    println!("Saved to {}", path.display());
    Ok(())
}

/// Decompresses a `.gz` file to the given output path.
fn decompress_gz(gz_path: &Path, out_path: &Path) -> Result<()> {
    let gz_file = File::open(gz_path)?;
    let mut decoder = flate2::read::GzDecoder::new(gz_file);
    let mut out_file = File::create(out_path)?;
    std::io::copy(&mut decoder, &mut out_file)?;
    Ok(())
}
