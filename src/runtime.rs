//! Device-scoped execution state owned outside the graph.
//!
//! Tinygrad keeps device identity in the graph and stores concrete buffers and
//! compiled kernels on the device side. This module follows the same split:
//! the lazy `UOp` graph knows *which* device a tensor lives on (via `DeviceId`),
//! but concrete resources — allocated memory buffers, compiled kernel objects,
//! and the `UOp` interner — live here in [`DeviceState`].
//!
//! ## Thread-local storage
//!
//! Each thread gets its own `DeviceState` per `DeviceId`, stored in the
//! `DEVICE_STATES` thread-local. This avoids interior-mutability contention
//! since the current execution model is single-threaded per device, matching
//! tinygrad's design where each `Device` instance owns its state.
//!
//! ## `BufferId` indirection
//!
//! Kernels and schedule items refer to buffers by [`BufferId`] rather than
//! holding direct `Buffer` references. This lets the scheduler *reserve* an
//! output slot before a kernel runs (so downstream kernels can reference it),
//! then fill it in after execution. It also decouples the graph layer from the
//! concrete buffer type, which will matter when we add GPU backends.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::{Rc, Weak};

use crate::device::{Buffer, CpuDevice, Device, DeviceId, Program};
use crate::dtype::DType;
use crate::uop::{Arg, Op, UOp, UOpInner, UOpKey};

/// Opaque identifier for a concrete device buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(usize);

impl BufferId {
    /// Return the raw numeric id, useful in debug output.
    #[must_use]
    pub fn raw(self) -> usize {
        self.0
    }
}

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One slot in the buffer registry. A record is created at reservation time
/// with metadata (dtype, numel) but no concrete buffer yet; the buffer is
/// filled in after kernel execution.
#[derive(Debug)]
struct BufferRecord {
    dtype: DType,
    numel: usize,
    /// `None` between reservation and execution; `Some` once a kernel has
    /// written the output or the buffer was inserted directly.
    buffer: Option<Buffer>,
}

/// Internal buffer registry with a two-phase lifecycle: `reserve` allocates an
/// id with metadata, `set` fills in the concrete buffer. This mirrors how the
/// scheduler plans output slots before kernels run.
#[derive(Debug, Default)]
struct BufferStore {
    next_id: usize,
    buffers: HashMap<BufferId, BufferRecord>,
}

impl BufferStore {
    fn reserve(&mut self, dtype: DType, numel: usize) -> BufferId {
        let id = BufferId(self.next_id);
        self.next_id += 1;
        self.buffers.insert(
            id,
            BufferRecord {
                dtype,
                numel,
                buffer: None,
            },
        );
        id
    }

    fn insert(&mut self, buffer: Buffer) -> BufferId {
        let id = self.reserve(buffer.dtype(), buffer.numel());
        self.set(id, buffer)
            .expect("inserted buffer must match reservation");
        id
    }

    fn set(&mut self, id: BufferId, buffer: Buffer) -> Result<(), RuntimeError> {
        let record = self
            .buffers
            .get_mut(&id)
            .ok_or(RuntimeError::UnknownBuffer(id))?;
        if record.dtype != buffer.dtype() || record.numel != buffer.numel() {
            return Err(RuntimeError::MetadataMismatch {
                id,
                expected_dtype: record.dtype,
                expected_numel: record.numel,
                actual_dtype: buffer.dtype(),
                actual_numel: buffer.numel(),
            });
        }
        record.buffer = Some(buffer);
        Ok(())
    }

    fn get(&self, id: BufferId) -> Result<Buffer, RuntimeError> {
        let record = self
            .buffers
            .get(&id)
            .ok_or(RuntimeError::UnknownBuffer(id))?;
        record
            .buffer
            .clone()
            .ok_or(RuntimeError::UninitializedBuffer(id))
    }
}

/// Errors from device-state buffer lookup or mutation.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The graph referenced an unknown buffer identifier.
    #[error("unknown device buffer {0}")]
    UnknownBuffer(BufferId),
    /// The scheduler reserved a buffer, but execution has not produced it yet.
    #[error("device buffer {0} is not initialized")]
    UninitializedBuffer(BufferId),
    /// The executor tried to write a buffer with mismatched metadata.
    #[error(
        "buffer {id} metadata mismatch: expected {expected_dtype:?}[{expected_numel}], got {actual_dtype:?}[{actual_numel}]"
    )]
    MetadataMismatch {
        /// Target buffer identifier.
        id: BufferId,
        /// Expected dtype.
        expected_dtype: DType,
        /// Expected element count.
        expected_numel: usize,
        /// Actual dtype.
        actual_dtype: DType,
        /// Actual element count.
        actual_numel: usize,
    },
}

/// Device-scoped execution state shared by tensors and compiled kernels.
pub(crate) struct DeviceState {
    device: Rc<dyn Device>,
    buffers: RefCell<BufferStore>,
    kernels: RefCell<HashMap<UOp, Rc<Program>>>,
    interner: RefCell<HashMap<UOpKey, Weak<UOpInner>>>,
}

impl DeviceState {
    /// Create a fresh device state wrapping the given backend.
    fn new(device: Rc<dyn Device>) -> Rc<Self> {
        Rc::new(Self {
            device,
            buffers: RefCell::new(BufferStore::default()),
            kernels: RefCell::new(HashMap::new()),
            interner: RefCell::new(HashMap::new()),
        })
    }

    /// Borrow the underlying device backend.
    #[must_use]
    pub(crate) fn device(&self) -> &dyn Device {
        self.device.as_ref()
    }

    /// Store a fully initialized buffer and return its id.
    #[must_use]
    pub(crate) fn store_buffer(&self, buffer: Buffer) -> BufferId {
        self.buffers.borrow_mut().insert(buffer)
    }

    /// Reserve a future output buffer.
    #[must_use]
    pub(crate) fn reserve_buffer(&self, dtype: DType, numel: usize) -> BufferId {
        self.buffers.borrow_mut().reserve(dtype, numel)
    }

    /// Load a buffer by id.
    ///
    /// # Errors
    ///
    /// Returns an error if the id is unknown or if the buffer has not been
    /// initialized yet.
    pub(crate) fn load_buffer(&self, id: BufferId) -> Result<Buffer, RuntimeError> {
        self.buffers.borrow().get(id)
    }

    /// Replace the concrete buffer backing a reserved output id.
    ///
    /// # Errors
    ///
    /// Returns an error if the id is unknown or if the buffer metadata does
    /// not match the reservation.
    pub(crate) fn write_buffer(&self, id: BufferId, buffer: Buffer) -> Result<(), RuntimeError> {
        self.buffers.borrow_mut().set(id, buffer)
    }

    pub(crate) fn cached_program(&self, sink: &UOp) -> Option<Rc<Program>> {
        self.kernels.borrow().get(sink).cloned()
    }

    pub(crate) fn insert_program(&self, sink: UOp, program: Rc<Program>) {
        self.kernels.borrow_mut().insert(sink, program);
    }

    /// Hash-cons a `UOp`: if an identical node (same op, dtype, srcs, arg) already
    /// exists and is still alive, return the existing `Rc` instead of allocating
    /// a new one. This keeps the graph compact and makes pointer equality a
    /// valid structural identity check during scheduling and rewriting.
    pub(crate) fn intern_uop(&self, op: Op, dtype: DType, srcs: Vec<UOp>, arg: Arg) -> UOp {
        let key = UOpKey {
            op,
            dtype,
            srcs,
            arg,
        };
        let mut cache = self.interner.borrow_mut();
        if let Some(existing) = cache.get(&key).and_then(Weak::upgrade) {
            return UOp::from_inner(existing);
        }

        let inner = Rc::new(UOpInner {
            op: key.op,
            dtype: key.dtype,
            srcs: key.srcs.clone(),
            arg: key.arg.clone(),
        });
        cache.insert(key, Rc::downgrade(&inner));
        UOp::from_inner(inner)
    }

    /// Reset all state so tests start with a clean device. Test-only because
    /// clearing buffers that live tensors still reference would cause panics.
    #[cfg(test)]
    pub(crate) fn clear_for_tests(&self) {
        *self.buffers.borrow_mut() = BufferStore::default();
        self.kernels.borrow_mut().clear();
        self.interner.borrow_mut().clear();
    }
}

// Per-thread registry of device states, lazily initialized on first access.
thread_local! {
    static DEVICE_STATES: RefCell<HashMap<DeviceId, Rc<DeviceState>>> = RefCell::new(HashMap::new());
}

/// Instantiate the concrete `Device` backend for a `DeviceId` and wrap it in
/// a new `DeviceState`.
fn new_state(device: DeviceId) -> Rc<DeviceState> {
    match device {
        DeviceId::Cpu => DeviceState::new(Rc::new(CpuDevice)),
    }
}

/// Return the device-scoped execution state for `device`.
#[must_use]
pub(crate) fn state(device: DeviceId) -> Rc<DeviceState> {
    DEVICE_STATES.with(|states| {
        let mut states = states.borrow_mut();
        states
            .entry(device)
            .or_insert_with(|| new_state(device))
            .clone()
    })
}

#[cfg(test)]
pub(crate) fn clear_for_tests(device: DeviceId) {
    state(device).clear_for_tests();
}
