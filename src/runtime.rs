//! Device-scoped execution state owned outside the graph.
//!
//! Tinygrad keeps device identity in the graph and stores compiled kernels on
//! the device side. This module follows the same split: the lazy `UOp` graph
//! knows *which* device a tensor lives on (via `DeviceId`), while compiled
//! kernels and the `UOp` interner live here in [`DeviceState`].
//!
//! ## Thread-local storage
//!
//! Each thread gets its own `DeviceState` per `DeviceId`, stored in the
//! `DEVICE_STATES` thread-local. This avoids interior-mutability contention
//! since the current execution model is single-threaded per device, matching
//! tinygrad's design where each `Device` instance owns its state.
//!
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use crate::device::{CpuDevice, Device, DeviceId, Program};
use crate::dtype::DType;
use crate::uop::{Arg, Op, UOp, UOpInner, UOpKey};

/// Device-scoped execution state shared by tensors and compiled kernels.
pub(crate) struct DeviceState {
    device: Rc<dyn Device>,
    kernels: RefCell<HashMap<UOp, Rc<Program>>>,
    interner: RefCell<HashMap<UOpKey, Weak<UOpInner>>>,
}

impl DeviceState {
    /// Create a fresh device state wrapping the given backend.
    fn new(device: Rc<dyn Device>) -> Rc<Self> {
        Rc::new(Self {
            device,
            kernels: RefCell::new(HashMap::new()),
            interner: RefCell::new(HashMap::new()),
        })
    }

    /// Borrow the underlying device backend.
    #[must_use]
    pub(crate) fn device(&self) -> &dyn Device {
        self.device.as_ref()
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
    /// only caches need resetting; buffer handles own their own storage.
    #[cfg(test)]
    pub(crate) fn clear_for_tests(&self) {
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
