use std::sync::Mutex;

use mass_spectrometry::prelude::{
    FlashIndexBuildPhase, FlashIndexBuildProgress, FlashRowSearchProgress,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEvent {
    Phase(FlashIndexBuildPhase, Option<u64>),
    Inc(u64),
    Finish,
    RowStart(u64),
    RowInc(u64),
    RowFinish,
}

#[derive(Debug, Default)]
pub struct RecordingProgress {
    events: Mutex<Vec<ProgressEvent>>,
}

impl RecordingProgress {
    pub fn events(&self) -> Vec<ProgressEvent> {
        self.events.lock().expect("progress mutex").clone()
    }
}

impl FlashIndexBuildProgress for RecordingProgress {
    fn start_phase(&self, phase: FlashIndexBuildPhase, len: Option<u64>) {
        self.events
            .lock()
            .expect("progress mutex")
            .push(ProgressEvent::Phase(phase, len));
    }

    fn inc(&self, delta: u64) {
        self.events
            .lock()
            .expect("progress mutex")
            .push(ProgressEvent::Inc(delta));
    }

    fn finish(&self) {
        self.events
            .lock()
            .expect("progress mutex")
            .push(ProgressEvent::Finish);
    }
}

impl FlashRowSearchProgress for RecordingProgress {
    fn start(&self, len: u64) {
        self.events
            .lock()
            .expect("progress mutex")
            .push(ProgressEvent::RowStart(len));
    }

    fn inc(&self, delta: u64) {
        self.events
            .lock()
            .expect("progress mutex")
            .push(ProgressEvent::RowInc(delta));
    }

    fn finish(&self) {
        self.events
            .lock()
            .expect("progress mutex")
            .push(ProgressEvent::RowFinish);
    }
}

pub fn assert_progress_reports_phase(
    events: &[ProgressEvent],
    phase: FlashIndexBuildPhase,
    len: Option<u64>,
) {
    assert!(
        events.contains(&ProgressEvent::Phase(phase, len)),
        "missing progress phase {phase:?} with len {len:?}: {events:?}"
    );
}
