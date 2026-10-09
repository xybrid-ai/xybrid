//! Resource-policy interruption and terminal outcomes through real TTS streams.
//!
//! Uses `kitten-tts-2` fixtures or `XYBRID_KITTEN_TEST_BUNDLE`, skipping when
//! unavailable. Run with `--features tts-zzz --test tts_abort` and a verified
//! native slice. The bundle should enable sentence chunks and prepared voices.
//! ONNX checks use `kokoro-82m` fixtures or `XYBRID_ONNX_TTS_TEST_BUNDLE`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
#[cfg(feature = "tts-zzz")]
use std::time::Instant;

use xybrid_core::device::{
    MemoryPressure, ResourceSnapshot, ResourceSnapshotProvider, ThermalState,
};
use xybrid_core::execution::TtsStatus;
use xybrid_core::ir::{Envelope, EnvelopeKind};
use xybrid_sdk::{AbortPolicy, AbortSignal, CancellationToken, ModelLoader, RunOptions, SdkError};

#[derive(Debug)]
struct PressureProvider {
    stressed: AtomicBool,
    snapshot: ResourceSnapshot,
}

impl ResourceSnapshotProvider for PressureProvider {
    fn current_snapshot(&self, _max_age: Duration) -> ResourceSnapshot {
        if self.stressed.load(Ordering::Acquire) {
            self.snapshot
        } else {
            ResourceSnapshot::default()
        }
    }
}

#[test]
#[cfg(feature = "tts-zzz")]
fn kitten_resource_abort_interrupts_compute_and_keeps_the_session_reusable() {
    let bundle = std::env::var_os("XYBRID_KITTEN_TEST_BUNDLE")
        .map(std::path::PathBuf::from)
        .or_else(|| integration_tests::fixtures::model_if_available("kitten-tts-2"));
    let Some(bundle) = bundle else {
        eprintln!("Skipping Kitten resource-abort test: prepared bundle unavailable");
        return;
    };
    let model = ModelLoader::from_directory(bundle).unwrap().load().unwrap();
    let short = Envelope::new(EnvelopeKind::Text("Your name sounds like trouble.".into()));
    let long = Envelope::new(EnvelopeKind::Text(
        "Tell me your name. Then explain your business. The village gate stays closed until I know who you are and what you want here today.".into(),
    ));
    let mut warm_packets = 0;
    let warm = model
        .run_tts_streaming(&short, &RunOptions::new(), |_| {
            warm_packets += 1;
            true
        })
        .unwrap();
    assert_eq!(warm.status, TtsStatus::Completed);
    assert!(warm_packets > 0);

    let token = CancellationToken::new();
    let mut delivered_end = 0;
    let cancelled = model
        .run_tts_streaming(
            &short,
            &RunOptions::new().with_cancellation_token(token.clone()),
            |packet| {
                delivered_end =
                    packet.first_sample + packet.pcm.len() as u64 / 2 / u64::from(packet.channels);
                if delivered_end == warm.samples {
                    token.cancel();
                }
                true
            },
        )
        .expect("cancellation from the final packet must remain terminal");
    assert_eq!(delivered_end, warm.samples);
    assert!(token.is_cancelled());
    assert_eq!(cancelled.status, TtsStatus::Cancelled);

    for (signal, snapshot, fallback, reason_text, core_reason) in [
        (
            AbortSignal::MemoryPressureCritical,
            ResourceSnapshot {
                memory_pressure: MemoryPressure::Critical,
                ..Default::default()
            },
            true,
            "memory_pressure_critical",
            xybrid_core::abort::AbortReason::StressMemory,
        ),
        (
            AbortSignal::ThermalCritical,
            ResourceSnapshot {
                thermal_state: ThermalState::Critical,
                ..Default::default()
            },
            false,
            "thermal_critical",
            xybrid_core::abort::AbortReason::StressThermal,
        ),
    ] {
        let provider = Arc::new(PressureProvider {
            stressed: AtomicBool::new(false),
            snapshot,
        });
        let token = CancellationToken::new();
        let options = RunOptions::new()
            .with_cancellation_token(token.clone())
            .with_resource_provider(provider.clone())
            .with_abort_policy(
                AbortPolicy::default()
                    .stop_on(signal)
                    .with_cloud_fallback(fallback),
            );
        let mut packets = 0;
        let started = Instant::now();
        let result = std::thread::scope(|scope| {
            // The resident session is ready; introduce stress during synthesis,
            // well before the engine can finish its first sentence waveform.
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(250));
                provider.stressed.store(true, Ordering::Release);
            });
            model.run_tts_streaming(&long, &options, |_| {
                packets += 1;
                true
            })
        });
        eprintln!(
            "{reason_text}: returned after {:?}, packets={packets}",
            started.elapsed()
        );
        assert_eq!(packets, 0, "resource pressure must stop before first audio");
        assert!(
            !token.is_cancelled(),
            "resource abort must not mutate the user's token"
        );
        let error = result.expect_err("resource pressure must abort local synthesis");
        if fallback {
            assert!(
                matches!(error, SdkError::AbortedForCloudFallback { reason } if reason == core_reason)
            );
        } else {
            assert!(
                matches!(error, SdkError::InferenceError { message, .. } if message.contains(reason_text))
            );
        }
    }

    // A consumer that closes delivery takes precedence over a simultaneous
    // resource abort, even when it has not supplied a cancellation token.
    let provider = Arc::new(PressureProvider {
        stressed: AtomicBool::new(false),
        snapshot: ResourceSnapshot {
            memory_pressure: MemoryPressure::Critical,
            ..Default::default()
        },
    });
    let options = RunOptions::new()
        .with_resource_provider(provider.clone())
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true),
        );
    let mut closed_packets = 0;
    let closed = model
        .run_tts_streaming(&short, &options, |_| {
            closed_packets += 1;
            provider.stressed.store(true, Ordering::Release);
            // Let the independent watcher observe pressure while the consumer
            // callback is active, then close rather than request cloud work.
            std::thread::sleep(Duration::from_millis(150));
            false
        })
        .expect("closing delivery must not trigger cloud fallback");
    assert_eq!(closed.status, TtsStatus::Cancelled);
    assert_eq!(closed_packets, 1);

    let mut reused_packets = 0;
    let reused = model
        .run_tts_streaming(&short, &RunOptions::new(), |_| {
            reused_packets += 1;
            true
        })
        .expect("resource aborts must drain and leave a reusable resident session");
    assert_eq!(reused.status, TtsStatus::Completed);
    assert!(reused_packets > 0);
}

#[test]
fn onnx_final_packet_pressure_keeps_completion_but_remaining_work_can_abort() {
    let bundle = std::env::var_os("XYBRID_ONNX_TTS_TEST_BUNDLE")
        .map(std::path::PathBuf::from)
        .or_else(|| integration_tests::fixtures::model_if_available("kokoro-82m"));
    let Some(bundle) = bundle else {
        eprintln!("Skipping ONNX terminal TTS test: Kokoro bundle unavailable");
        return;
    };
    let model = ModelLoader::from_directory(bundle).unwrap().load().unwrap();
    let short = Envelope::new(EnvelopeKind::Text("Your name sounds like trouble.".into()));
    for (signal, snapshot) in [
        (
            AbortSignal::MemoryPressureCritical,
            ResourceSnapshot {
                memory_pressure: MemoryPressure::Critical,
                ..Default::default()
            },
        ),
        (
            AbortSignal::ThermalCritical,
            ResourceSnapshot {
                thermal_state: ThermalState::Critical,
                ..Default::default()
            },
        ),
    ] {
        for fallback in [false, true] {
            let provider = Arc::new(PressureProvider {
                stressed: AtomicBool::new(false),
                snapshot,
            });
            let options = RunOptions::new()
                .with_resource_provider(provider.clone())
                .with_abort_policy(
                    AbortPolicy::default()
                        .stop_on(signal)
                        .with_cloud_fallback(fallback),
                );
            let mut packets = 0;
            let mut samples = 0;
            let result = model
                .run_tts_streaming(&short, &options, |packet| {
                    packets += 1;
                    samples += packet.pcm.len() as u64 / 2;
                    provider.stressed.store(true, Ordering::Release);
                    // Exceed the sampling interval so an erroneous post-final
                    // policy check would observe the new critical signal.
                    std::thread::sleep(Duration::from_millis(150));
                    true
                })
                .expect("pressure after final delivery must not restart finished speech");
            assert_eq!(packets, 1);
            assert_eq!(result.status, TtsStatus::Completed);
            assert_eq!(result.samples, samples);
            assert_eq!(result.chunks, 1);
        }
    }

    let token = CancellationToken::new();
    let result = model
        .run_tts_streaming(
            &short,
            &RunOptions::new().with_cancellation_token(token.clone()),
            |_| {
                token.cancel();
                true
            },
        )
        .expect("final-packet user cancellation is not a resource fallback");
    assert_eq!(result.status, TtsStatus::Cancelled);

    let provider = Arc::new(PressureProvider {
        stressed: AtomicBool::new(false),
        snapshot: ResourceSnapshot {
            memory_pressure: MemoryPressure::Critical,
            ..Default::default()
        },
    });
    let options = RunOptions::new()
        .with_resource_provider(provider.clone())
        .with_abort_policy(
            AbortPolicy::default()
                .stop_on(AbortSignal::MemoryPressureCritical)
                .with_cloud_fallback(true),
        );
    // Exceed the default 350-character chunk budget: there is still work after
    // packet one, so the next chunk's pre-inference resource check must abort.
    let long = Envelope::new(EnvelopeKind::Text(
        "The village gate stays closed until you tell me your true business. ".repeat(7),
    ));
    let mut packets = 0;
    let error = model
        .run_tts_streaming(&long, &options, |_| {
            packets += 1;
            provider.stressed.store(true, Ordering::Release);
            std::thread::sleep(Duration::from_millis(150));
            true
        })
        .expect_err("pressure must still stop remaining synthesis");
    assert_eq!(packets, 1);
    assert!(matches!(
        error,
        SdkError::AbortedForCloudFallback {
            reason: xybrid_core::abort::AbortReason::StressMemory
        }
    ));
}
