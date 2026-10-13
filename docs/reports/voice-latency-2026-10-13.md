# Orc demo voice latency — 13 October 2026

Voice synthesis and memory pressure dominate the observed waits. Changing the AI alone can shorten reply generation, but the fastest tested AI models also failed the character's recall or speaker-role requirements. Keep the current voice until a listening review accepts an alternative; first address shared-memory pressure and expired native work.

This is a survey and recommendation, with no production runtime change. The consumer is the Mac `orc-demo`, Unity 6000.6.4f1 Editor, Xybrid Unity SDK 0.9.0 and direct zzz Kitten profile v0.1.0-rc.4, on an M1 Pro with 10 CPU cores and 16 GiB unified memory. Cloud workspace checks use the repository's own version and do not validate that installed SDK.

## Where time goes

A queued submission waits for the current utterance. The AI then generates its **complete constrained JSON reply**. The serial voice worker must finish the first complete clause before playback starts on Unity's main thread. Subsequent clauses are generated in order and can leave pauses if they take longer than the preceding audio.

Rhubarb analyzes a cleaned copy concurrently; it does **not** gate audio. A display/preparation timeout does not cancel the outstanding native synthesis call, so later requests can wait behind expired work. A native stack sample during a 219-second call showed `zzz_embed_synthesize` and compute workers active, rather than an input-button wait.

Kitten uses CPU/Accelerate with six explicit threads. Gemma is configured through the installed Metal-capable llama backend. These share RAM on Apple Silicon: separate compute devices do not isolate memory. Accelerate's internal threads and the AI's CPU helper threads are automatic/unreported. Per-model offload and ONNX thread placement were not independently profiled.

## Voice comparison

Three identical lines were tested twice, after a separate warm-up:

1. “Give me your name.”
2. “The gate is open, Mira. Keep those turnips away from my boots.”
3. “You tried a bribe, and I let you through. Sell your turnips in the camp, then leave me to guard this gate in peace.”

First below means **first complete clause returned**, not first native packet or audible onset. Total includes every clause; RTF is total synthesis / generated audio duration. Values are seconds, pass 1 / pass 2.


First means the first complete clause returned. Each cell lists pass 1 / pass 2; missing values mean the resident run was stopped after severe stalls.

| Voice / AI state | Line | First clause | All clauses | Audio | RTF |
|---|---|---:|---:|---:|---:|
| Kitten-WarChief / AI-unloaded | Short | 1.33 / 2.54 | 1.33 / 2.54 | 1.08 / 1.08 | 1.23 / 2.36 |
| Kitten-WarChief / AI-unloaded | Medium | 2.11 / 2.39 | 6.56 / 6.89 | 5.12 / 5.12 | 1.28 / 1.35 |
| Kitten-WarChief / AI-unloaded | Long | 2.20 / 2.30 | 9.96 / 27.33 | 9.28 / 9.28 | 1.07 / 2.94 |
| Kokoro-Fenrir / AI-unloaded | Short | 4.65 / 0.70 | 4.65 / 0.70 | 1.36 / 1.36 | 3.41 / 0.51 |
| Kokoro-Fenrir / AI-unloaded | Medium | 6.50 / 0.73 | 20.41 / 2.82 | 5.04 / 5.04 | 4.05 / 0.56 |
| Kokoro-Fenrir / AI-unloaded | Long | 4.77 / 0.64 | 17.39 / 3.15 | 7.81 / 7.81 | 2.23 / 0.40 |
| Kitten-WarChief / Gemma4-resident | Short | 30.77 / 76.40 | 30.77 / 76.40 | 1.08 / 1.08 | 28.49 / 70.74 |
| Kitten-WarChief / Gemma4-resident | Medium | 243.84 / 35.21 | 394.22 / 263.36 | 5.12 / 5.12 | 77.00 / 51.44 |
| Kitten-WarChief / Gemma4-resident | Long | 71.11 | 419.76 | 9.28 | 45.23 |


Kitten open/warm-up: **10.69 / 5.67 s** unloaded, **61.69 / 205.03 s** with Gemma4 retained. Kokoro open was **1.57 s**. The resident run was stopped after severe stalls; its second long sample and resident Kokoro were not reached. The intentionally induced final permission error is excluded from backend failures.

One unsplit long Kitten observation took **219 s** for **8.08 s** of audio (RTF 27.1). Clause versions took **9.96 / 27.33 s** for **9.28 s** of audio. Different audio durations and changing host load prevent claiming a guaranteed clause speedup. Complete-clause playback remains useful because it permits earlier safe playback.

Kokoro's second pass was below RTF 1 and is a promising voice alternative, subject to listening review: Fenrir does not preserve WarChief's timbre. Its slower first pass illustrates why warm-up and host load must be recorded. These data do not establish a clean causal ranking.

Current WarChief uses Kitten Q2_0 language weights and an F32 meanflow decoder. Kitten Metal, selectable stable/expressive decoding presets, and a lighter supported quantization were not exposed by the installed SDK/direct provider; those comparisons could not be performed. No SDK upgrade or native/player build was made.

## AI comparison

Every model received the exact 2,698-character character card, the fixed Mira/turnips/bribe/passage/patience-3 handoff, the same grammar and 112-token cap. Three questions were asked twice. This tests a fixed scenario rather than an evolving live encounter. Handle creation was lazy; prime includes actual first inference. First-event timings are the first non-complete stream event counted by the helper, not verified token or audible-output timings.

| Model | Prime | Complete reply range | Median reply | First event range | Parsed replies |
|---|---:|---:|---:|---:|---:|
| Gemma4NPC | 22.53 | 1.51–11.35 | 4.69 | 0.12–0.26 | 6/6 |
| Gemma3NPC | 13.72 | 2.80–4.13 | 3.39 | 0.20–0.48 | 6/6 |
| LFM2.5-230M | 0.72 | 0.43–0.57 | 0.50 | 0.10–0.11 | 6/6 |
| SmolLM2-360M | 0.92 | 0.46–0.84 | 0.54 | 0.03–0.04 | 6/6 |

All replies parsed, but semantic quality failed:

- **Gemma4:** correctly answered player identity/business in one of two attempts; both prior-offer replies substituted turnips for the bribe.
- **Gemma3:** omitted Mira's name in both identity replies. Offer replies substituted turnips or invented a shiny button.
- **LFM:** returned “Mira” without the business, then “My name is Mira,” confusing player and NPC. Another reply reversed who offered the business/bribe; gate replies echoed the question.
- **SmolLM2:** confused identity or asked for the name again. Both identity outputs requested `ends_scene=true`; offer and gate replies were incorrect or uninformative.

Neither tiny model is ready as a drop-in NPC replacement. Gemma3 loaded and ran, but did not solve recall. Its pinned Q4_K_M GGUF is 4,237,062,304 bytes, architecture `gemma3n`, from `chimbiwide/Gemma3NPC-it-Q4-GGUF` revision `ac156eee306a9437ac702a6354f7c884f0e4a949`, SHA-256 `8d1d1a7f693237fcc6515274d2a6665fe867471a1d95005a39e21ec1a3c8c384`, verified on both machines. LFM and Smol used their cached Q4_K_M bundles. Their smaller resident footprint was not measured with Kitten in this survey.

## Host pressure and limits

Over 45.56 seconds, 38,027 16-KiB pages swapped in (**about 594 MiB**) and 51,020 swapped out (**about 797 MiB**). Swap occupancy was approximately **19 GiB** later in the session. Active paging supports investigating memory pressure; swap occupancy alone would not prove current contention.

The host workload varied and some early measurements overlapped model downloads. Unloaded and resident conditions were sampled sequentially, without randomization or a quiet-host control. Do not attribute all deterioration to Gemma residency or treat the ratios as causal speedups. Unrelated apps were left running. No first-audible-output/microphone timing was measured.

## Changes to prioritize

1. **Set a combined memory budget and verify a quiet-host run.** Monitor paging. Keep models resident only when they fit; assess unloading a large AI during speech or using a smaller AI after quality checks. Include reload/prime cost before choosing per-turn unloading.
2. **Offer Kokoro as a selectable voice and review the timbre.** Preserve Kitten startup, the paired WarChief embeddings, six threads, Accelerate and complete-clause playback until a replacement is approved.
3. **Cancel expired native work.** Tie a watchdog to the current call, await its return before disposal/reopen, and prevent incomplete output from being played as a complete clause. UI timeout alone cannot free the serial worker.
4. **Shorten replies and overlap stages carefully.** The current pipeline waits for the full JSON. An incremental parser could submit the earliest validated complete sentence, preserving escaping, emotion validation, patience clamps and scene authority. Measure this before promising lower latency.
5. **Keep player facts authoritative.** Gate identity/business/bribe recall and speaker roles separately from JSON parsing. Avoid allowing a fast but incorrect model to drive encounter endings.
6. **Instrument boundaries.** Record submission, queued-turn release, AI prime, first event, complete reply, speech queue/start, first complete clause, actual playback onset, last-clause completion, Rhubarb and cancellation. Total voice RTF alone conceals queue wait and gaps.

## Recording and verification

An eight-second Unity GameView recording was made from freshly synthesized WarChief speech after AI handles were disposed. Its line took **18.35 s** to synthesize for **7.64 s** of audio (RTF **2.40**). It is demonstration evidence, not another controlled benchmark. The capture produced 57 frames across eight seconds; the GIF preserves the eight-second duration. A contact sheet confirmed visible mouth movement. GIFs have no audio; saved WAVs were evidence only and were never replayed as a voice cache.

Temporary editor survey/capture scripts were removed after capture and Play-mode exit. The pre-existing consumer fixes were preserved. No runtime files or model binaries are committed by this survey. [Raw observations](voice-latency-2026-10-13.json) preserve boundaries and limitations; the recording lives in the workspace's gitignored `.context/voice-survey/troll-talking.gif`.

Repository checks passed: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace --features ort-download`. Model-gated tests may skip absent fixtures; these checks do not replace the real Mac observations above.
