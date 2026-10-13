# Phonon cloned-voice test — 13 October 2026

Follow-up: the [Gemma3NPC / Gemma4NPC E2B / E4B Q4 comparison](npc-q4-phonon-2026-10-13.md) tested this clone with each AI resident. The native Unity compatibility blocker described here remains unresolved.

The supplied voice synthesizes successfully with **upstream Phonon `ptts 0.4.0rc1`** and the available **7e71a02d@200 checkpoint**. Six measured replies generated finite, non-silent 24 kHz mono PCM in **0.09–0.38 seconds**, after warm-up. The existing zzz-integrations rc.4 native library rejected both the official newer GGUF and the custom clone GGUF with `InvalidRequiredMetadata`. Its original beta checkpoint still opens and synthesizes successfully.

Tests began after the voice survey and talking GIF were complete and committed as `eb3eca07`. This is an isolated runtime trial, not a Phonon integration into the Unity character.

## Compatibility and setup

The supplied 256,136-byte safetensors contains F32 `speaker_wavs` with shape **[1,512,125]**, metadata `sig=ba7a5a45`, `epoch=300`. This identifies the **speaker encoder**, not the speech language model. The newer checkpoint's documented encoder is also **ba7a5a45@300**. Treating that signature as an incompatible language-model ID would have been incorrect.

The older 202605 beta uses pre-projected `emb` [1,T,768] voices. Those are a different representation. The newer model projects speaker latents using its own 512→768 projection.

On the same M1 Pro Mac as the survey:

- Installed upstream `ptts==0.4.0rc1` and NumPy through an isolated `uv` environment with Python 3.12; no global Python package changes.
- Loaded the local checkpoint's own config, tokenizer and Q8 weights. Explicit CPU device, six threads, English normalization, temperature 0.3 and seed 0.
- Registered the supplied file with `tts.add_voice("trola_clone", path)`; registration took **0.00046 s**. Model opening took **0.064 s** and the separate warm-up took **0.186 s**.
- Used the same three short/medium/long texts as the Kitten/Kokoro survey, twice. Every call synthesized afresh.

The current C# `PhononSession` selects a voice inside the converted GGUF; it has no external-safetensors registration method. For the native trial, a custom GGUF was created in gitignored local test storage using the existing converter. The clone was registered as a test input with its measured SHA-256; existing model/config/tokenizer pins, encoder signature, dtype, shape and finite-data checks remained enforced. Staging used regular file copies because the converter rejects symlinks.

**No payload transpose is needed.** The channel-major safetensors payload is copied verbatim, with reversed GGUF dimension metadata **[125,512,1]**. The resulting private custom bundle is about 92.5 MB. Model weights, voice files, the custom bundle and raw private receipts are not committed.

## Actual clone measurements

First means the first streamed PCM array returned to the Python host, not Unity playback or first audible output. Total ends when the stream is exhausted and excludes writing the evidence WAV. Seconds, pass 1 / pass 2:

| Line | First PCM delivery | Complete synthesis | Audio duration | RTF |
|---|---:|---:|---:|---:|
| Short | 0.012 / 0.009 | 0.109 / 0.091 | 1.840 / 1.840 | 0.059 / 0.049 |
| Medium | 0.019 / 0.014 | 0.281 / 0.200 | 4.560 / 4.560 | 0.062 / 0.044 |
| Long | 0.020 / 0.017 | 0.384 / 0.318 | 7.600 / 7.600 | 0.051 / 0.042 |

All six outputs were finite and non-silent. Peak amplitudes were below 0.48; RMS was approximately 0.075–0.089. Generated WAVs are retained as evidence. Numerical validation establishes valid audio generation; intelligibility, timbre fidelity and suitability as Trola still need a listening review. The long-line sample is retained in `.context/phonon-survey/clone-sample.wav` in the cloud workspace, and all source WAVs remain under the Mac consumer's `.context/phonon-survey/`.

A minimal upstream reproduction, using the supplied checkpoint and voice path:

```python
import ptts

ptts.set_num_threads(6)
tts = ptts.TTS(config=checkpoint_directory, lang="en", device="cpu",
               quant="q8", temperature=0.3, seed=0)
tts.add_voice("trola_clone", clone_safetensors_path)
with tts.stream("Give me your name.", voice="trola_clone") as stream:
    for pcm in stream:
        consume_pcm(pcm)  # Copy/queue for playback; delivery is not audible onset.
```

## Pinned integration test

Followed the linked zzz-integrations [AGENTS](https://github.com/xybrid-ai/zzz-integrations/blob/main/AGENTS.md), README, integration lessons, TODO, validation, handoff and Unity setup. The Phonon library from the existing Space Jarv consumer was loaded by absolute path into an isolated process. It was never installed beside the Kitten plugin in the orc-demo editor.

The library's recorded source is **56385c4a3fbab8bc503341947af37e96ebb0071d**, Phonon profile **v0.1.0-rc.4**, ABI **1**. Its hash matched its local provenance. `supports_model("phonon")` returned 1; Kitten returned 0. Options retained native defaults: 512 frames/chunk, 256 chunk tokens, temperature 0.4, seed 42. This ABI exposes no Phonon thread override; no thread count was inferred.

| Input | Open result | Synthesis result |
|---|---|---|
| Official converted 7e71a02d@200, Freya | -3, `InvalidRequiredMetadata` | Not attempted after failed open |
| Validated custom 7e71a02d@200, supplied clone | -3, `InvalidRequiredMetadata` | Not attempted after failed open |
| Original 202605 beta, Marlowe | 0, successful (0.329 s) | 0, successful; 1.76 s audio, zero limited chunks |

The corrected original-beta probe delivered its first callback at **0.102 s** and completed its timed native call/shutdown at **0.393 s**. The probe used the exact ABI, including the nullable language argument, and kept callbacks rooted until return. It copied borrowed PCM, contained callback exceptions, retained a per-session cancellation watchdog and closed only after return. An initial harness call that omitted the language argument was corrected and is excluded from engine-failure evidence.

The standard newer model failed before the clone was involved, so the clone payload alone does not explain the rejection. ABI 1 and matching library basenames do not establish checkpoint/profile compatibility. The linked extraction's documented native recipe targets the original beta.

## What should change

1. **Update and pin a Phonon native slice that supports the speaker-latent checkpoint.** Validate the newer official model first, then the custom clone. Record a new engine/profile/model compatibility tuple; an ABI check alone is insufficient.
2. **Add validated cloned-voice import.** Either expose external voice registration through the native/C# boundary, or extend conversion to accept an explicit user voice while retaining the model, encoder, dtype, shape and finite-data checks. Do not reuse beta pre-projected-voice assumptions.
3. **Avoid `zzz_embed` plugin collisions.** The current Kitten and Phonon profiles share that basename. Use a deliberate combined build or distinct native names and corresponding wrapper imports when both providers must coexist.
4. **Keep complete-clause playback while testing the replacement.** Upstream can stream early, but this trial does not validate Unity buffering, Rhubarb, playback drain or interrupted speech. Integrate those deliberately and measure actual audio onset.
5. **Verify combined residency with the chosen AI.** The measured Phonon model is small and promising, but Kitten's stressed survey ran under different load. These trials do not prove a causal speedup with Gemma resident.

No zzz package code was changed, no native/player build was made, and no Phonon Unity or IL2CPP support is claimed. The extraction's lifecycle/cancellation/shutdown TODOs remain open. Source-package installation, fake-engine tests and ordinary successful speech cannot establish those acceptance conditions.

[Measured observations](phonon-clone-2026-10-13.json) omit private asset hashes and binary contents. The xybrid workspace's required fmt/Clippy/test checks passed during the survey; this follow-up changes documentation only and also passes `git diff --check`.
