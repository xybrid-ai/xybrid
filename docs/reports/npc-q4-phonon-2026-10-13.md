# NPC Q4 comparison with resident Phonon — 13 October 2026

**Keep Phonon as the voice direction and E2B as the comparison baseline.** With upstream Phonon and the supplied clone, the AI reply becomes the main measured delay: median **2.25–2.81 seconds**, followed by **0.06–0.16 seconds** to finish the first voice clause. E4B's median reply was only **0.12 seconds** faster than E2B, with a larger observed memory footprint and no improvement on player-name/business or bribe recall. None of the three models is reliable enough to own those facts.

These are backend measurements on the user's M1 Pro Mac. Phonon is **not integrated into the Unity character**: the earlier native rc.4 checkpoint incompatibility remains. The current character still uses Kitten/WarChief. See the [voice survey](voice-latency-2026-10-13.md) and [Phonon clone compatibility report](phonon-clone-2026-10-13.md).

## Comparable inputs and runtime

Tested only **Q4_K_M**, as requested. A standalone .NET 10 harness loaded the exact installed **Xybrid Unity SDK 0.9.0 assembly and native library**, without upgrading the SDK or rebuilding a Unity player/native engine. The host has **10 CPU cores and 16 GiB unified RAM**; the consumer editor version is **6000.6.4f1**.

Each model received the same **2,698-character character card**, 17 emotion tags, JSON grammar, **8,192-token context**, and fixed Mira / selling turnips / attempted bribe / Passed / patience-3 handoff. Six questions covered player identity/business, earlier offer, camp, NPC identity, passage and an insult. Each question ran twice: Gemma3 → E2B → E4B, then reversed. History stayed empty; native conversation context contained only the system card.

Generation matched the demo: **112 tokens**, temperature **1**, top-p **0.95**, top-k **64**, min-p **0**, repetition penalty **1**, and the existing stop strings. A separate one-token “Greetings” call primed each new handle. Only one AI handle was held at a time; Phonon stayed resident throughout. Voice generation followed each complete valid AI reply, with no overlapping AI/voice compute and no audio reuse.

The installed AI backend is Metal-capable and retained automatic offload/helper-thread settings. Its benchmark log supplied no placement diagnostics, so actual per-model offload and AI thread counts were not independently verified. Phonon explicitly used **CPU, six threads**, upstream **ptts 0.4.0rc1**, checkpoint **7e71a02d@200**, Q8 speech weights, the supplied clone, temperature **0.3** and seed **0**.

### Required Metal asset preparation

E2B's existing demo file differed from the upstream hash because its documented preparation converts `per_layer_model_proj.weight` from BF16 to FP16: the installed Metal library lacks the required BF16 matrix kernel. Its prepared hash matched the archived integration guide exactly. E4B contained the same BF16 tensor and received the same conversion in temporary test storage. The converter verified **identical bytes for every quantized tensor**. Gemma3 contained no BF16 tensors and needed no conversion.

All tested files passed SHA-256 verification before inference; downloads and conversion finished first. This auxiliary compatibility conversion preserves Q4_K_M and is not a higher-precision quantization trial. Upstream and tested hashes, pinned repository revisions, exact prompts and grammar are in the [observations](npc-q4-phonon-2026-10-13.json).

## AI latency and memory

Seconds unless indicated. Reply statistics cover 12 calls per model. First token counts the first **nonempty Token event**, excluding terminal empty events. “AI + voice clause” is the median of each reply's paired sum, not a microphone measurement.

| Model | File size, GB | Prime, pass 1 / 2 | Complete reply median (range) | First token median | AI + voice clause median | Largest end-of-call RSS, GiB |
|---|---:|---:|---:|---:|---:|---:|
| Gemma3NPC | 4.24 | 24.89 / 6.65 | 2.81 (2.31–3.48) | 0.227 | 2.92 | 3.42 |
| Gemma4NPC E2B | 3.43 | 4.15 / 3.28 | 2.37 (1.56–3.10) | 0.097 | 2.46 | 3.60 |
| Gemma4NPC E4B | 5.34 | 6.81 / 3.13 | 2.25 (1.85–3.37) | 0.147 | 2.32 | 5.71 |

Pass-1 / pass-2 reply medians were **2.69 / 2.95** for Gemma3, **2.47 / 2.26** for E2B, and **2.29 / 2.25** for E4B. E4B was not consistently faster across all answers: its total reply time across 12 calls was **28.60 s**, versus E2B's **28.09 s**.

Output lengths differ. Median nonempty token-event counts were **44 / 51.5 / 39** for Gemma3 / E2B / E4B. Approximate post-first-token event rates were **17.1 / 22.5 / 18.5 events/s**, using `(events − 1) / (complete − first)`. These are stream-event proxies rather than independently validated tokenizer throughput. E4B's shorter median output helps explain its lower reply median; this does not establish a decoding-speed advantage.

RSS is the isolated benchmark process's resident memory at call completion, including runtime state. It is not total model allocation or GPU physical footprint; compression, paging and allocator retention affect it. In particular, Gemma3's lower RSS despite a larger file does not establish a lower total memory requirement.

## Phonon with each AI resident

All **36 freshly generated outputs** were finite and non-silent, 24 kHz mono PCM. Each clause's stream was exhausted before starting the next, and first-clause completion was timestamped inside the voice process. The test HTTP response returned after the entire utterance; it did not deliver the first clause to Unity for playback. Evidence WAV writing was outside the synthesis timer. Values below are medians over that AI's 12 replies; text lengths differed.

| Resident AI | First PCM packet | First complete clause | All clauses | Generated audio | RTF |
|---|---:|---:|---:|---:|---:|
| Gemma3NPC | 0.012 | 0.094 | 0.252 | 5.00 | 0.048 |
| Gemma4NPC E2B | 0.010 | 0.089 | 0.328 | 7.08 | 0.047 |
| Gemma4NPC E4B | 0.012 | 0.075 | 0.204 | 3.92 | 0.051 |

Across all replies, the first complete clause took **0.060–0.164 s**, and full synthesis **0.086–0.748 s**, for **1.76–14.00 s** of audio. RTF stayed between **0.044 and 0.065**. No severe voice stalls occurred in this run. The differing reply texts prevent attributing voice-time differences to the AI variant; the similarly low RTF supports Phonon as the preferred candidate under combined residency.

The “AI + voice clause” timing excludes local HTTP/parsing overhead, speech queue wait, Unity scheduling, playback and Rhubarb. It demonstrates backend readiness only. Voice intelligibility and clone fidelity still require listening review; numerical audio validity does not establish either.

## Dialogue quality

All **36/36 replies passed JSON/schema checks**, including known emotion tags, bounded intensity/patience delta and boolean scene-ending fields. Factual checks failed despite that:

| Direct factual check | Gemma3NPC | E2B | E4B |
|---|---:|---:|---:|
| Gives Mira's name and selling turnips, in the correct player role | 1/2 | 0/2 | 0/2 |
| Identifies the earlier approach as a bribe without substituting another offered object | 0/2 | 0/2 | 0/2 |
| Identifies the NPC as Trola | 2/2 | 2/2 | 2/2 |

- **Gemma3:** one correct “Your name is Mira” answer; the other evaded the name. Prior-offer answers invented a cart of turnips or “A shiny button.” Camp answers dodged the question. One passage answer explicitly recalled admission; the other substituted a turnip-bribe story.
- **E2B:** both identity answers said “My name is Trola. I sell turnips,” reversing the player business into the NPC role. Offer answers substituted turnips. One camp answer invented wolves and requested a new toll despite the Passed handoff. The other mixed established wall/helmet facts with unsupported claims.
- **E4B:** identity answers omitted Mira; offer answers were “Turnips. What's the fuss?” and “Worthless to me.” One camp answer reused the established wall-building fact but did not explain camp activity; the other introduced an axe arm. Passage answers acknowledged admission but also introduced payment/turnip claims.

The two insult replies applied delta −1 in **1/2, 2/2, 1/2** cases respectively. Simple punctuation-based inspection found more than two sentences in **6/12, 10/12, 6/12** replies. JSON grammar therefore does not enforce character facts, appropriate state updates, or the requested speech length. Two stochastic samples per question are enough to expose these failures, not to estimate general reliability.

## Host pressure and limits

Over the sampled **151.3 seconds**, the host swapped in **91.6 MiB** and swapped out **3,073.7 MiB**. Swap occupancy rose from approximately **13.25 to 16.15 GiB**. Most sampled swap-out activity occurred in the first Gemma3 pass and first E4B pass. These are host-wide counters: other applications stayed open, and stage boundaries were sampled about every two seconds, so causation cannot be assigned to a model alone.

This run avoided overlapping downloads, used consistent settings and reversed model order. It still lacks a quiet-host control or randomized repeats. Files were hash-read before testing, so prime is not disk-cold startup. The high first Gemma3 prime should be treated as an observed startup risk under this workload, not an architectural constant. The raw host data preserve these limits. Empty engine diagnostic logs prevent stronger GPU-placement claims.

## What to change

1. **Finish the Phonon native compatibility work.** Pin a slice that opens the newer official checkpoint, then validate the supplied clone through external registration or verified conversion. Resolve the shared `zzz_embed` plugin basename before installing it alongside Kitten. Preserve complete-clause playback and verify actual Unity onset, drain, cancellation and lip sync. The existing rc.4 native rejection is still a blocker.
2. **Keep E2B as the baseline; defer an E4B switch.** E4B adds 1.91 GB of file size and had higher observed RSS without fixing recall. Its 0.12-second median advantage is too small and output-dependent to justify replacing E2B from this survey alone. Gemma3's one successful player-name answer is also insufficient to justify a switch.
3. **Make handoff facts authoritative in game code.** Answer direct identity/business/bribe/passage questions from the run state, then add generated banter if useful. Keep admission and expulsion under game control. Evaluate any prompt revision separately against these exact failures before allowing the model to drive state changes.
4. **Enforce concise speech and cancel expired native work.** The two-sentence instruction was frequently ignored. Validate length before submitting voice, preserve complete sentences, and instrument queued-turn release, first complete clause, actual playback onset and last-clause completion. A UI timeout alone must not leave the serial native worker occupied.
5. **Budget combined residency on the 16 GiB host.** Monitor active paging, warm the selected AI before interaction, and test a smaller context only after measuring actual prompt/history usage. Include reload cost before choosing per-turn unloading. With tested Phonon, optimizing the AI's complete-reply wait matters more than shaving milliseconds from voice synthesis.

## Artifacts and verification

The [sanitized JSON](npc-q4-phonon-2026-10-13.json) contains all 48 lifecycle/reply rows, 36 raw replies and voice measurements, exact inputs, model pins, auxiliary conversion receipt, and 75 host samples. Weights, private voice files and evidence WAVs remain in gitignored test storage. The runner exited successfully and its AI, Phonon and wrapper processes all exited.

The existing **eight-second talking GIF** remains at `.context/voice-survey/troll-talking.gif`; it shows freshly synthesized **WarChief**, not Phonon. The demo fixes were already committed on the Mac as **6ca2ecd**. This comparison makes no production runtime change; two unrelated generated ProjectSettings changes remain uncommitted there.

The benchmark built with zero warnings/errors, and its installed-SDK/native probe passed. Required xybrid fmt, Clippy and workspace tests with `ort-download` passed earlier in this survey; this follow-up changes reports only. JSON/count/pin consistency checks and `git diff --check` passed before committing the finished comparison.
