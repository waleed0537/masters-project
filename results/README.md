# Results

## Live-call session, 24 September 2026

[`live_call_2026-09-24_results.log`](live_call_2026-09-24_results.log) is the unedited
`~/qemu_share/results.log` written by the guest watcher loop while the author spoke on live calls to
the Asterisk detection line (extension 600). Each block is one 4-second window that went
**call recording → frontend (normal world) → classifier (secure world) → verdict**.

> **Reading the log.** The client app that produced it printed the two logits under swapped labels.
> The model's output order is `[spoof, bonafide]`, so in this log the number printed as
> *"Bonafide score"* is the **spoof** logit and the number printed as *"Spoof score"* is the
> **bonafide** logit. The `Verdict:` line was always computed correctly (bonafide logit > 0 → bonafide).
> The label bug is fixed in `secure-world/trusted-app/host/main.c`.

| Verdicts in log | What they are |
|---|---|
| 1 – 17 | One finished recording (window "69–73 s") scored over and over. The file had stopped growing, so the same input was re-scored; these are **not** independent windows. |
| 18 – 24 | **Call 1**: windows 0–4 s … 26–30 s, advancing in real time (7 windows). |
| 25 – 33 | **Call 2**: windows 0–4 s … 32–36 s, advancing in real time (9 windows). |
| 34 – 36 | The last window repeated after the call ended (the loop does not detect end-of-call). |

**Summary.** 36 verdicts, of which **16 are distinct windows** from two live calls. All were classified
**SPOOF** for genuine live speech: bonafide logit between −4.2 and −5.4, spoof logit between +4.1 and +5.3.
This is the domain-generalisation limitation observed live (see below).

## Verification against the reference implementation

| Check | Result |
|---|---|
| PyTorch vs ONNX (reference clip `LA_E_1000048.flac`) | bonafide logit −5.6216 in both |
| ONNX vs standalone C (native x86) | identical logits `5.391142 / −5.621588` (`scripts/native_check.sh`) |
| C vs secure-world Trusted Application (QEMU/OP-TEE) | identical logits |
| Second clip, `LA_E_1001928.flac` | PyTorch −4.9850 vs secure world −4.985014 |
| Secure-world memory | full model needs 319.5 MB of workspace vs a ~16 MB secure pool, so only the classifier runs in the TA |

## Real-world audio (domain-generalisation study)

A model trained on ASVspoof is tested on real recordings (the author's own voice, a text-to-speech
spoof, recordings from the supervisor). Candidate causes were eliminated one at a time:

| Candidate cause | Finding |
|---|---|
| Sample rate | Already handled (resampled to 16 kHz). Ruled out. |
| Loudness | Recording was ~12 dB quieter and not peak-normalised. Normalising fixed that confound but not the misclassification. |
| Speaker accent | A native Irish speaker was misclassified in the same way. Weakened as the main explanation. |
| Background noise | Denoising barely moved the scores. Ruled out. |

What remains is a genuine domain-generalisation gap: the detector does not transfer from clean
ASVspoof audio to real recordings. One of the supervisor's clips scored close to the decision
boundary, so the gap behaves more like a spectrum than a wall.
