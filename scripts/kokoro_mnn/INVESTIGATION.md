# Kokoro on MNN — investigation notes

Goal: Kokoro-v1.0 TTS currently runs 0.95–1.05× real-time on the target Android
phone via `piper-rs` + onnxruntime, using the int8-quantized ONNX. The codebase
already uses MNN for OCR with a 2–3× speedup over ORT on ARM. Question: can we
get a similar speedup for Kokoro by porting the model to MNN, and is the
engineering cost worth it?

Conclusion up front: **the duration bug is not in Kokoro's math path.** The
MNN-converted model was being fed / converted with `input_ids` as ONNX `INT64`,
and MNN 3.5 miscomputes the first word-embedding `Gather` with int64 indices.
That poisoned the whole encoder and showed up later as wrong duration
prediction. Narrowing `input_ids` to ONNX `INT32` before MNN conversion makes
the extracted duration predictor match ORT and fixes the full-model sample
count.

Current status: fp32 ONNX → fp32 MNN is viable for parity only on the While-LSTM
conversion, or on the first inference of the native-LSTM conversion. The native
MNN LSTM conversion mutates recurrent state across repeated `Module.forward()`
calls; warmups hide the good first run and make the second and later runs sound
wrong. The stock `mnnquant` command in MNN 3.5 has a Python wrapper bug and
aborts before quantization; bypassing it with `_tools.mnnquant([...])` lets
native-LSTM PTQ complete. The resulting full PTQ model is still not usable:
duration/pacing/intonation are subjectively much closer than the old int64
`Gather` failure, but audio quality is severely degraded, like an 8 kHz /
A-law style codec artifact. Converter-side weight-only int8
(`--weightQuantBits 8`) works, especially on the stable While-LSTM conversion,
but that is compression / weight-only quant, not the full activation int8 path
we wanted.

Terminology note: artifact names containing `i32` mean only that Kokoro's
`input_ids` token-index input was narrowed from ONNX `INT64` to `INT32` to avoid
MNN's `Gather` bug. They do **not** mean the model weights or activations are
int32. The best current artifact,
`kokoro-v1.0.patched.i32.while.wq8.block128.mnn`, uses `INT32` token ids,
fp32 `style`/`speed` inputs, While-subgraph LSTMs, and converter weight-only
8-bit quantization. It is not a full activation-int8 model like ORT's dynamic
int8 ONNX.

## Experimental setup

All artifacts live in `scripts/kokoro_mnn/`. Python env via `uv`:

- `MNN==3.5.0` (provides both `mnnconvert` and the Python inference runtime)
- `onnxruntime==1.26.0`
- `onnx==1.21.0`, `onnx_graphsurgeon==0.6.1`
- `numpy==2.4.4`

Bench harness: `bench.py` runs identical (phonemized) input through ORT and
MNN, reports latency and audio parity (sample-MAE, spectral relative MAE,
band-energy correlations, dumps WAVs). `bench_onnx.py` runs only ORT across
multiple model variants. `patch_resize.py` applies graph patches via
`onnx_graphsurgeon`. `probe_duration.py` is a temporary bisection helper that
rewrites selected intermediate tensors as graph outputs, converts the temporary
ONNX to MNN, and compares ORT vs MNN tensor values. `make_quant_calib.py`
generates MNN sequence-input calibration folders under `/tmp` for `mnnquant`.

Test sentence: `"The quick brown fox jumps over the lazy dog."`, en-us espeak
phonemes (52 tokens including 0-padding).

Reference voice style: row from `voices-v1.0.bin` at index
`min(token_count, 510)`, matching `piper-rs/src/kokoro.rs` behavior.

Host hardware: Linux desktop, x86_64, 16 logical cores (4 threads used in
benches). **Note: this is not the deployment target.** MNN is tuned for ARM;
final numbers must come from the phone.

## Phase 0 — what was tried first

### 1. int8 ONNX → MNN: refused

The shipped model `kokoro-v1.0.int8.onnx` (88 MB) uses ORT-emitted dynamic
quantization. `mnnconvert` rejects it with:

```
These Op Not Support: ONNX::ConvInteger | ONNX::DynamicQuantizeLSTM
  | ONNX::FastGelu | ONNX::FusedMatMul | ONNX::SkipLayerNormalization
```

These are valid ONNX ops but ORT-favored. MNN has its own quantizer
(`mnnquant`) intended to be run on an fp32 MNN model. So the workable path is
fp32 ONNX → fp32 MNN → mnnquant → int8 MNN.

### 2. fp32 ONNX → MNN: converts cleanly

Downloaded `kokoro-v1.0.onnx` from `onnx-community/Kokoro-82M-v1.0-ONNX`
(325 MB, ir_version 9, opset 20). Conversion succeeded:

- Default LSTM strategy: LSTMs unrolled into `While` subgraphs. Requires
  MNN's `Module` API at runtime.
- With `--useOriginRNNImpl`: native MNN LSTM op, no subgraphs.
- With `--fp16`: half-precision weights/activations. **Crashes at runtime** on
  `Broad cast error, dim1 = 89400, dim2 = 89100` inside
  `/decoder/decoder/generator/m_source/l_sin_gen/Mul_6`. Off-by-one F0 frame
  (89400 − 89100 = 300 = one frame at hop=300).

Resulting file sizes:

| Variant                              | Size    |
|--------------------------------------|---------|
| `kokoro-v1.0.mnn` (fp32, While LSTM) | 310 MB  |
| `kokoro-v1.0.rnn.mnn` (fp32, native) | 310 MB  |
| `kokoro-v1.0.fp16.mnn`               | 156 MB  |

## Phase 1 — performance (x86, indicative only)

ORT 4 threads, "Quick brown fox" → 80400 samples (3.35 s of audio):

| Model                          | File size | Median latency |
|--------------------------------|-----------|----------------|
| `kokoro-v1.0.int8.onnx` (ship) | 88 MB     | **3198 ms**    |
| `model_fp16.onnx`              | 156 MB    | 758 ms         |
| `kokoro-v1.0.onnx` (fp32)      | 310 MB    | **701 ms**     |
| `model_q8f16.onnx`             | 82 MB     | crashes (load) |
| MNN fp32 (native LSTM)         | 310 MB    | **930 ms**     |
| MNN fp32 (While LSTM)          | 310 MB    | 970 ms         |

**On x86, MNN is ~33% slower than ORT for this model.** Expected — MNN
optimizes for ARM. The shipped int8 ONNX is ~4× slower than fp32 here because
this desktop CPU lacks `i8sdot`/AVX-VNNI; on the target phone it pays off.

## Phase 2 — parity was broken

Across every working MNN variant, ORT vs MNN audio diverges badly:

| Metric                           | Value           |
|----------------------------------|-----------------|
| Sample count (ORT / MNN)         | 80400 / 82800   |
| Sample-MAE (first 80400)         | ~0.064          |
| RMS_ort / RMS_mnn                | 0.079 / 0.076   |
| Spectral L1 relative MAE         | ~0.65           |
| 8-band energy correlation        | band 0 ~0.5, bands 1–7 ~0 |

By ear (operator listening to `out_ort.wav` vs `out_mnn.wav` from `bench.py`):

- ORT: clean, intelligible English; "the quick brown fox jumps over the lazy
  dog".
- MNN: clean voice, recognizable words, **dragged-out pacing / drunk-sounding
  intonation**, ends mid-sentence on "the quick brown fox j-". Not static or
  noisy — the decoder body sounds healthy, the timing path is wrong.

### What we ruled out

Each hypothesis below was tested in sequence; each was eliminated by direct
measurement.

#### 2.1 Resize ops in the F0/upsample path

Six `Resize` nodes in the graph emit "empty input at index 1" warnings during
conversion:

- `/encoder/F0.1/upsample/Resize` — nearest, scale=2
- `/encoder/N.1/upsample/Resize` — nearest, scale=2
- `/decoder/.../f0_upsamp/Resize` — nearest, scale=300
- `/decoder/.../l_sin_gen/Resize` — linear, scale=1/300 (downsample)
- `/decoder/.../l_sin_gen/Resize_1` — nearest, scale=300
- `/decoder/.../decode.3/upsample/Resize` — nearest, scale=2

The "empty input at index 1" is **roi**, which ONNX explicitly allows to be
empty when `coordinate_transformation_mode != tf_crop_and_resize`. So the
warning is benign.

Patched the graph anyway via `patch_resize.py`:
- For nearest+integer scales, build an explicit `sizes` tensor as
  `Shape → Slice(time dim) → Mul(factor) → Concat([N,C,T_new])`, clear
  `scales`.
- For the linear downsample-by-300, same treatment with `Div`.

Result: patched ONNX is bit-identical to original under ORT (MAE = 0.0,
sample count matches). MNN output: **unchanged** — still 82800 samples,
same MAE 0.064. **Resize ops are innocent.**

#### 2.2 ONNX Round vs MNN Round

ONNX Round spec uses banker's rounding (round half to even). MNN's runtime
implementation is round-half-away-from-zero. Verified ORT's behavior:

```
input : [0.5, 1.5, 2.5, 3.5, 4.5]
ORT   : [0.0, 2.0, 2.0, 4.0, 4.0]    # banker's
```

Original hypothesis: `/encoder/Round` rounds per-phoneme float durations;
~8 phonemes with `x.5` values × +1 frame each under round-half-up would give
the +8 frame offset we saw.

Patched: replaced the single `Round` node with `Add(x, 0.5) → Floor(...)`,
which both runtimes implement identically. Verified the patch survives in
the saved ONNX (`encoder_Round/x_plus_half → /encoder/Round_as_floor`).

Result: **unchanged**. Same 82800/80400 split, same MAE 0.064.

#### 2.3 LSTM implementation

Cross-checked native MNN LSTM (`--useOriginRNNImpl`) vs unrolled While
subgraph. If the bug were in MNN's native LSTM op, the While version would
behave differently (using only primitive MatMul + sigmoid + tanh which are
well-tested elsewhere). It doesn't:

- Native LSTM:   MAE 0.0637, samples 82800
- While LSTM:    MAE 0.0624, samples 82800

Two completely different LSTM execution paths produce essentially the same
divergence. **LSTM is not the unique culprit.**

#### 2.4 Converter fusion passes

`mnnconvert` runs optimization at convert time. Tried `--optimizeLevel 1`
(safe-only fusions) vs the default `--optimizeLevel 2` (aggressive):

| Level | Samples | MAE    | Spectral rel |
|-------|---------|--------|--------------|
| 1     | 82800   | 0.0637 | 0.65         |
| 2     | 82800   | 0.0637 | 0.65         |

(`--optimizeLevel 0` only works on already-MNN-source models, not on the
ONNX import path.)

**Fusion passes are not the cause.** Bug is in the runtime, not the optimizer.

### 2.5 First conclusion — duration predictor body

Extracted a subgraph from `kokoro-v1.0.patched.onnx` ending at the duration
predictor's outputs (`/encoder/Div_output_0` raw float, `/encoder/Round_output_0`
post-round, `/encoder/Cast_output_0` integer durations) and ran both ORT and
MNN on identical inputs.

Float durations *before* rounding (54 phonemes, first/last shown):

```
ORT div_out: 11.74,  1.07,  1.37,  2.00,  2.00, 1.0,  1.20, 1.99, ...  6.24, 14.61
MNN div_out: 13.09,  2.39,  2.00,  2.00,  1.72, 1.32, 1.95, 2.44, ...  4.45,  8.60
```

- Max single-phoneme |diff|: **6.01 frames**.
- Sum ORT: 134 frames. Sum MNN: 138 frames.
- Round diff (max): 6.0. So the divergence is real numerical wrongness, not
  rounding artifacts.

This showed that the Round op was not the cause. But this was not yet the real
root cause. The duration predictor was receiving already-corrupted encoder
features.

The audio symptom maps cleanly:
- MNN allocates an output buffer for ~276 frames (82800 samples).
- The first ~268 frames carry actual audio (matching the buffer size ORT
  computes), with the last 8 frames zero-padded.
- But within those frames MNN spreads per-phoneme durations differently:
  some phonemes get 6 frames instead of 1, others get fewer, and the
  cumulative sum runs out before all 52 phonemes have been emitted.
- Result: the sentence is stretched and gets cut off at "...fox j-".

## Phase 3 — actual root cause: MNN `Gather` with `INT64` indices

`probe_duration.py` compared progressively earlier intermediate tensors. With
the original `INT64` input graph, the first bad tensor is node 1:

```
0001 Gather /encoder/bert/embeddings/word_embeddings/Gather
     ORT=(1, 54, 128) MNN=(1, 54, 128) mae=0.0645631 max=0.39547
```

That is the first embedding lookup:

```
Gather(
  encoder.bert.embeddings.word_embeddings.weight,
  input_ids
)
```

So the previous duration mismatch was downstream damage from the embedding
lookup, not an LSTM / Conv1d / LayerNorm semantic bug.

There are two related hazards:

1. In the Python runtime, `F.int` is 32-bit. Feeding `np.int64.tobytes()` with
   `F.int` gives interleaved zeros:

   ```
   ids int64:       [0, 46, 83, 16, 53]
   F.int readback:  [0,  0, 46,  0, 83]
   ```

   `bench.py` now chooses `F.int` vs `F.int64` from the ONNX input type.

2. Even with a correct `F.int64` feed, MNN 3.5's converted `Gather` does not
   match ORT when the indices input is ONNX `INT64`. Kokoro token ids are small,
   and ONNX `Gather` accepts int32 indices, so the practical fix is to narrow
   only the `input_ids` graph input to `INT32` before conversion.

Validation with a temporary duration ONNX whose `input_ids` graph input was
changed to `INT32`:

```
/encoder/Cast_output_0  mae 0.0        sum ORT=134 MNN=134
/encoder/Round_output_0 mae 0.0        sum ORT=134 MNN=134
/encoder/Div_output_0   mae 8.91e-06   max 2.35e-04
```

Full model validation after regenerating `kokoro-v1.0.patched.i32.onnx` and
`kokoro-v1.0.patched.i32.mnn` for a first inference:

```
ORT samples=80400 MNN samples=80400
MAE=0.00244
RMS_ort=0.0787 RMS_mnn=0.0787
spectral_mae_rel=0.0512
```

This is the key fix needed before trying fp32 MNN → int8 MNN.

Later testing found an important qualifier: the native-LSTM MNN model is not
stable across repeated forwards on the same loaded module. With copied outputs,
`probe_stability.py` shows the first native-LSTM fp32 inference is good, but
all later inferences differ from it by MAE ~0.054. The While-LSTM fp32 model is
stable across repeated forwards (MAE 0.0 vs first run). So use
`kokoro-v1.0.patched.i32.while.mnn` as the fp32 correctness reference unless
the runtime recreates the module for every utterance.

## Phase 4 — old bisection plan, now obsolete

Bisection: extract progressively smaller subgraphs from the duration
predictor (the BiLSTM + Conv1d chain inside `/encoder/predictor/`) and find
the first node where ORT and MNN disagree. Candidates:

- Conv1d with specific padding/dilation
- LayerNorm with non-default axis
- Some op chain inside `/encoder/predictor/lstm` or
  `/encoder/predictor/text_encoder/lstms.{0,2,4}`

Once found, either rewrite that op as equivalent primitives in the ONNX
(`onnx_graphsurgeon`) before conversion, or file the bug upstream against
MNN.

This is no longer the right direction. The first mismatch is before the
duration predictor, in the word embedding `Gather`. The local patch is small:
make `input_ids` `INT32` before conversion.

## Remaining risks before a user-visible win

Three risk points remain:

1. **`mnnquant` from fp32 MNN.** MNN does full PTQ (weights + activations),
   not dynamic int8 like ORT. Quality on Kokoro is uncertain; the F0
   generator and iSTFT subnets would need to be on `skip_quant_op_names` to
   preserve audio fidelity.

2. **Rust integration.** `src/mnn_inference.rs:30` exposes only a
   single-input/single-output Interpreter API
   (`run_dynamic_raw(input, shape)`). Kokoro needs 3 inputs / 1 output.
   Either extend the `mnn-sys` fork's `InferenceEngine` for multi-input, or
   migrate to MNN's `Module` API. Then add a `KokoroMnnModel` parallel to
   `KokoroModel` in `piper-rs`, dispatched on `.mnn` extension in
   `speech.rs::load_speech_model`. 1–2 days.

3. **ARM speedup unknown.** OCR's 2–3× on MNN comes from Conv2d, where MNN's
   ARM kernels shine. Kokoro is matmul + Conv1d + 1× STFT + LSTM heavy. The
   ratio may be much smaller — or MNN may even lose to ORT-int8-on-ARM
   thanks to `i8sdot`.

## Phase 5 — int8 MNN attempts

### 5.1 Full offline PTQ with `mnnquant`: wrapper bug, then quality bug

MNN's docs say sequence-input quantization uses a directory containing
`input_0`, `input_1`, ... subdirectories, each with per-input `.txt` files and
an `input.json` describing input shapes. `make_quant_calib.py` now generates
that format for Kokoro. Important detail: put `quant.json` **outside** the
sample directory. MNN's `readClibrationFiles()` treats every file or directory
under `path` as a calibration sample, so if `quant.json` lives in the sample
directory the quantizer tries to parse it as sample input and crashes.

```
/tmp/kokoro-mnn-quant-calib-clean/input_0/input_ids.txt
/tmp/kokoro-mnn-quant-calib-clean/input_0/style.txt
/tmp/kokoro-mnn-quant-calib-clean/input_0/speed.txt
/tmp/kokoro-mnn-quant-calib-clean/input_0/waveform.txt
/tmp/kokoro-mnn-quant-calib-clean/input_0/input.json
/tmp/kokoro-mnn-quant-clean-ema.json
```

The config uses:

```json
{
  "input_type": "sequence",
  "feature_quantize_method": "EMA",
  "weight_quantize_method": "MAX_ABS",
  "batch_size": 1,
  "quant_bits": 8
}
```

Attempting PTQ through the stock `mnnquant` CLI:

```sh
uv run mnnquant kokoro-v1.0.patched.i32.mnn \
  kokoro-v1.0.patched.i32.int8.mnn \
  /tmp/kokoro-mnn-quant-clean-ema.json
```

fails immediately:

```
terminate called after throwing an instance of 'std::length_error'
  what():  cannot create std::vector larger than max_size()
```

This is a PyMNN wrapper bug, not a model bug. GDB shows:

```
std::vector<char const*>::vector(__n=18446744073709551615)
PyTool_Quantization at /project/pymnn/src/MNNTools.cc:50
```

Local MNN source confirms `PyTool_Quantization` expects its first Python
argument to be a list and calls `PyList_Size(args)`. `mnnconvert.py` does this
correctly:

```py
Tools.mnnconvert(sys.argv)
```

But `mnnquant.py` calls the C++ wrapper with three separate strings:

```py
Tools.mnnquant(src_mnn, dst_mnn, config_json)
```

For a Python string, `PyList_Size()` returns `-1`; that is cast to the huge
unsigned vector length above. Workaround:

```py
import _tools
_tools.mnnquant([
    "mnnquant",
    "kokoro-v1.0.patched.i32.mnn",
    "kokoro-v1.0.patched.i32.ptq-ema.mnn",
    "/tmp/kokoro-mnn-quant-clean-ema.json",
])
```

With the clean calibration folder and direct `_tools.mnnquant([...])` call,
native-LSTM PTQ completes:

```
[00:58:49] @ 1761: Quantize model done!
```

The generated model is not usable yet:

| Variant | Size | Samples | MAE | Spectral rel |
|---------|------|---------|-----|--------------|
| fixed fp32 native LSTM | 311 MB | 80400 | 0.00244 | 0.0512 |
| PTQ EMA native LSTM | 135 MB | 82200 | 0.05649 | 1.4277 |

The `82200` sample count is only ~75 ms longer than ORT at 24 kHz and does not
match the old failure mode. Listening feedback: duration, pacing, and intonation
are correct; the problem is very poor fidelity, described as an 8 kHz / A-law
style low-quality rendition. So this is now a decoder/vocoder quantization
quality issue, not a duration bug and not the earlier int64 `Gather` conversion
bug.

Regenerated the fixed model without `--useOriginRNNImpl`, so LSTMs become
`While` subgraphs:

```sh
uv run mnnconvert -f ONNX \
  --modelFile kokoro-v1.0.patched.i32.onnx \
  --MNNModel kokoro-v1.0.patched.i32.while.mnn
```

The While model still has good fp32 parity:

```
ORT samples=80400 MNN samples=80400
MAE=0.00246
spectral_mae_rel=0.0516
```

The direct `_tools.mnnquant([...])` call still segfaults on the While model.
Backtraces:

- EMA: `Calibration::_quantizeModelEMA` →
  `ConvBNReluFusedModule::fakeQuantFeatureWithMinMax`.
- KL: `Calibration::_computeFeatureMapsRange` callback, inside
  `_featureInfo.find(weakPtr)`.

So the While/subgraph route is still blocked in MNN's quantizer.

Tried skipping every MNN op whose name contains `/encoder/` to preserve the
duration path in fp32. Quantization completes, but the resulting file is
byte-identical to the non-skip PTQ output:

```
sha256 kokoro-v1.0.patched.i32.ptq-ema.mnn
       kokoro-v1.0.patched.i32.ptq-ema-skip-encoder.mnn
       kokoro-v1.0.patched.i32.ptq-ema-skip-decoder.mnn
=> all f8292724a36de8396aae0c5b355dbfd63405931d2a72a55cb33de298e1012686
```

So these skip experiments did not actually skip anything in this MNN EMA PTQ
path; `skip_quant_op_names` is either ignored by `NN::turnQuantize(...)`, not
matching the names that matter, or only affects a different code path.

The observed benchmark for the skipped artifact is therefore not meaningful as
a skip result:

| Variant | Size | Samples | MAE | Spectral rel |
|---------|------|---------|-----|--------------|
| PTQ EMA native LSTM, skip `/encoder/` ops | 135 MB | 82200 | 0.04167 | 1.0000 |

The output RMS was effectively zero in the benchmark, so this skip strategy is
not viable.

Tried the opposite coarse skip: keep `/decoder/decoder/` ops in fp32, quantize
the rest. Quantization completed:

```py
_tools.mnnquant([
    "mnnquant",
    "kokoro-v1.0.patched.i32.mnn",
    "kokoro-v1.0.patched.i32.ptq-ema-skip-decoder.mnn",
    "/tmp/kokoro-mnn-quant-clean-ema-skip-decoder.json",
])
```

But this artifact is also byte-identical to the full PTQ model, so it does not
test the hypothesis that the codec-like degradation comes from quantizing the
decoder/generator/vocoder path.

Tried a fixed-shape static conversion as a possible workaround for dynamic
shape handling in `mnnquant`:

```txt
input_size = 3
input_names = input_ids,style,speed
input_dims = 1x54,1x256,1
```

Both static conversions segfault after printing the model inputs/outputs:

```sh
uv run mnnconvert -f ONNX \
  --modelFile kokoro-v1.0.patched.i32.onnx \
  --MNNModel kokoro-v1.0.patched.i32.static54.mnn \
  --saveStaticModel \
  --inputConfigFile /tmp/kokoro-static54-config.txt \
  --useOriginRNNImpl

uv run mnnconvert -f ONNX \
  --modelFile kokoro-v1.0.patched.i32.onnx \
  --MNNModel kokoro-v1.0.patched.i32.static54.while.mnn \
  --saveStaticModel \
  --inputConfigFile /tmp/kokoro-static54-config.txt
```

Both exit with code 139. No `.mnn` file is produced. So static-shape conversion
does not currently unblock the While/subgraph quantization path.

### 5.2 Converter weight-only int8 works, but is not full int8 PTQ

`mnnconvert --weightQuantBits 8` can emit smaller MNN files:

```sh
uv run mnnconvert -f ONNX \
  --modelFile kokoro-v1.0.patched.i32.onnx \
  --MNNModel kokoro-v1.0.patched.i32.wq8.mnn \
  --weightQuantBits 8 \
  --useOriginRNNImpl

uv run mnnconvert -f ONNX \
  --modelFile kokoro-v1.0.patched.i32.onnx \
  --MNNModel kokoro-v1.0.patched.i32.while.wq8.mnn \
  --weightQuantBits 8
```

Results:

| Variant | Size | Samples | MAE | Spectral rel |
|---------|------|---------|-----|--------------|
| fixed fp32 native LSTM | 311 MB | 80400 | 0.00244 | 0.0512 |
| fixed fp32 While LSTM | 311 MB | 80400 | 0.00246 | 0.0516 |
| weight-only int8 native LSTM | 127 MB | 80400 | 0.01457 | 0.2093 |
| weight-only int8 While LSTM | 96 MB | 80400 | 0.02141 | 0.2571 |

These models preserve output length, but parity is much worse than fixed fp32
MNN. They may or may not be listenable, and they are not expected to give the
same speed profile as a true activation-int8 PTQ model.

The same native-LSTM statefulness applies to weight-only native MNN:

| Variant | Repeated forwards on one module |
|---------|---------------------------------|
| native fp32 | second+ runs differ from first, MAE ~0.054 |
| native weight-only int8 | second+ runs differ from first, MAE ~0.055 |
| While fp32 | stable, MAE 0.0 |
| While weight-only int8 | stable, MAE 0.0 |

So if weight-only int8 is tested further, prefer
`kokoro-v1.0.patched.i32.while.wq8.mnn`.

## Phase 6 — current fidelity/debugging notes

### 6.1 `int32` precision is not the cause

Kokoro token ids are small vocabulary indices. `INT32` represents them exactly,
and the `input_ids: INT32` fp32 MNN model matches ORT on the duration predictor
and the full first-run waveform. The remaining low-fidelity PTQ output is not
an `i32` precision-loss problem.

### 6.2 Bench harness fixes

`bench.py` and `probe_mnn.py` now copy MNN `read()` buffers before returning
NumPy arrays. Without `.copy()`, a NumPy array can point at memory owned by an
MNN variable whose lifetime has ended, which made some internal probes report
impossible values. `bench.py` also defaults to `--warmup 0` because native MNN
LSTM models mutate state across forwards. Named WAV prefixes were added to
avoid comparing stale overwritten files:

```sh
uv run python bench.py ... --mnn kokoro-v1.0.patched.i32.while.mnn \
  --out-prefix cmp_fp32_while
uv run python bench.py ... --mnn kokoro-v1.0.patched.i32.while.wq8.mnn \
  --out-prefix cmp_wq8_while
uv run python bench.py ... --mnn kokoro-v1.0.patched.i32.ptq-ema.mnn \
  --out-prefix cmp_ptq_ema3_native
```

Latest named outputs:

| Prefix | Model | Samples | MAE | Spectral rel |
|--------|-------|---------|-----|--------------|
| `cmp_fp32_while` | While fp32 | 80400 | 0.00246 | 0.0516 |
| `cmp_wq8_while` | While weight-only int8 | 80400 | 0.02141 | 0.2571 |
| `cmp_wq8_block128_while` | While weight-only int8, block 128 | 80400 | 0.01331 | 0.2022 |
| `cmp_ptq_ema3_native` | native full PTQ EMA, 3 samples | 82200 | 0.05649 | 1.4277 |

Listening feedback: `cmp_wq8_while_mnn.wav` sounds good.
`cmp_wq8_block128_while_mnn.wav` has better metrics and should be the next
listening candidate. The overwritten `out_mnn.wav` should be ignored; it came
from the failed 1-sample PTQ run and is effectively silent/clipped.
`cmp_ptq_ema3_native_mnn.wav` is the confirmed low-quality full PTQ candidate.

Additional weight-only variants tested:

| Variant | Size | Samples | MAE | Spectral rel | Notes |
|---------|------|---------|-----|--------------|-------|
| symmetric, no block | 96 MB | 80400 | 0.02141 | 0.2571 | good by listening |
| block 64 | 99 MB | 80400 | 0.04078 | 0.2781 | worse |
| block 128 | 97 MB | 80400 | 0.01331 | 0.2022 | best metrics so far |
| block 256 | 97 MB | 80400 | 0.01814 | 0.2315 | worse than block128 |
| asymmetric | 96 MB | 81000 | 0.05245 | 0.3432 | bad length/parity |
| HQQ | 96 MB | 80400 | 0.02388 | 0.2394 | worse than block128 |

`--weightQuantBits 16` behaves like the plain int8 run in this converter
(same size and same metrics), consistent with the documented 2-8 bit range.

Tested fp16 converter combinations:

| Variant | Size | Samples | MAE | Spectral rel | Notes |
|---------|------|---------|-----|--------------|-------|
| While fp16 | 156 MB | 80400 | 0.04683 | 0.3253 | stable, worse parity |
| While `--fp16 --weightQuantBits 8 --weightQuantBlock 128` | 156 MB | 80400 | 0.04683 | 0.3253 | behaves like fp16, not like int8+fp16 |

So MNN 3.5's converter does not appear to produce a useful "int8 weights +
fp16 compute" hybrid for this model. Combining `--fp16` with weight quantization
made the model larger than the 97 MB block128 weight-only model and degraded
parity to the fp16 level. On this x86 host MNN reports `fp16:0`, so ARM speed
still needs direct device testing, but the file size and quality results do not
favor the fp16 combination.

Runtime precision/memory knobs were added to `bench.py`:

```sh
--mnn-precision normal|high|low|lowBF
--mnn-memory normal|high|low
--mnn-power normal|high|low
```

MNN docs say `precision=low` enables fp16 compute on supporting ARMv8.2
devices, and `memory=low` can change how weight-quantized models run. On the
x86 host used here MNN reports `fp16:0`, so these results are not predictive
for ARM speed. They do show quality is unchanged for the current best model:

| Model/config | MNN median, x86 | Samples | MAE | Spectral rel |
|--------------|-----------------|---------|-----|--------------|
| block128, normal | 1767 ms | 80400 | 0.01331 | 0.2022 |
| block128, `precision=low` | 1818 ms | 80400 | 0.01331 | 0.2022 |
| block128, `memory=low` | 1318 ms | 80400 | 0.01331 | 0.2022 |
| block128, `precision=low,memory=low` | 1852 ms | 80400 | 0.01331 | 0.2022 |
| fp32 While, `precision=low` | 1268 ms | 80400 | 0.00246 | 0.0516 |

On the target phone, the two configurations worth timing first are:

```sh
precision=low, memory=normal
precision=low, memory=low
```

For OCR, fp16 compute gave a large win; Kokoro may benefit too, but less
predictably because it has LSTM/MatMul/Conv1d/iSTFT-heavy structure rather than
mostly Conv2d.

Block128 was also checked on three additional sentences:

| Prefix | Samples ORT/MNN | MAE | Spectral rel |
|--------|------------------|-----|--------------|
| `cmp_wq8_block128_timing` | 103200 / 103200 | 0.01360 | 0.1847 |
| `cmp_wq8_block128_numbers` | 99000 / 99000 | 0.02094 | 0.2349 |
| `cmp_wq8_block128_duration` | 117600 / 117600 | 0.01629 | 0.2462 |

So block128 is not just a one-sentence artifact: length stays correct and the
parity metrics remain in the same range across several phoneme counts.

### 6.3 PTQ calibration experiments

A 5-sentence EMA calibration model (`kokoro-v1.0.patched.i32.ptq-ema5.mnn`)
was worse than the 3-sentence PTQ model: waveform values blew up before WAV
clipping. A 1-sentence EMA calibration model
(`kokoro-v1.0.patched.i32.ptq-ema1.mnn`) produced effectively silent/clipped
output in the benchmark. Native-LSTM state mutation likely contaminates MNN's
sequence calibration loop for samples after the first, but avoiding that alone
does not make PTQ usable.

KL calibration on the native-LSTM model also segfaulted. EMA/KL on the
While-LSTM model segfault as noted above. This points to MNN 3.5's PTQ tooling
being the blocker, not Kokoro's integer input type.

### 6.4 PTQ JSON ablations

Dumped the full PTQ model to JSON and stripped all 330 `extraTensorDescribe`
activation `quantInfo` entries, then converted JSON back to MNN:

```sh
uv run mnnconvert -f JSON \
  --modelFile /tmp/kokoro-ptq-noact.json \
  --MNNModel kokoro-v1.0.patched.i32.ptq-noact.mnn
```

Benchmark:

| Variant | Size | Samples | MAE | Spectral rel |
|---------|------|---------|-----|--------------|
| PTQ EMA native | 135 MB | 82200 | 0.05649 | 1.4277 |
| PTQ native with activation quantInfo stripped | 135 MB | 81000 | 0.05761 | 0.3984 |
| While weight-only int8 | 96 MB | 80400 | 0.02141 | 0.2571 |

Removing activation quantization improves the spectral metric substantially,
which supports the codec-artifact hypothesis: activation quantization is a big
part of the quality loss. It still does not beat the converter's While
weight-only int8 path, and it remains native-LSTM based.

Tried also removing decoder/vocoder `quanParameter` / `symmetricQuan` entries
from `/decoder/decoder/` convolution ops after stripping activation quantInfo.
That model does not run: MNN reports the converted conv/matmul ops have no
weight/bias. In PTQ JSON those quant parameters are the packed weight
representation, so deleting them is not a valid way to make a mixed fp32/int8
model.

## Phase 7 — Android ARM CPU measurements

Target phone connected over adb:

- ABI: `arm64-v8a`
- SoC platform: Qualcomm `lahaina`
- MNN CPU probe: `i8sdot:1`, `fp16:1`, `i8mm:0`, `sve2:0`, `sme2:0`
- CPU groups: `[0-3]` up to 1.8048 GHz, `[4-6]` up to 2.4192 GHz, `[7]`
  up to 2.8416 GHz

MNN Android tools were built from local MNN with:

- `CMAKE_BUILD_TYPE=Release`
- `ANDROID_ABI=arm64-v8a`
- `ANDROID_NATIVE_API_LEVEL=android-21`
- `MNN_ARM82=ON`
- `MNN_KLEIDIAI=ON`
- `MNN_USE_THREAD_POOL=ON`
- `MNN_OPENCL=OFF`, `MNN_VULKAN=OFF`, `MNN_OPENMP=OFF`

So these are CPU-backend measurements with ARMv8.2/fp16 support compiled in,
not GPU/NPU measurements.

The Android `ModuleBasic.out` input uses the same fox sentence as the host
bench:

```text
The quick brown fox jumps over the lazy dog.
```

Input shape is `input_ids [1,54]` (52 phoneme tokens plus sentinels), `style
[1,256]`, and `speed [1]`. The generated waveform is 80400 samples at 24000 Hz,
so the audio duration is 3.35 s.

The current best model, `kokoro-v1.0.patched.i32.while.wq8.block128.mnn`, was
pushed as `/data/local/tmp/kokoro_mnn_bench/kokoro.mnn` and run with:

```sh
LD_LIBRARY_PATH=. ./ModuleBasic.out kokoro.mnn input 0 0 <loops> <threads> <mask> <cache>
```

`forwardType=0` means CPU. Precision/memory/power mask `0` is normal mode;
mask `2` is `precision=low` (the fp16/low-precision runtime request). The user
kept the phone awake during these runs; earlier runs did not explicitly control
screen state.

Fresh-process, single-forward normal-mode runs for block128, 4 threads:

| Run | Module load/init | Forward time | Forward RTF | Load+forward RTF |
|-----|------------------|--------------|-------------|------------------|
| 1 | 265.5 ms | 3457.2 ms | 1.03 | 1.11 |
| 2 | 182.6 ms | 2424.1 ms | 0.72 | 0.78 |
| 3 | 172.0 ms | 2448.9 ms | 0.73 | 0.78 |
| 4 | 172.9 ms | 2454.3 ms | 0.73 | 0.78 |
| 5 | 171.8 ms | 2476.2 ms | 0.74 | 0.79 |

Interpretation: the very first process after killing `ModuleBasic.out` was
slower, but subsequent fresh processes were tightly clustered around 2.45 s
forward time plus ~175 ms load/init. That is about 0.73x realtime for the
forward alone, or ~0.78x including module load/init.

Longer in-process normal-mode run, block128, 4 threads, 20 loops:

| Avg | Min | Max | Avg RTF | Min RTF | Max RTF |
|-----|-----|-----|---------|---------|---------|
| 3138.3 ms | 2095.4 ms | 6747.6 ms | 0.94 | 0.63 | 2.01 |

There is substantial runtime variance on the phone. The best warm loops are
around 2.1 s, but the average can be pulled toward realtime or slower by
scheduler/thermal outliers.

Thread sweep for block128, normal mode, 5 loops:

| Threads | Avg | Min | Max | Avg RTF |
|---------|-----|-----|-----|---------|
| 1 | 4282.7 ms | 4205.5 ms | 4494.4 ms | 1.28 |
| 2 | 4202.8 ms | 3774.8 ms | 4811.8 ms | 1.25 |
| 3 | 3565.1 ms | 3430.3 ms | 3693.4 ms | 1.06 |
| 4 | 2912.3 ms | 2119.1 ms | 3329.2 ms | 0.87 |
| 6 | 4718.2 ms | 4607.6 ms | 4939.3 ms | 1.41 |
| 8 | 4596.9 ms | 4215.2 ms | 5251.2 ms | 1.37 |

Four threads is the best point found so far. More threads are worse on this
graph/device combination despite CPU usage reaching roughly 400%, likely due
to scheduling overhead and work spilling onto less useful cores.

Model comparison, normal mode, 4 threads, 5 loops:

| Model | Size | Load/init | Avg | Min | Max | Avg RTF |
|-------|------|-----------|-----|-----|-----|---------|
| fp32 While | 311 MB | 647.5 ms | 2823.6 ms | 2120.0 ms | 3293.7 ms | 0.84 |
| fp16 converter While | 156 MB | 336.1 ms | 3137.5 ms | 2479.8 ms | 3312.3 ms | 0.94 |
| wq8 block128 While | 97 MB | 266.1 ms | 2638.1 ms | 2075.0 ms | 3350.5 ms | 0.79 |

The converter-fp16 model is smaller than fp32 but slower here. The wq8 block128
model remains the best current Android candidate by size and speed.

Runtime `precision=low` mode, 4 threads, 3 loops:

| Model | Avg | Min | Max | Avg RTF |
|-------|-----|-----|-----|---------|
| fp32 While | 6912.5 ms | 5429.6 ms | 7683.9 ms | 2.06 |
| fp16 converter While | 7083.1 ms | 5693.6 ms | 7793.3 ms | 2.11 |
| wq8 block128 While | 6966.6 ms | 5227.7 ms | 7856.1 ms | 2.08 |

Despite the phone reporting fp16 support and the MNN build having
`MNN_ARM82=ON`, runtime `precision=low` is consistently much slower for this
Kokoro graph. A plausible explanation is that important ops in this graph
still execute through fp32 or conversion-heavy paths, so the low-precision path
adds layout/type conversion overhead without enough faster fp16 arithmetic to
pay for it. The same slowdown across fp32, fp16-converted, and wq8 artifacts
supports this being an MNN CPU backend/runtime path issue rather than simply a
model storage-size issue.

Native-LSTM speed-only comparison:

| Model | Size | Fresh single forwards | 5-loop avg/min/max | Notes |
|-------|------|-----------------------|--------------------|-------|
| native wq8 | 126 MB | 2374 / 2364 / 2485 ms | 2619 / 2072 / 3290 ms | native LSTM; repeated-forward output correctness is suspect |
| While wq8 block128 | 97 MB | 2496 / 2487 / 3362 ms | 2681 / 2154 / 3310 ms | stable repeated forwards |

The native-LSTM wq8 artifact is only marginally faster in these Android
speed-only timings. Because native-LSTM models mutate recurrent state across
repeated forwards, this is not a worthwhile tradeoff unless the app recreates
the MNN module per utterance or we find a reliable state-reset mechanism.

Full PTQ Android speed-only comparison:

| Model | Size | Fresh single forwards | 5-loop avg/min/max | Notes |
|-------|------|-----------------------|--------------------|-------|
| full PTQ EMA native | 135 MB | 3420 / 2454 / 2482 ms | 2790 / 2210 / 3394 ms | known bad codec-like audio quality; repeated native-LSTM forwards suspect |
| While wq8 block128 | 97 MB | 2492 / 2485 / 2433 ms | 2854 / 2124 / 3379 ms | stable; best current candidate |

The full PTQ artifact is not smaller than block128 and is not meaningfully
faster on the phone. Its 5-loop average was only ~2.3% faster in one run, which
is within the phone's normal variance, while the fresh single-forward timings
were effectively tied after the first PTQ outlier. Given the severe audio
quality regression, full PTQ does not look worth chasing for performance unless
a future MNN quantization path produces a much clearer speedup.

Why full PTQ can be larger than block128 weight-only:

- Activations are not stored in the `.mnn` file for every possible utterance.
  They are temporary runtime buffers. Activation quantization can reduce runtime
  memory traffic for quantized ops, but it does not make the model file 4x
  smaller.
- The file mostly contains graph structure, constants/weights, packed quantized
  weights, per-tensor/per-channel/per-block scales and zero points, and
  quantization metadata.
- The full PTQ artifact here is native-LSTM based and is not byte-for-byte the
  same weight packing as converter `--weightQuantBits 8 --weightQuantBlock
  128`. The fairer disk comparison is native wq8 at 126 MB vs full PTQ native
  at 135 MB. The PTQ metadata / packing overhead is enough to make full PTQ
  larger.
- The current block128 model is smaller because its converter-side weight-only
  packing is compact for this graph. It does not include activation-int8 runtime
  metadata, but that metadata also did not buy a meaningful speedup on the
  phone.

Why the MNN quantized files can still be larger/slower than the upstream
~86-88 MB ORT int8 ONNX:

- The upstream ONNX int8 model is not just "the fp32 graph with smaller
  tensors." It uses ORT-specific dynamic-quantized and fused operators such as
  `DynamicQuantizeLSTM`, `ConvInteger`, `FusedMatMul`, `FastGelu`, and
  `SkipLayerNormalization`. Those fusions can store the same logical model more
  compactly and route work to ORT's intended int8 kernels.
- MNN cannot import those ORT quantized ops, so the MNN path starts from fp32
  ONNX and asks MNN to re-quantize what it can. That can leave some constants or
  subgraphs unquantized, and it uses MNN's own packing/metadata format rather
  than ORT's.
- `--weightQuantBlock 128` adds scale/zero-point metadata per block. That is
  still compact enough to get 311 MB fp32 While down to 97 MB, but it is not
  guaranteed to beat ORT's encoding.
- Full PTQ adds activation quantization metadata but does not store activations
  in the file. It may also insert/depend on quant/dequant boundaries. If many
  ops fall back to float or dequantize internally, the model can be larger and
  not faster.
- Smaller weights only guarantee less model storage and potentially less memory
  bandwidth. They do not guarantee faster inference. Speed depends on whether
  the backend has optimized int8 kernels for the actual ops/shapes in Kokoro:
  LSTM/MatMul/Conv1d/normalization/iSTFT-style pieces, not just big Conv2d OCR
  layers. If MNN dequantizes around unsupported ops or uses a slower quantized
  kernel, int8 can be a wash or slower.

Decision point: even though MNN is not producing an ideal ORT-style fused int8
graph, the current `wq8 block128 While` result is still practically valuable.
The existing ONNX Runtime int8 path is around 0.95-1.05 RTF on the target phone,
which has little margin once the device warms up. The MNN block128 path is
roughly 0.73-0.79 RTF in the cleaner fresh-process Android measurements, with
some scheduler/thermal variance in longer loops. That is enough margin to be
worth pursuing as a deployable path if audio quality holds.

The near-term problem-solving target is therefore not "make MNN reach 0.1 RTF"
or "recreate ORT's exact fused quantized graph." It is:

1. Verify `kokoro-v1.0.patched.i32.while.wq8.block128.mnn` by listening on the
   real device/output path.
2. Build a real app-side MNN benchmark using the same preprocessing and MNN
   Module API that production would use, rather than relying only on
   `ModuleBasic.out`.
3. Measure cold load, first utterance, and warm repeated utterances end-to-end.
4. Compare against the current ORT path under the same screen/power/thermal
   conditions.
5. Only return to full PTQ/fused-op work if it shows a large speed win over
   block128, not a 1-3% noisy difference.

Phone-generated WAV for listening:

- Generated on-device with `ModuleBasic.out` from
  `/data/local/tmp/kokoro_mnn_bench/kokoro.mnn`
  (`kokoro-v1.0.patched.i32.while.wq8.block128.mnn`)
- Pulled waveform dump from `/data/local/tmp/kokoro_mnn_bench/output/0_0.txt`
- Local WAV: `android_block128_phone.wav`
- Samples: 80400, duration 3.35 s, min/max -0.5395 / 0.7136, RMS 0.0785
- ModuleBasic reported runtime memory around 407.9 MB during the dump forward

Listening/spectrogram note from the phone-generated block128 WAV: pacing and
pronunciation are good, but there is a high-pitched whistle/timbre/harmonic
overlaid while speech is active. It is absent during pauses, so it does not
look like a WAV/container artifact or DC/noise-floor issue. Spectrogram view
shows strong peaks around 4800 Hz and 9600 Hz. This is a pending quality issue
for the block128 candidate and is distinct from the much worse full-PTQ
codec-like artifact.

Local reproduction check: regenerated the same block128 MNN output on the x86
host with `bench.py` using the same fox sentence and voice file:

```sh
uv run python bench.py \
  --onnx kokoro-v1.0.patched.i32.onnx \
  --mnn kokoro-v1.0.patched.i32.while.wq8.block128.mnn \
  --voices /home/david/AndroidStudioProjects/bucket/tts/1/kokoro/voices-v1.0.bin \
  --threads 4 --iters 1 --warmup 0 \
  --out-prefix local_block128_same_config
```

Local outputs:

- `local_block128_same_config_mnn.wav`
- `local_block128_same_config_ort.wav`

The fresh local MNN WAV is byte-identical to the earlier local
`cmp_wq8_block128_while_mnn.wav` after WAV decoding (`MAE=0`). Comparing the
phone waveform dump to the local MNN WAV gives `MAE=0.00259`, `RMS diff
0.00561`, and both local and phone spectra show the same peaks:

| Output | 4800 Hz magnitude | 9600 Hz magnitude | Median spectral magnitude |
|--------|-------------------|-------------------|---------------------------|
| phone block128 | 73.69 | 34.48 | 2.84 |
| local block128 | 73.50 | 34.41 | 2.83 |

So the whistle/harmonic issue is reproducible off-phone and is almost certainly
in the block128 weight-quantized model output, not Android playback or the WAV
container.

Per-op trace for one timed block128 forward on phone:

| Op type | Time | Count | Notes |
|---------|------|-------|-------|
| Convolution | 1308 ms | 983 | dominant decoder/vocoder cost |
| While | 340 ms | 554 | control-flow/subgraph bucket; originally suspected LSTM, but later traces show this label is broader |
| UnaryOp | 129 ms | 4226 | activation/elementwise work |
| Stft | 81 ms | 1 | final generator STFT op |
| Raster | 64 ms | 12468 | layout/view/copy style overhead |
| BinaryOp | 58 ms | 6064 | elementwise arithmetic |
| Deconvolution | 35 ms | 3 | upsampling |
| Reduction | 30 ms | 132 | reductions |

Trace run total: `Avg=2353.6 ms`, while the op-summed time was 2049.8 ms.
The largest individual ops were all decoder/generator convolution blocks and
the final `STFT` op. The profile says the next optimization target is the
decoder/vocoder convolution stack, not the LSTM path.

KleidiAI check: although MNN was built with `MNN_KLEIDIAI=ON`, ModuleBasic
defaults `enableKleidiAI=0`. Explicitly passing `enableKleidiAI=1` did not help
for block128 in the quick 5-loop test:

| Mode | Avg | Min | Max |
|------|-----|-----|-----|
| KleidiAI off | 2397 ms | 2107 ms | 3248 ms |
| KleidiAI on | 2670 ms | 2133 ms | 3335 ms |

So the current app-side default should not force KleidiAI on without more
evidence.

Other quick runtime knobs on block128, 4 threads, 5 loops:

| Config | Load/init | Avg | Min | Max | Notes |
|--------|-----------|-----|-----|-----|-------|
| baseline | 171 ms | 2801 ms | 2127 ms | 3276 ms | normal mode |
| `mConfig.rearrange=true` | 489 ms | 2598 ms | 2093 ms | 3276 ms | slightly lower noisy avg, much higher load/init |
| `power=high` | 165 ms | 2901 ms | 2195 ms | 3305 ms | slower |
| `rearrange + power=high` | 494 ms | 2833 ms | 2133 ms | 3295 ms | no clear win |

`mConfig.rearrange=true` may be worth testing in the real app runner if module
load is amortized over many utterances, but it is not a clear first-utterance
win. The minimum times are nearly unchanged, suggesting this does not address
the main decoder convolution cost.

Performance-first convolution follow-up:

- The immediate goal here is performance attribution, not fixing the 4800/9600
  Hz whistle yet.
- MNN `Session_Debug` per-op tracing is useful for locating hot op classes, but
  it can perturb absolute timings. In a traced fp32 While run, total forward
  time inflated to 7274 ms and module load/init to 5126 ms, far worse than
  normal-mode fp32 measurements. Do not use traced fp32 milliseconds as
  production latency.
- In that debug trace, op counts match block128 exactly, and the traced hot
  class remains convolution:

| Type | block128 debug trace | fp32 debug trace | Count |
|------|----------------------|------------------|-------|
| Convolution | 1308 ms | 5843 ms | 983 |
| While | 340 ms | 388 ms | 554 |
| UnaryOp | 129 ms | 332 ms | 4226 |
| Stft | 81 ms | 113 ms | 1 |

The direction is still informative: the decoder/generator convolution stack is
where most CPU time sits, and block128 changes that path substantially. But for
production speed, use non-debug end-to-end timings.

Fresh non-debug check after the debug trace, 4 threads, 5 loops:

| Model | Load/init | Avg | Min | Max |
|-------|-----------|-----|-----|-----|
| fp32 While | 830 ms | 2764 ms | 2274 ms | 3293 ms |
| block128 While | 295 ms | 3050 ms | 2507 ms | 3384 ms |

This run shows the phone variance problem clearly: block128 was slower in this
particular short run, whereas earlier clean runs had block128 faster. So the
current evidence supports "convolutions are the right performance target," but
not "block128 is always faster than fp32 under every thermal/scheduler state."
The real app runner should measure both under matched thermal and screen state.

Tried Android `simpleperf` for production-mode attribution. It is available at
`/system/bin/simpleperf`, but `record -g` with callgraphs was too intrusive:
the 5-loop block128 run slowed to 6279 ms average and lost/truncated ~84% of
userspace samples. The report mostly showed unresolved `libMNN.so` offsets plus
kernel time and `sched_yield`, so this path needs a lower-overhead sample mode
or a symbolized/unstripped MNN build before it can replace MNN's own per-op
trace.

Built a separate Android profiling MNN build to resolve the sampling question:

```sh
cmake ../../../ \
  -DCMAKE_TOOLCHAIN_FILE=/home/david/Android/Sdk/ndk/28.1.13356709/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_STL=c++_static \
  -DMNN_USE_LOGCAT=false \
  -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_TEST=ON \
  -DANDROID_NATIVE_API_LEVEL=android-21 \
  -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
  -DNATIVE_LIBRARY_OUTPUT=. \
  -DNATIVE_INCLUDE_OUTPUT=. \
  -DMNN_ARM82=ON \
  -DMNN_KLEIDIAI=ON \
  -DCMAKE_C_FLAGS_DEBUG="-O2 -g -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS_DEBUG="-O2 -g -fno-omit-frame-pointer"
cmake --build . --target ModuleBasic.out -j4
```

`RelWithDebInfo` was not enough: MNN's Android command build still set hidden
symbols, `-fomit-frame-pointer`, and linker `-s`. The `Debug`/`-O2` profiling
build reports `Hidden: FALSE` and contains `.symtab` / `.debug_*` sections.
Artifacts are under:

- `/home/david/git/mnn-sys/3rd_party/MNN/project/android/build64profile`
- Pushed to `/data/local/tmp/kokoro_mnn_profile`

Sanity run for profile build, block128, 4 threads, 3 loops: load/init 171.9 ms,
avg 2568.6 ms, min 2104.9 ms, max 3248.9 ms. That is close enough to the
release build for sampling attribution.

Symbolized `simpleperf` findings for block128:

- Low-frequency flat profile, no callgraph:
  `simpleperf record -f 19 ... runLoops=10`
- No sample loss. Runtime still had outliers, but min loop stayed normal.
- Top user-space hotspot is `LoopL2` at vaddrs around `0x2734bc`.
- `addr2line` maps that `LoopL2` to
  `source/backend/cpu/arm/arm64/MNNPackedMatMul.S:87`.
- The frame-pointer callgraph confirms the main path:
  `MNN::DenseConvolutionTiledImpl::onExecute -> MNN::ThreadPool::enqueueInternal -> LoopL2`.

This is actual packed fp32 matmul work used by MNN's convolution lowering, not
obvious float/int8 conversion waste. `CPUFloatToInt8` and `CPUInt8ToFloat` were
present in the symbol table but did not appear as meaningful flat hotspots.
The current block128 model is therefore best understood as compact weight
storage feeding MNN's normal packed-matmul convolution compute, not as a true
int8-dot-product convolution runtime.

Profiled the full PTQ model with the same symbolized build. It also hotspots in
the same `MNNPackedMatMul.S` `LoopL2` cluster, not in obvious
`MNNGemmInt8AddBiasScale_*` kernels. This explains why full PTQ was not faster
despite being activation-quantized in metadata: for this model/path, MNN is not
delivering a materially different int8 convolution compute kernel.

Standalone conv microbench, using Kokoro-like hot decoder conv shapes, shows
that MNN does have much faster int8 kernels for these shapes when the graph
selects the right path. Test artifacts live in `conv_microbench/` and were run
on the phone from `/data/local/tmp/kokoro_conv_micro/conv_microbench`.

Representative shapes:

| Shape | Meaning |
|-------|---------|
| `hot_c128_w8040_k11_d1` | C=128, W=8040, K=11, dilation=1, pad=5 |
| `hot_c128_w8040_k11_d3` | C=128, W=8040, K=11, dilation=3, pad=15 |
| `hot_c128_w8040_k7_d1` | C=128, W=8040, K=7, dilation=1, pad=3 |
| `mid_c256_w1340_k11_d1` | C=256, W=1340, K=11, dilation=1, pad=5 |
| `mid_c256_w1340_k7_d1` | C=256, W=1340, K=7, dilation=1, pad=3 |

Standalone MNN conv microbench, 4 threads, 50 loops each. The averages are very
noisy because the phone scheduler still injected outliers, so the min column is
the useful "kernel can do this when not interrupted" signal:

| Shape | Variant | Avg ms | Min ms | Max ms |
|-------|---------|--------|--------|--------|
| hot C128 W8040 K11 d1 | fp32 | 22.555 | 21.506 | 71.219 |
| hot C128 W8040 K11 d1 | fp16 | 126.124 | 16.485 | 2706.925 |
| hot C128 W8040 K11 d1 | wq8 block128 | 171.851 | 16.506 | 1193.584 |
| hot C128 W8040 K11 d1 | PTQ int8 | 26.124 | 4.614 | 1041.072 |
| hot C128 W8040 K11 d3 | fp32 | 22.563 | 21.511 | 62.109 |
| hot C128 W8040 K11 d3 | fp16 | 23.210 | 16.517 | 63.724 |
| hot C128 W8040 K11 d3 | wq8 block128 | 23.492 | 21.570 | 62.118 |
| hot C128 W8040 K11 d3 | PTQ int8 | 6.014 | 4.614 | 48.477 |
| hot C128 W8040 K7 d1 | fp32 | 14.968 | 13.526 | 53.870 |
| hot C128 W8040 K7 d1 | fp16 | 15.189 | 13.733 | 55.977 |
| hot C128 W8040 K7 d1 | wq8 block128 | 15.194 | 13.602 | 53.174 |
| hot C128 W8040 K7 d1 | PTQ int8 | 3.980 | 3.160 | 21.857 |
| mid C256 W1340 K11 d1 | fp32 | 16.047 | 14.498 | 52.974 |
| mid C256 W1340 K11 d1 | fp16 | 16.303 | 14.763 | 51.956 |
| mid C256 W1340 K11 d1 | wq8 block128 | 19.634 | 15.366 | 124.067 |
| mid C256 W1340 K11 d1 | PTQ int8 | 3.852 | 2.913 | 25.438 |
| mid C256 W1340 K7 d1 | fp32 | 10.484 | 9.234 | 42.758 |
| mid C256 W1340 K7 d1 | fp16 | 10.474 | 9.231 | 44.923 |
| mid C256 W1340 K7 d1 | wq8 block128 | 10.410 | 9.169 | 40.425 |
| mid C256 W1340 K7 d1 | PTQ int8 | 2.777 | 1.884 | 21.976 |

Min-time ratios: standalone PTQ int8 is roughly 0.20-0.23x fp32 for all tested
shapes, i.e. about 4-5x faster for the core conv kernel. Converter-side
weight-only block128 mostly behaves like fp32 compute, except for a small win on
one C128 K11 case. Standalone fp16 is only clearly faster on C128 K11 and is
basically neutral on the other shapes; combined with the earlier full-model
`precision=low` slowdown, fp16 is not the primary path to chase right now.

Symbolized `simpleperf` on standalone `hot_c128_w8040_k11_d1.ptq.mnn` confirms
that the fast path is real int8 ARMv8.2 GEMM:

- Runtime with profile build: avg 4.957 ms, min 4.722 ms, max 34.002 ms.
- Hot symbols: `L8LoopSz_TILE_12`, `L8Tile12Quan`, and
  `_ArmBasicMNNPackC4ForMatMul_A_L4<12, 8>`.
- `addr2line` maps `L8LoopSz_TILE_12` to
  `source/backend/cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV82_Unit.S:184`.
- The previous full-model block128/fp32 hotspot, `LoopL2`, maps to
  `source/backend/cpu/arm/arm64/MNNPackedMatMul.S:87`.

Conclusion from step 1: MNN has a fast int8 conv kernel for the real Kokoro conv
shapes. The performance problem is not "MNN int8 conv is slow"; it is that the
full Kokoro PTQ graph is not selecting that int8 path, or is wrapped in a way
that makes the visible conv work fall back to packed fp32 matmul. This makes
instrumenting MNN's conv selection logic worthwhile.

Follow-up instrumentation found the concrete reason the hot full-model convs
missed the fast path in the normal Android build.

Temporary logging was added to the MNN profile build under
`MNN_KOKORO_CONV_LOG=1` in:

- `CPUBackend::onCreate`
- `CPUConvInt8Creator`
- `ConvolutionFloatFactory`
- `DenseConvolutionTiledExecutor`
- `DenseConvInt8TiledExecutor`

The standalone PTQ microbench conv selected the expected int8 executor:

```text
CPUBackend::onCreate name=output op=Convolution selected=ConvInt8 inputQuant=1 outputQuant=1 inputApply=1 outputApply=1
CPUConvInt8Creator name=output k=1x11 d=1x1 inC=128 outC=128 quanCommon=1 weight=180224 alpha=128 hasScaleInt=0
DenseConvInt8Tiled ctor name=output dynamic=0 quanCommon=1 asy=0 k=1x11 inC=128 outC=128
DenseConvInt8Tiled onResize ... UNIT=8 SRC_UNIT=4 DST_XUNIT=12
```

The full PTQ model did use int8 for some 1x1 matmul-converted convs, but not
for the hot decoder/generator K7/K11 conv stack. For example:

```text
CPUBackend::onCreate name=/decoder/decoder/generator/noise_res.1/convs1.0/Conv_output_0
  op=Convolution selected=Convolution inputQuant=0 outputQuant=1 inputApply=0 outputApply=0
FloatFactory loadedQuan ... hasScaleInt=0 lowMemoryArg=0 weight=0 weightFloat=180224 alpha=128
FloatFactory choice=DenseConvolutionTiled no-winograd ... passWeightQuant=0
```

CPUBackend only remaps `OpType_Convolution` to `OpType_ConvInt8` when both the
input and output tensors have int8 quant attributes. The standalone conv has a
quantized input and output, so it reaches `CPUConvInt8Creator`. The full-model
hot decoder convs receive float tensors from AdaIN/residual `BinaryOp` chains,
so `inputQuant=0`; MNN keeps them as float `Convolution`, dequantizes the int8
weights back to fp32 (`weightFloat=...`), and runs `DenseConvolutionTiled`.

For the first hot K11 conv, the input tensor chain is:

```text
Convolution adain1.0/fc -> Reshape -> Unsqueeze -> StridedSlice ->
BinaryOp Add_1 -> BinaryOp Mul_3 -> BinaryOp Add_3 ->
BinaryOp noise_res.1/Add -> ConvertTensor -> ConvertTensor -> hot K11 Conv
```

The quantized tensor chain breaks before the hot conv at those AdaIN/residual
float ops, not at the conv weight encoding.

Shape summary from the full PTQ log:

- 73 convs selected `ConvInt8`, mostly 1x1 matmul-converted/fc ops.
- 90 convs selected float `Convolution`.
- All hot decoder/generator K7/K11 convs selected float `Convolution` in the
  normal build.

The original Android MNN build had:

```text
MNN_LOW_MEMORY=OFF
MNN_CPU_WEIGHT_DEQUANT_GEMM=OFF
```

With those flags off, even `memory=low` cannot use MNN's dynamic-quant
float-input conv path. Rebuilding the Android profile variant with both flags
enabled:

```text
MNN_LOW_MEMORY=ON
MNN_CPU_WEIGHT_DEQUANT_GEMM=ON
```

and running full PTQ with runtime `memory=low` changed those 90 float-input
convs to dynamic int8 convs:

```text
FloatFactory loadedQuan ... lowMemoryArg=1 weight=180224 weightFloat=0 alpha=128
FloatFactory choice=DenseConvInt8Tiled dynamic-low-memory ...
DenseConvInt8Tiled ctor ... dynamic=1 quanCommon=1 ... k=1x11 inC=128 outC=128
```

Speed on the profile build for the full PTQ model, 4 threads, 5 loops:

| Build/runtime | Avg ms | Min ms | Max ms | Notes |
|---------------|--------|--------|--------|-------|
| low-memory/dequant-GEMM build, normal memory | 2745.5 | 2698.9 | 2770.6 | still not the dynamic hot-conv path |
| low-memory/dequant-GEMM build, `memory=low` | 1178.7 | 1134.5 | 1351.5 | all 90 float-input convs use dynamic int8 |

Per-op trace for the low-memory/dequant-GEMM build with `memory=low`:

| Type | Time | Count | Notes |
|------|------|-------|-------|
| Convolution | 331.6 ms | 90 | dynamic int8 path for formerly-float convs; MNN still labels these as `Convolution` |
| Convolution `[DT_INT8]` | 0.8 ms | 73 | tensor-quantized 1x1 convs |
| While | 397.2 ms | 590 | now comparable to conv cost |
| UnaryOp | 82.3 ms | 186 | secondary |
| BinaryOp | 55.8 ms | 408 | secondary |

So the fast-path issue is now understood:

1. Standalone PTQ convs go fast because their input/output tensors are quantized.
2. The full model's hot decoder convs are behind float AdaIN/residual ops, so
   they do not meet CPUBackend's `inputQuant && outputQuant` remap condition.
3. The default Android build also disabled MNN's dynamic-quant fallback for
   float-input quantized-weight convs.
4. Enabling `MNN_LOW_MEMORY` + `MNN_CPU_WEIGHT_DEQUANT_GEMM` and using
   `memory=low` makes those hot convs use dynamic int8 and gives a large speed
   win.

The remaining blocker is quality: this result is for the already-bad full PTQ
artifact. It proves the performance path is worth chasing, but does not fix the
codec-like audio degradation.

WAV check for the low-memory/dequant-GEMM fast path:

- Fast path command shape: low-memory/dequant-GEMM Android build,
  `kokoro-ptq-ema.mnn`, runtime `memory=low`, 4 threads, first forward dumped
  through `ModuleBasic`.
- Output file: `android_fast_ptq_lowmem_mnn.wav`.
- Waveform: 81600 samples, 3.400 s, min `-1.62e-10`, max `1.31e-10`,
  RMS `1.02e-12`.
- Repeated-forward check produced the same near-zero waveform for both dumped
  forwards:
  `android_fast_ptq_lowmem_repeat2_0_mnn.wav` and
  `android_fast_ptq_lowmem_repeat2_1_mnn.wav`.
- Normal-memory control on the same low-memory/dequant-GEMM build and same
  phone/model/input is not silent:
  `android_ptq_normal_mnn.wav`, 82200 samples, 3.425 s, min `-0.0796`,
  max `0.0684`, RMS `0.00581`.
- Trying `runMask=1024` / `DYNAMIC_QUANT_OPTIONS=2` with `memory=low`
  segfaulted.

So the dynamic low-memory path is fast but currently incorrect for the full PTQ
Kokoro artifact: it collapses the waveform to near-zero. This is a separate
correctness bug from the earlier full-PTQ codec-like quality issue.

Follow-up on the silent low-memory fast path:

- A standalone decoder-shaped weight-quantized convolution
  (`hot_c128_w8040_k11_d1.wq8b128.mnn`) does **not** collapse under
  `memory=low`. Normal vs low-memory output on the phone: RMS `0.375052` vs
  `0.375063`, MAE `0.00239`, correlation `0.999968`. This rules out the simple
  theory that MNN's dynamic int8 conv kernel always returns zeros.
- Full-model intermediate dump comparing normal memory vs low memory:
  - first text-encoder dynamic convs match very closely (RMS ratios ~1.0,
    correlations > `0.9998`);
  - duration logits/rounding are also close, but can differ by one frame;
  - decoder tensors diverge heavily after the F0/N path and decoder encode
    stack;
  - final waveform becomes non-silent **only when enough intermediate tensors
    are retained as requested outputs**.
- Prefix-output probe:
  - `waveform` alone, or with early encoder tensors, is silent
    (RMS `1.02e-12`);
  - retaining through `/encoder/F0_proj/Conv_output_0` is still silent;
  - retaining through both `/encoder/F0_proj/Conv_output_0` and
    `/encoder/N_proj/Conv_output_0` makes the low-memory output non-silent
    (RMS about `0.0506`);
  - retaining later decoder tensors gives a similar non-silent output
    (RMS about `0.0549`).
- Diagnostic WAV from the retained-output run:
  `android_fast_ptq_lowmem_debug_retained_mnn.wav`, 81600 samples / 3.400 s,
  raw tensor min `-2.54`, max `0.299`, RMS `0.0549`. The WAV clips the large
  negative spike to PCM16 range, but is useful for listening.

Current hypothesis: the remaining fast-path bug is probably not plain
convolution arithmetic. It looks like a low-memory graph scheduling / tensor
lifetime / in-place reuse issue exposed by the dynamic int8 path: asking for
intermediate tensors changes buffer retention and changes the final waveform
from near-zero to non-silent. Need prove this by finding the exact retained
tensor(s) that alter allocation/lifetime, then inspect MNN's low-memory
allocation or in-place rules around those tensors.

That hypothesis now has a concrete mechanism.

The tensors that matter are the decoder conditioning branches:

- `/decoder/decoder/F0_conv/Conv_output_0` (tensor index `2447`)
- `/decoder/decoder/N_conv/Conv_output_0` (tensor index `2651`)

In the MNN JSON, both are produced once and then reused by five concat ops:

- `/decoder/decoder/Concat_output_0`
- `/decoder/decoder/Concat_1_output_0`
- `/decoder/decoder/Concat_2_output_0`
- `/decoder/decoder/Concat_3_output_0`
- `/decoder/decoder/Concat_4_output_0`

But an instrumented MNN `Pipeline.cpp` lifetime log for the low-memory fast
path with `outputs=["waveform"]` shows both tensors allocated with `use=1` and
released as reusable immediately after that one counted use:

```
[kokoro-life] alloc idx=2447 label=F0_conv use=1 usage=0 ...
[kokoro-life] alloc idx=2651 label=N_conv use=1 usage=0 ...
[kokoro-life] release-input idx=2447 ... oldUse=1 newUse=0 ...
[kokoro-life] release-decision idx=2447 ... need=1 ...
[kokoro-life] release-input idx=2651 ... oldUse=1 newUse=0 ...
[kokoro-life] release-decision idx=2651 ... need=1 ...
```

When `/decoder/decoder/F0_conv/Conv_output_0` is requested as an additional
Module output, the same tensor is marked `usage=OUTPUT`, and release is blocked:

```
[kokoro-life] alloc idx=2447 label=F0_conv use=1 usage=2 storage=0
[kokoro-life] release-decision idx=2447 ... need=0 ...
```

That extra output is enough to make the low-memory waveform non-silent. The
same happens when requesting `N_conv`, or the earlier `N_proj/Squeeze/Reshape`
aliases. Requesting `F0_proj` alone does not fix it. This matches the graph:
`F0_conv` and `N_conv` are the small conditioning tensors reused across all
five decoder concat blocks; freeing either too early poisons later decoder
conditioning.

A concat probe supports the same story. With only the concat tensors and
waveform requested, later concat dumps (`Concat_1` through `Concat_4`) contain
NaNs/huge invalid values and the waveform is silent. With `F0_conv` and
`N_conv` pinned as extra outputs, all concat dumps are finite and the waveform
is non-silent.

Workaround tested on the phone:

- Load/run with extra output `/decoder/decoder/F0_conv/Conv_output_0` plus
  `waveform`, ignore the first output.
- Or load/run with extra output `/decoder/decoder/N_conv/Conv_output_0` plus
  `waveform`, ignore the first output.
- Both produce the same non-silent 81600-sample low-memory waveform:
  `android_fast_ptq_lowmem_pinned_f0conv_mnn.wav` and
  `android_fast_ptq_lowmem_pinned_nconv_mnn.wav`, RMS `0.0498`, min `-1.073`,
  max `0.295`.
- Quick 10-loop phone bench on the instrumented build:
  - pinned `F0_conv`: min `1111 ms`, average skewed by one `9501 ms` outlier;
  - pinned `N_conv`: min `1194 ms`, avg `1496 ms`, max `1684 ms`.

So the fast-mode silence is an MNN lifetime/refcount bug in the optimized
geometry/low-memory schedule, not an inherent int8 convolution arithmetic
failure. The workaround restores audible output, but the underlying full PTQ
artifact is still the low-quality/codec-like model, so this does not yet solve
final TTS quality.

Implementation note for the temporary MNN patch:

- A broad attempt to fix refcounting by recursively counting virtual/raster
  region origins kept the conditioning tensors live, but it perturbed Kokoro
  badly: waveform length changed and values became huge invalid numbers. That
  patch was backed out.
- The active local MNN patch is a narrower experiment in `Pipeline.cpp`:
  when `MNN_KOKORO_KEEP_CONDITIONING=1`, tensor indexes `2447` and `2651`
  are allocated as `Backend::STATIC` and `_needRelease` refuses to release
  them. This reproduces the extra-output pin behavior for the full PTQ model
  without changing Module outputs. It is model-index-specific and should not be
  considered the real upstream fix.
- For model-agnostic app integration, the safer workaround is still to request
  `/decoder/decoder/F0_conv/Conv_output_0` or
  `/decoder/decoder/N_conv/Conv_output_0` as an additional Module output and
  ignore it. That marks the tensor as an output by name instead of relying on
  hard-coded tensor indexes.

Fast + quality follow-up:

- Tested the best-quality `kokoro-v1.0.patched.i32.while.wq8.block128.mnn`
  artifact on the phone using the low-memory/dequant-GEMM MNN build and runtime
  `memory=low`.
- Used the output-pin workaround by requesting
  `/decoder/decoder/F0_conv/Conv_output_0` plus `waveform`, ignoring the first
  output.
- Output WAV: `android_block128_while_lowmem_f0pin_mnn.wav`.
- Waveform: 81000 samples, 3.375 s, min `-0.573`, max `0.748`, RMS `0.0794`.
- User listened: quality is good; the known 4800/9600 Hz tone/whistle remains,
  as expected, but the bad-codec full-PTQ artifact is not present.
- Warm phone timing, 10-loop `ModuleBasic` bench:

| Model/config | Avg | Min | Max | RTF avg | RTF min | Notes |
|--------------|-----|-----|-----|---------|---------|-------|
| block128 While WQ8, `memory=low`, F0 pin | 1259 ms | 1086 ms | 1603 ms | 0.37x | 0.32x | good-quality fast candidate |
| block128 While WQ8, normal memory, F0 pin | 4381 ms | 2708 ms | 8420 ms | 1.30x | 0.80x | same build/model, much slower |

These are warm benchmark-loop numbers. The `ModuleBasic` `cost time` line here
is model load/setup, not a clean first-generation/cold inference measurement.

Reference WAVs generated locally for listening comparison against the MNN
block128 low-memory sample:

- `original_fp32_onnx_reference.wav` from `kokoro-v1.0.onnx` via ORT, same text
  and voice selection; 80400 samples / 3.350 s, RMS `0.0787`.
- `original_int8_onnx_reference.wav` from upstream `int8.onnx` via ORT, same
  text and voice selection; 87000 samples / 3.625 s, RMS `0.0791`.

Both are host ORT reference renders, not phone timings. The fp32 host single run
was about `763 ms`; the int8 host single run was about `3160 ms`, which is
expected on this x86 host and not representative of ARM int8 performance.

Frequency artifact / whistle follow-up:

- Fast candidate MNN file:
  `kokoro-v1.0.patched.i32.while.wq8.block128.mnn`, 101,440,316 bytes
  (~96.7 MiB, shown as 97 MB by `ls -lh`). Phone copy
  `/data/local/tmp/kokoro_mnn_bench/kokoro.mnn` is the same byte size.
- The 4.8 kHz / 9.6 kHz lines are not Android-only and not caused by the
  low-memory fast path. They reproduce locally in the block128 MNN WAV and are
  present in the phone low-memory WAV at roughly the same level.
- They are also not general MNN fp32 conversion artifacts: fp32 While MNN and
  ORT have weaker lines. The block/plain converter weight-quantized MNN models
  boost them.
- Measured narrow-band energy, representative rows:

| WAV | 4.8 kHz vs broadband | 4.8 kHz vs neighbors | 9.6 kHz vs broadband | 9.6 kHz vs neighbors |
|-----|----------------------|----------------------|----------------------|----------------------|
| fp32 While MNN | -8.82 dB | +2.30 dB | -18.18 dB | +4.90 dB |
| block128 While MNN local | -6.00 dB | +5.13 dB | -14.36 dB | +8.64 dB |
| block128 While MNN phone lowmem | -5.87 dB | +5.35 dB | -14.71 dB | +8.34 dB |
| HQQ While MNN phone lowmem | -8.64 dB | +2.48 dB | -19.52 dB | +3.83 dB |

Interpretation: the artifact tracks MNN converter weight quantization mode, not
the phone, WAV writing, or the `memory=low` dynamic-conv path. The exact
frequencies are 1/5 and 2/5 of 24 kHz, which fits vocoder upsampling / periodic
alias or checkerboard energy being boosted by weight quantization error in the
decoder/generator stack.

Artifacts written to inspect this:

- `whistle_metrics.py` — reusable phase-insensitive STFT metric for this
  artifact. Example:
  `uv run python whistle_metrics.py --reference cmp_fp32_while_prec_low_mnn.wav local_block128_same_config_mnn.wav android_block128_while_lowmem_f0pin_mnn.wav notched_block128_remove_48_96.wav android_hqq_while_lowmem_f0pin_mnn.wav`.
  It reports broad high-frequency deltas and local tonal prominence at
  4.8/9.6 kHz.
- `diff_local_block128_minus_fp32_prec_low.wav` — aligned local block128 minus
  fp32 residual, gain-normalized for listening.
- `tonebands_fp32_prec_low_48_96.wav` — isolated 4.8/9.6 kHz bands from fp32.
- `tonebands_block128_48_96.wav` — isolated 4.8/9.6 kHz bands from block128.
- `tonebands_block128_minus_fp32_48_96.wav` — isolated 4.8/9.6 kHz residual.
- `notched_block128_remove_48_96.wav` — local block128 with narrow notches
  around 4.8/9.6 kHz, for a quick perceptual check.
- `android_hqq_while_lowmem_f0pin_mnn.wav` — HQQ weight-quant model on phone
  low-memory path. Spectrally much closer to fp32 at 4.8/9.6 kHz.

HQQ timing from the first phone run should be treated as invalid because the
phone screen was off during the bench. The spectral result remains useful, but
speed needs a rerun with the device awake.

Current local debugging round:

- Since the whistle reproduces on host, continue root cause locally before
  taking the phone again.
- Use the phase-insensitive `whistle_metrics.py` result as the acceptance
  metric: broad high-frequency energy is not the bug; narrow normalized
  time-axis energy around 0.2 and 0.4 cycles/sample is.
- Next probe: expose intermediate decoder/generator tensors from fp32,
  block128, and HQQ MNN models and find the first stage where block128's
  0.2/0.4-cycle tonal prominence diverges from fp32 while HQQ stays closer.
- Added `probe_whistle_layers.py`, which compares normalized 0.2/0.4
  cycles/sample tone prominence across named MNN intermediate tensors.
- Result: block128 stays essentially flat through the generator stack and first
  diverges at `/decoder/decoder/generator/istft/stft/ConvTranspose_output_0`.
  For the original block128 model, the layer probe reports about `+1.73 dB`
  at normalized 0.2 and `+3.03 dB` at normalized 0.4 immediately after that
  final iSTFT transpose-convolution, and the same deltas remain in `waveform`.
  With the new skip-iSTFT candidate, the same rows drop to about `+0.24 dB`
  and `-0.05 dB`.
- The practical cause is therefore not the earlier decoder convolution stack:
  the final iSTFT `ConvTranspose` basis weights are too sensitive to MNN
  converter weight-only quantization, producing a stride/periodic tone at
  1/5 and 2/5 of the 24 kHz sample rate.
- Tried direct MNN JSON patching as a diagnostic, but it is not a valid model
  creation route: converting an unmodified block128 MNN to JSON and back
  already degrades parity to about `MAE=0.0556`, `spectral_mae_rel=0.4006`.
  `--optimizeLevel 0` does not fix this. JSON hybrids can show tone direction,
  but should not be used as candidate artifacts.
- Valid fix route: generate a converter `--compressionParamsFile`, edit only
  `/decoder/decoder/generator/istft/stft/ConvTranspose_output_0` to
  `bits=0`, and reconvert from ONNX with the same block128 settings. Helper
  scripts added:
  - `hybridize_mnn_json.py` for diagnostic JSON experiments only.
  - `edit_compression_params.py` for the valid per-op converter config edit.
- New local candidate:
  `kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft.mnn`.
  It keeps all previous block128 quantization except the final iSTFT
  `ConvTranspose`. Exact size is `101,441,312` bytes, only `996` bytes larger
  than the original block128 artifact.
- Local bench for the new candidate, same sentence/voice:
  `80400` samples, `MAE=0.01313`, `spectral_mae_rel=0.1994`, RMS `0.0787`.
  This is slightly better than the original block128 run (`MAE=0.01331`,
  `spectral_mae_rel=0.2022`) and preserves length.
- `whistle_metrics.py` confirms the narrow tones are essentially back to fp32:

| WAV | highD | highNoToneD | 4.8tone | 9.6tone | 4.8D | 9.6D |
|-----|-------|-------------|---------|---------|------|------|
| original block128 local | +0.24 dB | +0.04 dB | +5.13 dB | +8.64 dB | +2.82 dB | +3.82 dB |
| block128 skip-iSTFT local | +0.01 dB | +0.01 dB | +2.51 dB | +4.89 dB | +0.17 dB | -0.07 dB |

The local WAV to listen to is
`cmp_wq8_block128_skip_istft_while_mnn.wav`. If listening confirms quality, the
next phone round should benchmark this artifact with `memory=low` plus the F0
conditioning output pin, because the final iSTFT op is small and unquantizing
it should not materially affect the low-memory convolution speedup.

Phone validation for the skip-iSTFT artifact:

- User listened to `cmp_wq8_block128_skip_istft_while_mnn.wav`: no whistle,
  great quality.
- Device was kept awake via adb before/during the run:
  `mWakefulness=Awake`, `mHoldingDisplaySuspendBlocker=true`. The phone was not
  USB-powered (`USB powered: false`), so these are screen-awake battery-power
  numbers.
- Important command hygiene: the fast low-memory/dequant-GEMM tool lives under
  `/data/local/tmp/kokoro_mnn_profile_lowmem`. Running from
  `/data/local/tmp/kokoro_mnn_bench` uses a smaller/non-profile tool and gives
  misleading ~2.7-2.8 s timings with ~408 MB reported memory. Those are not the
  valid fast-path numbers.
- Valid command shape:

```sh
cd /data/local/tmp/kokoro_mnn_profile_lowmem
LD_LIBRARY_PATH=. ./ModuleBasic.out \
  ../kokoro_mnn_bench/kokoro-block128-skip-istft.mnn \
  ../kokoro_mnn_bench/probe_views/f0conv \
  0 0 10 4 8 bench-block128-skip-istft-low-f0pin
```

Here `probe_views/f0conv` requests outputs
`/decoder/decoder/F0_conv/Conv_output_0` and `waveform`, so it applies the
existing F0 output-pin workaround. Mask `8` means normal precision,
`memory=low`, normal power.

Same-session 10-loop phone timings from the low-memory build:

| Model/config | Avg | Min | Max | RTF avg | RTF min | Notes |
|--------------|-----|-----|-----|---------|---------|-------|
| block128 skip-iSTFT, `memory=low`, F0 pin | 1308 ms | 1136 ms | 1646 ms | 0.39x | 0.34x | no-whistle candidate |
| original block128, `memory=low`, F0 pin rerun | 1371 ms | 1151 ms | 1645 ms | 0.41x | 0.34x | same session/control |

The skip-iSTFT fix therefore does not appear to cost speed; in this session it
was slightly faster than the original block128 run, within normal phone noise.
It also reports the expected low-memory footprint (`203.4 MB`) rather than the
wrong-tool `407.9 MB`.

Phone WAV generated from the valid low-memory run:

- `android_block128_skip_istft_lowmem_f0pin_mnn.wav`
- Raw dump: `android_block128_skip_istft_lowmem_f0pin_waveform.txt`
- Waveform: `81000` samples / `3.375 s`, min `-0.568`, max `0.757`, RMS
  `0.0796`.
- Phone spectral metric vs fp32 reference:
  `highD=-0.06 dB`, `highNoToneD=-0.09 dB`, `4.8tone=+2.83 dB`,
  `9.6tone=+4.27 dB`, `4.8D=+0.40 dB`, `9.6D=-0.76 dB`.
  This is close to fp32 and much lower than the original phone block128
  whistle (`4.8D=+2.95 dB`, `9.6D=+3.47 dB`).

Long-form listening check after phone feedback:

- User confirmed the local short skip-iSTFT WAV has no high-pitched whistle and
  great quality, but the phone WAV subjectively gets rougher/static-like as it
  progresses.
- Generated a longer host sample with four repeated fox sentences:
  `long_block128_skip_istft_mnn.wav`, with ORT reference
  `long_block128_skip_istft_ort.wav`.
- Length: `249600` samples / `10.4 s` (`208` phoneme tokens excluding
  sentinels).
- Host MNN long-run parity: `MAE=0.02432`, `spectral_mae_rel=0.2506`, RMS
  `0.0670`.
- Quarter-by-quarter host comparison against ORT did **not** show obvious
  progressive broadband high-frequency drift:

| Segment | Time | MAE | spectral rel | highD | highNoToneD |
|---------|------|-----|--------------|-------|-------------|
| 1 | 0.0-2.6 s | 0.02757 | 0.1688 | -0.01 dB | -0.08 dB |
| 2 | 2.6-5.2 s | 0.02929 | 0.1530 | -0.08 dB | -0.15 dB |
| 3 | 5.2-7.8 s | 0.02260 | 0.1399 | +0.06 dB | +0.01 dB |
| 4 | 7.8-10.4 s | 0.01778 | 0.1610 | +0.11 dB | +0.09 dB |

This does not rule out a phone/low-memory progressive artifact, but it suggests
the host skip-iSTFT model itself is not accumulating high-frequency noise over
the sentence.

Long-form phone check:

- Generated the same four-sentence input on the phone with
  `kokoro-block128-skip-istft.mnn`, `memory=low`, 4 threads, and the F0 output
  pin. Input folder:
  `/data/local/tmp/kokoro_mnn_bench/long_f0pin`.
- Output WAV: `android_long_block128_skip_istft_lowmem_f0pin_mnn.wav`.
- Raw dump: `android_long_block128_skip_istft_lowmem_f0pin_waveform.txt`.
- Waveform: `250800` samples / `10.45 s`, min `-0.578`, max `0.783`, RMS
  `0.0677`.
- 3-loop phone timing for the long input: avg `4019.9 ms`, min `3501.9 ms`,
  max `5010.5 ms`. For 10.45 s of audio this is about `0.38x` avg RTF,
  `0.34x` best RTF.
- Full-file spectral metric using the host long MNN as reference:

| WAV | highD | highNoToneD | 4.8tone | 9.6tone | 4.8D | 9.6D |
|-----|-------|-------------|---------|---------|------|------|
| host long skip-iSTFT MNN | +0.00 dB | +0.00 dB | +2.30 dB | +3.83 dB | +0.00 dB | +0.00 dB |
| phone long skip-iSTFT lowmem | -0.13 dB | -0.10 dB | +2.06 dB | +2.92 dB | -0.41 dB | -1.10 dB |
| host long ORT | -0.02 dB | +0.03 dB | +1.47 dB | +3.57 dB | -0.64 dB | -0.29 dB |

- Quarter-by-quarter phone-vs-host broad high-band deltas do not show an
  obvious monotonic high-frequency buildup:

| Segment | Time | MAE | spectral rel | highD | highNoToneD |
|---------|------|-----|--------------|-------|-------------|
| 1 | 0.0-2.6 s | 0.05098 | 0.2642 | -0.17 dB | -0.15 dB |
| 2 | 2.6-5.2 s | 0.05029 | 0.3223 | -0.09 dB | -0.10 dB |
| 3 | 5.2-7.8 s | 0.04113 | 0.2434 | -0.08 dB | -0.03 dB |
| 4 | 7.8-10.4 s | 0.02852 | 0.3211 | +0.04 dB | +0.07 dB |

If listening still reveals progressive roughness on the long phone WAV, the
metric to add next should target a different artifact than total high-band
energy, likely local/noise-like residual or alignment/pacing drift.

Follow-up on the long-phone roughness / "lazy" static:

- User heard definite degradation/static in the long phone low-memory WAV,
  especially around the `l` in "lazy"; host long MNN and ORT both sound perfect.
- User listened to the phone variants:
  - `android_long_block128_skip_istft_lowmem_f0npin_mnn.wav`: bad.
  - `android_long_block128_skip_istft_normal_f0pin_mnn.wav`: good.
  This confirms the remaining artifact is tied to Android `memory=low`, not to
  the skip-iSTFT model itself.
- Exported rough per-sentence and late-sentence listening slices under
  `phone_long_slices/`:
  - `phone_sentence_1.wav` ... `phone_sentence_4.wav`
  - `phone_sentence_1_late.wav` ... `phone_sentence_4_late.wav`
  - matching `host_*` slices
  - aligned residual probes:
    `phone_minus_host_sentence_{1..4}_aligned_residual.wav`
- Rendered the same long input on the phone with normal memory, same
  skip-iSTFT model, same F0 output pin:
  `android_long_block128_skip_istft_normal_f0pin_mnn.wav`.
  This output is `249600` samples / `10.4 s`, matching host length. The
  low-memory output is `250800` samples / `10.45 s`, so low-memory is changing
  the generated sequence slightly, not only making it faster.
- Normal-memory phone waveform stats: min `-0.517`, max `0.649`, RMS `0.0643`.
- Low-memory phone waveform stats: min `-0.578`, max `0.783`, RMS `0.0677`.
- Tried pinning both conditioning tensors (`F0_conv`, `N_conv`, `waveform`) for
  the long low-memory run. Output was byte/effectively identical to F0-only pin:
  `android_long_block128_skip_istft_lowmem_f0npin_mnn.wav`, `250800` samples /
  `10.45 s`, RMS `0.0677`. So the remaining roughness is not fixed by also
  pinning `N_conv`.
- Against the normal-memory phone render, low-memory has more high/air energy:

| WAV | highD | highNoToneD | airD | 4.8D | 9.6D |
|-----|-------|-------------|------|------|------|
| phone long normal memory | +0.00 dB | +0.00 dB | +0.00 dB | +0.00 dB | +0.00 dB |
| phone long low-memory F0 pin | +0.29 dB | +0.16 dB | +0.76 dB | +1.37 dB | -1.86 dB |
| phone long low-memory F0+N pin | +0.29 dB | +0.16 dB | +0.76 dB | +1.37 dB | -1.86 dB |

Current read: skip-iSTFT fixed the narrow whistle. The remaining long-phone
static/roughness is likely a low-memory runtime-path difference, since host and
phone normal-memory do not share the low-memory length change.

Duration-path probe for the low-memory roughness:

- Dumped phone normal vs low-memory intermediate outputs for the long input:
  `/encoder/Div_output_0`, `/encoder/Round_output_0`,
  `/encoder/Cast_output_0`, `/decoder/decoder/F0_conv/Conv_output_0`, and
  `waveform`.
- Normal memory duration sum: `416` frames.
- Low-memory skip-iSTFT duration sum: `418` frames. Two token durations round
  one frame higher:
  - tensor index `178`, phoneme `s` in the fourth "fox": `4 -> 5`;
  - tensor index `208`, final `ɡ` in the fourth "dog": `5 -> 6`.
  This exactly explains the long low-memory output length:
  `418 * 600 = 250800` samples vs normal `416 * 600 = 249600` samples.
- Built `kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-predictor.mnn`
  by keeping only `/encoder/predictor/` quantized-weight ops in fp32, plus the
  final iSTFT skip. Size: about `98 MB`.
  - Low-memory duration sum improved to `417`, but still had index `178`
    `4 -> 5`. So the remaining duration error comes from upstream encoder
    feature differences, not only from the duration projection head.
- Built `kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-encoder.mnn`
  by keeping all `/encoder/` quantized-weight ops in fp32, plus the final iSTFT
  skip. Size: about `130 MB`.
  - Low-memory duration sum matches normal exactly: `416`, no rounded duration
    differences.
  - Phone low-memory waveform length matches normal/host: `249600` samples /
    `10.4 s`.
  - Output WAV:
    `android_long_block128_skip_istft_encoder_lowmem_f0pin_mnn.wav`.
  - Waveform stats: min `-0.599`, max `0.790`, RMS `0.0677`.
  - 3-loop phone timing for the long input: avg `4075.7 ms`, min `3520.9 ms`,
    max `5069.5 ms`. For 10.4 s audio this is avg `0.39x` RTF, best `0.34x`
    RTF. This is basically the same speed class as the smaller skip-iSTFT-only
    model on the long input (`4019.9 ms` avg, `3501.9 ms` min), but with higher
    memory (`442.7 MB` reported vs `411.8 MB`) and larger file size.

Interpretation: the low-memory static complaint has at least one concrete
mechanism: low-memory dynamic execution perturbs encoder/duration values enough
to change rounded phoneme durations on long input. Keeping the encoder fp32
fixes the length/duration mismatch without giving up the decoder/generator
low-memory speed path. Need listening feedback on
`android_long_block128_skip_istft_encoder_lowmem_f0pin_mnn.wav` to know whether
the perceived static is fully fixed or only the pacing/length portion is fixed.

Listening feedback: `android_long_block128_skip_istft_lowmem_f0pin_mnn.wav` and
`android_long_block128_skip_istft_encoder_lowmem_f0pin_mnn.wav` sound
subjectively identical. Therefore the encoder-fp32 variant fixed the objective
duration/length mismatch, but **not** the audible "lazy"/static roughness. The
remaining quality issue is downstream of duration prediction, likely in the
decoder/generator low-memory dynamic-conv execution path.

Stage probe after that listening result:

- Compared normal vs low-memory on the fixed-length encoder-fp32 model, dumping
  one intermediate tensor at a time plus F0 pin and waveform to avoid excessive
  output pinning.
- Probe tensors:
  `/decoder/decoder/encode/Div_3_output_0`,
  `/decoder/decoder/decode.{0,1,2}/Div_3_output_0`,
  `/decoder/decoder/decode.3/Div_4_output_0`,
  `/decoder/decoder/generator/LeakyRelu_output_0`,
  `/decoder/decoder/generator/conv_post/Conv_output_0`, and
  `/decoder/decoder/generator/istft/stft/ConvTranspose_output_0`.
- Normal vs low-memory tensor relative-MAE/correlation:

| Stage | rel MAE | corr |
|-------|---------|------|
| decoder encode | 0.1368 | 0.9768 |
| decoder decode.0 | 0.1516 | 0.9728 |
| decoder decode.1 | 0.1816 | 0.9645 |
| decoder decode.2 | 0.1777 | 0.9616 |
| decoder decode.3 | 0.1919 | 0.8634 |
| generator input | 0.1477 | 0.9150 |
| conv_post | 0.0497 | 0.9896 |
| iSTFT deconv / waveform-like | 0.5941 | 0.4405 |

The decoder low-memory path is already measurably different before the
generator, but this alone is not the whole audible issue.

Built a broader diagnostic hybrid:
`kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-encoder-decoderpregenerator.mnn`
which keeps `/encoder/` and `/decoder/decoder/` fp32, excluding
`/decoder/decoder/generator/`, and still skips final iSTFT quantization. This
leaves only the generator path weight-quantized/low-memory among the main
vocoder stack.

- Size: about `225 MB`.
- Phone low-memory long WAV:
  `android_long_block128_skip_istft_enc_decpre_lowmem_f0pin_mnn.wav`.
- Waveform length: `249600` samples / `10.4 s`, RMS `0.0677`.
- 3-loop timing: avg `4380.2 ms`, min `3830.2 ms`, max `5411.1 ms` for
  10.4 s audio (`0.42x` avg RTF, `0.37x` best RTF).
- Memory: `538.1 MB` reported.
- Spectral metrics are almost identical to the encoder-only low-memory variant,
  so if this also sounds the same, the remaining roughness is very likely in
  the generator low-memory dynamic-conv path itself.
- Listening feedback: `android_long_block128_skip_istft_enc_decpre_lowmem_f0pin_mnn.wav`
  is bad and sounds about the same as the encoder-only low-memory variant. So
  keeping the pre-generator decoder fp32 does **not** address the static. This
  makes the generator low-memory dynamic-conv path the next isolation target.
- User guidance: stop running repeated timing loops while isolating this. Use
  single phone generations only until a candidate actually improves quality.
  The earlier "gets worse over time" hypothesis did not materialize across
  repeated sentences; the roughness appears to be a local per-utterance artifact
  rather than progressive accumulation.

Built a broad generator diagnostic hybrid:
`kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-encoder-generator.mnn`,
which keeps all `/encoder/` and `/decoder/decoder/generator/` ops fp32 while
leaving the middle decoder quantized. This is not shippable because it is large,
but it tests whether the remaining Android low-memory roughness is inside the
generator.

- Size: about `185 MB`.
- Phone low-memory long WAV:
  `android_long_block128_skip_istft_enc_gen_lowmem_f0pin_mnn.wav`.
- Waveform length: `249600` samples / `10.4 s`, RMS about `0.0670`.
- Listening feedback: good. This is the first low-memory phone hybrid in this
  pass that fixes the static/roughness while preserving the skip-iSTFT whistle
  fix. Therefore the audible Android low-memory issue is very likely inside the
  generator, not the encoder duration path or pre-generator decoder.

Next isolation split:

- `kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-encoder-genhead.mnn`
  keeps `/encoder/` plus generator head fp32 (`ups.0`, `noise_convs.0`,
  `noise_res.0`, `resblocks.0/1/2`); size about `173 MB`.
- `kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-encoder-gentail.mnn`
  keeps `/encoder/` plus generator tail fp32 (`ups.1`, `noise_convs.1`,
  `noise_res.1`, `resblocks.3/4/5`, `conv_post`, `istft`); size about `142 MB`.
  Both should be tested as single phone generations only, not benchmark loops.

Single phone generation results for that split:

| Hybrid | WAV | Samples | RMS | High/no-tone vs normal | Status |
|--------|-----|---------|-----|-------------------------|--------|
| encoder + generator head fp32 | `android_long_block128_skip_istft_enc_genhead_lowmem_f0pin_mnn.wav` | `249600` | `0.0677` | `+0.18 dB` | Metrics match the bad low-memory output. |
| encoder + generator tail fp32 | `android_long_block128_skip_istft_enc_gentail_lowmem_f0pin_mnn.wav` | `249600` | `0.0670` | `+0.30 dB` | Metrics match the good broad generator-fp32 output. |

This makes the second generator upsampling/tail region the current root-cause
area, not the first generator stage.

Narrowed the generator tail into three more single-generation phone tests, all
keeping encoder fp32 only to avoid duration drift while isolating the artifact:

| Hybrid | WAV | Samples | RMS | High/no-tone vs normal | Status |
|--------|-----|---------|-----|-------------------------|--------|
| encoder + tail upsample/noise fp32 | `android_long_block128_skip_istft_enc_gentail_upnoise_lowmem_f0pin_mnn.wav` | `249600` | `0.0678` | `+0.21 dB` | Metrics still match the bad low-memory family. |
| encoder + tail resblocks fp32 | `android_long_block128_skip_istft_enc_gentail_resblocks_lowmem_f0pin_mnn.wav` | `249600` | `0.0677` | `+0.17 dB` | Metrics still match the bad low-memory family. |
| encoder + `conv_post` fp32 | `android_long_block128_skip_istft_enc_gentail_convpost_lowmem_f0pin_mnn.wav` | `249600` | `0.0670` | `+0.31 dB` | Metrics match the good generator-tail/broad-generator family. |

Current best root-cause candidate: quantizing
`/decoder/decoder/generator/conv_post/Conv_output_0` interacts badly with the
Android low-memory dynamic-conv path; the small conv_post output difference is
likely amplified by the final iSTFT. Next test is a minimal model with only
`conv_post` plus final iSTFT left fp32, without the encoder-fp32 duration
workaround, to separate static quality from duration drift.

Minimal static-fix test:
`kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-convpost.mnn`, generated
from the skip-iSTFT block128 config with only
`/decoder/decoder/generator/conv_post/Conv_output_0` additionally set to
`bits=0`.

- Size: `101,499,956` bytes, only `58,644` bytes larger than
  `kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft.mnn`.
- Phone low-memory long WAV:
  `android_long_block128_skip_istft_convpost_lowmem_f0pin_mnn.wav`.
- Waveform length: `250800` samples / `10.45 s`, so the separate low-memory
  encoder duration drift remains when encoder ops stay quantized.
- RMS: `0.0670`.
- Metrics vs normal phone render match the good broad-generator / convpost-fp32
  family (`highNoToneD +0.28 dB`, `airD +0.31 dB`) rather than the bad static
  family (`airD +0.76 dB`).
- Listening feedback: good.

Listening also confirmed
`android_long_block128_skip_istft_enc_gentail_convpost_lowmem_f0pin_mnn.wav` is
good. This confirms the static/roughness fix is just `conv_post` fp32 plus the
existing final-iSTFT fp32 fix; generator head/tail resblocks do not need to stay
fp32 for this audible issue.

Approximate runtime for the confirmed-good variants:

- The minimal static-fix model should be essentially the same speed as
  `skip-istft`: earlier long-input timing for `skip-istft` was avg `4019.9 ms`
  for `10.45 s` audio, about `0.38x RTF` / `2.6x` realtime. The only additional
  fp32 op is one tiny final `conv_post`, and file size only grows by `58,644`
  bytes.
- The encoder+convpost variant should be close to the earlier encoder-fp32
  timing: avg `4075.7 ms` for `10.4 s` audio, about `0.39x RTF` / `2.55x`
  realtime. This fixes duration drift but is much larger.
- The broad encoder+generator diagnostic had a measured avg around `7286 ms`
  for `10.4 s` audio (`0.70x RTF` / `1.4x` realtime), so it is quality-useful
  as a diagnostic but not the target shipping shape.

Duration drift follow-up after the static fix:

Built duration-focused hybrids on top of
`kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-convpost.mnn` and probed
`/encoder/Round_output_0` on the phone in low-memory mode.

| Model subset kept fp32 | Rounded-duration sum | Diffs vs normal | Notes |
|------------------------|----------------------|-----------------|-------|
| baseline low-memory | `418` | idx `178: 4->5`, idx `208: 5->6` | Original duration drift. |
| predictor only | `417` | idx `178: 4->5` | Fixes the final-token `g` drift only. |
| predictor + CNN text encoder | `417` | idx `178: 4->5` | CNN weights are not the remaining cause. |
| predictor + BERT mapping + BERT encoder | `416` | none | Matches normal exactly. |
| predictor + BERT mapping only | `418` | idx `178: 4->5`, idx `209: 15->16` | Not sufficient; moves the second error. |
| predictor + BERT encoder only | `417` | idx `74: 4->5` | Not sufficient; moves the remaining error. |

This means the duration bug is not plain int32 precision loss. It is a
low-memory/dynamic-quant perturbation around the text/BERT-to-duration path:
the predictor weights and both BERT-side linear projections must stay fp32 to
match normal rounded durations on this long sentence.

Candidate combined quality+duration model:
`kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-convpost-duration-bert.mnn`

- Keeps final iSTFT fp32, `conv_post` fp32, predictor fp32, and the two
  BERT-side linear projections fp32.
- Size: `104,172,172` bytes, `2,672,216` bytes larger than the minimal
  static-fix model and `2,730,860` bytes larger than the skip-iSTFT base.
- Phone low-memory WAV:
  `android_long_block128_skip_istft_convpost_duration_bert_lowmem_f0pin_mnn.wav`.
- Waveform length: `249600` samples / `10.4 s`, matching normal.
- Metrics remain in the good static-fix family (`highNoToneD +0.27 dB`,
  `airD +0.26 dB` vs normal).

Regeneration is now persisted in `rebuild_best_mnn.sh`. The script recreates
the converter compression params locally under `compression_params/`, edits the
required ops to `bits=0`, and reconverts from
`kokoro-v1.0.patched.i32.onnx`. Run:

```sh
./rebuild_best_mnn.sh
```

The exact final conversion command emitted by the script is:

```sh
uv run mnnconvert -f ONNX \
  --modelFile kokoro-v1.0.patched.i32.onnx \
  --MNNModel kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-convpost-duration-bert.mnn \
  --weightQuantBits 8 \
  --weightQuantBlock 128 \
  --compressionParamsFile compression_params/block128.skip-istft-convpost-duration-bert.json
```

The listening WAV for this candidate was generated on the phone with:

```sh
adb -s 192.168.2.104:37733 shell \
  'input keyevent KEYCODE_WAKEUP; cd /data/local/tmp/kokoro_mnn_profile_lowmem && rm -f output/0_*.txt && LD_LIBRARY_PATH=. ./ModuleBasic.out ../kokoro_mnn_bench/kokoro-block128-skip-istft-convpost-duration-bert.mnn ../kokoro_mnn_bench/long_f0pin 0 0 0 4 8 dump-long-convpost-duration-bert-low-f0pin'

adb -s 192.168.2.104:37733 pull \
  /data/local/tmp/kokoro_mnn_profile_lowmem/output/0_1.txt \
  android_long_block128_skip_istft_convpost_duration_bert_lowmem_f0pin_waveform.txt
```

Then the waveform text was converted to 24 kHz mono PCM16 WAV as
`android_long_block128_skip_istft_convpost_duration_bert_lowmem_f0pin_mnn.wav`.

Japanese ONNX smoke sample:

- Added `~/git/piper-rs/examples/kokoro_text_wav.rs` so Kokoro ONNX can be run
  with arbitrary text, voice, and output path through the existing `piper-rs`
  Japanese `mucab` frontend.
- Command:

```sh
ORT_DYLIB_PATH=/home/david/git/piper-kokoro/libonnxruntime.so \
cargo run --features japanese --example kokoro_text_wav -- \
  /home/david/git/translator-rs/scripts/kokoro_mnn/int8.onnx \
  /home/david/Downloads/voices-v1.0.bin \
  /home/david/git/mucab/out/mucab.bin \
  ja jf_alpha \
  /home/david/git/translator-rs/scripts/kokoro_mnn/kokoro_ja_rainbow_jf_alpha_onnx.wav \
  '虹は、水滴の中で光が反射、屈折、分散することによって生じ、空に光のスペクトルが現れる気象現象です。'
```

- Voice selected by `piper-rs`: `jf_alpha=38`.
- IPA emitted by the frontend:
  `niʥi wa , suiteki no naka de hikari ɡa haɴɕa , kuʔseʦu , buɴsaɴ suru koto nijoʔte ɕoːʥi , sora ni hikari no su pe ku to ru ɡa arawareru kiɕoː ɡeɴɕoː desu .`
- Output WAV: `kokoro_ja_rainbow_jf_alpha_onnx.wav`, mono PCM16, 24 kHz,
  `263400` frames / `10.975 s`.

Japanese MNN sample with the same frontend IPA/style:

- Added `generate_mnn_wav.py` to run Kokoro MNN from an explicit phoneme string
  and named voice in `voices-v1.0.bin`.
- Command:

```sh
uv run python generate_mnn_wav.py \
  --mnn kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-convpost-duration-bert.mnn \
  --voices /home/david/Downloads/voices-v1.0.bin \
  --voice jf_alpha \
  --phonemes 'niʥi wa , suiteki no naka de hikari ɡa haɴɕa , kuʔseʦu , buɴsaɴ suru koto nijoʔte ɕoːʥi , sora ni hikari no su pe ku to ru ɡa arawareru kiɕoː ɡeɴɕoː desu .' \
  --out kokoro_ja_rainbow_jf_alpha_mnn.wav \
  --normalize
```

- Output WAV: `kokoro_ja_rainbow_jf_alpha_mnn.wav`, mono PCM16, 24 kHz,
  `262800` frames / `10.950 s`, peak-normalized like `piper-rs`.
- Also kept the unnormalized first render as
  `kokoro_ja_rainbow_jf_alpha_mnn_raw.wav`.

Phone generation of the same Japanese sample:

- Prepared input folder `android_ja_rainbow_jf_alpha_f0pin/` from the same IPA,
  `jf_alpha.npy`, float speed `1.0`, and outputs
  `/decoder/decoder/F0_conv/Conv_output_0` plus `waveform`.
- Pushed to phone as `/data/local/tmp/kokoro_mnn_bench/ja_rainbow_jf_alpha_f0pin`.
- Command:

```sh
adb -s 192.168.2.104:37733 shell \
  'input keyevent KEYCODE_WAKEUP; cd /data/local/tmp/kokoro_mnn_profile_lowmem && rm -f output/0_*.txt output/1_*.txt output/2_*.txt && LD_LIBRARY_PATH=. ./ModuleBasic.out ../kokoro_mnn_bench/kokoro-block128-skip-istft-convpost-duration-bert.mnn ../kokoro_mnn_bench/ja_rainbow_jf_alpha_f0pin 0 0 3 4 8 bench-ja-rainbow-jf-alpha-low-f0pin'
```

- Module load/init line: `162.579 ms`.
- 3-run timing: avg `3897.797 ms`, min `3403.864 ms`, max `4856.907 ms`.
- Output waveform: `262800` frames / `10.950 s`, so avg RTF `0.356x`, best
  RTF `0.311x`, worst RTF `0.444x`.
- WAVs:
  - `android_ja_rainbow_jf_alpha_mnn_phone.wav` peak-normalized like `piper-rs`;
  - `android_ja_rainbow_jf_alpha_mnn_phone_raw.wav` raw model amplitude.

Per-op/profile split for the current good-quality Japanese MNN candidate:

- Model:
  `kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-convpost-duration-bert.mnn`
- Phone-side model alias:
  `/data/local/tmp/kokoro_mnn_bench/kokoro-block128-skip-istft-convpost-duration-bert.mnn`
- Input:
  `/data/local/tmp/kokoro_mnn_bench/ja_rainbow_jf_alpha_f0pin`
- Trace file: `android_ja_rainbow_current_good_perop_3run_trace.txt`
- Command:

```sh
adb -s 192.168.2.104:37733 shell \
  'input keyevent KEYCODE_WAKEUP; svc power stayon true; settings put system screen_off_timeout 1800000; cd /data/local/tmp/kokoro_mnn_profile_lowmem && rm -f output/0_*.txt output/1_*.txt output/2_*.txt && LD_LIBRARY_PATH=. ./ModuleBasic.out ../kokoro_mnn_bench/kokoro-block128-skip-istft-convpost-duration-bert.mnn ../kokoro_mnn_bench/ja_rainbow_jf_alpha_f0pin 12 0 3 4 8 profile-ja-rainbow-current-low-f0pin-3run'
```

`runMask=12` means `Session_Debug` time tracing plus
`Session_Input_Inside`; `memory=low` is preserved by runtime mask `8`. This is
the profiling build, so absolute timing is slower/noisier than production. Use
the percentages for attribution.

3 timed forwards, profile totals:

| Op type | Time | Share of op-sum | Count | Notes |
|---------|------|-----------------|-------|-------|
| Convolution | 4450.5 ms | 38.4% | 7863 | still largest bucket, but no longer the overwhelming majority |
| While | 2986.8 ms | 25.8% | 1662 | control-flow/subgraph bucket; mostly BERT and decoder elementwise work, not the predictor LSTM |
| UnaryOp | 1303.1 ms | 11.2% | 37248 | mostly generator sine/activation style work |
| Stft | 949.1 ms | 8.2% | 3 | final STFT/iSTFT path kept fp32 to avoid the 4.8/9.6 kHz whistle |
| Raster | 635.8 ms | 5.5% | 108657 | view/layout/copy overhead |
| BinaryOp | 603.9 ms | 5.2% | 52590 | elementwise residual/AdaIN arithmetic |
| Deconvolution | 339.6 ms | 2.9% | 9 | upsampling convtranspose |
| Reduction | 286.8 ms | 2.5% | 396 | reductions |

`OP Summer=11593.9 ms`; end-to-end timed average was `3945.3 ms`, min
`3437.4 ms`, max `4910.8 ms`. Per-forward averages from the op trace are
approximately: convolution `1483.5 ms`, While `995.6 ms`, UnaryOp `434.4 ms`,
STFT `316.4 ms`, Raster `211.9 ms`, BinaryOp `201.3 ms`, Deconvolution
`113.2 ms`, Reduction `95.6 ms`.

The earlier block128 trace was `Convolution=1308 ms` out of `OP
Summer=2049.8 ms` for the shorter English sample, about 64% of op-summed time.
So the current quality-preserving fast candidate did shift the profile toward
roughly "one big third in convolution" on this Japanese sample. It is not down
to the full-PTQ dynamic-int8 experiment's `331.6 ms` conv bucket, because this
candidate deliberately keeps the final STFT path, `conv_post`, predictor, and
BERT-side projections in fp32 to preserve audio quality and duration.

Follow-up on the `While` bucket: this is not mainly the Kokoro predictor LSTM.
Parsing the trace by op name gives:

| `While` prefix | Time over 3 runs | Share of `While` |
|----------------|------------------|------------------|
| `/encoder/bert/` | 1749.1 ms | 58.6% |
| `/decoder/` | 1217.2 ms | 40.8% |
| `/encoder/predictor/` | 0.5 ms | ~0.0% |

The previous "LSTM subgraphs" shorthand was too broad. We are still using the
While-LSTM artifact overall because native MNN LSTM mutates recurrent state
across repeated forwards, but the current `While` profile cost is not an easy
"turn on native LSTM" win. The earlier native-vs-While speed-only test also
showed native LSTM was only marginally faster while being correctness-risky.

Largest individual timed ops in the 3-run trace:

| Op | Type | Time |
|----|------|------|
| `/decoder/decoder/generator/STFT_output_0` | Stft | 949.1 ms |
| `/decoder/decoder/generator/ups.1/ConvTranspose_output_0` | Deconvolution | 196.0 ms |
| `/decoder/decoder/generator/resblocks.5/convs2.1/Conv_output_0` | Convolution | 149.6 ms |
| `/decoder/decoder/generator/resblocks.5/convs1.0/Conv_output_0` | Convolution | 149.5 ms |
| `/decoder/decoder/generator/resblocks.5/convs2.2/Conv_output_0` | Convolution | 149.1 ms |
| `/decoder/decoder/generator/noise_res.1/convs*.*/Conv_output_0` family | Convolution | about 148 ms each |

Interpretation: convolution is still the first performance target, but it is
now one of several meaningful buckets. Further wins probably come from a mix of
remaining generator conv/deconv work, STFT, and graph overhead, not from one
single 80%-style hotspot.

Started branch-specific optimization checks for the two largest non-conv
profile areas:

1. Decoder-side `While`: parse shows mostly generator AdaIN/resblock
   elementwise ops (`Sub`, `Mul`, `Div`, `Add`) under
   `/decoder/decoder/generator/{noise_res.1,resblocks.3,resblocks.4,resblocks.5}`.
   This is not weight-heavy, so converter weight-int8 is unlikely to help
   directly; likely candidates are graph/elementwise fusion or avoiding layout
   churn.
2. Encoder/BERT `While`: parse shows BERT FFN/attention projections dominate.
   We already know both BERT-side fp32 exclusions are needed for exact rounded
   duration on the long sentence, but profiling the narrower variants can tell
   whether relaxing either one would be worth further correctness work.

Metric interpretation from `whistle_metrics.py`:

- Block128 does **not** have a huge broad high-frequency energy mismatch. It is
  only about `+0.17` to `+0.24 dB` in total 4-11 kHz energy vs fp32, and about
  `-0.06` to `+0.04 dB` if the two tone bands are excluded.
- The audible problem is the narrow tonal prominence:
  - local block128: `+2.82 dB` at 4.8 kHz and `+3.82 dB` at 9.6 kHz vs fp32;
  - phone block128 low-memory: `+2.95 dB` at 4.8 kHz and `+3.47 dB` at
    9.6 kHz vs fp32.
- The notch check removes those tones while preserving broad high-frequency
  balance (`highNoToneD` remains about `+0.05 dB`), matching listening feedback
  that `notched_block128_remove_48_96.wav` sounds much better.
- HQQ is close to fp32 by the same metric: phone HQQ low-memory is `+0.18 dB`
  at 4.8 kHz and `-1.34 dB` at 9.6 kHz vs fp32, with similar broad high-band
  balance. This was the best lead before the final-iSTFT skip experiment. The
  current lead is now `block128.skip-istft`, which keeps block128 speed/size and
  removes the whistle at the source.

Counter checks on the release build with `simpleperf stat`:

| Counter group | Result |
|---------------|--------|
| cycles/instructions/backend stalls | 87.46B cycles, 153.57B instructions, 38.18B backend-stalled cycles |
| L1D | 38.88B loads, 378.95M load misses |
| cache | 44.67B refs, 397.84M misses |

Approximate ratios from those runs: IPC ~1.76, backend-stalled cycles ~44% of
cycles, L1D miss rate ~0.97%, cache miss rate ~0.89%. These are whole-process
counters across a noisy phone run, but they do not look like a catastrophic
cache-miss/fp32-int8-conversion ping-pong problem. The dominant visible work is
packed convolution matmul plus some threadpool/scheduler overhead and smaller
elementwise/raster/sine/STFT costs.

Performance next-step decision tree:

1. Done: MNN does have faster int8 kernels for Kokoro's actual conv shapes.
   Standalone PTQ convs hit `MNNGemmInt8AddBiasScale_ARMV82_Unit.S` and are
   about 4-5x faster than fp32 by min loop time. Standalone fp16 is not a strong
   candidate.

2. If int8 kernels are faster in microbench, find why the full model does not
   select them. Done for the full PTQ artifact:
   - The hot decoder convs miss `ConvInt8` because their inputs are float after
     AdaIN/residual `BinaryOp` chains.
   - The default Android build had `MNN_LOW_MEMORY=OFF` and
     `MNN_CPU_WEIGHT_DEQUANT_GEMM=OFF`, so the dynamic-quant fallback for those
     float-input convs was unavailable.
   - A low-memory/dequant-GEMM Android build plus runtime `memory=low` routes
     those convs through `DenseConvInt8Tiled dynamic=1` and is much faster on
     the profile build.

3. If fp16 kernels are faster in microbench, find why `precision=low` is slower
   in the full graph.
   - Instrument selected conv executor and tensor dtype/layout.
   - Check whether only some ops use fp16 while others force fp32 conversions.
   - Try a converter-fp16 model plus normal runtime and a normal model plus
     low-precision runtime, but judge by selected kernels rather than only
     total latency.

4. If graph overhead is material after conv kernel selection is understood,
   consider fusion/layout work.
   - `Raster`, `BinaryOp`, `UnaryOp`, `MNNSin`, and `Stft` are secondary costs.
   - The current profiles do not show conversion as the main bottleneck, so do
     not start with generic quant/dequant fusion.
   - Fusion may still help if the callgraph shows repeated layout transforms
     around the same decoder conv blocks after instrumentation.

5. Keep a quality gate separate from performance.
   - The block128 whistle is parked for now.
   - If a faster int8/fp16 path is found, regenerate phone/local WAVs and check
     the 4800/9600 Hz peaks before calling it a candidate.

## Current next steps

1. Treat native-LSTM conversions as first-run-only unless the runtime recreates
   the module per utterance. Prefer While-LSTM artifacts for stability.
2. Patch or bypass MNN 3.5's Python `mnnquant.py` wrapper. The correct call is
   `_tools.mnnquant(sys.argv)` / `_tools.mnnquant([...])`, not three separate
   string arguments.
3. Treat the produced native-LSTM PTQ model as a failed fidelity candidate.
   Timing is subjectively correct; the blocker is severe codec-like audio
   degradation.
4. Treat `kokoro-v1.0.patched.i32.while.wq8.block128.mnn` as the best current
   MNN candidate by metrics, with `kokoro-v1.0.patched.i32.while.wq8.mnn` as
   the already-listened fallback. Both are stable and preserve sample count;
   block128 is also the fastest Android MNN candidate measured so far.
5. Do not use MNN runtime `precision=low` for this Kokoro graph on the tested
   phone unless a different backend/build proves otherwise. It is roughly 2x
   realtime, much slower than normal-mode CPU.
6. Keep the low-memory/dequant-GEMM Android build path:
   `MNN_LOW_MEMORY=ON`, `MNN_CPU_WEIGHT_DEQUANT_GEMM=ON`, runtime
   `memory=low`. It fixes the hot-conv fast-path issue for full PTQ.
7. Next performance/correctness branch: regenerate/listen to a WAV from the
   low-memory/dequant-GEMM full PTQ path only as a sanity check. Done: it is
   effectively silent. The real remaining work is correctness in MNN's dynamic
   low-memory quantized conv path and/or PTQ calibration/graph partitioning.

## File index

- `bench.py` — ORT vs MNN comparator with WAV dump. Defaults to no warmup and
  copies MNN output buffers before parity/WAV checks. Supports
  `--mnn-precision`, `--mnn-memory`, and `--mnn-power` for runtime config tests.
- `bench_onnx.py` — ORT-only multi-model bench.
- `patch_resize.py` — `onnx_graphsurgeon` script for the Resize + Round
  rewrites, and now also narrows `input_ids` from `INT64` to `INT32` for MNN
  `Gather` correctness.
- `probe_duration.py` — temporary ORT-vs-MNN intermediate tensor probe used to
  identify the first mismatch at the word embedding `Gather`.
- `probe_mnn.py` — MNN-vs-MNN intermediate tensor probe for comparing fp32 and
  PTQ MNN artifacts.
- `probe_stability.py` — repeated-forward stability probe; identified native
  MNN LSTM state mutation across forwards.
- `make_quant_calib.py` — generates MNN sequence-input calibration folders
  under `/tmp` and writes `quant.json` outside the sample folder for
  `mnnquant`.
- `kokoro-v1.0.onnx` — fp32 source from HuggingFace.
- `kokoro-v1.0.patched.onnx` — older fp32 source with Resize + Round patches
  applied. Still has `input_ids` as `INT64`, so it is not the fixed MNN source.
- `kokoro-v1.0.patched.i32.onnx` — fixed fp32 source with Resize + Round
  patches plus `input_ids: INT32`.
- `kokoro-v1.0.patched.i32.mnn` — fixed fp32 MNN conversion using native MNN
  LSTM. First inference has good parity, but repeated forwards on one module
  mutate recurrent state and diverge.
- `kokoro-v1.0.patched.i32.while.mnn` — fixed fp32 MNN conversion using While
  subgraphs for LSTMs. Stable repeated forwards and matching 80400-sample
  outputs.
- `kokoro-v1.0.patched.i32.ptq-ema.mnn` — native-LSTM full PTQ model produced
  via direct `_tools.mnnquant([...])`; 135 MB. Timing is subjectively correct,
  but audio fidelity is severely degraded.
- `kokoro-v1.0.patched.i32.ptq-noact.mnn` — JSON ablation of full PTQ with
  activation `quantInfo` stripped. Spectral metric improves, but quality/parity
  still does not beat While weight-only int8.
- `kokoro-v1.0.patched.i32.ptq-ema-skip-encoder.mnn` — PTQ attempt skipping
  `/encoder/` ops; 135 MB, wrong sample count and effectively silent output in
  the benchmark.
- `kokoro-v1.0.patched.i32.ptq-ema-skip-decoder.mnn` — PTQ attempt skipping
  `/decoder/decoder/` ops to keep the vocoder path fp32. Byte-identical to the
  full PTQ model, so the skip did not take effect.
- `kokoro-v1.0.patched.i32.wq8.mnn` — converter weight-only 8-bit model,
  native LSTM, 127 MB. Not full PTQ.
- `kokoro-v1.0.patched.i32.while.wq8.mnn` — converter weight-only 8-bit model,
  While LSTM, 96 MB. Not full PTQ. Good by listening.
- `kokoro-v1.0.patched.i32.while.wq8.block128.mnn` — converter weight-only
  8-bit model with `--weightQuantBlock 128`, While LSTM, 97 MB. Best current
  MNN candidate by metrics.
- `kokoro-v1.0.patched.i32.while.wq8.block64.mnn`,
  `kokoro-v1.0.patched.i32.while.wq8.block256.mnn`,
  `kokoro-v1.0.patched.i32.while.wq8.asym.mnn`,
  `kokoro-v1.0.patched.i32.while.wq8.hqq.mnn` — extra weight-only variants;
  all worse than block128 by the quick bench.
- `kokoro-v1.0.patched.i32.while.fp16.mnn` — fp16 converter output, 156 MB.
  Stable but worse parity than fp32 and block128 weight-only.
- `kokoro-v1.0.patched.i32.while.wq8.block128.fp16.mnn` — attempted int8+fp16
  hybrid. Size and parity match fp16 behavior, not the 97 MB block128
  weight-only model.
- `kokoro-v1.0.*.mnn` — older converted MNN models. The ones without the i32
  patch still produce the parity bug described above.
- `int8.onnx` — copy of the shipped int8 model.
- `model_fp16.onnx`, `model_q8f16.onnx` — alternative ONNX variants from
  HF for the recommended fp16 ORT bench.
- `out_ort.wav`, `out_mnn.wav` — last benchmark's audio output.
- `cmp_fp32_while_*.wav`, `cmp_wq8_while_*.wav`,
  `cmp_ptq_ema3_native_*.wav` — named comparison outputs from the latest
  debugging pass.
- `durations.onnx`, `durations.mnn` — extracted duration-predictor subgraph
  used to isolate the divergence to that submodule.
- `conv_microbench/android_micro_ptq_convlog.txt` — standalone PTQ conv
  executor-selection log; shows `selected=ConvInt8`.
- `conv_microbench/android_full_ptq_convlog.txt` — full PTQ executor-selection
  log on the original profile build; shows hot K7/K11 decoder convs falling
  back to float `Convolution`.
- `conv_microbench/android_full_ptq_lowmem_enabled_convlog.txt` and
  `conv_microbench/android_full_ptq_lowmem_enabled_speed.txt` — low-memory /
  dequant-GEMM build logs showing dynamic int8 conv selection and speed.
- `android_ja_rainbow_current_good_perop_3run_trace.txt` — current
  quality-preserving block128/skip-iSTFT/convpost-duration-bert Android
  profiling trace for the Japanese `jf_alpha` rainbow sample; conv is 38.4% of
  op-summed profile time.
- `android_fast_ptq_lowmem_mnn.wav` — phone-generated low-memory/dequant-GEMM
  full PTQ WAV; effectively silent due near-zero waveform.
- `android_ptq_normal_mnn.wav` — normal-memory control WAV from the same
  low-memory/dequant-GEMM build and input; not silent.
