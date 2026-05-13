# MNN Converter Issue Draft: duplicated external weights after `--optimizePrefer 2`

## Summary

`mnnconvert --optimizePrefer 2 --saveExternalData` can materialize duplicate
copies of shared ONNX weights in the generated external `.mnn.weight` sidecar.
The runtime accepts repeated references to a single external blob, so this
appears to be a converter/storage dedup limitation rather than a runtime
requirement.

## Repro Context

Model family: Kokoro v1.0 ONNX, after local graph cleanup for MNN conversion.
The source model has shared ALBERT/BERT-style projection weights. With
`--optimizePrefer 2`, MNN's MatMul-to-Conv optimization exposes/materializes
many of those projections as external quantized blobs.

Representative command:

```sh
uv run mnnconvert -f ONNX \
  --modelFile kokoro-v1.0.patched.i32.onnx \
  --MNNModel kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-convpost-duration-bert.optfast-external.mnn \
  --weightQuantBits 8 \
  --weightQuantBlock 128 \
  --compressionParamsFile compression_params/block128.optfast.skip-istft-convpost-duration-bert.json \
  --optimizePrefer 2 \
  --saveExternalData
```

The exact source model and compression params are not attached here, but the
pattern should apply to any graph where `--optimizePrefer 2` transforms shared
linear weights into repeated Conv/quantized-weight payloads.

## Observed Result

External sidecar before dedup:

- main `.mnn`: `1,027,352` bytes
- `.mnn.weight`: `151,709,247` bytes
- total: `152,736,599` bytes

After byte-identical blob dedup and direct flatbuffer external-offset patching:

- main `.mnn`: `1,027,352` bytes
- `.mnn.weight`: `86,949,555` bytes
- total: `87,976,907` bytes
- referenced external bytes: `151,709,247`
- unique external bytes: `86,949,555`
- removable duplicate bytes: `64,759,692`

The deduped model produces byte-identical PCM output to the non-deduped
external model and to the full single-file optfast model in our Kokoro test.
It also runs on Android with MNN `createFromFile`, provided the `.mnn.weight`
sidecar is next to the `.mnn` file.

## What Worked Locally

The local helper uses a JSON dump only as metadata, then patches int64 external
offsets in the original flatbuffer. It does not roundtrip JSON back to MNN,
because JSON -> MNN changes Kokoro waveform output in this case.

Important details from the helper:

- Include both top-level `oplists` and `subgraphs[].nodes`.
- Dedup key uses op type, main type, `main.common`, external blob tail metadata,
  and SHA-256 of the blob bytes.
- Repeated external offsets are accepted by MNN runtime.

## Expected Behavior

When converter output uses external weights, identical external payloads with
compatible metadata should be emitted once and referenced multiple times, or the
converter should otherwise preserve shared-source weights through
`--optimizePrefer 2` transformations when possible.

## Why This Matters

For this model, `--optimizePrefer 2` is the useful speed optimization, but
without external blob dedup it grows the artifact from roughly 104 MB to 153 MB.
With dedup, the optimized artifact is about 88 MB while preserving the same
runtime output.
