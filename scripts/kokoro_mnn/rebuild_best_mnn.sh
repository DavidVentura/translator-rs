#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SOURCE_ONNX="${SOURCE_ONNX:-kokoro-v1.0.patched.i32.onnx}"
OUT_MNN="${OUT_MNN:-kokoro-v1.0.patched.i32.while.wq8.block128.skip-istft-convpost-duration-bert.mnn}"
WORK_DIR="${WORK_DIR:-compression_params}"

BASE_COMPRESS="$WORK_DIR/block128.base.json"
FINAL_COMPRESS="$WORK_DIR/block128.skip-istft-convpost-duration-bert.json"
BASE_MNN="$WORK_DIR/block128.base.mnn"

mkdir -p "$WORK_DIR"
rm -f "$BASE_COMPRESS" "$FINAL_COMPRESS"

# First pass: ask MNN's converter to create a block128 weight-only compression
# parameter file. The model emitted in this pass is only an intermediate.
uv run mnnconvert -f ONNX \
  --modelFile "$SOURCE_ONNX" \
  --MNNModel "$BASE_MNN" \
  --weightQuantBits 8 \
  --weightQuantBlock 128 \
  --compressionParamsFile "$BASE_COMPRESS"

# Final pass config:
# - final iSTFT ConvTranspose fp32: removes the 4.8/9.6 kHz whistle;
# - generator conv_post fp32: fixes Android low-memory static/roughness;
# - predictor + BERT-side projections fp32: fixes Android low-memory rounded
#   duration drift on the long sentence.
uv run python edit_compression_params.py \
  --infile "$BASE_COMPRESS" \
  --outfile "$FINAL_COMPRESS" \
  --bits 0 \
  --op /decoder/decoder/generator/istft/stft/ConvTranspose_output_0 \
  --op-prefix /decoder/decoder/generator/conv_post/ \
  --op-prefix /encoder/predictor/ \
  --op-prefix /encoder/bert/ \
  --op-prefix /encoder/bert_encoder/

uv run mnnconvert -f ONNX \
  --modelFile "$SOURCE_ONNX" \
  --MNNModel "$OUT_MNN" \
  --weightQuantBits 8 \
  --weightQuantBlock 128 \
  --compressionParamsFile "$FINAL_COMPRESS"

stat -c '%n %s bytes' "$OUT_MNN"
