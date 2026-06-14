#!/usr/bin/env bash
# Bootstrap a fresh (ephemeral) GPU box to fine-tune the PP-OCRv6 small recognizer.
# Tested on vast.ai RTX 5070 Ti (Blackwell sm_120), CUDA 13 driver, Ubuntu 22.04,
# torch preinstalled, ~16G overlay disk + /dev/shm.
#
# Stage the rec files first from the local machine, then run this on the server:
#   scp -i ~/.ssh/davidkey -P <port> \
#       gen_hebrew.py prep_hebrew_corpus.py paddle/hebrew_latin_dict.txt \
#       paddle/hebrew_finetune.yml data/hebrew_corpus.txt \
#       root@<host>:/root/rec/
#   ssh -i ~/.ssh/davidkey -p <port> root@<host> 'bash /root/rec/setup_server.sh'
#
# hebrew_corpus.txt is optional — if absent it is rebuilt from Leipzig via
# prep_hebrew_corpus.py --download.
set -euo pipefail

REC_DIR=${REC_DIR:-/root/rec}
DATA_DIR=${DATA_DIR:-/dev/shm/heb}          # dataset in RAM: overlay disk is tiny
PADDLEOCR_DIR=${PADDLEOCR_DIR:-/root/PaddleOCR}
CONFIG=${CONFIG:-$REC_DIR/hebrew_finetune.yml}
PRETRAINED_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams"

N_WORKERS=${N_WORKERS:-20}                   # parallel generator processes
N_PER=${N_PER:-15000}                        # lines per worker (20*15000 = 300K)
VAL_N=${VAL_N:-8000}
EPOCHS=${EPOCHS:-20}
BS=${BS:-256}

echo "=== [1/6] reclaim disk (uv/pip/hf caches from any prior tenant) ==="
rm -rf /root/.cache/uv /root/.cache/pip /root/.cache/huggingface /root/.cache/torch /tmp/pip-* 2>/dev/null || true
df -h / | tail -1

echo "=== [2/6] paddlepaddle-gpu (cu129, runs on Blackwell) + GPU smoke test ==="
if ! python3 -c "import paddle" 2>/dev/null; then
  pip3 install --no-cache-dir "paddlepaddle-gpu==3.2.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
fi
python3 - <<'PY'
import paddle
assert paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() >= 1, "no GPU paddle"
z = paddle.matmul(paddle.randn([1024, 1024]), paddle.randn([1024, 1024]))
print("paddle", paddle.__version__, "GPU matmul ok on", str(z.place))
PY

echo "=== [3/6] system libs for opencv + PaddleOCR repo + rec/gen python deps ==="
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1
[ -d "$PADDLEOCR_DIR" ] || git clone --depth 1 -q https://github.com/PaddlePaddle/PaddleOCR.git "$PADDLEOCR_DIR"
pip3 install --no-cache-dir -q -r "$PADDLEOCR_DIR/requirements.txt" uharfbuzz python-bidi freetype-py

echo "=== [4/6] pretrained v6 small weights ==="
mkdir -p "$REC_DIR/pretrain"
[ -f "$REC_DIR/pretrain/PP-OCRv6_small_rec_pretrained.pdparams" ] || \
  wget -q -O "$REC_DIR/pretrain/PP-OCRv6_small_rec_pretrained.pdparams" "$PRETRAINED_URL"

echo "=== [5/6] dataset -> $DATA_DIR ($((N_WORKERS*N_PER)) train + $VAL_N val) ==="
cd "$REC_DIR"
[ -f hebrew_corpus.txt ] || python3 prep_hebrew_corpus.py --download --out hebrew_corpus.txt
rm -rf "$DATA_DIR" && mkdir -p "$DATA_DIR/images"
for k in $(seq 0 $((N_WORKERS-1))); do
  python3 gen_hebrew.py --out "$DATA_DIR" --n "$N_PER" --seed "$k" --prefix "w$k" --corpus hebrew_corpus.txt >/dev/null 2>&1 &
done
python3 gen_hebrew.py --out "$DATA_DIR" --n "$VAL_N" --seed 1000 --prefix val --corpus hebrew_corpus.txt >/dev/null 2>&1 &
wait
cat "$DATA_DIR"/labels_w*.txt > "$DATA_DIR/train_list.txt"
cat "$DATA_DIR"/labels_val*.txt > "$DATA_DIR/val_list.txt"
echo "train=$(wc -l < "$DATA_DIR/train_list.txt") val=$(wc -l < "$DATA_DIR/val_list.txt")"

echo "=== [6/6] fine-tune (detached; tail /root/rec/train.log) ==="
cd "$PADDLEOCR_DIR"
nohup python3 tools/train.py -c "$CONFIG" \
  -o Global.epoch_num="$EPOCHS" Global.save_epoch_step=5 \
     Train.sampler.first_bs="$BS" Train.loader.batch_size_per_card="$BS" \
  > "$REC_DIR/train.log" 2>&1 &
echo "train PID $!  ->  tail -f $REC_DIR/train.log"
