#!/usr/bin/env bash
# Provision a fresh GPU box (e.g. vast.ai) to train the ink model.
#
# Run this ON the box after copying the training files:
#   scp -i <key> -P <port> train.py gen_data.py model.py setup_vast.sh root@<host>:~/ink/
#   ssh -i <key> -p <port> root@<host> 'bash ~/ink/setup_vast.sh'
#
# PREFER A PYTORCH IMAGE. torch is ~3GB from the PyPI CDN (files.pythonhosted.org),
# and some vast instances cannot reach that CDN at all (apt mirror works, DNS works,
# but the Fastly CDN times out) — torch is then uninstallable. A PyTorch-template
# image ships torch preinstalled, so you only need the apt fonts below. This script
# skips any package already present, so it's a no-op for torch on such an image.
#
# Other notes baked in from a real run:
#   - Stock vast images can be bare Ubuntu (no torch, no pip, no fonts).
#   - astral.sh is often DNS-blocked while pypi.org resolves, so uv comes from pypi.
#   - The container CPU quota is far below the host core count; derive workers from
#     the cgroup, not nproc.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "== apt: pip, fontconfig, fonts (incl. CJK), dictionary =="
apt-get update -qq
apt-get install -y -qq \
  python3-pip fontconfig \
  fonts-dejavu fonts-liberation fonts-noto-core fonts-noto-cjk fonts-ubuntu \
  wamerican

# Only fetch Python packages that are missing (torch is preinstalled on a PyTorch image).
missing=""
for mod_pkg in torch:torch numpy:numpy PIL:pillow scipy:scipy; do
  mod=${mod_pkg%%:*}; pkg=${mod_pkg##*:}
  python3 -c "import $mod" 2>/dev/null || missing="$missing $pkg"
done
if [ -n "$missing" ]; then
  echo "== installing missing:$missing (via uv from pypi) =="
  python3 -m pip install --default-timeout=120 --retries 10 uv
  UV=$(command -v uv || echo /usr/local/bin/uv)
  UV_HTTP_TIMEOUT=180 "$UV" pip install --system $missing
else
  echo "== all python deps already present =="
fi

echo "== verify =="
python3 -c "import torch, numpy, PIL, scipy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python3 -c "import subprocess; n=len({l.split(':')[0] for l in subprocess.check_output(['fc-list',':lang=ko','file'],text=True).splitlines()}); print('cjk fonts:', n)"

# Worker count from the container's real CPU quota (cgroup v2 then v1), not nproc.
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r q p < /sys/fs/cgroup/cpu.max
  [ "$q" = max ] && cores=$(nproc) || cores=$(( q / p ))
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
  [ "$q" = -1 ] && cores=$(nproc) || cores=$(( q / p ))
else
  cores=$(nproc)
fi
workers=$(( cores > 1 ? cores - 1 : 1 ))

echo
echo "ready. ~${cores} usable cores. suggested run from ~/ink:"
echo "  python3 train.py --steps 8000 --batch 256 --workers ${workers} --lr 2.5e-3 --out ckpt"
echo "(batch*steps = strips seen; dial --steps to your strip budget)"
