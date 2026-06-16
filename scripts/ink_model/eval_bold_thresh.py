"""Bold threshold sweep for the asymmetric preference: never embolden THIN text
(false positive), but missing a bold (rendering it normal) is fine. For each bold
threshold t, report on uniform lines: FP (normal->bold), FN (bold->normal, benign),
and the false-positive rate split by the regular font's real weight — light (<=300,
genuinely thin: must stay ~0) vs mid (==400, the heaviest 'regular', bold-ish is OK).

  python3 eval_bold_thresh.py ckpt/ink-latest.pt
"""

import random
import sys

import numpy as np
import torch

import gen_data as g
from model import InkUNet

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "ckpt/ink-latest.pt", map_location="cpu")
model = InkUNet(base=ck["base"], levels=ck["levels"], bold_from=ck.get("bold_from", 1),
                bold_head=ck.get("bold_head", "dilated"))
model.load_state_dict(ck["model"])
model.eval()
g._cjk_units = lambda r, txt, ib: [(txt, ib)]


def pooled_scores(plan_fn, n, pool=None):
    if pool is not None:
        g._weighted_fonts = lambda script, bold: pool
    rng = random.Random(7)
    out = []
    for _ in range(n):
        g._run_plan = plan_fn(rng)
        img, cov, bold = g.sample(rng, width=320)
        ink = cov > 0.5
        if ink.sum() < 40:
            continue
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]
        with torch.no_grad():
            pred = torch.sigmoid(model(x))[0, 1].numpy()
        out.append(float(pred[ink].mean()))
    return np.array(out)


def uniform(target):
    def make(_rng):
        def plan(r):
            o = []
            for _ in range(r.randint(1, 4)):
                rr = r.random()
                s = "cjk" if rr < 0.18 else (g.SHAPED_NAMES[r.randrange(len(g.SHAPED_NAMES))] if rr < 0.36 else "latin")
                o.append((s, target))
            return o
        return plan
    return make


light = tuple(p for p in g.font_paths() if (w := g._font_weight(p)) is not None and w <= 300)
mid = tuple(p for p in g.font_paths() if (w := g._font_weight(p)) is not None and w == 400)

reg = pooled_scores(uniform(False), 1000)
bold = pooled_scores(uniform(True), 1000)
g._run_plan = lambda r: [("latin", False)]
light_s = pooled_scores(lambda _r: g._run_plan, 600, pool=light)
mid_s = pooled_scores(lambda _r: g._run_plan, 600, pool=mid)

print(f"head={ck.get('bold_head','dilated')} base={ck['base']}  (reg {len(reg)} / bold {len(bold)} / light {len(light_s)} / mid {len(mid_s)})\n")
print("bold_thr   FP(norm→bold)  FN(bold→norm)   thin-light FP   mid-400 FP")
for t in (0.50, 0.60, 0.70, 0.80):
    fp = (reg >= t).mean()
    fn = (bold < t).mean()
    lfp = (light_s >= t).mean()
    mfp = (mid_s >= t).mean()
    print(f"  {t:.2f}      {fp:6.1%}        {fn:6.1%}         {lfp:6.1%}        {mfp:6.1%}")
