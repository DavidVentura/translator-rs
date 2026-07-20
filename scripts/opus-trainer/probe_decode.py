#!/usr/bin/env python3
"""Decode a probe file with Hy-MT2 under vLLM, into an arbitrary target language.

Separate from vllm_decode.py on purpose: that script is the memoized KD decode
step, hardcoded to English, and editing it would invalidate the KD chain's step
keys mid-campaign. This one is eval-only.

Probe sets have no references — they exist because FLORES is clean news and hides
the failure mode that matters (hallucinated entities, mangled numbers/units,
register collapse on informal text). The output is read, not scored.

    probe_decode.py MODEL SRC TGT_LANG_NAME OUT
"""

import sys
import time


def main() -> None:
    from vllm import LLM, SamplingParams

    model, src, tgt, out = sys.argv[1:5]

    # Hy-MT2's own English instruction template (matches hy_mt2_gate.py exactly).
    template = ("Translate the following text into {tgt}. Note that you should only "
                "output the translated result without any additional explanation: {text}")

    lines = [l for l in open(src, encoding="utf-8").read().splitlines() if l.strip()]

    llm = LLM(model=model, dtype="auto", trust_remote_code=True, max_model_len=2048,
              gpu_memory_utilization=0.90, enforce_eager=True)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": template.format(tgt=tgt, text=s)}],
            add_generation_prompt=True, tokenize=False,
        )
        for s in lines
    ]

    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
    dt = time.time() - t0

    with open(out, "w", encoding="utf-8") as f:
        for o in outs:
            f.write(o.outputs[0].text.strip().replace("\n", " ").replace("\t", " ") + "\n")
    print(f"THROUGHPUT {len(lines)} lines in {dt:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
