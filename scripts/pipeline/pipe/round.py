"""Skeleton for a training round: the steps a corpus MUST pass before it is trained on.

WHY THIS EXISTS. Round 1 of uig shipped a student trained on a corpus that was
~15-20% Arabic-script Kazakh. Nothing was broken and nothing errored — prep_data.py
ran alpha_ratio, length and fastText LID exactly as designed, and lid.176 only knows
Cyrillic Kazakh, so töte scored as Uyghur and passed. The teacher then hallucinated
on it ("جون" jön/"proper" -> the name "John") and the student learned from that. A
corpus defect is silent by construction: training succeeds, only the scores are bad.

pipe is the pipeline and stays the pipeline — this does not execute or cache
anything, it declares the SHAPE of a round and hands the steps to pipe. What it adds
is that a round cannot quietly skip a stage: the next pair's author has to answer
each question rather than remember to ask it.

THE SPLIT, which is the whole point:

  ABSTRACT (varies per pair, you must implement it)
      sources / lid / gold / known_good

  CONCRETE (invariant, the base class runs it, you cannot skip or override it)
      gate()

Everything the 2026-07-17 uig investigation caught was caught by the gate, not by
having the right rules — the rules were mostly WRONG on first write:
  - "-دىڭ/-تىڭ is a Kazakh genitive"      -> also Uyghur 2sg past (ئالدىڭ, "you took");
                                             0.55% false-drop for 3.7% recall.
  - "ەمەس is a Kazakh token"               -> Uyghur ئەمەس CONTAINS it; 2.6% -> 0.001%
                                             once matched on word boundaries.
  - "ئ ې ۈ are the Uyghur markers"         -> missed غ خ ھ چ entirely, which are the
                                             complementary halves of ع ح.
  - "lines with lots of #&!? are junk"     -> 11.9% of HPLT, but also 7.7% of FLORES.
Hand-written rules are a hypothesis. The gate is what makes them a finding.

CALIBRATION, learned the hard way: gold alone is NOT a false-positive gate. FLORES
devtest is 1012 lines, and the bad -دىڭ rule scored 0/1012 on it — 0.55% of 1012 is
~6 lines, indistinguishable from noise. It was the 2.46M known-good bucket that
caught it. So `known_good` is the load-bearing reference and `gold` is only a
sanity floor. known_good = the corpus that trained the last student that worked; it
is large and it has the real distribution.

WHAT THIS CANNOT DO. It cannot invent the unknown unknown. Nobody knew töte Kazakh
was in there, and no skeleton makes you think of it. What it does is force the
QUESTION into every round ("what shares this script?", "what is my known-good?") and
make the answer measured rather than asserted. `fertility_histogram` is the one
generic smell that does not need you to know what you are looking for: a corpus that
is 85% language A and 15% language B is BIMODAL against a vocab trained on A (the
uig buckets measured 1.81 vs 2.12 pieces/word) — a single blended number hides that,
the shape does not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .step import Run
from .types import Artifact


@dataclass(frozen=True)
class Source:
    """Where a corpus comes from. Fetched ONCE on the hub, never per box: a US box's
    peering to HF/the hub measured ~1KB/s, and the vast `inet_down` filter does not
    catch it (it measures peak to a speedtest server, not the path that matters)."""

    name: str
    url: str
    text_field: str | None = None  # set for document-level JSONL (HPLT, MADLAD)


@dataclass(frozen=True)
class GateReport:
    rule: str
    fires_on_target: float   # share of the corpus this would drop
    fp_on_gold: float        # share of gold (small; sanity floor only)
    fp_on_known_good: float  # share of the last working corpus — THE number

    @property
    def ratio(self) -> float:
        return self.fires_on_target / max(self.fp_on_known_good, 1e-9)


class TrainRound(ABC):
    """One training round for one pair. Subclass per pair, not per attempt."""

    # ---- varies per pair: you must answer these ----

    @abstractmethod
    def sources(self) -> list[Source]:
        """Every corpus this round trains on, including carried-over ones.

        Carried-over corpora are NOT exempt from filtering. The round-1 uig corpus
        had already trained a working student and still carried 0.92% forum/CJK junk
        and 0.22% byte-fallback mojibake — the filter is a pass over ALL corpora, and
        being a no-op on a clean one is the expected outcome, not a reason to skip it.
        """

    @abstractmethod
    def lid(self) -> object:
        """Pair-specific language separator (script_lid-shaped: text -> verdict).

        Required because the generic LID is the thing that fails: fastText lid.176
        classifies Kazakh from Cyrillic only. Ask what shares this script — ug/kk is
        not the only pair with a neighbour hiding behind a shared script.
        """

    @abstractmethod
    def gold(self) -> Path:
        """Held-out gold for the pair (FLORES devtest). Sanity floor, not the gate."""

    @abstractmethod
    def known_good(self) -> Path | None:
        """The corpus that trained the last student that worked, for FP calibration.

        None is allowed — a first round has no predecessor — but the gate then has no
        sensitive false-positive reference and says so loudly. It is not a free pass.
        """

    # ---- invariant: the base class does this, a subclass never overrides it ----

    def gate(self, rule_name: str, fires: float, gold_fp: float, kg_fp: float | None,
             budget: float) -> GateReport:
        """Refuse a filter rule whose cost on real text exceeds its budget.

        Deliberately concrete: this is the step no round may skip, because it is the
        only one that turned a plausible rule into a measured one every time it ran.
        """
        if kg_fp is None:
            raise ValueError(
                f"{rule_name}: no known_good corpus, so its false-positive rate is "
                f"UNMEASURED. gold is {gold_fp:.3%} but gold is too small to bound it "
                f"(a rule costing 0.55% scored 0/1012 on FLORES). Supply known_good, "
                f"or state explicitly that this round accepts an unmeasured filter."
            )
        report = GateReport(rule_name, fires, gold_fp, kg_fp)
        if kg_fp > budget:
            raise ValueError(
                f"{rule_name} drops {kg_fp:.3%} of known-good text to catch "
                f"{fires:.2%} of the target (ratio {report.ratio:.1f}:1); budget is "
                f"{budget:.3%}. This is the -دىڭ shape: plausible rule, real cost."
            )
        return report

    @abstractmethod
    def fertility_histogram(self, corpus: Path) -> list[float]:
        """Pieces/word quantiles against the round's vocab, for the bimodality smell.

        Abstract only because it needs the pair's vocab; it is not optional — build()
        calls it and prints it for every source. Two languages in one corpus show up
        as two humps against a vocab trained on one of them.
        """

    def build(self, run: Run) -> Artifact:
        """Fetch -> filter -> gate -> report. The fixed order, wired into pipe.

        Left unimplemented in this sketch: it should call the shared filter step and
        return the artifact `train` consumes, so that a round which skips filtering
        has nothing to hand to train — a missing artifact, not a silent 15% Kazakh.
        """
        raise NotImplementedError("sketch: wire to filter_corpus.py + the flow's steps")
