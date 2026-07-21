"""The pair pipeline as a type. A pair supplies parts; it never writes the order.

`pipe.gate` and the eval steps were each built after a failure, documented, and
then absent from the next flow — because a flow is copied, and a copy keeps what
someone remembered to keep. Comments do not survive that; a signature does.

So the graph lives in `run_pair` ONLY. A pair implements `Pair`, which cannot be
instantiated without every part, and `run_pair` is the only thing that decides
what runs when. There is no supported way to express "pair without a teacher
gate" or "KD straight off the raw corpus" — not because it is forbidden, but
because no signature accepts it:

- `Pair.kd()` takes `Filtered`. The only source of a `Filtered` is `pipe.gate`,
  which refuses when `known_good` is absent. So KD cannot see an ungated corpus:
  uig round 1's 15-20% Kazakh becomes a type error rather than a bad run.
- `run_pair` calls `require_check_set` before renting anything, so a pair with no
  check set stops before spending. en-tl and sw-en were trained end to end on
  teachers that pass FLORES and produce unusable output; the check set is the
  requirement that would have stopped both, so it ranks with "needs a teacher"
  and "needs a GPU".
- `PairResult` is frozen and total: a run that did not evaluate the teacher, the
  KD checkpoint AND the packed student cannot be
  constructed, so "we skipped the eval" has no representation.

An implementation could still no-op its own `train()`. That is out of scope —
this makes the honest path the only *convenient* one, not the only possible one.

    class UigPair(Pair):
        lang, src = "ug", "en"
        def teacher_decode(self): return hy_decode_step
        def kd(self, run, filtered): ...
        def pack(self, run, model, vocab): ...

    def main(run, argv):
        return run_pair(run, UigPair()).to_json()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pipe.artstore import Filtered
from pipe.gate import gate
from pipe.step import Run, StepDef
from pipe.target import Bigserver
from pipe.types import Artifact

from . import evalsteps


@dataclass(frozen=True)
class Evaluated:
    """One system's numbers plus the pairs a human or agent reads.

    Both, always. The metrics rank; only the review shows that a teacher renders
    *Banyo* as "Reflection" — chrF called that direction a tie, COMET ranked the
    broken model above its own median, and the mechanical checks scored it 100%
    clean.
    """

    metrics: Artifact
    review: Artifact

    def to_json(self) -> dict:
        return {"metrics": self.metrics.to_json(), "review": self.review.to_json()}


@dataclass(frozen=True)
class PairResult:
    """Total by construction: every stage that must be evaluated has a field."""

    teacher: Evaluated
    kd: Evaluated
    student: Evaluated

    def to_json(self) -> dict:
        return {
            "teacher": self.teacher.to_json(),
            "kd": self.kd.to_json(),
            "student": self.student.to_json(),
        }


class Pair(ABC):
    """The pair-specific parts, and only those.

    Everything here varies by pair for a real reason: the teacher family decodes
    differently (vLLM / transformers / Marian), and uigen_r2 merges several KD
    sources where uigen_v3 has one. Ordering does not vary, so it is not here.
    """

    lang: str
    src: str = "en"
    filter_budget: float = 0.02

    @abstractmethod
    def teacher_decode(self) -> StepDef:
        """Step decoding FLORES + the check set with the teacher.

        Must emit flores_hyp / flores_ref / flores_src / check_hyp.
        """

    @abstractmethod
    def kd(self, run: Run, filtered: Filtered) -> Artifact:
        """Filtered corpus -> aligned train_tsv (split/decode/gather/cefilter/align).

        Takes `Filtered`, not `Artifact`: the corpus gate is upstream of KD in the
        type system, so it cannot be skipped by editing a call site.
        """

    @abstractmethod
    def train(self) -> StepDef:
        """One box: train -> decode(FLORES) -> decode(check set).

        One step because pipe leases per step key, so splitting it rents a box per
        phase and ships a checkpoint between them; the decodes are seconds on a
        GPU already in hand.

        Finetune is deliberately NOT here. It bought +2.87 chrF on tl and +1.3 on
        sw, always on FLORES, and its effect on deployment-shaped input has never
        been measured for any pair — so it is an experiment to run against this
        baseline (train a ft variant, eval it the same way, compare check deltas),
        not a required stage. Adding it back means a second Evaluated on
        PairResult, which is the point: a stage that is not measured does not
        belong in the required shape.

        Must emit: model, flores_hyp, check_hyp.
        """

    @abstractmethod
    def pack(self, run: Run, model: Artifact, vocab: Artifact) -> dict[str, Artifact]:
        """Quantize + shortlist -> {"model", "vocab", "shortlist"}."""


def _score(run: Run, flores_hyp: Artifact, check_hyp: Artifact,
           flores_ref: Artifact, flores_src: Artifact,
           check_src: Artifact, check_ref: Artifact) -> Evaluated:
    """Scoring is shared and always on Bigserver: identical code per pair is what
    makes numbers comparable across pairs and rounds. Decoding stays on the
    rented box, which is model-specific and already paid for."""
    out = run.do(
        evalsteps.eval_score, timeout=3600,
        flores_hyp=flores_hyp, flores_ref=flores_ref, flores_src=flores_src,
        check_hyp=check_hyp, check_src=check_src, check_ref=check_ref,
    )
    return Evaluated(metrics=out["metrics"], review=out["review"])


def run_pair(run: Run, pair: Pair) -> PairResult:
    """The graph. The only place it exists."""
    a = run.ledger.artifact
    check_src, check_ref = evalsteps.require_check_set(run)
    flores_ref, flores_src = a("flores_ref"), a("flores_src")

    filtered = gate(
        run,
        raw=a("raw"),
        vocab=a("vocab"),
        gold=a("gold"),
        known_good=a("known_good"),
        budget=pair.filter_budget,
        image="prep:next",
        target=Bigserver(cpus=16),
        label=f"{pair.src}{pair.lang}",
    )

    # Before the KD rental, not after: a student cannot exceed its teacher, so
    # every GPU-hour past a bad gate distils a known-bad model.
    decoded = run.do(
        pair.teacher_decode(), timeout=2 * 3600,
        filtered=filtered, check_src=check_src, check_ref=check_ref,
    )
    teacher = _score(run, decoded["flores_hyp"], decoded["check_hyp"],
                     flores_ref, flores_src, check_src, check_ref)

    train_tsv = pair.kd(run, filtered)

    trained = run.do(
        pair.train(), timeout=12 * 3600,
        train_tsv=train_tsv, vocab=a("vocab"), valid=a("valid"),
        flores_src=flores_src, check_src=check_src,
    )
    kd_eval = _score(run, trained["flores_hyp"], trained["check_hyp"],
                     flores_ref, flores_src, check_src, check_ref)

    packed = pair.pack(run, trained["model"], a("vocab"))
    student_out = evalsteps.student_eval(
        run, pair.lang, pair.src,
        packed["model"], packed["vocab"], packed["shortlist"],
    )
    student = Evaluated(metrics=student_out["metrics"], review=student_out["review"])

    return PairResult(teacher=teacher, kd=kd_eval, student=student)
