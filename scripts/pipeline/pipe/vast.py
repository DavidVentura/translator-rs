from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid as uuidlib
from dataclasses import dataclass
from pathlib import Path

VAST_BIN = Path(sys.executable).parent / "vastai"


class VastError(RuntimeError):
    pass


@dataclass(frozen=True)
class Offer:
    id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    dph_total: float
    reliability: float
    inet_down: float
    disk_space: float
    geolocation: str

    @staticmethod
    def from_json(d: dict) -> Offer:
        return Offer(
            id=d["id"],
            machine_id=d.get("machine_id") or 0,
            gpu_name=d["gpu_name"],
            num_gpus=d["num_gpus"],
            dph_total=d["dph_total"],
            reliability=d["reliability2"],
            inet_down=d.get("inet_down") or 0.0,
            disk_space=d["disk_space"],
            geolocation=d.get("geolocation") or "?",
        )


@dataclass(frozen=True)
class Instance:
    id: int
    status: str
    ssh_host: str | None
    ssh_port: int | None
    label: str | None
    gpu_name: str

    @staticmethod
    def from_json(d: dict) -> Instance:
        # cur_state is the DESIRED state ("running" from the moment of creation);
        # actual_status is what the box's agent reports. Falling back to cur_state
        # made never-started boxes look running.
        return Instance(
            id=d["id"],
            status=d.get("actual_status") or "created",
            ssh_host=d.get("ssh_host"),
            ssh_port=d.get("ssh_port"),
            label=d.get("label"),
            gpu_name=d.get("gpu_name") or "?",
        )

    @property
    def reachable(self) -> bool:
        return self.status == "running" and bool(self.ssh_host) and bool(self.ssh_port)


@dataclass(frozen=True)
class Lease:
    uuid: str
    instance_id: int
    run: str
    step: str
    created: float
    expires: float

    @property
    def label(self) -> str:
        return f"pipe:{self.uuid}"

    def to_json(self) -> dict:
        return {
            "uuid": self.uuid,
            "instance_id": self.instance_id,
            "run": self.run,
            "step": self.step,
            "created": self.created,
            "expires": self.expires,
        }

    @staticmethod
    def from_json(d: dict) -> Lease:
        return Lease(**d)


class VastApi:
    def __init__(self, leases_dir: Path) -> None:
        self.leases_dir = leases_dir

    def _cli(self, args: list[str]) -> object:
        done = subprocess.run(
            [str(VAST_BIN), *args, "--raw"], capture_output=True, text=True
        )
        if done.returncode != 0:
            raise VastError(f"vastai {' '.join(args)} failed: {done.stderr.strip()}")
        try:
            return json.loads(done.stdout)
        except json.JSONDecodeError as e:
            raise VastError(f"vastai {' '.join(args)} gave non-JSON: {done.stdout[:400]}") from e

    def search(
        self,
        gpu_name: str,
        min_reliability: float = 0.98,
        min_inet_down: float = 300.0,
        disk_gb: int = 100,
        num_gpus: int = 1,
        max_dph: float = 1.0,
        geo_in: tuple[str, ...] = (),
        min_cuda: float = 0.0,
    ) -> list[Offer]:
        # inet_down is filtered at search time: a slow box otherwise only reveals
        # itself by taking tens of minutes to pull the image.
        query = (
            f"reliability > {min_reliability} "
            f"gpu_name={gpu_name} "
            f"num_gpus={num_gpus} "
            f"inet_down > {min_inet_down} "
            f"disk_space > {disk_gb} "
            f"dph_total < {max_dph} "
            f"rentable=true verified=true"
        )
        if geo_in:
            query += f" geolocation in [{','.join(geo_in)}]"
        if min_cuda:
            # An image's CUDA runtime needs a driver at least that new; a too-old
            # driver fails at first CUDA call with "forward compatibility was
            # attempted on non supported HW" — after renting and pulling.
            query += f" cuda_max_good >= {min_cuda}"
        raw = self._cli(["search", "offers", query, "-o", "dph_total"])
        assert isinstance(raw, list)
        offers = [Offer.from_json(o) for o in raw]
        # One offer per physical machine: cheap hosts list several slots, and
        # concurrent renters otherwise pile onto the same (possibly broken) box.
        seen: set[int] = set()
        unique = []
        for o in offers:
            if o.machine_id in seen:
                continue
            seen.add(o.machine_id)
            unique.append(o)
        return unique

    def create(self, offer: Offer, image: str, disk_gb: int, label: str) -> int:
        # --cancel-unavail: without it a failed schedule leaves a stopped instance
        # sitting there looking like a live box.
        raw = self._cli(
            [
                "create", "instance", str(offer.id),
                "--image", image,
                "--disk", str(disk_gb),
                "--label", label,
                "--ssh",
                "--cancel-unavail",
                # vast's default onstart wraps ssh sessions in tmux, which breaks
                # non-interactive exec ("open terminal failed: not a terminal").
                "--onstart-cmd", "touch ~/.no_auto_tmux",
            ]
        )
        assert isinstance(raw, dict)
        if not raw.get("success"):
            raise VastError(f"create failed on offer {offer.id}: {raw}")
        return int(raw["new_contract"])

    def instance(self, instance_id: int) -> Instance:
        raw = self._cli(["show", "instance", str(instance_id)])
        if isinstance(raw, list):
            if not raw:
                raise VastError(f"instance {instance_id} not found")
            raw = raw[0]
        assert isinstance(raw, dict)
        return Instance.from_json(raw)

    def instances(self) -> list[Instance]:
        raw = self._cli(["show", "instances"])
        assert isinstance(raw, list)
        return [Instance.from_json(i) for i in raw]

    def destroy(self, instance_id: int) -> None:
        # Not via _cli: `vastai destroy` prompts y/N on stdin and answers in prose,
        # not JSON — the 2026-07-15 box leak was every cleanup destroy "failing"
        # here while the instance kept billing.
        done = subprocess.run(
            [str(VAST_BIN), "destroy", "instance", str(instance_id)],
            capture_output=True, text=True, input="y\n",
        )
        if done.returncode != 0 or "destroying instance" not in done.stdout:
            raise VastError(
                f"destroy {instance_id} failed: {done.stdout.strip()} {done.stderr.strip()}"
            )

    def new_lease(self, instance_id: int, run: str, step: str, max_hours: float) -> Lease:
        now = time.time()
        lease = Lease(
            uuid=uuidlib.uuid4().hex[:12],
            instance_id=instance_id,
            run=run,
            step=step,
            created=now,
            expires=now + max_hours * 3600,
        )
        self.write_lease(lease)
        return lease

    def write_lease(self, lease: Lease) -> None:
        self.leases_dir.mkdir(parents=True, exist_ok=True)
        (self.leases_dir / f"{lease.uuid}.json").write_text(json.dumps(lease.to_json(), indent=2))

    def leases(self) -> dict[str, Lease]:
        if not self.leases_dir.is_dir():
            return {}
        out = {}
        for f in self.leases_dir.glob("*.json"):
            lease = Lease.from_json(json.loads(f.read_text()))
            out[lease.uuid] = lease
        return out

    def drop_lease(self, uuid: str) -> None:
        (self.leases_dir / f"{uuid}.json").unlink(missing_ok=True)

    # Seconds a box may sit in one status without progressing to the next.
    # A healthy rental moves created -> loading (scheduling + agent start) ->
    # running (image pulled); a box that stalls in ANY stage is a dud, and
    # waiting the full overall budget on it just delays the retry on a fresh
    # offer. Budgets, not one flat window.
    STAGE_BUDGETS = {"created": 240.0, "loading": 600.0, "running": 120.0}

    def wait_ready(self, instance_id: int, timeout: float, poll: float = 15.0) -> Instance:
        deadline = time.monotonic() + timeout
        last_status: str | None = None
        since = time.monotonic()
        while True:
            inst = self.instance(instance_id)
            if inst.reachable:
                return inst
            now = time.monotonic()
            if inst.status != last_status:
                last_status, since = inst.status, now
            budget = self.STAGE_BUDGETS.get(inst.status, 300.0)
            if now - since > budget:
                self.destroy(instance_id)
                raise TimeoutError(
                    f"instance {instance_id} stalled in {inst.status!r} for "
                    f"{int(now - since)}s (stage budget {int(budget)}s); destroyed"
                )
            if now > deadline:
                self.destroy(instance_id)
                raise TimeoutError(
                    f"instance {instance_id} still {inst.status} after {timeout}s; destroyed"
                )
            time.sleep(poll)
