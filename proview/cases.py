"""Cases: a folder per patient case with its information, notes and a saved snapshot of the
models after every step (so any earlier step can be restored, also after closing ProVixel).

Folder layout::

    <cases root>/2026-11-14_Ahmed Ali/
        case.json          information, notes, list of steps
        steps/0001.json    the scene after step 1 (models, positions, materials)
        meshes/<hash>.ply  model geometry, stored once and shared by all steps that use it

A model that was imported and not changed is not copied: its step entry points to the
original file (the case keeps only its location).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .mesh import Mesh, load_mesh, save_ply

CASE_FILE = "case.json"
WORK_TYPES = ["Study model", "Crown", "Bridge", "Implant", "Veneer", "Inlay / Onlay", "Denture",
              "Night guard", "Aligner", "Other"]


def default_root() -> str:
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    return os.path.join(docs if os.path.isdir(docs) else os.path.expanduser("~"), "ProVixel Cases")


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def safe_name(text: str) -> str:
    """Text usable as a Windows folder name (Arabic and other letters are kept)."""
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    return text[:80] or "Case"


def mesh_hash(mesh: Mesh) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(np.ascontiguousarray(mesh.vertices).tobytes())
    h.update(np.ascontiguousarray(mesh.faces).tobytes())
    if mesh.colors is not None:
        h.update(np.ascontiguousarray(mesh.colors).tobytes())
    return h.hexdigest()


@dataclass
class CaseInfo:
    patient: str
    doctor: str = ""
    date: str = field(default_factory=lambda: _dt.date.today().isoformat())
    work_type: str = ""
    case_no: str = ""
    teeth: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "CaseInfo":
        known = {k: str(d.get(k, "")) for k in cls.__dataclass_fields__}
        known["patient"] = known["patient"] or "Patient"
        known["date"] = known["date"] or _dt.date.today().isoformat()
        return cls(**known)

    def title(self) -> str:
        parts = [self.patient]
        if self.doctor:
            parts.append(self.doctor)
        return " · ".join(parts)


@dataclass
class ModelState:
    """Everything needed to put a model back in the scene."""

    name: str
    hash: str
    mesh_file: str | None  # relative to the case folder, None = the original file is used
    source: str | None  # original file the model was imported from
    matrix: list
    material: str
    color: list
    opacity: float = 1.0
    visible: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ModelState":
        return cls(name=d.get("name", "Model"), hash=d.get("hash", ""), mesh_file=d.get("mesh_file"),
                   source=d.get("source"), matrix=d.get("matrix") or np.eye(4).ravel().tolist(),
                   material=d.get("material", ""), color=d.get("color") or [0.8, 0.8, 0.8],
                   opacity=float(d.get("opacity", 1.0)), visible=bool(d.get("visible", True)))


class CaseError(Exception):
    pass


class Case:
    def __init__(self, path: str, info: CaseInfo, steps=None, created=None, modified=None):
        self.path = path
        self.info = info
        self.steps: list[dict] = steps or []
        self.created = created or _now()
        self.modified = modified or self.created

    # ------------------------------------------------------------------ create / load
    @classmethod
    def create(cls, root: str, info: CaseInfo) -> "Case":
        os.makedirs(root, exist_ok=True)
        base = safe_name(f"{info.date}_{info.patient}")
        path = os.path.join(root, base)
        k = 2
        while os.path.exists(path):
            path = os.path.join(root, f"{base} ({k})")
            k += 1
        os.makedirs(os.path.join(path, "steps"))
        os.makedirs(os.path.join(path, "meshes"))
        case = cls(path, info)
        case.save()
        return case

    @classmethod
    def load(cls, path: str) -> "Case":
        f = os.path.join(path, CASE_FILE)
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError) as exc:
            raise CaseError(f"Could not read the case in {path}:\n{exc}") from exc
        return cls(path, CaseInfo.from_dict(d.get("info", {})), d.get("steps", []), d.get("created"),
                   d.get("modified"))

    def save(self) -> None:
        self.modified = _now()
        d = {"app": "ProVixel", "version": 1, "info": asdict(self.info), "created": self.created,
             "modified": self.modified, "steps": self.steps}
        tmp = os.path.join(self.path, CASE_FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(self.path, CASE_FILE))

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    # ------------------------------------------------------------------ steps
    def store_mesh(self, mesh: Mesh, h: str | None = None) -> tuple[str, str]:
        """Write the mesh into the case (once per content). Returns (hash, relative file)."""
        h = h or mesh_hash(mesh)
        rel = os.path.join("meshes", h + ".ply")
        full = os.path.join(self.path, rel)
        if not os.path.exists(full):
            tmp = full + ".tmp.ply"
            save_ply(tmp, mesh)
            os.replace(tmp, full)
        return h, rel

    def add_step(self, label: str, states: list[ModelState], replace_last: bool = False) -> int:
        """Save the scene as a new step (or over the last one). Returns the step number."""
        if replace_last and self.steps:
            n = self.steps[-1]["n"]
            self.steps.pop()
        else:
            n = (self.steps[-1]["n"] + 1) if self.steps else 1
        rel = os.path.join("steps", f"{n:04d}.json")
        with open(os.path.join(self.path, rel + ".tmp"), "w", encoding="utf-8") as fh:
            json.dump({"label": label, "time": _now(), "models": [asdict(s) for s in states]}, fh,
                      ensure_ascii=False, indent=1)
        os.replace(os.path.join(self.path, rel + ".tmp"), os.path.join(self.path, rel))
        self.steps.append({"n": n, "label": label, "time": _now(), "file": rel.replace("\\", "/"),
                           "models": len(states)})
        self.save()
        return n

    def step_states(self, n: int | None = None) -> list[ModelState]:
        """Model states of step n (default: the last step)."""
        if not self.steps:
            return []
        step = self.steps[-1] if n is None else next((s for s in self.steps if s["n"] == n), None)
        if step is None:
            raise CaseError(f"Step {n} does not exist.")
        with open(os.path.join(self.path, step["file"]), encoding="utf-8") as fh:
            d = json.load(fh)
        return [ModelState.from_dict(m) for m in d.get("models", [])]

    def load_state_mesh(self, s: ModelState) -> Mesh:
        if s.mesh_file:
            full = os.path.join(self.path, s.mesh_file)
            if os.path.exists(full):
                return load_mesh(full)
        if s.source and os.path.exists(s.source):
            return load_mesh(s.source)
        where = s.source or s.mesh_file or "?"
        raise CaseError(f'The file of "{s.name}" was not found:\n{where}')

    def cleanup(self) -> int:
        """Delete mesh files no step uses any more (left by merged steps). Returns the count."""
        used = set()
        for st in self.steps:
            try:
                with open(os.path.join(self.path, st["file"]), encoding="utf-8") as fh:
                    for m in json.load(fh).get("models", []):
                        if m.get("mesh_file"):
                            used.add(os.path.basename(m["mesh_file"]))
            except (OSError, ValueError):
                return 0  # never delete anything when a step can't be read
        removed = 0
        folder = os.path.join(self.path, "meshes")
        for f in os.listdir(folder) if os.path.isdir(folder) else []:
            if f not in used:
                try:
                    os.remove(os.path.join(folder, f))
                    removed += 1
                except OSError:
                    pass
        return removed


def list_cases(root: str) -> list[Case]:
    """All cases in the root folder, newest date first."""
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, CASE_FILE)):
            try:
                out.append(Case.load(path))
            except CaseError:
                continue
    out.sort(key=lambda c: (c.info.date, c.modified), reverse=True)
    return out
