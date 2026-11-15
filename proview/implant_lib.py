"""Implant libraries on the user's computer (exocad-style folders).

Each library folder has a ``config.xml`` (ImplantLibraryEntry) that names the scan body
(MarkerFilename), the implant analog (ImplantFilename) and the ti-bases / abutments
(SupportFilename, InterfaceFilename) of each implant type, plus the implant axis
(AxisOcclusal). All geometry files of one entry share one coordinate system. ``*.partInfo``
files next to the geometry carry manufacturer and part names.

Only readable STL geometry is used; nothing is copied out of the user's library.
"""
from __future__ import annotations

import hashlib
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh, load_stl


@dataclass
class Subtype:
    name: str
    support: str | None
    interface: str | None
    height: float | None  # parsed from names like "H5.5"


@dataclass
class ImplantType:
    name: str
    analog: str | None  # full path
    subtypes: list[Subtype] = field(default_factory=list)


@dataclass
class LibEntry:
    folder: str
    library: str  # folder name
    name: str  # display information
    marker: str  # scan body STL (full path)
    part_name: str = ""
    manufacturer: str = ""
    part_number: str = ""
    axis: tuple = (0.0, 0.0, 1.0)  # implant axis, occlusal direction, in the library frame
    asc_max: float | None = None  # largest angled screw channel angle
    types: list[ImplantType] = field(default_factory=list)

    @property
    def title(self) -> str:
        scan = self.part_name or os.path.splitext(os.path.basename(self.marker))[0]
        return f"{scan}  —  {self.name or self.library}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _text(el, name, default=""):
    c = _child(el, name)
    return (c.text or "").strip() if c is not None and c.text else default


def _vec(el, name):
    c = _child(el, name)
    if c is None:
        return None
    try:
        return tuple(float(_text(c, k)) for k in ("x", "y", "z"))
    except ValueError:
        return None


def _height(name: str) -> float | None:
    m = re.search(r"(?:^|[_\-\s])H\s?(\d+(?:[.,]\d+)?)", name or "", re.IGNORECASE)
    return float(m.group(1).replace(",", ".")) if m else None


def _find_file(folder: str, name: str | None) -> str | None:
    if not name:
        return None
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        return path
    low = name.lower()  # libraries made on Windows: names may differ in case
    for f in os.listdir(folder):
        if f.lower() == low:
            return os.path.join(folder, f)
    return None


def read_part_info(path: str) -> dict:
    """Manufacturer / part name / number from ``<file>.partInfo`` (any letter case)."""
    folder, base = os.path.split(path)
    want = (base + ".partinfo").lower()
    for f in os.listdir(folder):
        if f.lower() == want:
            try:
                root = ET.parse(os.path.join(folder, f)).getroot()
            except (ET.ParseError, OSError):
                return {}
            out = {}
            for el in root.iter():
                tag = _local(el.tag)
                if tag in ("Manufacturer", "PartName", "PartNumber") and el.text:
                    out[tag] = el.text.strip()
            return out
    return {}


def read_config(folder: str) -> list[LibEntry]:
    """Library entries described by ``folder/config.xml`` (empty when there is none)."""
    cfg = _find_file(folder, "config.xml")
    if cfg is None:
        return []
    try:
        root = ET.parse(cfg).getroot()
    except (ET.ParseError, OSError):
        return []
    entries = []
    elements = [root] if _child(root, "MarkerFilename") is not None else []
    elements += [el for el in root.iter() if el is not root and _child(el, "MarkerFilename") is not None]
    for el in elements:
        marker = _find_file(folder, _text(el, "MarkerFilename"))
        if marker is None:
            continue
        info = read_part_info(marker)
        axis = _vec(el, "AxisOcclusal") or (0.0, 0.0, 1.0)
        asc = _text(el, "AxisScrewChannelMaxUserAngle")
        types = []
        tc = _child(el, "TypeConfig")
        for it in (tc if tc is not None else []):
            if _local(it.tag) != "ImplantTypeConfig":
                continue
            subs = []
            sc = _child(it, "SubtypeConfig")
            for st in (sc if sc is not None else []):
                if _local(st.tag) != "ImplantSubtypeConfig":
                    continue
                name = _text(st, "DisplayInformation") or _text(st, "Keyword")
                subs.append(Subtype(name, _find_file(folder, _text(st, "SupportFilename")),
                                    _find_file(folder, _text(st, "InterfaceFilename")), _height(name)))
            types.append(ImplantType(_text(it, "DisplayInformation") or _text(it, "Keyword"),
                                     _find_file(folder, _text(it, "ImplantFilename")), subs))
        try:
            asc_max = float(asc) if asc else None
        except ValueError:
            asc_max = None
        entries.append(LibEntry(folder=folder, library=os.path.basename(folder),
                                name=_text(el, "DisplayInformation"), marker=marker,
                                part_name=info.get("PartName", ""), manufacturer=info.get("Manufacturer", ""),
                                part_number=info.get("PartNumber", ""), axis=axis, asc_max=asc_max, types=types))
    return entries


def load_geometry(path: str) -> Mesh | None:
    """A readable STL of plausible size (mm), or None (encrypted / other formats)."""
    try:
        m = load_stl(path)
    except (ValueError, OSError, struct.error):
        return None
    if m.n_faces < 20 or not np.isfinite(m.vertices).all():
        return None
    size = float(np.linalg.norm(m.vertices.max(0) - m.vertices.min(0)))
    if not (1.0 < size < 80.0):
        return None
    return m


def file_hash(path: str) -> str:
    h = hashlib.blake2b(digest_size=10)
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def scan_library(root: str, progress=lambda text: None) -> tuple[list[LibEntry], list[dict]]:
    """All entries of all library folders under ``root`` and a report row per folder."""
    entries, report = [], []
    if not os.path.isdir(root):
        return entries, report
    folders = [root] + sorted(os.path.join(root, d) for d in os.listdir(root)
                              if os.path.isdir(os.path.join(root, d)))
    for k, folder in enumerate(folders):
        progress(f"Reading {os.path.basename(folder)} ({k + 1}/{len(folders)})")
        try:
            es = read_config(folder)
        except OSError:
            es = []
        if not es and folder == root:
            continue
        readable = 0
        for e in es:
            if load_geometry(e.marker) is not None:
                readable += 1
                entries.append(e)
        report.append({"library": os.path.basename(folder), "entries": len(es), "readable": readable,
                       "config": _find_file(folder, "config.xml") is not None,
                       "types": sum(len(e.types) for e in es)})
    return entries, report
