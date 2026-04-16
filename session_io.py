"""
session_io.py
─────────────
Save / load a lattice session to a single .pkl file.

The tricky part: slicesY / slicesX are lists of instances of
`SlicedLenghts`, a class defined *locally* inside build_lattice().
Local classes can't be pickled by reference, so we flatten every
such object to a plain dict before saving.

EXCEPTION: pyvista DataSet objects (PolyData, UnstructuredGrid, etc.)
ARE natively picklable and must NOT be flattened — they are stored as-is.

On load, flattened dicts are restored as types.SimpleNamespace so that
attribute access (obj.foo) works identically to the original class.
"""

import os
import glob
import pickle
import types

SESSION_DIR = "sessions"


def _is_pyvista(obj) -> bool:
    """True for any object whose type lives in the pyvista package tree."""
    return type(obj).__module__.split(".")[0] == "pyvista"


# ── serialisation helpers ─────────────────────────────────────────────────────

def _to_serialisable(obj):
    """
    Recursively convert local-class instances to plain dicts.
    Any pyvista object is left untouched — pickle handles the entire
    pyvista class hierarchy natively.
    numpy arrays, ints, strings, etc. pass through unchanged.
    """
    if _is_pyvista(obj):                   # any pyvista type — keep as-is
        return obj
    if isinstance(obj, list):
        return [_to_serialisable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    # Custom class instance (has __dict__ but isn't a bare type)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        flat = {"__class_name__": type(obj).__name__}
        flat.update({k: _to_serialisable(v) for k, v in vars(obj).items()})
        return flat
    return obj


def _from_serialisable(obj):
    """
    Inverse of _to_serialisable.
    Dicts that carry __class_name__ are rebuilt as SimpleNamespace.
    pyvista objects come back from pickle already intact.
    """
    if _is_pyvista(obj):                   # already a live pyvista object
        return obj
    if isinstance(obj, list):
        return [_from_serialisable(item) for item in obj]
    if isinstance(obj, dict):
        restored = {k: _from_serialisable(v) for k, v in obj.items()}
        if "__class_name__" in restored:
            restored.pop("__class_name__")
            return types.SimpleNamespace(**restored)
        return restored
    return obj


# ── path helpers ──────────────────────────────────────────────────────────────

def _ensure_dir():
    os.makedirs(SESSION_DIR, exist_ok=True)


def _resolve_path(name: str) -> str:
    if os.sep in name or "/" in name:
        return name if name.endswith(".pkl") else name + ".pkl"
    _ensure_dir()
    stem = name[:-4] if name.endswith(".pkl") else name
    return os.path.join(SESSION_DIR, stem + ".pkl")


def list_sessions() -> list:
    _ensure_dir()
    return sorted(glob.glob(os.path.join(SESSION_DIR, "*.pkl")))

# ── public API ────────────────────────────────────────────────────────────────

def save_session(
    slicesY, slicesX, lattice_nodes,
    n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount,
    name: str,
) -> str:
    path = _resolve_path(name)

    payload = {
        "slicesY":        _to_serialisable(slicesY),
        "slicesX":        _to_serialisable(slicesX),
        "lattice_nodes":  _to_serialisable(lattice_nodes),
        "n_origin_shift": int(n_origin_shift),
        "n_root":         int(n_root),
        "n_lead":         int(n_lead),
        "n_tip":          int(n_tip),
        "boundary_dir":   int(boundary_dir),
        "VCcount":        int(VCcount),
    }

    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[session_io] Saved → {path}")
    return path


def load_session(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = pickle.load(fh)

    payload = {
        "slicesY":        _from_serialisable(raw["slicesY"]),
        "slicesX":        _from_serialisable(raw["slicesX"]),
        "lattice_nodes":  _from_serialisable(raw["lattice_nodes"]),
        "n_origin_shift": raw["n_origin_shift"],
        "n_root":         raw["n_root"],
        "n_lead":         raw["n_lead"],
        "n_tip":          raw["n_tip"],
        "boundary_dir":   raw["boundary_dir"],
        "VCcount":        raw["VCcount"],
    }

    print(f"[session_io] Loaded ← {path}")
    _print_summary(payload)
    return payload

def load_boundary_params(path: str):
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    return (
        raw["n_origin_shift"],
        raw["n_root"],
        raw["n_lead"],
        raw["n_tip"],
        raw["boundary_dir"],
        raw["VCcount"],
    )

def _print_summary(payload: dict):
    print(
        f"  boundary → shift={payload['n_origin_shift']}, "
        f"root={payload['n_root']}, lead={payload['n_lead']}, "
        f"tip={payload['n_tip']}, dir={payload['boundary_dir']}, "
        f"VCcount={payload['VCcount']}"
    )
    for key in ("slicesY", "slicesX", "lattice_nodes"):
        val = payload.get(key)
        if val is None:
            print(f"  {key:14s} → None")
        elif hasattr(val, "__len__"):
            print(f"  {key:14s} → {type(val).__name__}  len={len(val)}")
        else:
            print(f"  {key:14s} → {val}")
