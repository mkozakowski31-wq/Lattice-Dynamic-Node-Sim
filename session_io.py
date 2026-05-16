import os
import glob
import pickle
import re
import types
import numpy as np

SESSION_DIR = "sessions"

def _natural_key(name: str):
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def isolate_differing_characters(folder_path, isfile):
    # 1. Select items and sort them alphabetically
    try:
        if isfile:
            # Filter for files
            items = [f for f in os.listdir(folder_path) 
                     if os.path.isfile(os.path.join(folder_path, f))]
            label = "File"
        else:
            # Filter for directories
            items = [f for f in os.listdir(folder_path) 
                     if os.path.isdir(os.path.join(folder_path, f))]
            label = "Folder"
            
        # Natural sort: alphabetical on text segments, numeric on digit segments
        items.sort(key=_natural_key)
        
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found.")
        return []

    print(f"--- {label} Paths (Alphabetical) ---")
    
    item_paths = []
    char_sets = []
    all_chars_union = set()

    for name in items:
        path = os.path.abspath(os.path.join(folder_path, name))
        item_paths.append(path)
        print(path)

        # 2. Identify characters to compare
        if isfile:
            # For files, we analyze the text INSIDE the file
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"Could not read file {name}: {e}")
                continue
        else:
            # For folders, we analyze the characters in the FOLDER NAME
            content = name
            
        char_set = set(content)
        char_sets.append(char_set)
        all_chars_union.update(char_set)

    if not char_sets:
        print(f"No valid {label.lower()}s found to compare.")
        return item_paths

    # 3. Find characters common to ALL items (Intersection)
    common_to_all = set.intersection(*char_sets)

    # 4. Isolate characters NOT common to all (Union - Intersection)
    differing_chars = all_chars_union - common_to_all
    sorted_diff = sorted(list(differing_chars))
    
    # 5. Output the results
    print(f"\n--- Differing Characters in {label} {'Content' if isfile else 'Names'} ---")
    if sorted_diff:
        print(" ".join(sorted_diff))
    else:
        print(f"All {label.lower()}s contain the exact same set of characters.")

    return item_paths

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
    root_pts, tip_pts, lead_pts, trail_pts, junction_points,
    VWcount, size,
    excluded_indices, stage_overrides,
    name: str,
) -> str:
    path = _resolve_path(name)

    payload = {
        # ── lattice ────────────────────────────────────────────────────
        "slicesY":          _to_serialisable(slicesY),
        "slicesX":          _to_serialisable(slicesX),
        "lattice_nodes":    _to_serialisable(lattice_nodes),
        # ── global boundary params ─────────────────────────────────────
        "n_origin_shift":   int(n_origin_shift),
        "n_root":           int(n_root),
        "n_lead":           int(n_lead),
        "n_tip":            int(n_tip),
        "boundary_dir":     int(boundary_dir),
        "VCcount":          int(VCcount),
        # ── resampled boundary geometry ────────────────────────────────
        "root_pts":         _to_serialisable(root_pts),
        "tip_pts":          _to_serialisable(tip_pts),
        "lead_pts":         _to_serialisable(lead_pts),
        "trail_pts":        _to_serialisable(trail_pts),
        "junction_points":  _to_serialisable(junction_points),
        "VWcount":          int(VWcount),
        "size":             float(size),
        # ── filter / stage state ───────────────────────────────────────
        "excluded_indices": sorted(int(i) for i in excluded_indices),
        "stage_overrides":  {int(k): dict(v) for k, v in stage_overrides.items()},
    }

    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[session_io] Saved → {path}")
    return path


def load_session(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = pickle.load(fh)

    payload = {
        # ── lattice ────────────────────────────────────────────────────
        "slicesY":          _from_serialisable(raw["slicesY"]),
        "slicesX":          _from_serialisable(raw["slicesX"]),
        "lattice_nodes":    _from_serialisable(raw["lattice_nodes"]),
        # ── global boundary params ─────────────────────────────────────
        "n_origin_shift":   raw["n_origin_shift"],
        "n_root":           raw["n_root"],
        "n_lead":           raw["n_lead"],
        "n_tip":            raw["n_tip"],
        "boundary_dir":     raw["boundary_dir"],
        "VCcount":          raw["VCcount"],
        # ── resampled boundary geometry (None for old sessions) ────────
        "root_pts":        (np.asarray(raw["root_pts"])
                            if "root_pts" in raw else None),
        "tip_pts":         (np.asarray(raw["tip_pts"])
                            if "tip_pts" in raw else None),
        "lead_pts":        (np.asarray(raw["lead_pts"])
                            if "lead_pts" in raw else None),
        "trail_pts":       (np.asarray(raw["trail_pts"])
                            if "trail_pts" in raw else None),
        "junction_points": (np.asarray(raw["junction_points"])
                            if "junction_points" in raw else None),
        "VWcount":          raw.get("VWcount"),
        "size":             raw.get("size"),
        # ── filter / stage state ───────────────────────────────────────
        "excluded_indices": set(raw.get("excluded_indices", [])),
        "stage_overrides":  {int(k): dict(v)
                             for k, v in raw.get("stage_overrides", {}).items()},
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
        f"VCcount={payload['VCcount']},  VWcount={payload.get('VWcount')},  "
        f"size={payload.get('size')}"
    )
    for key in ("slicesY", "slicesX", "lattice_nodes"):
        val = payload.get(key)
        if val is None:
            print(f"  {key:14s} → None")
        elif hasattr(val, "__len__"):
            print(f"  {key:14s} → {type(val).__name__}  len={len(val)}")
        else:
            print(f"  {key:14s} → {val}")
    # boundary geometry
    for key in ("root_pts", "tip_pts", "lead_pts", "trail_pts"):
        arr = payload.get(key)
        if arr is not None:
            print(f"  {key:14s} → ndarray  shape={arr.shape}")
        else:
            print(f"  {key:14s} → None  (old session — Resampler will rerun)")
    # filter state
    excl = payload.get("excluded_indices", set())
    ovrd = payload.get("stage_overrides",  {})
    print(f"  excluded       → {sorted(excl) if excl else 'none'}")
    print(f"  stage_overrides→ {len(ovrd)} stage(s) with custom boundaries")

