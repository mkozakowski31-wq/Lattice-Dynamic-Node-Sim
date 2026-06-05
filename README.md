# Rhombic Lattice Simulation for Conformal Skin Tracking

**A geodesic lattice simulation that tracks how a rhombic lattice deforms across the concave lower surface of a chordwise morphing wing.**

![status](https://img.shields.io/badge/status-research%20prototype-orange)
![python](https://img.shields.io/badge/python-3.13%2B-blue)
![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

> Companion code for the paper *Rhombic Lattice Simulation for Conformal Skin Tracking on a Chordwise Morphing Wing* (M. Kozakowski, 2026). See [Citing this work](https://github.com/user-attachments/files/28650327/rLatticeSimForDynamicNodes.pdf).

<img width="1920" height="1109" alt="ezgif com-crop" src="https://github.com/user-attachments/assets/c8552f6f-dbc9-4c23-bd22-5833b3a545fd" />
<!-- TODO: drop a render or GIF of the lattice deforming across stages here -->

## Overview

A simplified, telescopically extending (chordwise morphing) wing based on the highly cambered **CH-10 airfoil** has a concave lower surface. An elastic skin stretched over that region under spanwise tension will *bowstring* (bridging the concavity instead of following it), which ruins the aerodynamic profile and concentrates stress at the anchor points.

The proposed fix is a **2-D rhombic lattice** bonded between the wing structure and the skin. Its tip-edge corner nodes are fixed anchors; the rest are free. As the wing extends chordwise, the Poisson contraction of the rhombic geometry draws those free nodes spanwise, so the skin conforms to the surface at every point rather than bridging it.

This repository simulates the central question that design raises: **how do the free lattice nodes move across the curved surface over the full extension cycle?** Given a sequence of surface meshes (contracted → extended), the pipeline builds a surface-conforming reference lattice on the contracted state and propagates it onto every later state, recovering the global path each node traces.

Two pieces of the method were developed for this project:

1. **Geodesic lattice construction**: exact polyhedral geodesics (FlipOut) between arc-length-resampled boundary points, with nodes located at the geodesics' closest approach in 3-D.
2. **Marching trilateration**: a row-by-row propagation that re-expresses the lattice on each deformed mesh using only the reference strut lengths and the deformed surface geometry; no deformation field and no global solve.

## Features

- Surface-conforming reference lattice from a single combined surface mesh.
- Exact geodesic ribs via the edge-flip method (`potpourri3d`), avoiding the faceting of edge-constrained shortest paths.
- Closest-approach node extraction refined to sub-sample precision.
- Marching trilateration across an arbitrary number of deformed mesh stages, parallelised with `joblib`.
- Interactive CLI for boundary definition, per-stage overrides, stage exclusion, and lattice sizing.
- Session save/load — skip recomputation and resume directly at the deformation step.
- 3-D visualisation throughout (PyVista).
- Optional, experimental relative-path tracking on the dynamic surface region.

## How it works

The pipeline runs in five stages, coordinated by `Main.py`.

| Stage | File | Role |
|-------|------|------|
| 1 | `stage_1_boundary.py` | Load meshes, extract the boundary loop, split it into Root/Lead/Tip/Trail edges, arc-length resample, run the CLI. |
| 2 | `stage_2_geodesics.py` | Weld resampled boundary points into the mesh, compute the X and Y geodesic ribs (FlipOut). |
| 3 | `stage_3_lattice.py` | Locate lattice nodes at geodesic closest-approach, build straight struts, index slices via a closed-form formula. |
| 4 | `stage_4_deformation.py` | Marching trilateration: propagate the lattice across each deformed mesh from sphere–sphere circle / surface intersections. |
| 5 | `stage_5_relative_plotting.py` | *(Experimental)* reconstruct contact points on the dynamic region via barycentric coordinates. |

Supporting module: `session_io.py` handles pickle-based session save/load and natural-sort stage ordering.

### Reference lattice (stages 1–3)

The mesh perimeter is found as the set of edges belonging to a single triangle, walked into an ordered loop from an origin vertex, and partitioned into four edges by vertex count. Opposing edges are resampled to matching counts by cumulative arc length so the geodesic ribs pair one-to-one. Ribs are computed as geodesics on a boundary-stitched mesh; nodes are the closest-approach points of crossing ribs, and the straight segments between consecutive nodes are the physical struts.

### Deformation tracking (stage 4)

Each deformed stage is seeded from its resampled tip-edge nodes and marched spanwise toward the root, one row at a time. Each new node is placed at the two reference strut lengths from two known anchors: the locus is the intersection circle of two spheres, and the node is taken where that circle meets the deformed surface. Rows alternate between an interior mode and an edge mode (endpoints constrained to the Lead/Trail boundary). Because each node depends only on the previous row, rows are evaluated in parallel.

## Repository structure

```
.
├── Main.py                       # Entry point — runs the full pipeline
├── stage_1_boundary.py           # Boundary extraction, resampling, CLI
├── stage_2_geodesics.py          # Boundary stitching + FlipOut geodesic ribs
├── stage_3_lattice.py            # Node extraction, struts, slice indexing
├── stage_4_deformation.py        # Marching trilateration
├── stage_5_relative_plotting.py  # Experimental relative-path reconstruction
├── session_io.py                 # Session save/load, natural sort
└── sessions/                     # Saved sessions (.pkl), created on first save
```

## Requirements

- **Python 3.13** (pinned via `.python-version`). The project is not run on 3.14 yet: one dependency, `potpourri3d`, does not currently ship a prebuilt wheel for it.
- **[uv](https://docs.astral.sh/uv/)** for dependency management.
- A **3-D display**, the program opens an interactive PyVista viewer and a file dialog, so it needs a graphical session rather than a headless one.

### Install and run

```bash
uv python pin 3.13     # if not already pinned
uv sync                # fetches Python 3.13, resolves dependencies, builds the .venv
uv run python Main.py
```

`uv sync` installs everything from the committed `uv.lock`, so the environment is reproducible across machines, and `uv run` executes inside that environment without manual activation.

### Notes

- `PyQt5` is the Qt backend the `pyvistaqt` viewer needs; `pyvista` does not pull it in automatically.

## Input data

The program operates on the **concave lower-surface mesh** of the wing, exported as `.obj` for each stage of the extension cycle. A minimum of three stages is required (contracted, at least one intermediate, and fully extended) and more intermediate stages improve the deformation mapping. Stages are processed in natural order (`Stage_2` before `Stage_10`).

Two input layouts are selected with `master_folder_setting`.

**`master_folder_setting = False`** *(recommended)* — one combined mesh per stage:

```
MasterFolder/
├── Stage_1.obj
├── Stage_2.obj
└── Stage_3.obj
```

**`master_folder_setting = True`** — combined + static + dynamic mesh per stage, which also enables the experimental relative-path step:

```
MasterFolder/
├── Stage_1/
│   ├── Mesh_A.obj   # combined surface
│   ├── Mesh_B.obj   # static region
│   └── Mesh_C.obj   # dynamic region
├── Stage_2/
│   └── ...
└── Stage_3/
    └── ...
```

> **Origin detection.** With `based_on_extrema = True` the boundary origin is found from geometric extrema, so any clean mesh works. With `based_on_extrema = False` it is read from a marker vertex in an OBJ group named `origin`; use `True` for externally prepared meshes, and use `False` for tutorial folders. 

See Part II of the paper for the full CAD-to-mesh export workflow.

## Usage

```bash
python Main.py
```

1. A folder selection dialog opens, allowing you to choose your `MasterFolder`.
2. The 3-D viewer and the boundary CLI launch. Adjust the boundary parameters until the four edges are colored correctly (root = red, lead = blue, tip = green, trail = orange).
3. Type `continue` to build the reference lattice, then optionally save the session.
4. The lattice is propagated across all stages and the global node paths are drawn.

### CLI commands

| Command | Action |
|---------|--------|
| `shift root lead tip dir` | Set the global boundary (five integers, e.g. `-55 87 337 87 1`). |
| `latticesize` | Set the target cell size. |
| `load` | Restore a saved session; typing `continue` straight after skips the lattice rebuild. |
| `filter` | Step through every stage mesh to validate/override its boundary or exclude it (`Enter` = next, `b` = edit this stage, `r` = reset, `m` = mark/exclude, `q` = quit). |
| `continue` | Commit the configuration and run the lattice plotter. |
| `help` | Show the color legend and command reference. |

## Configuration

Parameters are set at the top of `Main.py`; most can also be adjusted live in the CLI.

| Parameter | Type | Description |
|-----------|------|-------------|
| `boundary_dir` | int | Boundary traversal direction. `1` = right wing (Root→Lead→Tip→Trail). `-1` (left wing) is **not currently supported**. |
| `n_origin_shift` | int | Boundary vertices between the detected origin and the root–trail corner. |
| `n_root`, `n_lead`, `n_tip` | int | Number of boundary vertices on each edge; trail is the remainder. |
| `size` | float | Target lattice cell size (smaller → finer lattice). |
| `increase_lattice_stretch` | float | Rhombic cell aspect ratio (higher → more spanwise elongation). |
| `based_on_extrema` | bool | Origin from geometric extrema (`True`) or an OBJ `origin` marker (`False`). |
| `master_folder_setting` | bool | Input layout and whether relative plotting runs. See [Input data](#input-data). |
| `smoothing_level` | int | Window of the moving-average filter applied to the final paths (`0` disables). |

## Output

- The global 3-D path traced by each lattice node across the extension cycle (orange polylines), drawn over the contracted and extended surfaces.
- Node positions on the contracted (teal) and extended (blue) meshes.
- With `master_folder_setting = True`, an experimental second view of node paths reconstructed onto the dynamic-region surface.

## Limitations

This is a research prototype. Known constraints (discussed in the paper):

- **Spanwise coverage loss.** As the lattice contracts spanwise under extension, a strip near the root edge is left uncovered in the fully extended state; the simulation tracks only the covered region.
- **Right wing only.** Only `boundary_dir = 1` is functional.
- **Manual boundary calibration.** Boundary vertex counts must be tuned by hand per dataset; incorrect values produce silently malformed partitions.
- **Drift in marching trilateration.** Errors accumulate row to row and grow on high-curvature or large-deformation geometry; the magnitude is not yet characterised.
- **No lattice-size validation.** An incompatible `size` errors at the plotting stage rather than on entry.
- **Endpoint contraction from smoothing.** The moving-average filter is not pinned to the path endpoints, so a high `smoothing_level` pulls them inward.
- **`based_on_extrema = False`** requires the `origin` vertex marking and will fail on meshes prepared without it.

## Tutorial

The repository includes sample data so you can run the full pipeline without preparing your own meshes. Each example is selected by setting two parameters at the top of `Main.py`, then running the program. The default boundary parameters in `Main.py` (`-55 87 337 87 1`) are already calibrated for the included meshes, after filtering out or adjusting boundaries for any stage with large boundary diffrence, you can simply type `continue` at the boundary prompt.

### 1. Combined + static + dynamic meshes (`MasterFolder`)

Runs the full pipeline including the experimental relative-path tracking, which needs the separate static and dynamic surface meshes.

In `Main.py`:

```python
based_on_extrema = False   # origin is read from the `origin` marker in the OBJ
master_folder_setting = True    # expects Stage_*/ folders, each with Mesh_A/B/C
```

Then:

```bash
uv run python Main.py
```

1. In the folder dialog, select the included **`MasterFolder`** directory.
2. The viewer shows the mesh with its colored boundary edges (root = red, lead = blue, tip = green, trail = orange). The defaults match most of the data, so after you `filter` out any meshes with inconsistent boundaries, type `continue`.
3. (Optional) Save the session when prompted.
4. The global node paths are drawn over the contracted (teal) and extended (blue) meshes.
5. Press **Enter** at the `EXPERIMENTAL` prompt to also reconstruct the relative node paths on the dynamic surface (purple).

### 2. Combined meshes only (`MasterFolderWithOnlyCombined`)

The simpler, recommended mode: one combined mesh per stage, global paths only, no relative-path step.

In `Main.py`:

```python
based_on_extrema = False
master_folder_setting = False   # expects Stage_1.obj, Stage_2.obj, … in one folder
```

Then:

```bash
uv run python Main.py
```

1. Select the included **`MasterFolderWithOnlyCombined`** directory.
2. Type `filter`, look through the stages, and if the boundaries are inconsistent, type `m` to remove them from the pipeline or `b` to adjust the local boundary for that specific mesh.
3. Type `continue` at the boundary prompt.
4. (Optional) Save the session when prompted.
5. The global node paths are drawn across the stages. There is no relative-path step in this mode, since the dynamic surface is not provided.

### 3. Loading a saved session (`Tutorial.pkl`)

`Tutorial.pkl` is a session saved from the `MasterFolder` run. Loading it restores the reference lattice and **skips the geodesic and lattice computation**, jumping straight to deformation tracking (much faster than rebuilding from scratch). The meshes are still required: the marching stage runs on the actual stage geometry, and the session only restores the reference lattice.

In `Main.py` (match the dataset the session was built from):

```python
based_on_extrema = False
master_folder_setting = True
```

Then:

```bash
uv run python Main.py
```

1. Select the included **`MasterFolder`** directory, the meshes are needed for the deformation stage.
2. At the boundary prompt type `load`, choose **`Tutorial.pkl`** from the list (sessions are read from the `sessions/` folder; a full path is also accepted), then type `continue`.
3. Because a full session is loaded, the reference lattice is reused and the pipeline proceeds directly to marching the lattice across the stages, producing the same paths as Example 1.

## More Documentation / Article

Additional information regarding the program’s design methodology, implementation, and usage can be found in the accompanying paper:
[Mesh Morphing and Dynamic Lattice Simulation](https://github.com/user-attachments/files/28650327/rLatticeSimForDynamicNodes.pdf)
