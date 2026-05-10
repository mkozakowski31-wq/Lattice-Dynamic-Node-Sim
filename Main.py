import pyvista as pv
import time
from pyvistaqt import BackgroundPlotter
import numpy as np
from scipy.ndimage import uniform_filter1d
from session_io import save_session
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

from session_io import isolate_differing_characters
from stage_1_boundary import load_meshes, define_boundaries
from stage_2_geodesics import compute_geodesics
from stage_3_lattice import build_lattice
from stage_4_deformation import DeformedMesh
from stage_5_relative_plotting import reconstruct_on_surface, find_contact_points 
root = tk.Tk()
root.withdraw()

Master_folder_selected = filedialog.askdirectory()
print(f"Selected folder: {Master_folder_selected}")

Folder_paths = isolate_differing_characters(Master_folder_selected, isfile=False)
print(f'folders selected are: {str(Folder_paths)}')

class SurfaceMeshSections:
    def __init__(self, stage, FilePaths):
        self.stage = stage
        self.combined_mesh = FilePaths[0]
        self.static_mesh_sect = FilePaths[1]

        transSects = []
        for i in range(2,len(FilePaths),1):
            transSects.append(FilePaths[i])

        self.trans_mesh_sects = transSects

TotalSurfaceObjects = []
for i in range(len(Folder_paths)):
    filePaths = isolate_differing_characters(Folder_paths[i], isfile=True)
    currOBJ = SurfaceMeshSections(stage=i, FilePaths=filePaths)
    TotalSurfaceObjects.append(currOBJ)
    print(f'Stage = {str(currOBJ.stage)}')
    print(f"Combined Mesh = {str(currOBJ.combined_mesh)}")
    print(f"Static Mesh = {str(currOBJ.static_mesh_sect)}")
    print(f"Transiant Meshes = {str(currOBJ.trans_mesh_sects)}")

obj_paths_intermediate = []

# ---- Paths ----
obj_path_contract = TotalSurfaceObjects[0].combined_mesh
for paths in range(1,len(TotalSurfaceObjects)-1):
    obj_paths_intermediate.append(TotalSurfaceObjects[paths].combined_mesh)
obj_path_extended = TotalSurfaceObjects[-1].combined_mesh

# ---- All combined mesh paths (used by the boundary filter) ----
all_obj_paths = [obj_path_contract] + obj_paths_intermediate + [obj_path_extended]

# ---- Parameters ----
boundary_dir = 1
n_origin_shift = -102
n_root = 103
n_lead = 501
n_tip = 72
size = 1
# ---- Dev Parameters ----
increase_lattice_stretch = 2
based_on_extrema = False

p = BackgroundPlotter()

(mesh, mesh_extended,
 points, faces,
 points_ex, faces_ex,
 visEdges, origin) = load_meshes(obj_path_contract, obj_path_extended)

(root_edges, lead_edges, tip_edges, trail_edges,
 root_pts, lead_pts, tip_pts, trail_pts,
 junction_points, leadEdge, trailEdge,
 VWcount, VCcount,
 n_origin_shift, n_root, n_lead, n_tip, boundary_dir, size,
 loaded, slicesY, slicesX, lattice_nodes,
 excluded_indices, stage_overrides) = define_boundaries(
    p, mesh, points, faces, origin,
    visEdges, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, size, increase_lattice_stretch, based_on_extrema, all_obj_paths)

# ---- Apply exclusions from the filter step ----
if excluded_indices:
    print(f"\nRemoving {len(excluded_indices)} excluded stage(s): {sorted(excluded_indices)}")
    TotalSurfaceObjects = [obj for i, obj in enumerate(TotalSurfaceObjects)
                           if i not in excluded_indices]
    all_obj_paths = [p for i, p in enumerate(all_obj_paths)
                     if i not in excluded_indices]
    print(f"Continuing with {len(TotalSurfaceObjects)} stage(s).\n")

if not loaded:
    GeodesicStart = time.time()

    geo_linesX, geo_linesY, geo_lineX, geo_lineY = compute_geodesics(
        mesh, root_pts, tip_pts, lead_pts, trail_pts, VWcount, VCcount)

    slicesY, slicesX, lattice_nodes = build_lattice(
        p, mesh, mesh_extended, geo_linesX, geo_linesY,
        root_pts, lead_pts, tip_pts, trail_pts,
        junction_points, leadEdge, trailEdge,
        points_ex, faces_ex,
        n_origin_shift, n_root, n_lead, n_tip, boundary_dir,
        VWcount, VCcount, visEdges)
    
    GeodesicEnd = time.time()
    length = GeodesicEnd - GeodesicStart
    print(f"It took {str(length)} seconds to perform lattice plotting")

    _name = input("Save session? Enter a filename (no extension) or press Enter to skip: ").strip()
    if _name:
        save_session(
            slicesY, slicesX, lattice_nodes,
            n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount,
            name=_name,
        )
    else:
        print("Session not saved.")

MarchingStart = time.time()

# Rebuild path lists from (potentially filtered) TotalSurfaceObjects
obj_path_contract     = TotalSurfaceObjects[0].combined_mesh
obj_paths_intermediate = [TotalSurfaceObjects[i].combined_mesh for i in range(1, len(TotalSurfaceObjects) - 1)]
obj_path_extended     = TotalSurfaceObjects[-1].combined_mesh

results = []

def _stage_params(stage_i):
    """Return (shift, root, lead, tip, dir) using per-stage override if present."""
    ov = stage_overrides.get(stage_i, {})
    return (
        ov.get('n_origin_shift', n_origin_shift),
        ov.get('n_root',         n_root),
        ov.get('n_lead',         n_lead),
        ov.get('n_tip',          n_tip),
        ov.get('boundary_dir',   boundary_dir),
    )

_s, _r, _le, _ti, _bd = _stage_params(0)
results.append(DeformedMesh(obj_path_contract, _s, _r, _le, _ti, _bd, VCcount, based_on_extrema))
for i in range(len(obj_paths_intermediate)):
    _s, _r, _le, _ti, _bd = _stage_params(i + 1)
    results.append(DeformedMesh(obj_paths_intermediate[i], _s, _r, _le, _ti, _bd, VCcount, based_on_extrema))
_s, _r, _le, _ti, _bd = _stage_params(len(TotalSurfaceObjects) - 1)
results.append(DeformedMesh(obj_path_extended, _s, _r, _le, _ti, _bd, VCcount, based_on_extrema))

p.clear()

intersect_pts = []
relative_intersect_pts = []
bary_recordsArr = []
for i in range(len(results)):
    currentPts = results[i].Z_SegTopographicalDeformation(p, slicesY, slicesX)
    contact_pts, bary_records, contact_indices = find_contact_points(pv.read(TotalSurfaceObjects[i].trans_mesh_sects[0]), currentPts, tolerance=0.01)
    # print(f'In stage {str(i)}, there are {str(len(bary_records))} in contact with dynamic mesh')

    for orig_idx, bary_rec in zip(contact_indices, bary_records):
        relative_intersect_pts.append((i, orig_idx, bary_rec))
    bary_recordsArr.extend(bary_records)
    intersect_pts.append(currentPts)

intersect_ptsC = intersect_pts[0]
intersect_ptsE = intersect_pts[-1]

p.add_points(intersect_ptsC, color = "teal", point_size=15, render_points_as_spheres=True)
p.add_points(intersect_ptsE, color = "blue", point_size=15, render_points_as_spheres=True)

p.add_mesh(results[0].mesh, color ="red", opacity=0.5)
p.add_mesh(results[-1].mesh, color ="blue", opacity=0.2)
p.add_mesh(pv.read(TotalSurfaceObjects[0].static_mesh_sect), color ="green", opacity=0.4)

MarchingEnd = time.time()
length = MarchingEnd - MarchingStart
print(f"It took {str(length)} seconds to perform lattice mapping on {len(intersect_pts)+1} meshes")

def _smooth_path(pts, window=5):
    """Smooth a short list/array of 3-D points along the stage axis."""
    arr = np.asarray(pts, dtype=float)
    n   = len(arr)
    if n < 3:
        return arr
    w = min(window, n if n % 2 == 1 else n - 1)
    out = np.empty_like(arr)
    for dim in range(3):
        out[:, dim] = uniform_filter1d(arr[:, dim], size=w, mode='nearest')
    return out

for i in tqdm(range(len(intersect_ptsC)), desc= "Plotting lattice: "):
    pts = []
    pts.append(intersect_ptsC[i])
    for y in range(1, len(intersect_pts)-1, 1):
        pts.append(intersect_pts[y][i])
    pts.append(intersect_ptsE[i])
    Npts = pts
    pts = _smooth_path(pts, window=5)
    p.add_mesh(pv.lines_from_points(pts), line_width = 2, color = "orange")
    p.add_mesh(pv.lines_from_points(Npts), line_width = 2, color = "green")

input("Press enter to visualize relative paths")

p.clear()

# Load the single reference mesh once — all stages reconstruct onto this
reference_mesh = pv.read(TotalSurfaceObjects[0].trans_mesh_sects[0])

reconstructedPoints = reconstruct_on_surface(reference_mesh, bary_recordsArr)
p.add_points(reconstructedPoints, color='green', point_size=8, render_points_as_spheres=True)
# Group bary records by original point index across all stages
point_stage_map = defaultdict(dict)  # {orig_pt_idx: {stage_i: bary_record}}

for stage_i, orig_idx, bary_rec in relative_intersect_pts:
    point_stage_map[orig_idx][stage_i] = bary_rec

total_stages = len(results)

# Reconstruct and draw a path per node — only if present in every stage
for orig_idx, stage_dict in tqdm(point_stage_map.items(), desc="Plotting relative paths: "):
    if len(stage_dict) != total_stages:
        continue  # discard any node missing from any stage

    path_pts = []
    for stage_i in sorted(stage_dict.keys()):
        # Always reconstruct onto the same reference mesh — this is the key fix
        reconstructed = reconstruct_on_surface(reference_mesh, [stage_dict[stage_i]])
        path_pts.append(reconstructed[0])

    path_pts = np.array(path_pts)
    path_pts = _smooth_path(path_pts, window=3)
    p.add_mesh(pv.lines_from_points(path_pts), line_width=2, color="purple")

p.add_mesh(reference_mesh, color='blue', opacity=0.2)
p.add_mesh(pv.read(TotalSurfaceObjects[0].static_mesh_sect), color='red', opacity=0.2)

input("Press enter to close viewer")


