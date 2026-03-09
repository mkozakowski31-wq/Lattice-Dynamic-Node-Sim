import pyvista as pv
from pyvistaqt import BackgroundPlotter
import numpy as np

from stage_1_boundary import load_meshes, define_boundaries
from stage_2_geodesics import compute_geodesics, smooth_geodesic
from stage_3_lattice   import build_lattice
from stage_4_deformation import DeformedMesh

# ---- Paths ----
obj_path_contract = "Models/LatticeTestbenchPYplotTaper.obj"
obj_path_extended = "Models/ExtendedLatticeTestbenchPYplotTaper.obj"

# ---- Parameters ----
boundary_dir   = 1
n_origin_shift = -102
n_root         = 103
n_lead         = 501
n_tip          = 72
size           = 1

# ---- Viewer ----
p = BackgroundPlotter()

# ---- Stage 1: define boundaries interactively ----
(mesh, mesh_extended,
 points, faces,
 points_ex, faces_ex,
 visEdges) = load_meshes(obj_path_contract, obj_path_extended)

(root_edges, lead_edges, tip_edges, trail_edges,
 root_pts, lead_pts, tip_pts, trail_pts,
 junction_points, leadEdge, trailEdge,
 VWcount, VCcount,
 n_origin_shift, n_root, n_lead, n_tip, boundary_dir, size) = define_boundaries(
    p, mesh, points, faces, visEdges,
    n_origin_shift, n_root, n_lead, n_tip, boundary_dir, size)


# ---- Stage 2: compute geodesics, updates viewer ----
geo_linesX, geo_linesY, geo_lineX, geo_lineY = compute_geodesics(
    mesh, root_pts, tip_pts, lead_pts, trail_pts, VWcount, VCcount)

# ---- Stage 3: build lattice, extended mesh, final viz ----
Geolines = build_lattice(
    p, mesh, mesh_extended,
    geo_linesX, geo_linesY,
    root_pts, lead_pts, tip_pts, trail_pts,
    junction_points, leadEdge, trailEdge,
    points_ex, faces_ex,
    n_origin_shift, n_root, n_lead, n_tip, boundary_dir,
    VWcount, VCcount, visEdges)

result = DeformedMesh(obj_path_contract, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount)

x = result.Z_SegTopographicalDeformation(Geolines,0)
print(x)
input("Press enter to close viewer")