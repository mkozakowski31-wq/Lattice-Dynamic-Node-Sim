import pyvista as pv
from pyvistaqt import BackgroundPlotter
import numpy as np

from stage_1_boundary import load_meshes, define_boundaries
from stage_2_geodesics import compute_geodesics, smooth_geodesic
from stage_3_lattice   import build_lattice
from stage_4_deformation import DeformedMesh

# ---- Paths ----
obj_path_contract = "Models/ExtendedLatticeTestbenchPYplotTaper.obj"
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
z_segments, Geolines = build_lattice(
    p, mesh, mesh_extended,
    geo_linesX, geo_linesY,
    root_pts, lead_pts, tip_pts, trail_pts,
    junction_points, leadEdge, trailEdge,
    points_ex, faces_ex,
    n_origin_shift, n_root, n_lead, n_tip, boundary_dir,
    VWcount, VCcount, visEdges)

result = DeformedMesh(obj_path_contract, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount)

(segs, shot, CsegCurve) = result.Z_SegTopographicalDeformation(Geolines, z_segments,0)
(segs, shot, CsegCurves) = result.Z_SegTopographicalDeformation(Geolines, z_segments, len(z_segments)-1)
CsegCurves = np.vstack(CsegCurves)
CsegCurve = np.vstack(CsegCurve)

result = DeformedMesh(obj_path_extended, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount)

(segs, shot, segCurve) = result.Z_SegTopographicalDeformation(Geolines, z_segments,0)
(segs, shot, segCurves) = result.Z_SegTopographicalDeformation(Geolines, z_segments, len(z_segments)-1)
segCurves = np.vstack(segCurves)
segCurve = np.vstack(segCurve)
# print(segCurve[-1])
# print(segCurves[-1])
print("_______________")
print(len(CsegCurves))
print(CsegCurves)
print("_______________")
print(len(CsegCurve))
print(CsegCurve)
print("_______________")
slice = smooth_geodesic(result.mesh, segCurve[-1], segCurves[-1],n_points=60, iters=80)

p.add_mesh(pv.lines_from_points(segCurve), color="orange", line_width = 15)
p.add_mesh(pv.lines_from_points(segCurves), color="orange", line_width = 15)
p.add_mesh(pv.lines_from_points(CsegCurve), color="orange", line_width = 15)
p.add_mesh(pv.lines_from_points(CsegCurves), color="orange", line_width = 15)
p.add_points(segCurve, color="blue",render_points_as_spheres=True, point_size = 20)
p.add_points(segCurves, color="blue",render_points_as_spheres=True, point_size = 20)
p.add_points(CsegCurve, color="blue",render_points_as_spheres=True, point_size = 20)
p.add_points(CsegCurves, color="blue",render_points_as_spheres=True, point_size = 20)
p.add_mesh(shot,color="green", line_width = 14)
p.add_mesh(segs,color="green", line_width = 14)
p.add_mesh(pv.lines_from_points(slice),color="green", line_width = 14)
p.add_mesh(result.mesh)

# for i in range(20):
#     (segs, shot, segCurve) = result.LatticeTopographicalDeformation(Geolines, z_segments,i)
#     segCurves.append(pv.lines_from_points(segCurve))

# p.add_mesh(pv.merge(segCurves), color="orange", line_width = 14)

#result.LatticeTopographicalDeformation(Geolines, z_segments, 8)
input("Press enter to close viewer")

