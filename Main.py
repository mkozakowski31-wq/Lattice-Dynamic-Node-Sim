import pyvista as pv
import time
from pyvistaqt import BackgroundPlotter
import numpy as np
from session_io import save_session
from tqdm import tqdm


from stage_1_boundary import load_meshes, define_boundaries
from stage_2_geodesics import compute_geodesics, smooth_geodesic
from stage_3_lattice import build_lattice
from stage_4_deformation import DeformedMesh
obj_paths_intermediate = []
# ---- Paths ----
obj_path_contract = "/Users/marko/Documents/GitHub/Mesh2CleanSurface/GroupMeshCV1.obj"
obj_paths_intermediate.append("MultiModels/1_LatticeTestbenchPYplotTaper.obj")
obj_paths_intermediate.append("MultiModels/2_LatticeTestbenchPYplotTaper.obj")
obj_path_extended = "MultiModels/V5BoundConsisCleanC.obj"

# ---- Parameters ----
boundary_dir   = 1
n_origin_shift = -102
n_root         = 103
n_lead         = 501
n_tip          = 72
size           = 1
# ---- Dev Parameters ----
increase_lattice_stretch = 1.5 
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
 loaded, slicesY, slicesX, lattice_nodes) = define_boundaries(
    p, mesh, points, faces, origin,
    visEdges, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, size, increase_lattice_stretch, based_on_extrema)

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

results = []
results.append(DeformedMesh(obj_path_contract, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount, based_on_extrema))
results.append(DeformedMesh(obj_paths_intermediate[0], n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount, based_on_extrema))
results.append(DeformedMesh(obj_paths_intermediate[1], n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount, based_on_extrema))
results.append(DeformedMesh(obj_path_extended, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount, based_on_extrema))

p.clear()

intersect_ptsC = results[0].Z_SegTopographicalDeformation(p, slicesY, slicesX)
intersect_ptsI1 = results[1].Z_SegTopographicalDeformation(p, slicesY, slicesX)
intersect_ptsI2 = results[2].Z_SegTopographicalDeformation(p, slicesY, slicesX)
intersect_ptsE = results[1].Z_SegTopographicalDeformation(p, slicesY, slicesX)

p.add_points(intersect_ptsC, color = "teal", point_size=15, render_points_as_spheres=True)
p.add_points(intersect_ptsE, color = "blue", point_size=15, render_points_as_spheres=True)

p.add_mesh(results[0].mesh, color ="red", opacity=0.5)
p.add_mesh(results[1].mesh, color ="blue", opacity=0.2)
MarchingEnd = time.time()
length = MarchingEnd - MarchingStart
print(f"It took {str(length)} seconds to perform lattice mapping on 4 meshes")

for i in tqdm(range(len(intersect_ptsC)), desc= "Plotting lattice: "):
    pts = []
    pts.append(intersect_ptsC[i])
    pts.append(intersect_ptsI1[i])
    pts.append(intersect_ptsI2[i])
    pts.append(intersect_ptsE[i])
    p.add_mesh(pv.lines_from_points(pts), line_width = 2, color = "orange")

input("Press enter to close viewer")
