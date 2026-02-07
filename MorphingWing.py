import numpy as np 
import pyvista as pv
import vtk
from pyvistaqt import BackgroundPlotter
from scipy.spatial import cKDTree
from tqdm import tqdm
from collections import Counter
import time
from joblib import Parallel, delayed


obj_path = "Models/LatticeTestBenchComplicatedGeo.obj"
plotter = pv.Plotter()
p = BackgroundPlotter()
mesh = pv.read(obj_path)
mesh = mesh.triangulate()
visMesh = mesh
mesh = mesh.clean(tolerance=1e-6)
visEd = True
points = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]  # (N,3)

#--------------- PARAMETER VARIABLES ----------------------
boundary_dir = 1
n_origin_shift = -102  # optional shift of start point
n_root  = 103
n_lead  = 501
n_tip   = 72
#Lattice adjustment values
size = 1

def EdgeLength(edges, points):
    edges = np.asarray(edges)
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    return np.linalg.norm(p1 - p0, axis=1).sum()

def reorder_curve(points):

    points = np.asarray(points)
    used = np.zeros(len(points), dtype=bool)

    # start from one endpoint (furthest from centroid)
    centroid = points.mean(axis=0)
    start = np.argmax(np.linalg.norm(points - centroid, axis=1))

    ordered = [points[start]]
    used[start] = True
    current = points[start]

    for _ in range(len(points) - 1):
        dists = np.linalg.norm(points - current, axis=1)
        dists[used] = np.inf
        idx = np.argmin(dists)

        ordered.append(points[idx])
        used[idx] = True
        current = points[idx]

    return np.array(ordered)

def resample_curve_equal(points, N):
    points = np.asarray(points)

    if N < 2:
        raise ValueError("N must be >= 2")

    # Segment vectors and lengths
    segs = points[1:] - points[:-1]
    lens = np.linalg.norm(segs, axis=1)

    total_len = lens.sum()
    if total_len == 0:
        return np.repeat(points[:1], N, axis=0)

    # Target arc-length positions
    target_s = np.linspace(0.0, total_len, N)

    # Cumulative lengths
    cum_len = np.concatenate([[0.0], np.cumsum(lens)])

    out = np.zeros((N, 3))
    seg_idx = 0

    for i, s in enumerate(target_s):
        while seg_idx < len(lens) - 1 and cum_len[seg_idx + 1] < s:
            seg_idx += 1

        seg_len = lens[seg_idx]
        if seg_len == 0:
            out[i] = points[seg_idx]
        else:
            t = (s - cum_len[seg_idx]) / seg_len
            out[i] = points[seg_idx] + t * segs[seg_idx]

    return out

def pn_triangle_point(P, N, bary):
    u, v, w = bary

    # Corner points
    b300 = P[0]
    b030 = P[1]
    b003 = P[2]

    # Edge control points
    b210 = (2*P[0] + P[1] - np.dot(P[1]-P[0], N[0]) * N[0]) / 3
    b120 = (2*P[1] + P[0] - np.dot(P[0]-P[1], N[1]) * N[1]) / 3

    b021 = (2*P[1] + P[2] - np.dot(P[2]-P[1], N[1]) * N[1]) / 3
    b012 = (2*P[2] + P[1] - np.dot(P[1]-P[2], N[2]) * N[2]) / 3

    b102 = (2*P[2] + P[0] - np.dot(P[0]-P[2], N[2]) * N[2]) / 3
    b201 = (2*P[0] + P[2] - np.dot(P[2]-P[0], N[0]) * N[0]) / 3

    # Center point
    E = (b210 + b120 + b021 + b012 + b102 + b201) / 6
    V = (P[0] + P[1] + P[2]) / 3
    b111 = E + (E - V) / 2

    # Bernstein basis
    return (
        b300*u**3 + b030*v**3 + b003*w**3 +
        3*(b210*u**2*v + b120*u*v**2 +
           b021*v**2*w + b012*v*w**2 +
           b102*w**2*u + b201*w*u**2) +
        6*b111*u*v*w
    )

def smooth_geodesic(mesh, start, end, n_points=50, iters=50):

    path = np.linspace(start, end, n_points)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    for _ in range(iters):
        for i in range(1, n_points - 1):
            p = path[i]

            # --- VTK closest-point query ---
            closest_point = [0.0, 0.0, 0.0]
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.mutable(0.0)

            locator.FindClosestPoint(p,closest_point,cell_id,sub_id,dist2)

            cid = int(cell_id)
            cell = mesh.get_cell(cid)
            vids = cell.point_ids

            P = mesh.points[vids]
            N = mesh.point_normals[vids]

            # Barycentric coordinates
            v0, v1, v2 = P
            v = np.array(closest_point) - v0
            a = v1 - v0
            b = v2 - v0

            d00 = np.dot(a, a)
            d01 = np.dot(a, b)
            d11 = np.dot(b, b)
            d20 = np.dot(v, a)
            d21 = np.dot(v, b)

            denom = d00 * d11 - d01 * d01
            v_coord = (d11 * d20 - d01 * d21) / denom
            w_coord = (d00 * d21 - d01 * d20) / denom
            u_coord = 1.0 - v_coord - w_coord

            bary = np.clip([u_coord, v_coord, w_coord], 0.0, 1.0)
            bary /= bary.sum()

            path[i] = pn_triangle_point(P, N, bary)

    return path

def sample_polyline(points, n=200):
    t = np.linspace(0, 1, len(points))
    ti = np.linspace(0, 1, n)
    return np.vstack([
        np.interp(ti, t, points[:, k])
        for k in range(3)
    ]).T

def curve_curve_closest_points(curveA, curveB):
    tree = cKDTree(curveB)
    dists, idx = tree.query(curveA)
    i = np.argmin(dists)
    return curveA[i], curveB[idx[i]]

def EdgeSolver(n_origin_shift, n_root, n_lead, n_tip, boundary_dir):
    edges = []

    for a, b, c in faces:
        edges.append(tuple(sorted((a, b))))
        edges.append(tuple(sorted((b, c))))
        edges.append(tuple(sorted((c, a))))

    edge_counts = Counter(edges)
    boundary_edges = [e for e, count in edge_counts.items() if count == 1]

    boundary_edges = np.array(boundary_edges)
    boundary_vertices = np.unique(boundary_edges.flatten())

    print("Boundary edges:", len(boundary_edges))
    print("Boundary vertices:", len(boundary_vertices))

    # Build adjacency list
    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Choose start vertex (root reference)
    # Here: minimum X — change axis if desired
    start = min(boundary_vertices, key=lambda i: points[i, 0])

    ordered_vertices = [start]
    prev = None
    current = start

    while True:
        neighbors = adj[current]
        next_v = neighbors[0] if neighbors[0] != prev else neighbors[1]

        if next_v == start:
            break

        ordered_vertices.append(next_v)
        prev, current = current, next_v

    ordered_vertices = np.array(ordered_vertices)

    if boundary_dir == -1:
        ordered_vertices = ordered_vertices[::-1]

    # cyclic shift (edge-based)
    shift = n_origin_shift % len(ordered_vertices)
    ordered_vertices = np.roll(ordered_vertices, -shift)

    # Build ordered edges
    ordered_edges = [
        (ordered_vertices[i], ordered_vertices[(i + 1) % len(ordered_vertices)])
        for i in range(len(ordered_vertices))
    ]

    print("Ordered boundary vertices:", len(ordered_vertices))

    N = len(ordered_edges)


    # trailing auto-fills

    n_trail = N - (n_root + n_lead + n_tip)
    if n_trail <= 0:
        raise ValueError("Segment sizes exceed boundary length")

    # Slice ordered edges
    root_edges  = ordered_edges[0 : n_root]
    lead_edges  = ordered_edges[n_root : n_root + n_lead]
    tip_edges   = ordered_edges[n_root + n_lead : n_root + n_lead + n_tip]
    trail_edges = ordered_edges[n_root + n_lead + n_tip :]

    # Convert to vertex sets
    root_pts  = points[np.unique(np.array(root_edges).flatten())]
    lead_pts  = points[np.unique(np.array(lead_edges).flatten())]
    tip_pts   = points[np.unique(np.array(tip_edges).flatten())]
    trail_pts = points[np.unique(np.array(trail_edges).flatten())]

    L_R = ordered_vertices[n_root]
    L_T = ordered_vertices[n_root + n_lead]
    T_T = ordered_vertices[n_root + n_lead + n_tip]
    T_R = ordered_vertices[0]

    junction_vertices = np.array([L_R, L_T, T_T, T_R])
    junction_points = points[junction_vertices]

    return (
    ordered_vertices,
    root_edges, lead_edges, tip_edges, trail_edges,
    root_pts, lead_pts, tip_pts, trail_pts, junction_points)

(ordered_vertices, root_edges, lead_edges, tip_edges, trail_edges,
root_pts, lead_pts, tip_pts, trail_pts, junction_points) = EdgeSolver(n_origin_shift, n_root, n_lead, n_tip , boundary_dir)

visEdges = visMesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)

def updateGeo(visEd):
    p.clear()

    p.add_mesh(mesh, color="red", opacity=0.8)
    if visEd == True:
        p.add_mesh(visEdges, color="blue", line_width=1)

    #Boundries
    p.add_points(root_pts,  color="red", point_size=12, render_points_as_spheres=True)
    p.add_points(lead_pts,  color="blue", point_size=12, render_points_as_spheres=True)
    p.add_points(tip_pts,   color="green", point_size=12, render_points_as_spheres=True)
    p.add_points(trail_pts, color="orange",   point_size=12, render_points_as_spheres=True)

    #corners
    p.add_points(junction_points[0], color="pink", point_size=20, render_points_as_spheres=True) # LEAD-ROOT
    p.add_points(junction_points[1], color="teal", point_size=20, render_points_as_spheres=True) # TIP-LEAD
    p.add_points(junction_points[2], color="black", point_size=20, render_points_as_spheres=True) # TIP-TRAIL
    p.add_points(junction_points[3], color="white", point_size=20, render_points_as_spheres=True) # ROOT-TRAIL
updateGeo(visEd)
while True:
    print("\nAdjust edge parameters:")
    print(f"shift={n_origin_shift}, root={n_root}, lead={n_lead}, tip={n_tip}, boundary direction={boundary_dir}")

    cmd = input("Enter: shift root lead tip boundry  OR   type 'continue' OR type 'help' OR type 'latticesize'\n> ").strip()

    if cmd.lower() == "continue":
        break
    if cmd.lower() == "help":
        print("")
        print("Each index defines the number of vertices in each boundry edge")
        print("Adjust boundry direction (Counter vs. Clockwise), by inputing 1 or -1 to boundry index")
        print("Boundary Edges: Root chord = Red, Leading edge = Blue, Tip chord = Green, Trailing edge = Orange")
        print("Corners: Root_Lead = Pink, Lead_Tip = Green, Tip_trail = Black, Trail_root = White")
        input("Press enter to continue")
    if cmd.lower() == "latticesize":
        print("WARNING: Entering a lattice size too large may result in error")
        while True:
            print("Current lattice size = "+ str(size))
            cmd1 = input('Enter Lattice size: ')
            try:
                userLattice= int(cmd1)
                size = userLattice
                print(f"Valid input. Lattice size: {str(size)}")
                break
            except:
                print("Not an integer, try again. ")
    else:
        try:
            n_origin_shift, n_root, n_lead, n_tip, boundary_dir = map(int, cmd.split())
        except ValueError:
            print("Invalid input. Example: -102 103 501 72 1 or continue")
            continue

    # ---- recompute edges ----
    (
        ordered_vertices,
        root_edges, lead_edges, tip_edges, trail_edges,
        root_pts, lead_pts, tip_pts, trail_pts, junction_points
    ) = EdgeSolver(n_origin_shift, n_root, n_lead, n_tip, boundary_dir)
    updateGeo(visEd)

visEd = False
root_len  = EdgeLength(root_edges, points)
tip_len   = EdgeLength(tip_edges, points)
lead_len  = EdgeLength(lead_edges, points)
trail_len = EdgeLength(trail_edges, points)

if root_len >= tip_len:
    VCcount = int(np.ceil(root_len/size))
else:
    VCcount = int(np.ceil(root_len/size))
if lead_len >= trail_len:
    VWcount = int(np.ceil(lead_len/size))
else:
    VWcount = int(np.ceil(trail_len/size))

root_pts = resample_curve_equal(reorder_curve(root_pts),VCcount)
tip_pts = resample_curve_equal(reorder_curve(tip_pts), VCcount)
lead_pts = resample_curve_equal(reorder_curve(lead_pts), VWcount)
trail_pts = resample_curve_equal(reorder_curve(trail_pts), VWcount)

print('Verify count, root vertices: '+str(len(root_pts))+", tip vertices: "+str(len(tip_pts)))
print('Verify count, lead vertices: '+str(len(lead_pts))+", trail vertices: "+str(len(trail_pts)))

#Algorithm to plot corrisponding diagonal vertices
geo_linesX = []
start_time = time.perf_counter()

mesh.compute_normals(inplace=True)
def make_geo(start, end):
    curve = smooth_geodesic(mesh, start, end, n_points=60, iters=80)
    return pv.lines_from_points(curve)
# ---- Root → Trail (X) ----
geo_linesX_1 = Parallel(n_jobs=-1)(
    delayed(make_geo)(trail_pts[VWcount - 1 - x], root_pts[VCcount - 1 - x])
    for x in tqdm(range(VCcount), desc="Processing RootTrailX Geodesics")
)
# ---- Lead → Trail (X) ----
geo_linesX_2 = Parallel(n_jobs=-1)(
    delayed(make_geo)(trail_pts[x], lead_pts[x + VCcount - 1]) for x in tqdm(range(0, VWcount - VCcount), desc="Processing LeadTrailX Geodesics")
)
# ---- Tip → Lead (X) ----
geo_linesX_3 = Parallel(n_jobs=-1)(
    delayed(make_geo)(tip_pts[x],lead_pts[x])for x in tqdm(range(VCcount), desc="Processing TipLeadX Geodesics")
)
# ---- Combine + merge ONCE ----
geo_linesX = geo_linesX_1 + geo_linesX_2 + geo_linesX_3
geo_lineX = pv.merge(geo_linesX)

span_dir = tip_pts.mean(axis=0) - root_pts.mean(axis=0)
span_dir /= np.linalg.norm(span_dir)

def curve_key(curve):
    return np.dot(curve.points.mean(axis=0), span_dir)

geo_linesX = sorted(geo_linesX, key=curve_key)
geo_linesY=[]
# ---- Root → Lead (Y) ----
geo_linesY_1 = Parallel(n_jobs=-1)(
    delayed(make_geo)(lead_pts[VWcount - y],root_pts[y - 1]) for y in tqdm(range(VCcount, 0, -1), desc="Processing RootLeadY Geodesics")
)
# ---- Lead → Trail (Y) ----
geo_linesY_2 = Parallel(n_jobs=-1)(
    delayed(make_geo)(trail_pts[y + VCcount - 1],lead_pts[y])
    for y in tqdm(range(VWcount - VCcount), desc="Processing LeadTrailY Geodesics")
)
# ---- Tip → Trail (Y) ----
geo_linesY_3 = Parallel(n_jobs=-1)(
    delayed(make_geo)(tip_pts[VCcount - y - 1],trail_pts[y])for y in tqdm(range(VCcount), desc="Processing TipTrailY Geodesics")
)
# ---- Combine + merge ONCE ----
geo_linesY = geo_linesY_1 + geo_linesY_2 + geo_linesY_3
geo_lineY = pv.merge(geo_linesY)
end_time = time.perf_counter()

span_dir = tip_pts.mean(axis=0) - root_pts.mean(axis=0)
span_dir /= np.linalg.norm(span_dir)

def curve_key(curve):
    return np.dot(curve.points.mean(axis=0), span_dir)

geo_linesY = sorted(geo_linesY, key=curve_key)

elapsed_time = end_time - start_time
print(f"Execution took: {elapsed_time:.4f} seconds")

lattice_nodes = []
polyConnectY = []
polyConnectX = []
for cx in tqdm(geo_linesX, desc="Processing Y Inter-Lattice straight lines:"):
    cx_s = sample_polyline(cx.points, 300)
    connectY = []
    for cy in geo_linesY:
        cy_s = sample_polyline(cy.points, 300)
        pA, _ = curve_curve_closest_points(cx_s, cy_s)
        lattice_nodes.append(pA)
        connectY.append(pA)
    for l in range(len(connectY) - 1):
        Ypts = np.vstack([connectY[l], connectY[l + 1]])
        polyConnectY.append(pv.lines_from_points(Ypts))

for cy in tqdm(geo_linesY, desc="Processing X Inter-Lattice Straight lines:"):
    cy_s = sample_polyline(cy.points, 300)
    connectX = [] 
    for cx in geo_linesX:
        cx_s = sample_polyline(cx.points, 300)
        pA, _ = curve_curve_closest_points(cy_s, cx_s)
        connectX.append(pA)
    for l in range(len(connectX) - 1):
        Xpts = np.vstack([connectX[l], connectX[l + 1]])
        polyConnectX.append(pv.lines_from_points(Xpts))

polyConnectY_mesh = pv.merge(polyConnectY)
polyConnectX_mesh = pv.merge(polyConnectX)
lattice_nodes = np.asarray(lattice_nodes)

#Lattice
updateGeo(visEd)

p.add_points(lattice_nodes, color="cyan", point_size=6,render_points_as_spheres=True)
# p.add_mesh(geo_lineX, line_width=3, color='white')
# p.add_mesh(geo_lineY, line_width=3, color='gray')
p.add_mesh(polyConnectY_mesh, line_width=3, color='black')
p.add_mesh(polyConnectX_mesh, line_width=3, color='gray')

input("Press enter to close viewer")