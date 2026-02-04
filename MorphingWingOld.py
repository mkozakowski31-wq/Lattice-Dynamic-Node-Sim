import numpy as np
import pyvista as pv
from collections import Counter
from scipy.spatial import cKDTree

# -------------------------------------------------
# LOAD + CLEAN MESH
# -------------------------------------------------
obj_path = "/Users/marko/Documents/GitHub/marko/Models/LatticeTestbenchPYplotTaper.obj"

mesh = pv.read(obj_path)
mesh = mesh.triangulate()
mesh = mesh.clean(tolerance=1e-6)

points = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]

# -------------------------------------------------
# FIND BOUNDARY EDGES
# -------------------------------------------------
edges = []
for a, b, c in faces:
    edges.append(tuple(sorted((a, b))))
    edges.append(tuple(sorted((b, c))))
    edges.append(tuple(sorted((c, a))))

edge_counts = Counter(edges)
boundary_edges = np.array([e for e, c in edge_counts.items() if c == 1])
boundary_vertices = np.unique(boundary_edges.flatten())

print("Boundary vertices:", len(boundary_vertices))

# -------------------------------------------------
# ORDER BOUNDARY LOOP
# -------------------------------------------------
adj = {}
for a, b in boundary_edges:
    adj.setdefault(a, []).append(b)
    adj.setdefault(b, []).append(a)

start = min(boundary_vertices, key=lambda i: points[i, 0])

ordered = [start]
prev = None
cur = start

while True:
    nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
    if nxt == start:
        break
    ordered.append(nxt)
    prev, cur = cur, nxt

ordered = np.array(ordered)

# -------------------------------------------------
# USER-CONTROLLED SEGMENT SIZES
# -------------------------------------------------
n_root = 103
n_lead = 501
n_tip  = 72
n_trail = len(ordered) - (n_root + n_lead + n_tip)

if n_trail <= 0:
    raise ValueError("Segment sizes invalid")

root_idx  = ordered[:n_root]
lead_idx  = ordered[n_root:n_root+n_lead]
tip_idx   = ordered[n_root+n_lead:n_root+n_lead+n_tip]
trail_idx = ordered[n_root+n_lead+n_tip:]

root_pts  = points[root_idx]
trail_pts = points[trail_idx]

# -------------------------------------------------
# STRUCTURAL CURVE FUNCTIONS
# -------------------------------------------------
def arc_param(P):
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s / s[-1]

def interp_boundary(P, t):
    s = arc_param(P)
    i = np.searchsorted(s, t)
    if i == 0:
        return P[0]
    if i >= len(P):
        return P[-1]
    w = (t - s[i-1]) / (s[i] - s[i-1])
    return (1-w)*P[i-1] + w*P[i]

def ruled_curve(A, B, n=150):
    curve = []
    for t in np.linspace(0, 1, n):
        pa = interp_boundary(A, t)
        pb = interp_boundary(B, t)
        curve.append((1-t)*pa + t*pb)
    return np.array(curve)

def project_to_mesh(curve, V):
    tree = cKDTree(V)
    _, idx = tree.query(curve)
    return V[idx]

def smooth_curve(curve, iters=15, lam=0.4):
    C = curve.copy()
    for _ in range(iters):
        C[1:-1] += lam * (C[:-2] + C[2:] - 2*C[1:-1])
    return C

# -------------------------------------------------
# BUILD STRUCTURAL CONNECTION
# -------------------------------------------------
print("Connecting root_pts[0] to trail_pts[39]")

curve = ruled_curve(root_pts, trail_pts, n=200)
curve = project_to_mesh(curve, points)
curve = smooth_curve(curve)

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgray", opacity=0.85)

plotter.add_points(root_pts, color="purple", point_size=10)
plotter.add_points(trail_pts, color="cyan", point_size=10)

plotter.add_mesh(pv.PolyData(curve), color="yellow", line_width=4)
plotter.add_points(curve, color="yellow", point_size=6)

plotter.show_axes()
plotter.show_bounds()
plotter.show()