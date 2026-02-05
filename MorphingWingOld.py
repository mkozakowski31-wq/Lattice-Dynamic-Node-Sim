import numpy as np
import pyvista as pv
from collections import Counter

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
obj_path = "Models/LatticeTestbenchPYplotTaper.obj"

# Desired approximate spacing between lattice nodes
TARGET_SPACING = 1.0

# -------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------
def compute_edge_length(edges, pts):
    """Sum of lengths of an array of edges."""
    e = np.asarray(edges)
    p0 = pts[e[:,0]]
    p1 = pts[e[:,1]]
    return np.linalg.norm(p1 - p0, axis=1).sum()

def reorder_curve(points):
    """Order an unordered point set into a continuous open curve."""
    pts = np.asarray(points)
    used = np.zeros(len(pts), dtype=bool)
    centroid = pts.mean(axis=0)
    start = np.argmax(np.linalg.norm(pts - centroid, axis=1))
    ordered = [pts[start]]
    used[start] = True
    current = pts[start]
    for _ in range(len(pts) - 1):
        d = np.linalg.norm(pts - current, axis=1)
        d[used] = np.inf
        idx = np.argmin(d)
        ordered.append(pts[idx])
        used[idx] = True
        current = pts[idx]
    return np.array(ordered)

def resample_curve(points, spacing):
    """Resample an ordered curve to approximate equal spacing."""
    pts = np.asarray(points)
    if pts.shape[0] < 2:
        return pts.copy()
    segs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(segs, axis=1)
    total_len = seg_lens.sum()
    n_samples = max(2, int(np.ceil(total_len / spacing)))
    target_s = np.linspace(0.0, total_len, n_samples)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    resampled = []
    j = 0
    for ts in target_s:
        while j < len(seg_lens)-1 and cum[j+1] < ts:
            j += 1
        seg_len = seg_lens[j]
        if seg_len == 0:
            p = pts[j]
        else:
            t = (ts - cum[j]) / seg_len
            p = (1-t)*pts[j] + t*pts[j+1]
        resampled.append(p)
    return np.array(resampled)

# -------------------------------------------------
# LOAD + CLEAN MESH
# -------------------------------------------------
mesh = pv.read(obj_path).triangulate().clean(tolerance=1e-6)
points = mesh.points
faces = mesh.faces.reshape(-1, 4)[:,1:]

# -------------------------------------------------
# EXTRACT BOUNDARY LOOP
# -------------------------------------------------
# Find unique boundary edges (count == 1)
all_edges = []
for a,b,c in faces:
    all_edges.append(tuple(sorted((a,b))))
    all_edges.append(tuple(sorted((b,c))))
    all_edges.append(tuple(sorted((c,a))))

edge_counts = Counter(all_edges)
boundary_edges = np.array([e for e,c in edge_counts.items() if c == 1])
boundary_vertices = np.unique(boundary_edges.flatten())

# Build adjacency for loop walk
adj = {}
for a,b in boundary_edges:
    adj.setdefault(a, []).append(b)
    adj.setdefault(b, []).append(a)

# Walk boundary loop
start = min(boundary_vertices, key=lambda i: points[i,0])
ordered_boundary = [start]
prev = None
cur = start

while True:
    nbrs = adj[cur]
    nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
    if nxt == start:
        break
    ordered_boundary.append(nxt)
    prev, cur = cur, nxt

ordered_boundary = np.array(ordered_boundary)

# -------------------------------------------------
# SPLIT INTO SEGMENTS
# -------------------------------------------------
# Adjust these if your boundary configuration is different
n_root = 103
n_lead = 501
n_tip  = 72

N_total = len(ordered_boundary)
n_trail = N_total - (n_root + n_lead + n_tip)
if n_trail < 0:
    raise ValueError("Boundary segmentation counts exceed boundary length")

root_idxs  = ordered_boundary[:n_root]
lead_idxs  = ordered_boundary[n_root:n_root+n_lead]
tip_idxs   = ordered_boundary[n_root+n_lead:n_root+n_lead+n_tip]
trail_idxs = ordered_boundary[n_root+n_lead+n_tip:]

root_pts  = reorder_curve(points[root_idxs])
lead_pts  = reorder_curve(points[lead_idxs])
tip_pts   = reorder_curve(points[tip_idxs])
trail_pts = reorder_curve(points[trail_idxs])

# Resample to approximate equal arc-length spacing
root_pts  = resample_curve(root_pts, TARGET_SPACING)
lead_pts  = resample_curve(lead_pts, TARGET_SPACING)
tip_pts   = resample_curve(tip_pts, TARGET_SPACING)
trail_pts = resample_curve(trail_pts, TARGET_SPACING)

# -------------------------------------------------
# BUILD STRAIGHT LATTICE MEMBERS
# -------------------------------------------------
lattice_segments = []

def add_line(a, b):
    """Store a straight 3D bar between a and b."""
    return pv.Line(a, b)

# 1) Spanwise bars
spanN = min(len(root_pts), len(tip_pts))
for i in range(spanN):
    lattice_segments.append(add_line(root_pts[i], tip_pts[i]))

# 2) Chordwise bars
chordN = min(len(lead_pts), len(trail_pts))
for j in range(chordN):
    lattice_segments.append(add_line(lead_pts[j], trail_pts[j]))

# 3) Diagonal bars (simple cross pattern)
Nmin = min(spanN, chordN)
for k in range(Nmin):
    lattice_segments.append(add_line(root_pts[k], lead_pts[k]))
    lattice_segments.append(add_line(tip_pts[k], trail_pts[k]))

# Optional cross diagonals (if desired)
for k in range(Nmin-1):
    lattice_segments.append(add_line(root_pts[k+1], lead_pts[k]))
    lattice_segments.append(add_line(tip_pts[k], trail_pts[k+1]))

# Combine into one PolyData
lattice = pv.merge(lattice_segments)

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
visEdges = mesh.extract_feature_edges(
    boundary_edges=True,
    non_manifold_edges=False,
    feature_edges=False,
    manifold_edges=False
)

plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgray", opacity=0.7)
plotter.add_mesh(visEdges, color="blue", line_width=1)

plotter.add_mesh(lattice, color="black", line_width=2)

plotter.add_points(root_pts, color="red",    point_size=8)
plotter.add_points(lead_pts, color="yellow", point_size=8)
plotter.add_points(tip_pts, color="green",   point_size=8)
plotter.add_points(trail_pts, color="orange",point_size=8)

plotter.show_axes()
plotter.show_bounds(grid="back", location="all")
plotter.show()