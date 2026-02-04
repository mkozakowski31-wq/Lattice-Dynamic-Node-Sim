import numpy as np
import pyvista as pv
from collections import Counter

obj_path = "Models/LatticeTestbenchPYplotTaper.obj"

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




mesh = pv.read(obj_path)
mesh = mesh.triangulate()
visMesh = mesh
mesh = mesh.clean(tolerance=1e-6)

points = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]  # (N,3)

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
# ---- adjust THESE ----
n_origin_shift = -102  # optional shift of start point
n_root  = 103
n_lead  = 501
n_tip   = 72
#Lattice adjustment values
size = 1

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
root_verts  = np.unique(np.array(root_edges).flatten())
lead_verts  = np.unique(np.array(lead_edges).flatten())
tip_verts   = np.unique(np.array(tip_edges).flatten())
trail_verts = np.unique(np.array(trail_edges).flatten())

root_pts  = points[root_verts]
lead_pts  = points[lead_verts]
tip_pts   = points[tip_verts]
trail_pts = points[trail_verts]

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

# ---- Junction vertices (derived from segmentation) ----
L_R = ordered_vertices[n_root]
L_T = ordered_vertices[n_root + n_lead]
T_T = ordered_vertices[n_root + n_lead + n_tip]
T_R = ordered_vertices[0]

junction_vertices = np.array([L_R, L_T, T_T, T_R])
junction_points = points[junction_vertices]


visEdges = visMesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)

#visualize
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="red", opacity=0.8)
plotter.add_mesh(visEdges, color="blue", line_width=1)
plotter.show_axes()
plotter.show_bounds(grid="back", location="all")


#segments
plotter.add_points(root_pts,  color="red", point_size=12, render_points_as_spheres=True)
plotter.add_points(lead_pts,  color="blue", point_size=12, render_points_as_spheres=True)
plotter.add_points(tip_pts,   color="green", point_size=12, render_points_as_spheres=True)
plotter.add_points(trail_pts, color="orange",   point_size=12, render_points_as_spheres=True)

#corners
# plotter.add_points(junction_points[0], color="pink", point_size=20, render_points_as_spheres=True) # LEAD-ROOT
# plotter.add_points(junction_points[1], color="teal", point_size=20, render_points_as_spheres=True) # TIP-LEAD
# plotter.add_points(junction_points[2], color="black", point_size=20, render_points_as_spheres=True) # TIP-TRAIL
# plotter.add_points(junction_points[3], color="white", point_size=20, render_points_as_spheres=True) # ROOT-TRAIL

plotter.show()
