import numpy as np
import pyvista as pv
from collections import Counter

obj_path = "/Users/marko/Documents/GitHub/marko/Models/LatticeTestbenchPYplotTaper.obj"

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
n_origin_shift = -103  # optional shift of start point
n_root  = 105
n_lead  = 500
n_tip   = 72


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

visEdges = visMesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)

#visualize
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="red", opacity=0.8)
plotter.add_mesh(visEdges, color="blue", line_width=1)

plotter.add_points(root_pts,  color="purple", point_size=12, render_points_as_spheres=True)
plotter.add_points(lead_pts,  color="yellow", point_size=12, render_points_as_spheres=True)
plotter.add_points(tip_pts,   color="orange", point_size=12, render_points_as_spheres=True)
plotter.add_points(trail_pts, color="cyan",   point_size=12, render_points_as_spheres=True)

plotter.show_axes()
plotter.show_bounds(grid="back", location="all")
plotter.show()