import numpy as np
import pyvista as pv
from collections import Counter


obj_path = "/Users/marko/Documents/GitHub/marko/Models/LatticeTestBenchComplicatedGeo.obj"

mesh = pv.read(obj_path)
visMesh = mesh
mesh = mesh.triangulate()
mesh = mesh.clean(tolerance=1e-6)
points = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]  # shape (N, 3)

edges = []

for a, b, c in faces:
    edges.append(tuple(sorted((a, b))))
    edges.append(tuple(sorted((b, c))))
    edges.append(tuple(sorted((c, a))))

edge_counts = Counter(edges)
boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
print("Boundary edges:", len(boundary_edges))
boundary_vertices = np.unique(np.array(boundary_edges).flatten())
print("Boundary vertices:", len(boundary_vertices))


boundary_points = mesh.points[boundary_vertices]

visEdges = visMesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)

plotter = pv.Plotter()

plotter.add_mesh(mesh, opacity=1, color="red")
plotter.add_mesh(
    boundary_points,
    color="green",
    point_size=10,
    render_points_as_spheres=True
)
plotter.add_mesh(visEdges, color="blue", line_width=1) 
plotter.show_axes()
plotter.show_bounds(grid='back', location='all', show_xlabels=True, show_ylabels=True, show_zlabels=True)
plotter.show()