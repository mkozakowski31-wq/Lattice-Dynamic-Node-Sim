import pyvista as pv
import numpy as np


m1 = pv.read('/Users/marko/Documents/GitHub/marko/MultiModels/6.obj')  # starting shape
m2 = pv.read('/Users/marko/Documents/GitHub/marko/MultiModels/41.obj')  # ending shape

if m1.n_points != m2.n_points:
    raise ValueError("Meshes have different numbers of vertices!")

# Optional: check max difference
#diff = np.linalg.norm(m1.points - m2.points, axis=1)
#print("Max vertex displacement:", diff.max())

base_mesh = m1.copy()

plotter = pv.Plotter()
actor = plotter.add_mesh(base_mesh, color="red", show_edges=True, edge_color="blue", line_width=1)

def deform(t):
    """
    t: slider value from 0.0 to 1.0
    Interpolates vertices linearly between m1 and m2
    """
    base_mesh.points = (1 - t) * m1.points + t * m2.points
    plotter.render()

plotter.add_slider_widget(
    deform,
    rng=[0.0, 1.0],
    value=0.0,
    title="Deformation",
    pointa=(0.02, 0.9),
    pointb=(0.3, 0.9),
)

plotter.show()