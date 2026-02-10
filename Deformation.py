import pyvista as pv
from pyvistaqt import BackgroundPlotter
import numpy as np
p = BackgroundPlotter()


m1 = pv.read('Models/LatticeTestbenchPYplotTaper.obj')  # starting shape
m2 = pv.read('Models/ExtendedLatticeTestbenchPYplotTaper.obj')  # ending shape

if m1.n_points != m2.n_points:
    raise ValueError("Meshes have different numbers of vertices!")

# Optional: check max difference
#diff = np.linalg.norm(m1.points - m2.points, axis=1)
#print("Max vertex displacement:", diff.max())

base_mesh = m1.copy()

actor = p.add_mesh(base_mesh, color="red", show_edges=True, edge_color="blue", line_width=1)

# def span_axis(axis):
#     if axis == 'x' or 'X':
#         axis = 0
#     elif axis == 'y' or 'Y':
#         axis == 1
#     elif axis == 'z' or 'Z':
#         axis =2 
def deform(t):
        new_pts = m1.points.copy()


        # new_pts[:, 0] = (1 - t) * m1.points[:, 0] + t * m2.points[:, 0]
        new_pts[:, 1] = (1 - t) * m1.points[:, 1] + t * m2.points[:, 1]

        # Z stays unchanged (from m1)
        new_pts[:, 2] = m1.points[:, 2]

        base_mesh.points = new_pts
        actor
    
# while True:
#     p.add_mesh(m1, color="red", show_edges=True, edge_color="blue", line_width=1)
#     p.show_bounds()
#     axis = input("Input span wise axis (x,y,z):")
#     if axis == 'x' or 'X' or 'y' or 'Y' or 'z' or 'Z':
#         span_axis(axis)
#         break
#     else:
#         print('Invalid input, try again.')

p.add_slider_widget(
    deform,
    rng=[0.0, 1.0],
    value=0.0,
    title="Deformation",
    pointa=(0.02, 0.9),
    pointb=(0.3, 0.9),
)

input("Press enter to close program")