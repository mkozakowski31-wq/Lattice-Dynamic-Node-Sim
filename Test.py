import pyvista as pv
import numpy as np
s1 = pv.Sphere(phi_resolution=500, theta_resolution=500)
s2 = s1.copy()
s2.points += np.array([0.25, 0, 0])
intersection, s1_split, s2_split = s1.intersection(s2)
pl = pv.Plotter()
_ = pl.add_mesh(intersection, color='r', line_width=10)
pl.show()