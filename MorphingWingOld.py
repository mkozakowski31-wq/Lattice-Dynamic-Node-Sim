import pyvista as pv
import numpy as np

base = pv.read("Models/LatticeTestbenchPYplotTaper.obj")
ext  = pv.read("Models/ExtendedLatticeTestbenchPYplotTaper.obj")

if base.n_points != ext.n_points:
    raise ValueError("Vertex count mismatch")
if base.n_points == ext.n_points:
    print("67")
# OPTIONAL but recommended
if base.n_cells != ext.n_cells:
    raise ValueError("Topology mismatch")