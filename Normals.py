import pyvista as pv

# Load mesh
mesh = pv.read("/Users/marko/Documents/GitHub/marko/Models/LatticeTestbenchPYplotTaper.obj")
mesh = mesh.triangulate().clean()

# Compute vertex normals (THIS replaces OBJ vn safely)
mesh = mesh.compute_normals(
    point_normals=True,
    cell_normals=False,
    auto_orient_normals=True,
    consistent_normals=True
)

# Create glyphs (arrows) for normals
arrows = mesh.glyph(
    orient="Normals",
    scale=False,
    factor=5.0  # adjust size if needed
)

# Plot
p = pv.Plotter()
p.add_mesh(mesh, color="lightgray", opacity=0.4)
p.add_mesh(arrows, color="blue")
p.show()