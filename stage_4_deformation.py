
import numpy as np
import pyvista as pv
import vtk

from stage_1_boundary import EdgeSolver, resample_curve_equal, reorder_curve
from stage_3_lattice import collect_lattice_segments_along_geodesic, segment_lengths

class DeformedMesh:
    def __init__(self, mesh_path, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount):
        # load mesh
        self.mesh = pv.read(mesh_path)
        self.mesh = self.mesh.clean(tolerance=1e-6)
        self.mesh.compute_normals(inplace=True)

        # extract boundary
        points = self.mesh.points
        faces  = self.mesh.faces.reshape(-1, 4)[:, 1:]

        (root_edges, lead_edges, tip_edges, trail_edges,
         root_pts, lead_pts, tip_pts, trail_pts,
         junction_points, leadEdge, trailEdge) = EdgeSolver(
            n_origin_shift, n_root, n_lead, n_tip,
            boundary_dir, points, faces)

        self.tip_pts  = resample_curve_equal(reorder_curve(tip_pts, junction_points[1], junction_points[2]), VCcount)
        self.leadEdge  = pv.lines_from_points(lead_pts)
        self.trailEdge = pv.lines_from_points(trail_pts)
        self.corners = junction_points

        # build locator once
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(self.mesh)
        self._locator.BuildLocator()

    def Z_SegTopographicalDeformation(self, Geolines, idx):
        print("67")
        x = 5
        return x