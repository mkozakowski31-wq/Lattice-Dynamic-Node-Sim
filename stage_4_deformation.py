import numpy as np
import pyvista as pv
from tqdm import tqdm
from joblib import Parallel, delayed
import vtk

from stage_1_boundary import EdgeSolver, resample_curve_equal, reorder_curve, get_origin_coords

def find_triangulation_point(center1, center2, r1, r2, semi_dir, mesh_pts, mesh_faces, tol=1e-5, debug=False):
    c1 = np.asarray(center1, float)
    c2 = np.asarray(center2, float)
    semi_dir = np.asarray(semi_dir, float)

    if np.linalg.norm(semi_dir) > 0:
        semi_dir = semi_dir / np.linalg.norm(semi_dir)

    d = np.linalg.norm(c2 - c1)
    if d < 1e-12:
        if debug:
            print("Degenerate sphere centers")
        return c1

    u = (c2 - c1) / d
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h2 = r1*r1 - a*a
    h = np.sqrt(max(0.0, h2))
    circle_center = c1 + a*u

    if debug:
        print("circle_center:", circle_center)
        print("circle_radius:", h)

    candidates = []

    for tri in mesh_faces:
        v0, v1, v2 = mesh_pts[tri]

        d0 = np.dot(v0 - circle_center, u)
        d1 = np.dot(v1 - circle_center, u)
        d2 = np.dot(v2 - circle_center, u)

        edges = [(v0, v1, d0, d1),
                (v1, v2, d1, d2),
                (v2, v0, d2, d0),]

        seg_pts = []
        for p0, p1, s0, s1 in edges:
            if s0 * s1 < 0.0:
                t = s0 / (s0 - s1)
                seg_pts.append(p0 + t * (p1 - p0))

        if len(seg_pts) != 2:
            continue

        p0, p1 = seg_pts
        dvec = p1 - p0
        f = p0 - circle_center

        A = np.dot(dvec, dvec)
        B = 2 * np.dot(f, dvec)
        C = np.dot(f, f) - h * h
        disc = B * B - 4 * A * C

        if disc < 0:
            continue

        sqrt_disc = np.sqrt(disc)
        for t in ((-B - sqrt_disc) / (2 * A), (-B + sqrt_disc) / (2 * A)):
            if 0 <= t <= 1:
                p = p0 + t * dvec
                if np.dot(p - c1, semi_dir) > 0:
                    candidates.append(p)

    if len(candidates) == 0:
        if debug:
            print("No triangle intersection candidates")
        return circle_center

    candidates = np.array(candidates)
    expected = circle_center + semi_dir * h
    best = np.argmin(np.linalg.norm(candidates - expected, axis=1))
    return candidates[best]

def trianglationLengthEdge(cPt1, Length, edge_pts, Dir):
    tol = Length * 0.1
    dists = np.linalg.norm(edge_pts - cPt1, axis=1)
    candidates = edge_pts[np.abs(dists - Length) < tol]

    if len(candidates) == 0:
        for scale in [0.1, 0.2, 0.5]:
            candidates = edge_pts[np.abs(dists - Length) < Length * scale]
            if len(candidates) > 0:
                print(f"Warning: used wider tolerance {scale*100:.0f}%")
                break
        if len(candidates) == 0:
            print("Warning: no candidates found on edge within any tolerance")
            return None

    normal = np.asarray(Dir, dtype=float)
    normal /= np.linalg.norm(normal)
    dots = np.dot(candidates - cPt1, normal)
    filtered = candidates[dots > 0]

    if len(filtered) == 0:
        print("Warning: all candidates filtered out by half-space - check Dir")
        return None

    dists_filtered = np.linalg.norm(filtered - cPt1, axis=1)
    best = np.argmin(np.abs(dists_filtered - Length))
    return filtered[best]

def _compute_point_edge(i, intersect_ptsP, polyLenX, polyLenY, mesh_pts, mesh_faces, lead_pts, trail_pts, direction):
    if i == 0:
        p1 = trianglationLengthEdge(intersect_ptsP[0], polyLenY[i], lead_pts, direction)
        p2 = find_triangulation_point(intersect_ptsP[0], intersect_ptsP[1], polyLenX[0], polyLenY[1], direction, mesh_pts, mesh_faces)
        return [p1, p2]
    elif i == len(intersect_ptsP) - 2:
        p1 = find_triangulation_point(intersect_ptsP[-2], intersect_ptsP[-1], polyLenX[-2], polyLenY[-1], direction, mesh_pts, mesh_faces)
        p2 = trianglationLengthEdge(intersect_ptsP[-1], polyLenX[-1], trail_pts, direction)
        return [p1, p2]
    else:
        p = find_triangulation_point(intersect_ptsP[i + 1], intersect_ptsP[i], polyLenY[i+1], polyLenX[i], direction, mesh_pts, mesh_faces)
        return [p]

def _compute_point_normal(i, intersect_ptsP, polyLenX, polyLenY, mesh_pts, mesh_faces, direction):
    p = find_triangulation_point(intersect_ptsP[i + 1], intersect_ptsP[i], polyLenY[i], polyLenX[i], direction, mesh_pts, mesh_faces)
    return [p]

class DeformedMesh:
    def __init__(self, mesh_path, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VCcount, based_on_extrema):
        self.mesh = pv.read(mesh_path)
        self.mesh = self.mesh.clean(tolerance=1e-6)
        self.mesh.compute_normals(inplace=True)

        points = self.mesh.points
        faces = self.mesh.faces.reshape(-1, 4)[:, 1:]
        origin = get_origin_coords(mesh_path)

        (root_edges, lead_edges, tip_edges, trail_edges,
         root_pts, lead_pts, tip_pts, trail_pts,
         junction_points, leadEdge, trailEdge) = EdgeSolver(
            n_origin_shift, n_root, n_lead, n_tip,
            boundary_dir, points, faces, based_on_extrema, origin)

        self.tip_pts = resample_curve_equal(reorder_curve(tip_pts,  junction_points[1], junction_points[2]), VCcount)
        self.root_pts = resample_curve_equal(reorder_curve(root_pts, junction_points[3], junction_points[0]), VCcount)
        self.leadEdge = pv.lines_from_points(lead_pts)
        self.trailEdge = pv.lines_from_points(trail_pts)
        self.corners = junction_points

        self._mesh_pts = np.array(self.mesh.points)
        self._mesh_faces = self.mesh.faces.reshape(-1, 4)[:, 1:].copy()
        self._lead_pts = np.array(self.leadEdge.points)
        self._trail_pts = np.array(self.trailEdge.points)

        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(self.mesh)
        self._locator.BuildLocator()

    def Z_SegTopographicalDeformation(self, p, slicesY, slicesX):
        def SliceLengths(idx):
            polyLenX = []
            polyLenY = []
            for i in range(len(slicesX[idx].PolLengths)):
                arc = slicesX[idx].PolLengths[i].compute_arc_length()
                print(f"arc lengthX: {arc['arc_length'][-1]}")
                polyLenX.append(arc['arc_length'][-1])
            for i in range(len(slicesY[idx].PolLengths)):
                arc = slicesY[idx].PolLengths[i].compute_arc_length()
                print(f"arc lengthY: {arc['arc_length'][-1]}")
                polyLenY.append(arc['arc_length'][-1])
            return polyLenX, polyLenY

        direction = self.corners[0] - self.corners[1]
        direction = direction / np.linalg.norm(direction)

        mesh_pts = self._mesh_pts
        mesh_faces = self._mesh_faces
        lead_pts = self._lead_pts
        trail_pts = self._trail_pts

        intersect_pts = []
        intersect_ptsP = self.tip_pts

        isEdge = False
        for x in tqdm(range(len(slicesX) - 1, -1, -1), desc="Lattice Deformation: "):
            polyLenX, polyLenY = SliceLengths(x)
            n = len(intersect_ptsP) - 1

            # Freeze current row as a plain list so workers get a clean copy.
            ptsP_snap = list(intersect_ptsP)

            if isEdge:
                results = Parallel(n_jobs=-1)(
                    delayed(_compute_point_edge)(i, ptsP_snap, polyLenX, polyLenY,
                    mesh_pts, mesh_faces, lead_pts, trail_pts, direction)
                    for i in range(n)
                )
                isEdge = False
            else:
                results = Parallel(n_jobs=-1)(
                    delayed(_compute_point_normal)(i, ptsP_snap, polyLenX, polyLenY,
                        mesh_pts, mesh_faces, direction)
                    for i in range(n)
                )
                isEdge = True

            intersect_ptsC = [pt for sublist in results for pt in sublist]
            intersect_pts.extend(intersect_ptsC)
            intersect_ptsP = intersect_ptsC

        return np.array(intersect_pts)

    def sphereCheck(self, p, slicesY, slicesX):
        def SliceLengths(idx):
            polyLenX = []
            polyLenY = []
            for i in range(len(slicesX[idx].PolLengths)):
                arc = slicesX[idx].PolLengths[i].compute_arc_length()
                polyLenX.append(arc['arc_length'][-1])
            for i in range(len(slicesY[idx].PolLengths)):
                arc = slicesY[idx].PolLengths[i].compute_arc_length()
                polyLenY.append(arc['arc_length'][-1])
            return polyLenX, polyLenY

        spheres = []
        spheres2 = []
        polyLenX, polyLenY = SliceLengths(-1)

        for i in range(len(self.root_pts) - 1):
            spheres.append(pv.Sphere(radius=polyLenX[i], center=self.tip_pts[i],theta_resolution=100, phi_resolution=100))
            spheres2.append(pv.Sphere(radius=polyLenY[i], center=self.tip_pts[i],theta_resolution=100, phi_resolution=100))

        p.add_mesh(self.mesh)
        p.add_mesh(pv.merge(spheres),  style='wireframe', color="green")
        p.add_mesh(pv.merge(spheres2), style='wireframe', color="blue")
