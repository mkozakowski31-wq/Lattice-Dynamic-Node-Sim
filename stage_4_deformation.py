
import numpy as np
import pyvista as pv
import vtk

from stage_1_boundary import EdgeSolver, resample_curve_equal, reorder_curve
from stage_3_lattice import collect_lattice_segments_along_geodesic, segment_lengths

def sphere_curve_intersection(center_pt, radius, mesh):
    sphere = vtk.vtkSphere()
    sphere.SetCenter(center_pt)
    sphere.SetRadius(radius)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(sphere)
    cutter.SetInputData(mesh)
    cutter.Update()

    intersection = pv.wrap(cutter.GetOutput())
    if intersection.n_points == 0:
        print(f"Warning: no intersection at {center_pt} radius {radius}")
        return None

    pts        = intersection.points
    center_pt  = np.array(center_pt)

    # project points onto plane perpendicular to mesh normal at center
    # then sort by angle around center
    vecs    = pts - center_pt
    
    # build local 2D coordinate frame at center
    u       = vecs[0] / np.linalg.norm(vecs[0])
    normal  = np.cross(u, vecs[1])
    normal /= np.linalg.norm(normal)
    v       = np.cross(normal, u)

    # compute angle of each point in local frame
    angles  = np.arctan2(np.dot(vecs, v), np.dot(vecs, u))
    order   = np.argsort(angles)

    ordered_pts = pts[order]

    return ordered_pts

def direction_between(d1, d2):
    d1 = np.array(d1, dtype=float)
    d2 = np.array(d2, dtype=float)

    # normalize inputs
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)

    mid = d1 + d2

    # handle opposite directions (would cancel to zero)
    if np.linalg.norm(mid) < 1e-8:
        raise ValueError("Directions are opposite; no unique between-direction")

    return mid / np.linalg.norm(mid)

def geodesic_shoot(mesh, start_pt, direction, length, step=1.0):
    if mesh.point_normals is None:
        mesh.compute_normals(inplace=True)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    pts = [np.array(start_pt, dtype=float)]

    p = pts[0].copy()
    d = direction / np.linalg.norm(direction)

    traveled = 0.0

    while traveled < length:

        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)

        locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)

        p = np.array(closest)

        cell = mesh.get_cell(int(cell_id))
        vids = cell.point_ids
        N = mesh.point_normals[vids].mean(axis=0)
        N /= np.linalg.norm(N)

        d = d - np.dot(d, N) * N
        d /= np.linalg.norm(d)

        p = p + d * step
        pts.append(p.copy())

        traveled += step

    return pv.lines_from_points(np.array(pts))

def segments_on_polyline_forward(start_pt, polyline_pts, lengths, eps=1e-12):
    curve = np.asarray(polyline_pts, dtype=float)

    pts = [np.asarray(start_pt, float)]

    # find closest index on curve
    idx = np.argmin(np.linalg.norm(curve - start_pt, axis=1))

    for L in lengths:

        center = pts[-1]
        found = False

        # search ONLY forward along curve
        for i in range(idx, len(curve) - 1):

            a = curve[i]
            b = curve[i + 1]
            d = b - a

            A = np.dot(d, d)

            # skip degenerate segments (zero length)
            if A < eps:
                continue

            B = 2 * np.dot(d, a - center)
            C = np.dot(a - center, a - center) - L**2

            disc = B*B - 4*A*C

            # no real intersection
            if disc < 0:
                continue

            sqrt_disc = np.sqrt(disc)

            t1 = (-B + sqrt_disc) / (2*A)
            t2 = (-B - sqrt_disc) / (2*A)

            # check forward intersections only
            for t in (t1, t2):
                if 0.0 <= t <= 1.0:
                    new_pt = a + t * d
                    pts.append(new_pt)
                    idx = i  # continue forward from here
                    found = True
                    break

            if found:
                break

        if not found:
            print("No forward intersection found for length", L)
            break

    return np.array(pts)

def polylineIntersect(polyA, polyB, percent_error=0.1):

    A = np.asarray(polyA)
    B = np.asarray(polyB)

    def avg_seg_len(P):
        if len(P) < 2:
            return 0.0
        return np.mean(np.linalg.norm(P[1:] - P[:-1], axis=1))

    scale = max(avg_seg_len(A), avg_seg_len(B))
    tol = scale * (percent_error / 100.0)

    # --- segment-to-segment distance ---
    def seg_dist(p1, p2, q1, q2):
        u = p2 - p1
        v = q2 - q1
        w0 = p1 - q1

        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)

        denom = a*c - b*b

        if denom < 1e-12:
            # segments are parallel
            s = 0.0
            t = (b > c and d/b) or (e/c)
        else:
            s = (b*e - c*d) / denom
            t = (a*e - b*d) / denom

        s = np.clip(s, 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0)

        closest_p = p1 + s * u
        closest_q = q1 + t * v

        return np.linalg.norm(closest_p - closest_q)
    for i in range(len(A) - 1):
        p1, p2 = A[i], A[i + 1]

        for j in range(len(B) - 1):
            q1, q2 = B[j], B[j + 1]

            if seg_dist(p1, p2, q1, q2) <= tol:
                return True

    return False


def calculate_line_angle(mesh, startPt,lengths, sEdge, seg_bound_lineS, seg_bound_lineC):
    new_polyArr = []
    print(lengths)
    print(len(lengths))
    for x in range(25):
        seg_dir = direction_between(seg_bound_lineS, seg_bound_lineC)
        geo_trace = geodesic_shoot(mesh, start_pt=startPt, direction=seg_dir,length=50,step=0.01)
        new_poly = segments_on_polyline_forward(start_pt=startPt, polyline_pts=geo_trace.points, lengths=lengths)
        intersection = polylineIntersect((pv.lines_from_points(new_poly)).points, sEdge.points, percent_error=0.1)
        if intersection == True:
            print('contact')
            seg_bound_lineC = seg_dir
        else: 
            print('no contact')
            seg_bound_lineS = seg_dir
    new_polyArr.extend(new_poly)
            
    return geo_trace

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

    def Z_SegTopographicalDeformation(self, Geolines, z_segments, idx):
        segments_on_curve = []
        z_segment = z_segments[idx]
        gl = z_segment.PolyDatArr[0]
        if z_segment.StartLineX == True:
            pts_along_gx, straight_segs = collect_lattice_segments_along_geodesic(gl, Geolines.Geo_linesY)
            lengths = segment_lengths(straight_segs)
            seg_dir = self.corners[1] - self.tip_pts[int(z_segment.TipPt)-1]
            sEdge = self.leadEdge
        else: 
            pts_along_gx, straight_segs = collect_lattice_segments_along_geodesic(gl, Geolines.Geo_linesX)
            lengths = segment_lengths(straight_segs)
            seg_dir = self.corners[2] - self.tip_pts[int(z_segment.TipPt)-1]
            sEdge = self.trailEdge
        print("Total lattice length:", lengths.sum())

        if z_segment.TipPt == 1: 
            seg_dir = self.corners[2] - self.corners[1]
            CDir = seg_dir / np.linalg.norm(seg_dir)
            seg_dir = self.corners[0]-self.corners[1]
            SDir = seg_dir / np.linalg.norm(seg_dir)

            shot = calculate_line_angle(self.mesh, self.tip_pts[int(z_segment.TipPt)-1], lengths, sEdge, SDir, CDir)
            segments_on_curve.append(segments_on_polyline_forward(self.tip_pts[int(z_segment.TipPt)-1], shot.points, lengths))
        elif z_segment.TipPt == len(self.tip_pts):
            seg_dir = self.corners[1] - self.corners[2]
            CDir = seg_dir / np.linalg.norm(seg_dir)
            seg_dir = self.corners[3]-self.corners[2]
            SDir = seg_dir / np.linalg.norm(seg_dir)

            shot = calculate_line_angle(self.mesh, self.tip_pts[int(z_segment.TipPt)-1], lengths, sEdge, SDir, CDir)
            segments_on_curve.append(segments_on_polyline_forward(self.tip_pts[int(z_segment.TipPt)-1], shot.points, lengths))
        else:
            print("No Corner")
            CDir = seg_dir / np.linalg.norm(seg_dir)
            intersectionPts = resample_curve_equal(sphere_curve_intersection(self.tip_pts[int(z_segment.TipPt)-1],1, self.mesh),50)
            seg_dir = intersectionPts[(int(len(intersectionPts)/2))] - self.tip_pts[int(z_segment.TipPt)-1]
            SDir = seg_dir / np.linalg.norm(seg_dir)

            shot = calculate_line_angle(self.mesh, self.tip_pts[int(z_segment.TipPt)-1], lengths, sEdge, SDir, CDir)
            segments_on_curve.append(segments_on_polyline_forward(self.tip_pts[int(z_segment.TipPt)-1], shot.points, lengths))
        print("_________________________")
        print(segments_on_curve[-1][-1])
        print("__________________")
        # i = 1
        # if sEdge == self.leadEdge:
        #     print("running this if to create ")
        #     sEdge = self.trailEdge
        #     pts_along_gx, straight_segs = collect_lattice_segments_along_geodesic(z_segment.PolyDatArr[i], Geolines.Geo_linesY)
        #     lengths = segment_lengths(straight_segs)

        #     seg_dir = self.corners[2] - self.corners[1]
        #     CDir = seg_dir / np.linalg.norm(seg_dir)
        #     seg_dir = self.corners[0]-self.corners[1]
        #     SDir = seg_dir / np.linalg.norm(seg_dir)

        #     shot = calculate_line_angle(self.mesh, segments_on_curve[-1][-1], lengths, sEdge, SDir, CDir)
        #     segments_on_curve.append(segments_on_polyline_forward(self.tip_pts[int(z_segment.TipPt)-1], shot.points, lengths))
        # else:
        #     sEdge = self.leadEdge
        #     pts_along_gx, straight_segs = collect_lattice_segments_along_geodesic(z_segment.PolyDatArr[i], Geolines.Geo_linesY)
        #     lengths = segment_lengths(straight_segs)

        #     seg_dir = self.corners[1] - self.corners[2]
        #     CDir = seg_dir / np.linalg.norm(seg_dir)
        #     seg_dir = self.corners[3]-self.corners[2]
        #     SDir = seg_dir / np.linalg.norm(seg_dir)

        #     shot = calculate_line_angle(self.mesh, segments_on_curve[-1][-1], lengths, sEdge, SDir, CDir)
        #     segments_on_curve.append(segments_on_polyline_forward(self.tip_pts[int(z_segment.TipPt)-1], shot.points, lengths))
        #     sEdge == self.leadEdge

        return straight_segs, shot, segments_on_curve
