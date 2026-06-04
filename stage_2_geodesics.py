import numpy as np
import pyvista as pv
from collections import Counter, defaultdict
from tqdm import tqdm
import potpourri3d as pp3d
import time

def stitch_boundary_points_into_mesh(mesh, tip_pts, trail_pts, root_pts, lead_pts):
    V = np.asarray(mesh.points, dtype=float).copy()
    F = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int64).copy()

    groups = {'tip': np.asarray(tip_pts, float),
              'trail': np.asarray(trail_pts, float),
              'root': np.asarray(root_pts, float),
              'lead': np.asarray(lead_pts, float)}

    V, F, idx_map = _stitch_arrays(V, F, groups)

    faces_vtk = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()
    return pv.PolyData(V, faces_vtk), idx_map


def _boundary_edges(F):
    cnt = Counter()
    for tri in F:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for e in ((a, b), (b, c), (c, a)):
            cnt[tuple(sorted(e))] += 1
    return [e for e, c in cnt.items() if c == 1]


def _point_segment_dist(p, a, b):
    ab = b - a
    L2 = ab.dot(ab)
    if L2 == 0:
        return np.linalg.norm(p - a), 0.0
    t = np.clip((p - a).dot(ab) / L2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab)), t


def _nearest_boundary_edge(pt, bedges, V):
    best, best_d, best_t = None, np.inf, 0.0
    for (a, b) in bedges:
        d, t = _point_segment_dist(pt, V[a], V[b])
        if d < best_d:
            best_d, best, best_t = d, (a, b), t
    return best, best_t, best_d


def _stitch_arrays(V, F, groups):
    F = [list(map(int, tri)) for tri in F]
    bedges = _boundary_edges(np.array(F))

    per_edge = defaultdict(list)
    idx_map = {name: [None] * len(pts) for name, pts in groups.items()}

    for name, pts in groups.items():
        for li, pt in enumerate(pts):
            edge, _, _ = _nearest_boundary_edge(pt, bedges, V)
            a, b = tuple(sorted(edge))
            _, t_sorted = _point_segment_dist(pt, V[a], V[b])
            per_edge[(a, b)].append((t_sorted, pt, name, li))

    for (a, b), entries in per_edge.items():
        entries.sort(key=lambda e: e[0])              # order along a->b

        chain = [a]
        for (t, pt, name, li) in entries:
            m = len(V)
            V = np.vstack([V, pt])
            idx_map[name][li] = m
            chain.append(m)
        chain.append(b)

        ti = next((fi for fi, tri in enumerate(F) if a in tri and b in tri), None)
        if ti is None:
            raise RuntimeError(f"No triangle found on boundary edge {(a, b)}")
        tri = F[ti]
        c = next(x for x in tri if x not in (a, b))
        a_to_b = (tri[(tri.index(a) + 1) % 3] == b)    # winding direction

        F.pop(ti)
        seq = chain if a_to_b else chain[::-1]
        for s0, s1 in zip(seq[:-1], seq[1:]):
            F.append([s0, s1, c])

    F = np.array(F, dtype=np.int64)
    idx_map = {name: np.array(v, dtype=np.int64) for name, v in idx_map.items()}
    return V, F, idx_map

def make_geo(start, end, mesh, solver, corner_tol):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    si = int(mesh.find_closest_point(start))
    ei = int(mesh.find_closest_point(end))
    
    if si == ei or np.linalg.norm(end - start) < corner_tol:
        return pv.lines_from_points(np.vstack([start, end]))

    pts = solver.find_geodesic_path(v_start=si, v_end=ei)
    if pts is None or len(pts) < 2:
        # solver returned nothing usable — fall back to straight segment
        return pv.lines_from_points(np.vstack([start, end]))

    return pv.lines_from_points(pts)


def compute_geodesics(mesh, root_pts, tip_pts, lead_pts, trail_pts, VWcount, VCcount):
    mesh.compute_normals(inplace=True)

    V = np.asarray(mesh.points, dtype=np.float64)
    F = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int64)
    solver = pp3d.EdgeFlipGeodesicSolver(V, F)

    bnd = _boundary_edges(F)
    med_edge = np.median([np.linalg.norm(V[a] - V[b]) for a, b in bnd])
    corner_tol = 0.25 * med_edge

    start_time_geo = time.perf_counter()

    #X family 
    geo_linesX_1 = [
        make_geo(trail_pts[VWcount - 1 - x], root_pts[x - VCcount], mesh, solver, corner_tol)
        for x in tqdm(range(VCcount), desc="Processing RootTrailX Geodesics")
    ]
    geo_linesX_2 = [
        make_geo(trail_pts[x], lead_pts[VWcount - x - VCcount], mesh, solver, corner_tol)
        for x in tqdm(range(0, VWcount - VCcount), desc="Processing LeadTrailX Geodesics")
    ]
    geo_linesX_3 = [
        make_geo(tip_pts[x], lead_pts[VWcount - x - 1], mesh, solver, corner_tol)
        for x in tqdm(range(VCcount-1), desc="Processing TipLeadX Geodesics")
    ]
    geo_linesX = geo_linesX_1 + geo_linesX_2 + geo_linesX_3
    geo_lineX = pv.merge(geo_linesX)

    span_dir = tip_pts.mean(axis=0) - root_pts.mean(axis=0)
    span_dir /= np.linalg.norm(span_dir)
    geo_linesX = sorted(geo_linesX, key=lambda c: np.dot(c.points.mean(axis=0), span_dir))

    # Y family
    geo_linesY_1 = [
        make_geo(lead_pts[VCcount - y], root_pts[y - 1], mesh, solver, corner_tol)
        for y in tqdm(range(VCcount, 0, -1), desc="Processing RootLeadY Geodesics")
    ]
    geo_linesY_2 = [
        make_geo(trail_pts[y + VCcount - 1], lead_pts[VWcount - y - 1], mesh, solver, corner_tol)
        for y in tqdm(range(VWcount - VCcount), desc="Processing LeadTrailY Geodesics")
    ]
    geo_linesY_3 = [
        make_geo(tip_pts[VCcount - y - 1], trail_pts[y], mesh, solver, corner_tol)
        for y in tqdm(range(VCcount-1), desc="Processing TipTrailY Geodesics")
    ]
    geo_linesY = geo_linesY_1 + geo_linesY_2 + geo_linesY_3
    geo_lineY = pv.merge(geo_linesY)

    span_dir = tip_pts.mean(axis=0) - root_pts.mean(axis=0)
    span_dir /= np.linalg.norm(span_dir)
    geo_linesY = sorted(geo_linesY, key=lambda c: np.dot(c.points.mean(axis=0), span_dir))

    elapsed = time.perf_counter() - start_time_geo
    print(f"Execution of Geodesics took: {elapsed:.4f} seconds")

    return geo_linesX, geo_linesY, geo_lineX, geo_lineY
