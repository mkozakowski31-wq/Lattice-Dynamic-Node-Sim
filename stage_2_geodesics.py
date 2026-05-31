import numpy as np
import pyvista as pv
import vtk
from tqdm import tqdm
from joblib import Parallel, delayed
import time


def pn_triangle_point(P, N, bary):
    u, v, w = bary
    b300 = P[0]; b030 = P[1]; b003 = P[2]
    b210 = (2*P[0] + P[1] - np.dot(P[1]-P[0], N[0]) * N[0]) / 3
    b120 = (2*P[1] + P[0] - np.dot(P[0]-P[1], N[1]) * N[1]) / 3
    b021 = (2*P[1] + P[2] - np.dot(P[2]-P[1], N[1]) * N[1]) / 3
    b012 = (2*P[2] + P[1] - np.dot(P[1]-P[2], N[2]) * N[2]) / 3
    b102 = (2*P[2] + P[0] - np.dot(P[0]-P[2], N[2]) * N[2]) / 3
    b201 = (2*P[0] + P[2] - np.dot(P[2]-P[0], N[0]) * N[0]) / 3
    E = (b210 + b120 + b021 + b012 + b102 + b201) / 6
    V = (P[0] + P[1] + P[2]) / 3
    b111 = E + (E - V) / 2
    return (
        b300*u**3 + b030*v**3 + b003*w**3 +
        3*(b210*u**2*v + b120*u*v**2 +
           b021*v**2*w + b012*v*w**2 +
           b102*w**2*u + b201*w*u**2) +
        6*b111*u*v*w
    )


def smooth_geodesic(mesh, start, end, n_points=50, iters=50):
    path = np.linspace(start, end, n_points)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    for _ in range(iters):
        for i in range(1, n_points - 1):
            p = path[i]
            closest_point = [0.0, 0.0, 0.0]
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.mutable(0.0)

            locator.FindClosestPoint(p, closest_point, cell_id, sub_id, dist2)

            cid = int(cell_id)
            cell = mesh.get_cell(cid)
            vids = cell.point_ids
            P = mesh.points[vids]
            N = mesh.point_normals[vids]

            v0, v1, v2 = P
            v = np.array(closest_point) - v0
            a = v1 - v0
            b = v2 - v0
            d00 = np.dot(a, a); d01 = np.dot(a, b); d11 = np.dot(b, b)
            d20 = np.dot(v, a); d21 = np.dot(v, b)
            denom = d00 * d11 - d01 * d01
            v_coord = (d11 * d20 - d01 * d21) / denom
            w_coord = (d00 * d21 - d01 * d20) / denom
            u_coord = 1.0 - v_coord - w_coord

            bary = np.clip([u_coord, v_coord, w_coord], 0.0, 1.0)
            bary /= bary.sum()
            path[i] = pn_triangle_point(P, N, bary)

    return path


def make_geo(start, end, mesh):
    curve = smooth_geodesic(mesh, start, end, n_points=60, iters=80)
    return pv.lines_from_points(curve)


def compute_geodesics(mesh, root_pts, tip_pts, lead_pts, trail_pts, VWcount, VCcount):
    mesh.compute_normals(inplace=True)

    start_time_geo = time.perf_counter()

    # X family
    geo_linesX_1 = Parallel(n_jobs=-1)(
        delayed(make_geo)(trail_pts[VWcount - 1 - x], root_pts[x - VCcount], mesh)
        for x in tqdm(range(VCcount), desc="Processing RootTrailX Geodesics")
    )
    geo_linesX_2 = Parallel(n_jobs=-1)(
        delayed(make_geo)(trail_pts[x], lead_pts[VWcount - x - VCcount], mesh)
        for x in tqdm(range(0, VWcount - VCcount), desc="Processing LeadTrailX Geodesics")
    )
    geo_linesX_3 = Parallel(n_jobs=-1)(
        delayed(make_geo)(tip_pts[x], lead_pts[VWcount - x - 1], mesh)
        for x in tqdm(range(VCcount), desc="Processing TipLeadX Geodesics")
    )
    geo_linesX = geo_linesX_1 + geo_linesX_2 + geo_linesX_3
    geo_lineX  = pv.merge(geo_linesX)

    span_dir = tip_pts.mean(axis=0) - root_pts.mean(axis=0)
    span_dir /= np.linalg.norm(span_dir)
    geo_linesX = sorted(geo_linesX, key=lambda c: np.dot(c.points.mean(axis=0), span_dir))

    # Y family
    geo_linesY_1 = Parallel(n_jobs=-1)(
        delayed(make_geo)(lead_pts[VCcount - y], root_pts[y - 1], mesh)
        for y in tqdm(range(VCcount, 0, -1), desc="Processing RootLeadY Geodesics")
    )
    geo_linesY_2 = Parallel(n_jobs=-1)(
        delayed(make_geo)(trail_pts[y + VCcount - 1], lead_pts[VWcount - y - 1], mesh)
        for y in tqdm(range(VWcount - VCcount), desc="Processing LeadTrailY Geodesics")
    )
    geo_linesY_3 = Parallel(n_jobs=-1)(
        delayed(make_geo)(tip_pts[VCcount - y - 1], trail_pts[y], mesh)
        for y in tqdm(range(VCcount), desc="Processing TipTrailY Geodesics")
    )
    geo_linesY = geo_linesY_1 + geo_linesY_2 + geo_linesY_3
    geo_lineY  = pv.merge(geo_linesY)

    span_dir  = tip_pts.mean(axis=0) - root_pts.mean(axis=0)
    span_dir /= np.linalg.norm(span_dir)
    geo_linesY = sorted(geo_linesY, key=lambda c: np.dot(c.points.mean(axis=0), span_dir))

    elapsed = time.perf_counter() - start_time_geo
    print(f"Execution of Geodesics took: {elapsed:.4f} seconds")

    return geo_linesX, geo_linesY, geo_lineX, geo_lineY
