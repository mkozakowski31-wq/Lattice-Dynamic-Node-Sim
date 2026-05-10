from dataclasses import dataclass
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import time

from stage_1_boundary import updateGeo

def sample_polyline(points, n=200):
    t = np.linspace(0, 1, len(points))
    ti = np.linspace(0, 1, n)
    return np.vstack([np.interp(ti, t, points[:, k]) for k in range(3)]).T

def curve_curve_closest_points(curveA, curveB):
    # --- coarse search ---
    tree = cKDTree(curveB)
    dists, idx = tree.query(curveA)
    i_coarse = int(np.argmin(dists))
    j_coarse = int(idx[i_coarse])

    # --- build arc-length parameterisations over a local window ---
    def local_window(curve, center, half=3):
        lo = max(0, center - half)
        hi = min(len(curve) - 1, center + half)
        return curve[lo:hi + 1], lo

    segA, loA = local_window(curveA, i_coarse)
    segB, loB = local_window(curveB, j_coarse)

    def arclen_params(seg):
        """Return cumulative arc-length t values normalised to [0,1]."""
        deltas = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(deltas)])
        total = cum[-1]
        return cum / total if total > 0 else cum

    tA = arclen_params(segA)
    tB = arclen_params(segB)

    def interp_seg(seg, t_nodes, t):
        """Linearly interpolate a polyline segment at parameter t in [0,1]."""
        t = np.clip(t, 0.0, 1.0)
        k = np.searchsorted(t_nodes, t, side='right') - 1
        k = np.clip(k, 0, len(seg) - 2)
        dt = t_nodes[k + 1] - t_nodes[k]
        alpha = (t - t_nodes[k]) / dt if dt > 0 else 0.0
        return seg[k] + alpha * (seg[k + 1] - seg[k])

    # --- for each t on A, find the closest t on B, then minimise over A ---
    def dist_at_tA(t):
        pA = interp_seg(segA, tA, t)
        # inner minimisation: best t on B for this point on A
        res = minimize_scalar(
            lambda s: np.linalg.norm(pA - interp_seg(segB, tB, s)),
            bounds=(0.0, 1.0), method='bounded',
            options={'xatol': 1e-7}
        )
        return res.fun

    res_outer = minimize_scalar(
        dist_at_tA,
        bounds=(0.0, 1.0), method='bounded',
        options={'xatol': 1e-7}
    )

    t_best_A = res_outer.x
    pA_best = interp_seg(segA, tA, t_best_A)

    res_inner = minimize_scalar(
        lambda s: np.linalg.norm(pA_best - interp_seg(segB, tB, s)),
        bounds=(0.0, 1.0), method='bounded',
        options={'xatol': 1e-7}
    )
    pB_best = interp_seg(segB, tB, res_inner.x)

    return pA_best, pB_best

def collect_lattice_segments_along_geodesic(geodesic, opposing_geodesics, samples=300):
    g_s = sample_polyline(geodesic.points, samples)
    intersection_points = []

    for opp in opposing_geodesics:
        opp_s = sample_polyline(opp.points, samples)
        pA, _ = curve_curve_closest_points(g_s, opp_s)
        intersection_points.append(pA)

    intersection_points = np.asarray(intersection_points)
    tree = cKDTree(g_s)
    _, idx = tree.query(intersection_points)
    order = np.argsort(idx)
    ordered_points = intersection_points[order]

    segments = []
    for i in range(len(ordered_points) - 1):
        seg_pts = np.vstack([ordered_points[i], ordered_points[i + 1]])
        segments.append(pv.lines_from_points(seg_pts))

    return ordered_points, segments

def segment_lengths(segments):
    lengths = []
    for seg in segments:
        pts = seg.points if hasattr(seg, "points") else np.asarray(seg)
        if len(pts) < 2:
            continue
        L = np.linalg.norm(pts[1] - pts[0])
        if L > 0:
            lengths.append(L)
    return np.asarray(lengths)

def build_lattice(p, mesh, mesh_extended, geo_linesX, geo_linesY, root_pts, lead_pts, tip_pts, trail_pts,junction_points, leadEdge, trailEdge,
                  points_ex, faces_ex, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, VWcount, VCcount, visEdges):
    # ---- Build lattice nodes + straight struts ----
    start_time_str = time.perf_counter()
    lattice_nodes = []
    polyConnectY = []
    polyConnectX = []

    for cx in tqdm(geo_linesX, desc="Processing Y Inter-Lattice straight lines:"):
        cx_s = sample_polyline(cx.points, 300)
        connectY = []
        for cy in geo_linesY:
            cy_s = sample_polyline(cy.points, 300)
            pA, _  = curve_curve_closest_points(cx_s, cy_s)
            lattice_nodes.append(pA)
            connectY.append(pA)
        for l in range(len(connectY) - 1):
            Ypts = np.vstack([connectY[l], connectY[l + 1]])
            polyConnectY.append(pv.lines_from_points(Ypts))

    for cy in tqdm(geo_linesY, desc="Processing X Inter-Lattice Straight lines:"):
        cy_s = sample_polyline(cy.points, 300)
        connectX = []
        for cx in geo_linesX:
            cx_s = sample_polyline(cx.points, 300)
            pA, _ = curve_curve_closest_points(cy_s, cx_s)
            connectX.append(pA)
        for l in range(len(connectX) - 1):
            Xpts = np.vstack([connectX[l], connectX[l + 1]])
            polyConnectX.append(pv.lines_from_points(Xpts))

    polyConnectY = [seg for seg in polyConnectY if seg.length > 1e-4]
    polyConnectX = [seg for seg in polyConnectX if seg.length > 1e-4]

    polyConnectY_mesh = pv.merge(polyConnectY)
    polyConnectX_mesh = pv.merge(polyConnectX)
    lattice_nodes = np.asarray(lattice_nodes)
    
    elapsed_str = time.perf_counter() - start_time_str

    def lattice_index(r, c):
        maxSeg = (len(tip_pts) - 1) * 2
        k = c // 2
        if r + k < len(tip_pts) - 1:
            indexF = (r + k)**2 + r + 3*k + (c % 2)
        elif r + k <= len(lead_pts) - 2 :
            cI = (len(tip_pts) - 2) - r
            if c % 2:
                cI = (2 * cI) + 1
            else:
                cI = 2 * cI
            k = cI // 2
            indexF = (r + k)**2 + r + 3*k + (cI % 2) + (maxSeg * (c - cI)//2)
        else:
            newR = len(lead_pts) - 2 - k
            initial_index = lattice_index(newR,c) + 2*(maxSeg-1)
            firstDif = 2*(len(tip_pts)-3)
            for i in range((r-newR-1)):
                initial_index += firstDif
                firstDif -= 2

            indexF = initial_index
        
        return int(indexF)

    @dataclass
    class SlicedLenghts:
        isforward:bool
        PolLengths: list


    # ---- Final visualisation — contracted mesh ----
    updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
              junction_points, visEdges, visEd=False, clear=True)

    # ---- Add lattice to viewer ----
    # def label_segments(p, segments, color="black"):
    #     midpoints = np.array([(seg.points[0] + seg.points[1]) / 2 for seg in segments])
    #     labels = [str(i) for i in range(len(segments))]
    #     p.add_point_labels(midpoints, labels, font_size=8, text_color=color,
    #                     fill_shape=False, margin=0, always_visible=True)

    # label_segments(p, polyConnectY, color="yellow")
    # label_segments(p, polyConnectX, color="cyan")
    p.add_points(lattice_nodes, color="cyan", point_size=6, render_points_as_spheres=True)
    p.add_mesh(leadEdge, color="blue", line_width=3)
    p.add_mesh(trailEdge, color="orange", line_width=3)
    p.add_mesh(polyConnectY_mesh, line_width=3, color="yellow")
    polyConnectArr = []
    print(len(lead_pts))
    print(len(tip_pts))
    slicesX = []
    for x in tqdm(range(0, len(lead_pts)*2 - 2, 1), desc="Calculating X slices: "):
        sliceArray = []
        for y in range(0, len(tip_pts)-1, 1):
            polyC = lattice_index(y, x)
            sliceArray.append(polyConnectX[polyC])
            if x % 2 == 0:
                isforward = False
            else:
                isforward = True
        slicesX.append(SlicedLenghts(PolLengths=sliceArray, isforward=isforward))

    slicesY = []
    for x in tqdm(range(0, len(lead_pts)*2 - 2, 1), desc="Calculating Y slices: "):
        sliceArray = []
        for y in range(0, len(tip_pts)-1, 1):
            li = lattice_index(y, x)
            # print(f"x: {str(x)}")
            # print(f"y: {str(y)}")
            # print(f"lattice index: {str(li)}")
            sliceArray.append(polyConnectY[li])
            if x % 2 == 0:
                isforward = False
            else:
                isforward = True
        sliceArray.reverse()
        slicesY.append(SlicedLenghts(PolLengths=(sliceArray), isforward=isforward))
    

    p.add_mesh(polyConnectX_mesh, line_width=3, color="blue")

    return slicesY, slicesX, lattice_nodes
