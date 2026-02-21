import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import time

from stage_1_boundary import (EdgeSolver, resample_curve_equal, reorder_curve, updateGeo)
from stage_4_deformation import ZSequence, ZStart

def sample_polyline(points, n=200):
    t  = np.linspace(0, 1, len(points))
    ti = np.linspace(0, 1, n)
    return np.vstack([np.interp(ti, t, points[:, k]) for k in range(3)]).T

def curve_curve_closest_points(curveA, curveB):
    tree = cKDTree(curveB)
    dists, idx = tree.query(curveA)
    i = np.argmin(dists)
    return curveA[i], curveB[idx[i]]

def collect_lattice_segments_along_geodesic(geodesic, opposing_geodesics, samples=300):
    g_s = sample_polyline(geodesic.points, samples)
    intersection_points = []

    for opp in opposing_geodesics:
        opp_s = sample_polyline(opp.points, samples)
        pA, _ = curve_curve_closest_points(g_s, opp_s)
        intersection_points.append(pA)

    intersection_points = np.asarray(intersection_points)
    tree   = cKDTree(g_s)
    _, idx = tree.query(intersection_points)
    order  = np.argsort(idx)
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

    polyConnectY_mesh = pv.merge(polyConnectY)
    polyConnectX_mesh = pv.merge(polyConnectX)
    lattice_nodes = np.asarray(lattice_nodes)

    elapsed_str = time.perf_counter() - start_time_str
    print(f"Execution of Calculating Straight Lines took: {elapsed_str:.4f} seconds")

    gx = geo_linesX[9]
    pts_along_gx, straight_segs = collect_lattice_segments_along_geodesic(gx, geo_linesY)
    lengths = segment_lengths(straight_segs)
    print("Total lattice length:", lengths.sum())
    geo_lines_seqX = []
    geo_lines_seqY = []
    z_sequences = []

    for f in range(1, len(tip_pts)+1, 1):
        x=0
        geo_lines_seqX = []
        if f != len(tip_pts):
            Corner = False
            if f == 1:
                Corner = True
            geo_lines_seqX.append(geo_linesX[VWcount+VCcount-f])
            while (VWcount-f)-(x*VCcount)+x >= 0:
                if x%2 == 0:
                    geo_lines_seqX.append(geo_linesY[(VWcount-f)-(x*VCcount)+x])
                else:
                    geo_lines_seqX.append(geo_linesX[(VWcount-f)-(x*VCcount)+x])
                x+= 1
            z_sequences.append(ZSequence(StartLineX= True,TipPt= f, PolyDatArr=geo_lines_seqX, TopCorner= Corner, BotCorner= False))
        x = 0
        geo_lines_seqY = []
        if f != 1:
            Corner = False
            if f == len(tip_pts):
                Corner = True
            geo_lines_seqY.append(geo_linesY[VWcount+f-1])
            while (VWcount-f)-(x*VCcount)+x >= 0:
                if x%2 == 0:
                    geo_lines_seqY.append(geo_linesX[(VWcount+f)-((x+1)*VCcount)+x-1])           
                else:
                    geo_lines_seqY.append(geo_linesY[(VWcount+f)-((x+1)*VCcount)+x-1])
                x+= 1
            z_sequences.append(ZSequence(StartLineX= False, TipPt= f, PolyDatArr=geo_lines_seqY, TopCorner= False, BotCorner= Corner))

    print("lenght" + str(len(z_sequences)))


    geo_lines_seqX = []
    geo_lines_seqY = []

    for seq in z_sequences:
        if seq.StartLineX == True:
            geo_lines_seqX.extend(seq.PolyDatArr)
        else:
            geo_lines_seqY.extend(seq.PolyDatArr)
    
    geo_lines_seqX = pv.merge(geo_lines_seqX)
    geo_lines_seqY = pv.merge(geo_lines_seqY)

    # ---- Final visualisation — contracted mesh ----
    updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
              junction_points, visEdges, visEd=False, clear=True)

    p.add_mesh(geo_lines_seqY, color="yellow", line_width=7)
    p.add_mesh(geo_lines_seqX, color="blue", line_width=7)

    # ---- Add lattice to viewer ----
    p.add_points(pts_along_gx, color="yellow", point_size=10)
    p.add_mesh(pv.merge(straight_segs), color="purple", line_width=4)
    p.add_points(lattice_nodes, color="cyan", point_size=6, render_points_as_spheres=True)
    p.add_mesh(leadEdge, color="blue", line_width=10)
    p.add_mesh(trailEdge, color="orange", line_width=10)
    p.add_mesh(polyConnectY_mesh, line_width=3, color="black")
    p.add_mesh(polyConnectX_mesh, line_width=3, color="gray")

    input("Press enter to close viewer")
