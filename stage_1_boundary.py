import numpy as np
import pyvista as pv
from collections import Counter


def load_meshes(obj_path_contract, obj_path_extended):
    mesh = pv.read(obj_path_contract)
    mesh_extended = pv.read(obj_path_extended)

    mesh = mesh.triangulate()
    visMesh = mesh
    mesh = mesh.clean(tolerance=1e-6)

    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    mesh_extended = mesh_extended.clean(tolerance=1e-6)
    points_ex = mesh_extended.points
    faces_ex = mesh_extended.faces.reshape(-1, 4)[:, 1:]

    visEdges = visMesh.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False
    )

    return mesh, mesh_extended, points, faces, points_ex, faces_ex, visEdges

def EdgeLength(edges, points):
    edges = np.asarray(edges)
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    return np.linalg.norm(p1 - p0, axis=1).sum()

def reorder_curve(points, pointA, pointB):
    points = np.asarray(points)
    pointA = np.asarray(pointA)
    pointB = np.asarray(pointB)
    direction  = pointB - pointA
    direction /= np.linalg.norm(direction)
    projections = np.dot(points - pointA, direction)
    return points[np.argsort(projections)]

def resample_curve_equal(points, N):
    points = np.asarray(points)
    if N < 2:
        raise ValueError("N must be >= 2")
    segs = points[1:] - points[:-1]
    lens = np.linalg.norm(segs, axis=1)
    total_len = lens.sum()
    if total_len == 0:
        return np.repeat(points[:1], N, axis=0)
    target_s = np.linspace(0.0, total_len, N)
    cum_len = np.concatenate([[0.0], np.cumsum(lens)])
    out = np.zeros((N, 3))
    seg_idx  = 0
    for i, s in enumerate(target_s):
        while seg_idx < len(lens) - 1 and cum_len[seg_idx + 1] < s:
            seg_idx += 1
        seg_len = lens[seg_idx]
        if seg_len == 0:
            out[i] = points[seg_idx]
        else:
            t      = (s - cum_len[seg_idx]) / seg_len
            out[i] = points[seg_idx] + t * segs[seg_idx]
    return out

def EdgeSolver(n_origin_shift, n_root, n_lead, n_tip, boundary_dir, points, faces):
    edges = []
    for a, b, c in faces:
        edges.append(tuple(sorted((a, b))))
        edges.append(tuple(sorted((b, c))))
        edges.append(tuple(sorted((c, a))))

    edge_counts = Counter(edges)
    boundary_edges = [e for e, count in edge_counts.items() if count == 1]
    boundary_edges = np.array(boundary_edges)
    boundary_vertices = np.unique(boundary_edges.flatten())

    print("Boundary edges:",len(boundary_edges))
    print("Boundary vertices:",len(boundary_vertices))

    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    start = min(boundary_vertices, key=lambda i: points[i, 0])
    ordered_vertices = [start]
    prev, current = None, start

    while True:
        neighbors = adj[current]
        next_v = neighbors[0] if neighbors[0] != prev else neighbors[1]
        if next_v == start:
            break
        ordered_vertices.append(next_v)
        prev, current = current, next_v

    ordered_vertices = np.array(ordered_vertices)

    if boundary_dir == -1:
        ordered_vertices = ordered_vertices[::-1]

    shift = n_origin_shift % len(ordered_vertices)
    ordered_vertices = np.roll(ordered_vertices, -shift)

    ordered_edges = [(ordered_vertices[i], ordered_vertices[(i + 1) % len(ordered_vertices)])for i in range(len(ordered_vertices))]

    print("Ordered boundary vertices:", len(ordered_vertices))

    N = len(ordered_edges)
    n_trail = N - (n_root + n_lead + n_tip)
    if n_trail <= 0:
        raise ValueError("Segment sizes exceed boundary length")

    root_edges = ordered_edges[0:n_root]
    lead_edges = ordered_edges[n_root:n_root + n_lead]
    tip_edges = ordered_edges[n_root + n_lead:n_root + n_lead + n_tip]
    trail_edges = ordered_edges[n_root + n_lead + n_tip:]

    root_pts = points[np.unique(np.array(root_edges).flatten())]
    lead_pts = points[np.unique(np.array(lead_edges).flatten())]
    tip_pts = points[np.unique(np.array(tip_edges).flatten())]
    trail_pts = points[np.unique(np.array(trail_edges).flatten())]

    leadEdge = pv.lines_from_points(lead_pts)
    trailEdge = pv.lines_from_points(trail_pts)

    L_R = ordered_vertices[n_root]
    L_T = ordered_vertices[n_root + n_lead]
    T_T = ordered_vertices[n_root + n_lead + n_tip]
    T_R = ordered_vertices[0]

    junction_points = points[np.array([L_R, L_T, T_T, T_R])]

    return (root_edges, lead_edges, tip_edges, trail_edges, root_pts, lead_pts, tip_pts, trail_pts, junction_points, leadEdge, trailEdge)

def Resampler(root_pts, tip_pts, lead_pts, trail_pts, root_edges, tip_edges, lead_edges, trail_edges,junction_points, points, size):
    root_len = EdgeLength(root_edges, points)
    tip_len = EdgeLength(tip_edges, points)
    lead_len = EdgeLength(lead_edges, points)
    trail_len = EdgeLength(trail_edges, points)

    if root_len >= tip_len:
        VCcount = int(np.ceil(root_len/size))
    else:
        VCcount = int(np.ceil(root_len/size))

    if lead_len >= trail_len:
        VWcount = int(np.ceil(lead_len/size))
    else:
        VWcount = int(np.ceil(trail_len/size))

    root_pts  = resample_curve_equal(reorder_curve(root_pts, junction_points[3], junction_points[0]), VCcount)
    tip_pts   = resample_curve_equal(reorder_curve(tip_pts, junction_points[1], junction_points[2]), VCcount)
    lead_pts  = resample_curve_equal(reorder_curve(lead_pts, junction_points[0], junction_points[1]), VWcount)
    trail_pts = resample_curve_equal(reorder_curve(trail_pts, junction_points[2], junction_points[3]), VWcount)

    return root_pts, tip_pts, lead_pts, trail_pts, VWcount, VCcount


def updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
              junction_points, visEdges, visEd, clear):
    if clear:
        p.clear()
    p.add_mesh(mesh, color="red", opacity=0.8)
    if visEd:
        p.add_mesh(visEdges, color="blue", line_width=1)
    p.add_points(root_pts, color="red", point_size=12, render_points_as_spheres=True)
    p.add_points(lead_pts, color="blue", point_size=12, render_points_as_spheres=True)
    p.add_points(tip_pts, color="green", point_size=12, render_points_as_spheres=True)
    p.add_points(trail_pts, color="orange", point_size=12, render_points_as_spheres=True)
    p.add_points(junction_points[0], color="pink",  point_size=20, render_points_as_spheres=True)
    p.add_points(junction_points[1], color="teal",  point_size=20, render_points_as_spheres=True)
    p.add_points(junction_points[2], color="black", point_size=20, render_points_as_spheres=True)
    p.add_points(junction_points[3], color="white", point_size=20, render_points_as_spheres=True)


def define_boundaries(p, mesh, points, faces, visEdges,
                      n_origin_shift, n_root, n_lead, n_tip, boundary_dir, size):
    """Interactive terminal loop. Returns fully solved + resampled boundary data."""

    (root_edges, lead_edges, tip_edges, trail_edges, root_pts, lead_pts, tip_pts, trail_pts,
     junction_points, leadEdge, trailEdge) = EdgeSolver(n_origin_shift, n_root, n_lead, n_tip, boundary_dir, points, faces)

    updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
              junction_points, visEdges, visEd=True, clear=True)

    while True:
        print("\nAdjust edge parameters:")
        print(f"shift={n_origin_shift}, root={n_root}, lead={n_lead}, "
              f"tip={n_tip}, boundary direction={boundary_dir}")

        cmd = input("Enter: shift root lead tip boundary  OR  'continue'  OR  'help'  OR  'latticesize'\n> ").strip()

        if cmd.lower() == "continue":
            break

        elif cmd.lower() == "help":
            print("")
            print("Each index defines the number of vertices in each boundary edge")
            print("Adjust boundary direction (Counter vs. Clockwise) with 1 or -1")
            print("Root=Red, Lead=Blue, Tip=Green, Trail=Orange")
            print("Corners: Root_Lead=Pink, Lead_Tip=Teal, Tip_Trail=Black, Trail_Root=White")
            input("Press a key to return")

        elif cmd.lower() == "latticesize":
            print("WARNING: Entering a lattice size too large may result in error")
            while True:
                print("Current lattice size = " + str(size))
                cmd1 = input("Enter Lattice size: ")
                try:
                    size = float(cmd1)
                    print(f"Valid input. Lattice size: {str(size)}")
                    break
                except:
                    print("Not a float, try again.")
        else:
            try:
                n_origin_shift, n_root, n_lead, n_tip, boundary_dir = map(int, cmd.split())
            except ValueError:
                print("Invalid input. Example: -102 103 501 72 1 or continue")
                continue

            (root_edges, lead_edges, tip_edges, trail_edges,
             root_pts, lead_pts, tip_pts, trail_pts,
             junction_points, leadEdge, trailEdge) = EdgeSolver(
                n_origin_shift, n_root, n_lead, n_tip, boundary_dir, points, faces)

            updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
                      junction_points, visEdges, visEd=True, clear=True)

    (root_pts, tip_pts, lead_pts, trail_pts, VWcount, VCcount) = Resampler(
        root_pts, tip_pts, lead_pts, trail_pts,
        root_edges, tip_edges, lead_edges, trail_edges,
        junction_points, points, size)

    print(f"Verify count, root vertices: {len(root_pts)}, tip vertices: {len(tip_pts)}")
    print(f"Verify count, lead vertices: {len(lead_pts)}, trail vertices: {len(trail_pts)}")

    return (root_edges, lead_edges, tip_edges, trail_edges, root_pts, lead_pts, tip_pts, trail_pts,junction_points,
            leadEdge, trailEdge, VWcount, VCcount, n_origin_shift, n_root, n_lead, n_tip, boundary_dir, size)