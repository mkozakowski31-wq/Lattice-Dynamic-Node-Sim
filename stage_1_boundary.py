import os
import numpy as np
import pyvista as pv
from collections import Counter
from session_io import load_session, list_sessions

def get_origin_vertex_index(obj_path):
    current_group = None
    vertex_index = 0

    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('g '):
                current_group = line.split(maxsplit=1)[1]

            elif line.startswith('v '):
                if current_group == 'origin':
                    print(f"vertex_index: {vertex_index}")
                    return vertex_index
                vertex_index += 1

    raise ValueError("No vertex found in group 'origin'")

def get_origin_coords(obj_path):
    current_group = None
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('g '):
                current_group = line.split(maxsplit=1)[1]
            elif line.startswith('v ') and current_group == 'origin':
                coords = list(map(float, line.split()[1:]))
                return np.array(coords)
    raise ValueError("No vertex found in group 'origin'")


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

    visEdges = visMesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
    origin = get_origin_coords(obj_path_contract)
    return mesh, mesh_extended, points, faces, points_ex, faces_ex, visEdges, origin

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
    seg_idx = 0
    for i, s in enumerate(target_s):
        while seg_idx < len(lens) - 1 and cum_len[seg_idx + 1] < s:
            seg_idx += 1
        seg_len = lens[seg_idx]
        if seg_len == 0:
            out[i] = points[seg_idx]
        else:
            t = (s - cum_len[seg_idx]) / seg_len
            out[i] = points[seg_idx] + t * segs[seg_idx]
    return out

def EdgeSolver(n_origin_shift, n_root, n_lead, n_tip, boundary_dir, points, faces, based_on_extrema, origin):
    edges = []
    for a, b, c in faces:
        edges.append(tuple(sorted((a, b))))
        edges.append(tuple(sorted((b, c))))
        edges.append(tuple(sorted((c, a))))

    edge_counts = Counter(edges)
    boundary_edges = [e for e, count in edge_counts.items() if count == 1]
    boundary_edges = np.array(boundary_edges)
    boundary_vertices = np.unique(boundary_edges.flatten())

    print("Boundary edges:", len(boundary_edges))
    print("Boundary vertices:", len(boundary_vertices))

    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    
    if based_on_extrema == True:
        start = min(boundary_vertices, key=lambda i: points[i, 0])
    else:
        bv_coords = points[boundary_vertices]
        dists = np.linalg.norm(bv_coords - origin, axis=1)
        start = boundary_vertices[np.argmin(dists)]
    print(f"start: {str(start)}")
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

    ordered_edges = [
        (ordered_vertices[i], ordered_vertices[(i + 1) % len(ordered_vertices)])
        for i in range(len(ordered_vertices))
    ]

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

    return (root_edges, lead_edges, tip_edges, trail_edges,
            root_pts, lead_pts, tip_pts, trail_pts,
            junction_points, leadEdge, trailEdge)


def Resampler(root_pts, tip_pts, lead_pts, trail_pts,
              root_edges, tip_edges, lead_edges, trail_edges,
              junction_points, points, size, increase_lattice_stretch):
    root_len = EdgeLength(root_edges, points)
    tip_len = EdgeLength(tip_edges, points)
    lead_len = EdgeLength(lead_edges, points)
    trail_len = EdgeLength(trail_edges, points)

    VCcount = int(np.ceil(root_len / size))

    if lead_len >= trail_len:
        VWcount = int(np.ceil(lead_len / size))
    else:
        VWcount = int(np.ceil(trail_len / size))
    VWcount = int(VWcount // increase_lattice_stretch)

    root_pts = resample_curve_equal(reorder_curve(root_pts, junction_points[3], junction_points[0]), VCcount)
    tip_pts = resample_curve_equal(reorder_curve(tip_pts, junction_points[1], junction_points[2]), VCcount)
    lead_pts = resample_curve_equal(reorder_curve(lead_pts, junction_points[0], junction_points[1]), VWcount)
    trail_pts = resample_curve_equal(reorder_curve(trail_pts, junction_points[2], junction_points[3]), VWcount)

    return root_pts, tip_pts, lead_pts, trail_pts, VWcount, VCcount


def filter_meshes(p, all_obj_paths, n_origin_shift, n_root, n_lead, n_tip,
                  boundary_dir, based_on_extrema, excluded_indices,
                  stage_overrides=None):
    if stage_overrides is None:
        stage_overrides = {}
    COL_W = 10

    header = (f"{'Stage':<8} {'BndVerts':<{COL_W}} {'Trail':>{COL_W}} "
              f"{'Root':>{COL_W}} {'Lead':>{COL_W}} {'Tip':>{COL_W}} "
              f"{'Status':>{COL_W}} {'Excl':>{6}}")
    rows = []

    print(f"\n{'='*65}")
    print(f"  Boundary filter — {len(all_obj_paths)} meshes  "
          f"({len(excluded_indices)} currently excluded, "
          f"{len(stage_overrides)} with custom boundaries)")
    print(f"  Global: shift={n_origin_shift}  root={n_root}  lead={n_lead}  "
          f"tip={n_tip}  dir={boundary_dir}")
    print(f"  Controls:  Enter = next   m = mark/unmark   "
          f"b = edit boundary   r = reset boundary   q = quit filter")
    print(f"{'='*65}\n")

    for stage_idx, obj_path in enumerate(all_obj_paths):
        stage_label  = f"S{stage_idx}"
        short_path   = os.path.basename(obj_path)
        is_excluded  = stage_idx in excluded_indices
        excl_badge   = "  *** EXCLUDED ***" if is_excluded else ""

        # resolve effective boundary params (per-stage override wins) 
        ov = stage_overrides.get(stage_idx, {})
        s_shift = ov.get('n_origin_shift', n_origin_shift)
        s_root = ov.get('n_root', n_root)
        s_lead = ov.get('n_lead', n_lead)
        s_tip = ov.get('n_tip', n_tip)
        s_dir = ov.get('boundary_dir', boundary_dir)
        has_override = bool(ov)
        override_badge = "  [CUSTOM BOUNDARY]" if has_override else ""

        print(f"[{stage_label}]  {short_path}{excl_badge}{override_badge}")
        if has_override:
            print(f"  Override: shift={s_shift}  root={s_root}  lead={s_lead}  "
                  f"tip={s_tip}  dir={s_dir}")

        # load & clean mesh 
        load_ok = True
        try:
            raw = pv.read(obj_path).triangulate()
            clean = raw.clean(tolerance=1e-6)
            pts = clean.points
            fcs = clean.faces.reshape(-1, 4)[:, 1:]
            vis = raw.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
            origin_coords = get_origin_coords(obj_path)
        except Exception as exc:
            print(f"  LOAD ERROR: {exc}")
            load_ok = False

        # run EdgeSolver
        status = "FAIL"
        warn_str = ""
        diag_cols = f"{'-':<{COL_W}} {'-':>{COL_W}} {'-':>{COL_W}} {'-':>{COL_W}} {'-':>{COL_W}}"

        if load_ok:
            try:
                (root_edges, lead_edges, tip_edges, trail_edges,
                 root_pts, lead_pts, tip_pts, trail_pts,
                 junction_points, leadEdge, trailEdge) = EdgeSolver(
                    s_shift, s_root, s_lead, s_tip,
                    s_dir, pts, fcs, based_on_extrema, origin_coords)

                n_bnd = len(np.unique(np.array(root_edges + lead_edges +tip_edges  + trail_edges).flatten()))
                n_trail = len(trail_edges)

                warnings = []
                if n_trail <= 0:
                    warnings.append("trail<=0")
                if len(root_pts) < 2:
                    warnings.append("root<2pts")
                if len(lead_pts) < 2:
                    warnings.append("lead<2pts")
                if len(tip_pts) < 2:
                    warnings.append("tip<2pts")

                status = "WARN" if warnings else "OK"
                warn_str = "  [" + ", ".join(warnings) + "]" if warnings else ""
                diag_cols = (f"{n_bnd:<{COL_W}} {n_trail:>{COL_W}} "
                             f"{len(root_edges):>{COL_W}} {len(lead_edges):>{COL_W}} "
                             f"{len(tip_edges):>{COL_W}}")

                print(f"  boundary_verts={n_bnd}  trail={n_trail}  "
                      f"root={len(root_edges)}  lead={len(lead_edges)}  "
                      f"tip={len(tip_edges)}")
                if warnings:
                    print(f"  ⚠  {', '.join(warnings)}")

                # color the excluded mesh grey so the user sees it is inactive
                mesh_col = "grey" if is_excluded else "red"
                mesh_opa = 0.3   if is_excluded else 0.8
                p.clear()
                p.add_mesh(clean, color=mesh_col, opacity=mesh_opa)
                p.add_mesh(vis, color="blue", line_width=1)
                if not is_excluded:
                    p.add_points(root_pts, color="red", point_size=12, render_points_as_spheres=True)
                    p.add_points(lead_pts, color="blue", point_size=12, render_points_as_spheres=True)
                    p.add_points(tip_pts, color="green",point_size=12, render_points_as_spheres=True)
                    p.add_points(trail_pts, color="orange", point_size=12, render_points_as_spheres=True)
                    p.add_points(junction_points[0], color="pink", point_size=20, render_points_as_spheres=True)
                    p.add_points(junction_points[1], color="teal", point_size=20, render_points_as_spheres=True)
                    p.add_points(junction_points[2], color="black", point_size=20, render_points_as_spheres=True)
                    p.add_points(junction_points[3], color="white", point_size=20, render_points_as_spheres=True)

            except Exception as exc:
                print(f"  EdgeSolver FAILED: {exc}")
                warn_str = f"  [{exc}]"

        excl_col = "YES" if stage_idx in excluded_indices else "no"
        rows.append(f"{stage_label:<8} {diag_cols} {status:>{COL_W}} "
                    f"{excl_col:>{6}}{warn_str}  {short_path}")

        # helper: redraw viewer with current effective params 
        def _redraw(r_pts, le_pts, t_pts, tr_pts, jpts, clean_m, vis_m, excl):
            mesh_col = "grey" if excl else "red"
            mesh_opa = 0.3   if excl else 0.8
            p.clear()
            p.add_mesh(clean_m, color=mesh_col, opacity=mesh_opa)
            p.add_mesh(vis_m, color="blue", line_width=1)
            if not excl:
                p.add_points(r_pts, color="red", point_size=12, render_points_as_spheres=True)
                p.add_points(le_pts, color="blue", point_size=12, render_points_as_spheres=True)
                p.add_points(t_pts, color="green", point_size=12, render_points_as_spheres=True)
                p.add_points(tr_pts, color="orange", point_size=12, render_points_as_spheres=True)
                p.add_points(jpts[0], color="pink", point_size=20, render_points_as_spheres=True)
                p.add_points(jpts[1], color="teal", point_size=20, render_points_as_spheres=True)
                p.add_points(jpts[2], color="black", point_size=20, render_points_as_spheres=True)
                p.add_points(jpts[3], color="white", point_size=20, render_points_as_spheres=True)

        # per-mesh prompt
        last = stage_idx == len(all_obj_paths) - 1
        while True:
            if last:
                prompt = "  [Enter=finish | m=mark/unmark | b=edit boundary | r=reset boundary] > "
            else:
                prompt = "  [Enter=next | m=mark/unmark | b=edit boundary | r=reset boundary] > "
            action = input(prompt).strip().lower()

            if action == "m":
                if stage_idx in excluded_indices:
                    excluded_indices.discard(stage_idx)
                    print(f"  ✓  S{stage_idx} UN-marked — will be included in pipeline")
                    excl_col = "no"
                else:
                    excluded_indices.add(stage_idx)
                    print(f"  ✗  S{stage_idx} MARKED — will be excluded from pipeline")
                    excl_col = "YES"
                # Update the last row's exclusion column
                rows[-1] = (f"{stage_label:<8} {diag_cols} {status:>{COL_W}} "
                            f"{excl_col:>{6}}{warn_str}  {short_path}")
                # Update viewer tint if EdgeSolver had succeeded
                if load_ok and status != "FAIL":
                    is_excluded = stage_idx in excluded_indices
                    _redraw(root_pts, lead_pts, tip_pts, trail_pts,
                            junction_points, clean, vis, is_excluded)

            elif action == "b":
                # ── edit boundary params for this specific stage ─────────────
                ov_cur = stage_overrides.get(stage_idx, {})
                c_shift = ov_cur.get('n_origin_shift', n_origin_shift)
                c_root  = ov_cur.get('n_root',         n_root)
                c_lead  = ov_cur.get('n_lead',         n_lead)
                c_tip   = ov_cur.get('n_tip',          n_tip)
                c_dir   = ov_cur.get('boundary_dir',   boundary_dir)

                print(f"\n  ── Editing boundary for S{stage_idx} ──")
                print(f"     Global : shift={n_origin_shift}  root={n_root}  "
                      f"lead={n_lead}  tip={n_tip}  dir={boundary_dir}")
                print(f"     Current: shift={c_shift}  root={c_root}  "
                      f"lead={c_lead}  tip={c_tip}  dir={c_dir}")
                print(f"     Enter five integers: shift root lead tip boundary_dir")
                print(f"     (Press Enter with no input to keep current values)")

                while True:
                    raw = input(f"  S{stage_idx} boundary> ").strip()
                    if raw == "":
                        print("  No change.")
                        break
                    try:
                        vals = list(map(int, raw.split()))
                        if len(vals) != 5:
                            raise ValueError("Need exactly 5 integers")
                        c_shift, c_root, c_lead, c_tip, c_dir = vals

                        # Validate by running EdgeSolver immediately
                        if load_ok:
                            try:
                                (re2, le2, te2, tre2,
                                 rp2, lp2, tp2, trp2,
                                 jp2, _, _) = EdgeSolver(
                                    c_shift, c_root, c_lead, c_tip,
                                    c_dir, pts, fcs, based_on_extrema, origin_coords)

                                stage_overrides[stage_idx] = {
                                    'n_origin_shift': c_shift, 'n_root': c_root,
                                    'n_lead': c_lead, 'n_tip': c_tip,
                                    'boundary_dir': c_dir,
                                }
                                # Update local working vars for the current mesh display
                                root_edges, lead_edges, tip_edges, trail_edges = re2, le2, te2, tre2
                                root_pts, lead_pts, tip_pts, trail_pts = rp2, lp2, tp2, trp2
                                junction_points = jp2

                                is_excluded = stage_idx in excluded_indices
                                _redraw(root_pts, lead_pts, tip_pts, trail_pts, junction_points, clean, vis, is_excluded)

                                n_bnd2 = len(np.unique(np.array(root_edges + lead_edges + tip_edges  + trail_edges).flatten()))
                                n_trail2 = len(trail_edges)
                                print(f"  ✓  Override applied — "
                                      f"bnd_verts={n_bnd2}  trail={n_trail2}  "
                                      f"root={len(root_edges)}  lead={len(lead_edges)}  "
                                      f"tip={len(tip_edges)}")
                                # Rebuild diag_cols / status for updated summary row
                                diag_cols = (f"{n_bnd2:<{COL_W}} {n_trail2:>{COL_W}} "
                                             f"{len(root_edges):>{COL_W}} "
                                             f"{len(lead_edges):>{COL_W}} "
                                             f"{len(tip_edges):>{COL_W}}")
                                status = "OK"
                                warn_str = "  [CUSTOM]"
                                rows[-1] = (f"{stage_label:<8} {diag_cols} "
                                             f"{status:>{COL_W}} "
                                             f"{excl_col:>{6}}{warn_str}  {short_path}")
                                break

                            except Exception as exc:
                                print(f"  EdgeSolver failed with those params: {exc}")
                                print("  Try again or press Enter to cancel.")
                        else:
                            # Mesh didn't load — store the override speculatively
                            stage_overrides[stage_idx] = {
                                'n_origin_shift': c_shift, 'n_root': c_root,
                                'n_lead':         c_lead,  'n_tip':  c_tip,
                                'boundary_dir':   c_dir,
                            }
                            print("  Override stored (mesh load had failed — cannot validate live).")
                            break

                    except ValueError as ve:
                        print(f"  Invalid input ({ve}). Example: -102 103 501 72 1")

            elif action == "r":
                # ── reset this stage to global boundary params ───────────────
                if stage_idx in stage_overrides:
                    stage_overrides.pop(stage_idx)
                    print(f"  ↺  S{stage_idx} boundary reset to global params")

                    if load_ok:
                        try:
                            (root_edges, lead_edges, tip_edges, trail_edges,
                             root_pts, lead_pts, tip_pts, trail_pts,
                             junction_points, _, _) = EdgeSolver(
                                n_origin_shift, n_root, n_lead, n_tip,
                                boundary_dir, pts, fcs, based_on_extrema, origin_coords)

                            is_excluded = stage_idx in excluded_indices
                            _redraw(root_pts, lead_pts, tip_pts, trail_pts,
                                    junction_points, clean, vis, is_excluded)

                            n_bnd_r = len(np.unique(
                                np.array(root_edges + lead_edges +
                                         tip_edges  + trail_edges).flatten()))
                            n_trail_r = len(trail_edges)
                            diag_cols = (f"{n_bnd_r:<{COL_W}} {n_trail_r:>{COL_W}} "
                                         f"{len(root_edges):>{COL_W}} "
                                         f"{len(lead_edges):>{COL_W}} "
                                         f"{len(tip_edges):>{COL_W}}")
                            status = "OK"
                            warn_str = ""
                            rows[-1] = (f"{stage_label:<8} {diag_cols} "
                                        f"{status:>{COL_W}} "
                                        f"{excl_col:>{6}}{warn_str}  {short_path}")
                        except Exception as exc:
                            print(f"  EdgeSolver failed after reset: {exc}")
                else:
                    print(f"  S{stage_idx} has no boundary override — nothing to reset.")

            else:
                break   # Enter — advance

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FILTER SUMMARY  —  {len(excluded_indices)} mesh(es) excluded, "
          f"{len(stage_overrides)} with custom boundaries")
    print(f"{'='*65}")
    print(header)
    print('-' * (len(header) + 4))
    for r in rows:
        print(r)
    if excluded_indices:
        print(f"\n  Excluded stages: {sorted(excluded_indices)}")
        print("  These will be removed from the pipeline on 'continue'.")
    if stage_overrides:
        print(f"\n  Custom boundary overrides:")
        for si, ov in sorted(stage_overrides.items()):
            print(f"    S{si}: shift={ov['n_origin_shift']}  root={ov['n_root']}  "
                  f"lead={ov['n_lead']}  tip={ov['n_tip']}  dir={ov['boundary_dir']}")
    print(f"{'='*65}\n")

    return excluded_indices, stage_overrides


def updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
              junction_points, visEdges, visEd, clear):
    if clear:
        p.clear()
    p.add_mesh(mesh, color="red", opacity=0.8)
    if visEd:
        p.add_mesh(visEdges, color="blue", line_width=1)
    p.add_points(root_pts,  color="red",    point_size=12, render_points_as_spheres=True)
    p.add_points(lead_pts,  color="blue",   point_size=12, render_points_as_spheres=True)
    p.add_points(tip_pts,   color="green",  point_size=12, render_points_as_spheres=True)
    p.add_points(trail_pts, color="orange", point_size=12, render_points_as_spheres=True)
    p.add_points(junction_points[0], color="pink",  point_size=20, render_points_as_spheres=True)
    p.add_points(junction_points[1], color="teal",  point_size=20, render_points_as_spheres=True)
    p.add_points(junction_points[2], color="black", point_size=20, render_points_as_spheres=True)
    p.add_points(junction_points[3], color="white", point_size=20, render_points_as_spheres=True)


def define_boundaries(p, mesh, points, faces, origin, visEdges,
                      n_origin_shift, n_root, n_lead, n_tip,
                      boundary_dir, size, increase_lattice_stretch, based_on_extrema,
                      all_obj_paths=None):

    # Full-session payload — populated only when the user runs 'load'
    _loaded = False
    _slicesY = None
    _slicesX = None
    _lattice_nodes = None
    _excluded_indices = set()   # indices into all_obj_paths marked for removal
    _stage_overrides = {}      # per-stage boundary parameter overrides

    # Boundary geometry from session (bypasses Resampler when present)
    _loaded_root_pts = None
    _loaded_tip_pts = None
    _loaded_lead_pts = None
    _loaded_trail_pts = None
    _loaded_junction_pts = None
    _loaded_VWcount = None
    _loaded_VCcount = None
    _loaded_size = None

    (root_edges, lead_edges, tip_edges, trail_edges,
     root_pts, lead_pts, tip_pts, trail_pts,
     junction_points, leadEdge, trailEdge) = EdgeSolver(
        n_origin_shift, n_root, n_lead, n_tip, boundary_dir, points, faces, based_on_extrema, origin)

    updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
              junction_points, visEdges, visEd=True, clear=True)

    while True:
        print("\nAdjust edge parameters:")
        print(f"shift={n_origin_shift}, root={n_root}, lead={n_lead}, "
              f"tip={n_tip}, boundary direction={boundary_dir}")
        if _loaded:
            print("  [full session loaded — 'continue' will skip geodesics & lattice]")
        if _excluded_indices:
            print(f"  [excluded stages: {sorted(_excluded_indices)}]")
        if _stage_overrides:
            print(f"  [custom boundaries on stages: {sorted(_stage_overrides.keys())}]")

        cmd = input(
            "Enter: shift root lead tip boundary  |  'continue'  |  "
            "'load'  |  'latticesize'  |  'filter'  |  'help'\n> "
        ).strip()

        if cmd.lower() == "continue":
            break

        elif cmd.lower() == "load":
            saved = list_sessions()
            if saved:
                print("\nAvailable sessions:")
                for i, f in enumerate(saved):
                    print(f"  [{i}] {f}")
                shortcut = input("Enter a list index to pick one, or type a full path: ").strip()
                path = (saved[int(shortcut)]
                if shortcut.isdigit() and int(shortcut) < len(saved)
                else shortcut)
            else:
                print("No sessions found in sessions/ folder.")
                path = input("Enter full path to a .pkl session file: ").strip()

            if not path:
                print("Load cancelled.")
                continue
            if not os.path.isfile(path):
                print(f"File not found: {path}")
                continue

            try:
                sess = load_session(path)

                # Store full lattice payload
                _slicesY = sess["slicesY"]
                _slicesX = sess["slicesX"]
                _lattice_nodes = sess["lattice_nodes"]

                # Restore boundary params so the viewer reflects the session
                n_origin_shift = sess["n_origin_shift"]
                n_root = sess["n_root"]
                n_lead = sess["n_lead"]
                n_tip = sess["n_tip"]
                boundary_dir = sess["boundary_dir"]
                _loaded = True

                # Restore filter state
                _excluded_indices = set(sess.get("excluded_indices", []))
                _stage_overrides = {int(k): dict(v)
                                     for k, v in sess.get("stage_overrides", {}).items()}
                if _excluded_indices:
                    print(f"  → {len(_excluded_indices)} excluded stage(s) restored: "
                          f"{sorted(_excluded_indices)}")
                if _stage_overrides:
                    print(f"  → {len(_stage_overrides)} custom boundary override(s) restored")

                # Restore pre-computed resampled boundary geometry (new sessions only)
                if sess.get("root_pts") is not None:
                    _loaded_root_pts = np.asarray(sess["root_pts"])
                    _loaded_tip_pts = np.asarray(sess["tip_pts"])
                    _loaded_lead_pts = np.asarray(sess["lead_pts"])
                    _loaded_trail_pts = np.asarray(sess["trail_pts"])
                    _loaded_junction_pts = np.asarray(sess["junction_points"])
                    _loaded_VWcount = sess["VWcount"]
                    _loaded_VCcount = sess["VCcount"]
                    _loaded_size = float(sess["size"])
                    print(f"  → Boundary geometry restored from session  "
                          f"(root={len(_loaded_root_pts)}, tip={len(_loaded_tip_pts)}, "
                          f"lead={len(_loaded_lead_pts)}, trail={len(_loaded_trail_pts)})")
                else:
                    print("  → Old session format — Resampler will rerun on 'continue'")

                print(f"  → shift={n_origin_shift}, root={n_root}, lead={n_lead}, "
                        f"tip={n_tip}, dir={boundary_dir}  (VCcount={sess['VCcount']})")

                (root_edges, lead_edges, tip_edges, trail_edges,
                    root_pts, lead_pts, tip_pts, trail_pts,
                    junction_points, leadEdge, trailEdge) = EdgeSolver(
                    n_origin_shift, n_root, n_lead, n_tip,
                    boundary_dir, points, faces, based_on_extrema, origin)

                updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
                            junction_points, visEdges, visEd=True, clear=True)

            except Exception as exc:
                print(f"Failed to load session: {exc}")
       # ── help ──────────────────────────────────────────────────────────
        elif cmd.lower() == "help":
            print("")
            print("Each index defines the number of vertices in each boundary edge")
            print("Adjust boundary direction (Counter vs. Clockwise) with 1 or -1")
            print("Root=Red, Lead=Blue, Tip=Green, Trail=Orange")
            print("Corners: Root_Lead=Pink, Lead_Tip=Teal, Tip_Trail=Black, Trail_Root=White")
            print("'load'   — restore a full saved session (skips geodesics & lattice on 'continue')")
            print("'filter' — cycle through all stage meshes and validate current boundary params")
            print("           In filter: 'm' marks/unmarks a stage for exclusion")
            print("                      'b' edits boundary params for that specific stage only")
            print("                      'r' resets a stage back to the global boundary params")
            input("Press a key to return")

        elif cmd.lower() == "filter":
            if not all_obj_paths:
                print("No mesh paths provided to filter against. "
                      "Pass all_obj_paths to define_boundaries().")
            else:
                _excluded_indices, _stage_overrides = filter_meshes(
                    p, all_obj_paths,
                    n_origin_shift, n_root, n_lead, n_tip,
                    boundary_dir, based_on_extrema,
                    _excluded_indices, _stage_overrides)
                # Restore the primary mesh view after filter finishes
                updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
                          junction_points, visEdges, visEd=True, clear=True)

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
            # Any manual edit invalidates a previously loaded session
            _loaded = False
            _slicesY = _slicesX = _lattice_nodes = None
            _loaded_root_pts = _loaded_tip_pts = _loaded_lead_pts = None
            _loaded_trail_pts = _loaded_junction_pts = None
            _loaded_VWcount = _loaded_VCcount = _loaded_size = None
            try:
                n_origin_shift, n_root, n_lead, n_tip, boundary_dir = map(int, cmd.split())
            except ValueError:
                print("Invalid input. Example: -102 103 501 72 1 or continue")
                continue

            (root_edges, lead_edges, tip_edges, trail_edges,
             root_pts, lead_pts, tip_pts, trail_pts,
             junction_points, leadEdge, trailEdge) = EdgeSolver(
                n_origin_shift, n_root, n_lead, n_tip, boundary_dir, points, faces, based_on_extrema, origin)

            updateGeo(p, mesh, root_pts, lead_pts, tip_pts, trail_pts,
                      junction_points, visEdges, visEd=True, clear=True)

    if _loaded_root_pts is not None:
        # Resampled boundary restored from session — skip Resampler entirely
        root_pts = _loaded_root_pts
        tip_pts = _loaded_tip_pts
        lead_pts = _loaded_lead_pts
        trail_pts = _loaded_trail_pts
        junction_points = _loaded_junction_pts
        VWcount = _loaded_VWcount
        VCcount = _loaded_VCcount
        size = _loaded_size
        print(f"  [session] Boundary reuse — skipping Resampler")
    else:
        (root_pts, tip_pts, lead_pts, trail_pts, VWcount, VCcount) = Resampler(
            root_pts, tip_pts, lead_pts, trail_pts,
            root_edges, tip_edges, lead_edges, trail_edges,
            junction_points, points, size, increase_lattice_stretch)

    print(f"Verify count, root vertices: {len(root_pts)}, tip vertices: {len(tip_pts)}")
    print(f"Verify count, lead vertices: {len(lead_pts)}, trail vertices: {len(trail_pts)}")

    return (root_edges, lead_edges, tip_edges, trail_edges,
            root_pts, lead_pts, tip_pts, trail_pts,
            junction_points, leadEdge, trailEdge,
            VWcount, VCcount,
            n_origin_shift, n_root, n_lead, n_tip, boundary_dir, size,
            _loaded, _slicesY, _slicesX, _lattice_nodes,
            _excluded_indices, _stage_overrides)

