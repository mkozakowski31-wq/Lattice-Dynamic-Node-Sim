(root_edges_ex, lead_edges_ex, tip_edges_ex, trail_edges_ex,
    root_pts_ex, lead_pts_ex, tip_pts_ex, trail_pts_ex,
    junction_points_ex, leadEdge_ex, trailEdge_ex) = EdgeSolver(
    n_origin_shift, n_root, n_lead, n_tip, boundary_dir,
    points_ex, faces_ex)

root_pts_ex = resample_curve_equal(reorder_curve(root_pts_ex, junction_points_ex[3], junction_points_ex[0]), VCcount)
tip_pts_ex  = resample_curve_equal(reorder_curve(tip_pts_ex,  junction_points_ex[1], junction_points_ex[2]), VCcount)

updateGeo(p, mesh_extended, root_pts_ex, lead_pts_ex, tip_pts_ex, trail_pts_ex,
            junction_points_ex, visEdges, visEd=False, clear=False)
