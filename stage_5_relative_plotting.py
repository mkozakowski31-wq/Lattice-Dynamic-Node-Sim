from tkinter import W
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

def compute_barycentric(A, B, C, P):
    """
    Returns (u, v, w) such that P ≈ u*A + v*B + w*C
    Uses v0 = B-A, v1 = C-A so weights map directly to (A, B, C).
    """
    v0 = B - A  
    v1 = C - A  
    v2 = P - A

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None

    v = (d11 * d20 - d01 * d21) / denom   # weight of B
    w = (d00 * d21 - d01 * d20) / denom   # weight of C
    u = 1.0 - v - w                        # weight of A

    return np.array([u, v, w])             # now correctly [weight_A, weight_B, weight_C]


def find_contact_points(surface, intersect_points, tolerance):
    surface = surface.triangulate().clean()
    faces = surface.faces.reshape(-1, 4)[:, 1:]
    pts = surface.points

    contact_points  = []
    bary_records    = []
    contact_indices = []  

    for pt_idx, pt in enumerate(intersect_points):
        cell_id = surface.find_closest_cell(pt)
        tri = faces[cell_id]
        A, B, C = pts[tri[0]], pts[tri[1]], pts[tri[2]]

        normal = np.cross(B - A, C - A)
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        dist = np.dot(pt - A, normal)

        if abs(dist) <= tolerance:
            proj = pt - dist * normal
            bary = compute_barycentric(A, B, C, proj)

            if bary is not None:
                contact_points.append(pt)
                contact_indices.append(pt_idx)  # <-- store original index
                bary_records.append({
                    'barycentric':       bary,
                    'triangle_centroid': (A + B + C) / 3.0,
                    'triangle_normal':   normal,
                    'projected_point':   proj
                })

    return np.array(contact_points), bary_records, contact_indices

def reconstruct_on_surface(target_surface, bary_records):
    target_surface = target_surface.triangulate().clean()
    faces = target_surface.faces.reshape(-1, 4)[:, 1:]
    pts = target_surface.points

    centroids = pts[faces].mean(axis=1)
    normals = np.cross(
        pts[faces[:, 1]] - pts[faces[:, 0]],
        pts[faces[:, 2]] - pts[faces[:, 0]]
    )
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-12)

    centroid_tree = cKDTree(centroids)

    reconstructed = []
    for rec in bary_records:
        src_centroid = rec['triangle_centroid']
        src_normal = rec['triangle_normal']
        bary = rec['barycentric']

        k = min(20, len(centroids))
        _, idxs = centroid_tree.query(src_centroid, k=k)

        best_idx = idxs[np.argmax(normals[idxs] @ src_normal)]

        tri = faces[best_idx]
        A, B, C = pts[tri[0]], pts[tri[1]], pts[tri[2]]

        reconstructed_pt = bary[0] * A + bary[1] * B + bary[2] * C
        reconstructed.append(reconstructed_pt)

    return np.array(reconstructed)
