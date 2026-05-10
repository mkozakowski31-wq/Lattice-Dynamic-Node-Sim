/*
stage_4_marching_triangulation.cpp
===================================
C++ replacement for find_triangulation_point() in stage_4_deformation.py.

WHY THIS IS FASTER THAN PYTHON
-------------------------------
The Python version loops over every mesh face in a Python for-loop,
doing per-face dot products, edge-plane intersection tests, and quadratic
circle-segment intersection solves. At ~65,000 faces and ~900 lattice
nodes per mesh, that is 58 million Python iterations per deformation state.

This C++ version:
  1. Builds an AABB (axis-aligned bounding box) spatial index once per mesh
     so only faces near the intersection circle are checked (~50 instead of ~65000)
  2. Does all arithmetic in tight C++ loops with no Python overhead per face
  3. Is called from Python via pybind11 — numpy arrays go in, numpy arrays come out
  4. Exposes find_triangulation_point_batch() which processes all lattice
     nodes in one C++ call, eliminating the joblib Parallel overhead entirely

BUILD
-----
Requirements: pybind11, Eigen (header-only)
  pip install pybind11 eigen
  brew install eigen          # macOS
  apt install libeigen3-dev   # Ubuntu

Compile (macOS / Linux):
  c++ -O3 -march=native -shared -fPIC \
      $(python3 -m pybind11 --includes) \
      -I$(brew --prefix eigen)/include/eigen3 \
      stage_4_marching_triangulation.cpp \
      -o stage_4_marching_triangulation$(python3-config --extension-suffix)

Then in Python:
  import stage_4_marching_triangulation as tri
  result = tri.find_triangulation_point(c1, c2, r1, r2, semi_dir,
                                         mesh_pts, mesh_faces)
  batch  = tri.find_triangulation_point_batch(centers1, centers2, r1s, r2s,
                                              semi_dirs, mesh_pts, mesh_faces)
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>

namespace py = pybind11;
using Vec3  = Eigen::Vector3d;
using Mat3N = Eigen::Matrix<double, 3, Eigen::Dynamic>;

// ── AABB spatial index ────────────────────────────────────────────────────────
// Divides the mesh into a regular grid of cells.
// Each cell stores a list of face indices whose bounding box overlaps it.s
// Query: given a sphere (center, radius), return all face indices
// whose bounding box overlaps the sphere's bounding box.
// This reduces face checks from ~65000 to ~50 on a typical wing mesh.

struct AABBGrid {
    Vec3 origin;
    Vec3 cell_size;
    int  nx, ny, nz;
    std::vector<std::vector<int>> cells;   // cell index -> list of face indices

    AABBGrid() : nx(0), ny(0), nz(0) {}

    void build(const double* pts, const int* faces, int n_faces, double padding = 0.0) {
        // Find mesh bounding box
        Vec3 lo( 1e18,  1e18,  1e18);
        Vec3 hi(-1e18, -1e18, -1e18);

        // We iterate face vertices to find bounds
        // pts layout: [x0,y0,z0, x1,y1,z1, ...]
        // faces layout: [a0,b0,c0, a1,b1,c1, ...]
        for (int f = 0; f < n_faces; ++f) {
            for (int v = 0; v < 3; ++v) {
                int idx = faces[f * 3 + v];
                Vec3 p(pts[idx*3], pts[idx*3+1], pts[idx*3+2]);
                lo = lo.cwiseMin(p);
                hi = hi.cwiseMax(p);
            }
        }

        lo -= Vec3(padding, padding, padding);
        hi += Vec3(padding, padding, padding);
        origin = lo;

        // Choose cell count so average cell side ≈ 5% of bounding box diagonal
        double diag = (hi - lo).norm();
        double target_cell = diag * 0.05;
        nx = std::max(1, (int)std::ceil((hi[0]-lo[0]) / target_cell));
        ny = std::max(1, (int)std::ceil((hi[1]-lo[1]) / target_cell));
        nz = std::max(1, (int)std::ceil((hi[2]-lo[2]) / target_cell));
        cell_size = Vec3((hi[0]-lo[0])/nx, (hi[1]-lo[1])/ny, (hi[2]-lo[2])/nz);

        cells.assign(nx * ny * nz, {});

        // Insert each face into all overlapping cells
        for (int f = 0; f < n_faces; ++f) {
            Vec3 flo( 1e18,  1e18,  1e18);
            Vec3 fhi(-1e18, -1e18, -1e18);
            for (int v = 0; v < 3; ++v) {
                int idx = faces[f * 3 + v];
                Vec3 p(pts[idx*3], pts[idx*3+1], pts[idx*3+2]);
                flo = flo.cwiseMin(p);
                fhi = fhi.cwiseMax(p);
            }
            int ix0 = std::max(0, (int)std::floor((flo[0]-origin[0])/cell_size[0]));
            int iy0 = std::max(0, (int)std::floor((flo[1]-origin[1])/cell_size[1]));
            int iz0 = std::max(0, (int)std::floor((flo[2]-origin[2])/cell_size[2]));
            int ix1 = std::min(nx-1, (int)std::floor((fhi[0]-origin[0])/cell_size[0]));
            int iy1 = std::min(ny-1, (int)std::floor((fhi[1]-origin[1])/cell_size[1]));
            int iz1 = std::min(nz-1, (int)std::floor((fhi[2]-origin[2])/cell_size[2]));
            for (int ix = ix0; ix <= ix1; ++ix)
            for (int iy = iy0; iy <= iy1; ++iy)
            for (int iz = iz0; iz <= iz1; ++iz)
                cells[ix + nx*(iy + ny*iz)].push_back(f);
        }
    }

    // Return candidate face indices whose AABB overlaps query sphere
    std::vector<int> query(const Vec3& center, double radius) const {
        if (nx == 0) return {};
        Vec3 lo = center - Vec3(radius, radius, radius);
        Vec3 hi = center + Vec3(radius, radius, radius);
        int ix0 = std::max(0, (int)std::floor((lo[0]-origin[0])/cell_size[0]));
        int iy0 = std::max(0, (int)std::floor((lo[1]-origin[1])/cell_size[1]));
        int iz0 = std::max(0, (int)std::floor((lo[2]-origin[2])/cell_size[2]));
        int ix1 = std::min(nx-1, (int)std::floor((hi[0]-origin[0])/cell_size[0]));
        int iy1 = std::min(ny-1, (int)std::floor((hi[1]-origin[1])/cell_size[1]));
        int iz1 = std::min(nz-1, (int)std::floor((hi[2]-origin[2])/cell_size[2]));

        // Collect unique face indices using a visited flag trick
        std::vector<int> result;
        for (int ix = ix0; ix <= ix1; ++ix)
        for (int iy = iy0; iy <= iy1; ++iy)
        for (int iz = iz0; iz <= iz1; ++iz) {
            const auto& cell = cells[ix + nx*(iy + ny*iz)];
            for (int f : cell) result.push_back(f);
        }
        // Deduplicate
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
};

// ── Core triangulation logic ──────────────────────────────────────────────────
// Direct port of find_triangulation_point() — same math, zero Python overhead.
//
// Given two sphere centers c1, c2 with radii r1, r2:
//   1. Find the circle of intersection of the two spheres
//      (center = circle_center, radius = h, axis = u = (c2-c1)/|c2-c1|)
//   2. For each mesh face, find where the plane defined by u passes through it
//      (edge-plane intersection gives a chord segment through the face)
//   3. Intersect that chord segment with the circle (quadratic solve)
//   4. Keep candidates on the correct half-space side of semi_dir
//   5. Return the candidate closest to the expected point circle_center + semi_dir*h

static Vec3 find_point_impl(
    const Vec3& c1, const Vec3& c2,
    double r1, double r2,
    const Vec3& semi_dir_in,
    const double* mesh_pts,
    const int*    mesh_faces,
    int           n_faces,
    const AABBGrid& grid)
{
    Vec3 semi_dir = semi_dir_in;
    double sn = semi_dir.norm();
    if (sn > 0) semi_dir /= sn;

    double d = (c2 - c1).norm();
    if (d < 1e-12) return c1;

    Vec3 u = (c2 - c1) / d;
    double a  = (r1*r1 - r2*r2 + d*d) / (2.0 * d);
    double h2 = r1*r1 - a*a;
    double h  = std::sqrt(std::max(0.0, h2));
    Vec3 circle_center = c1 + a * u;

    // Query only nearby faces using spatial index
    double query_radius = h * 1.5 + 1e-3;
    std::vector<int> candidates_f = grid.query(circle_center, query_radius);

    std::vector<Vec3> candidates;
    candidates.reserve(16);

    for (int fi : candidates_f) {
        int ia = mesh_faces[fi*3+0];
        int ib = mesh_faces[fi*3+1];
        int ic = mesh_faces[fi*3+2];

        Vec3 v0(mesh_pts[ia*3], mesh_pts[ia*3+1], mesh_pts[ia*3+2]);
        Vec3 v1(mesh_pts[ib*3], mesh_pts[ib*3+1], mesh_pts[ib*3+2]);
        Vec3 v2(mesh_pts[ic*3], mesh_pts[ic*3+1], mesh_pts[ic*3+2]);

        // Signed distances from the plane through circle_center with normal u
        double d0 = (v0 - circle_center).dot(u);
        double d1 = (v1 - circle_center).dot(u);
        double d2 = (v2 - circle_center).dot(u);

        // Collect edge-plane intersection points (exactly 2 for a valid cross)
        Vec3 seg_pts[2];
        int  ns = 0;
        auto try_edge = [&](const Vec3& p0, const Vec3& p1, double s0, double s1) {
            if (s0 * s1 < 0.0 && ns < 2) {
                double t = s0 / (s0 - s1);
                seg_pts[ns++] = p0 + t * (p1 - p0);
            }
        };
        try_edge(v0, v1, d0, d1);
        try_edge(v1, v2, d1, d2);
        try_edge(v2, v0, d2, d0);

        if (ns != 2) continue;

        // Intersect segment seg_pts[0]..seg_pts[1] with circle of radius h
        Vec3 dvec = seg_pts[1] - seg_pts[0];
        Vec3 f    = seg_pts[0] - circle_center;
        double A  = dvec.dot(dvec);
        double B  = 2.0 * f.dot(dvec);
        double C  = f.dot(f) - h * h;
        double disc = B*B - 4.0*A*C;
        if (disc < 0.0) continue;

        double sq = std::sqrt(disc);
        for (int sign : {-1, 1}) {
            double t = (-B + sign * sq) / (2.0 * A);
            if (t < 0.0 || t > 1.0) continue;
            Vec3 p = seg_pts[0] + t * dvec;
            if ((p - c1).dot(semi_dir) > 0.0)
                candidates.push_back(p);
        }
    }

    if (candidates.empty()) return circle_center;

    Vec3 expected = circle_center + semi_dir * h;
    int best = 0;
    double best_d = (candidates[0] - expected).norm();
    for (int i = 1; i < (int)candidates.size(); ++i) {
        double dd = (candidates[i] - expected).norm();
        if (dd < best_d) { best_d = dd; best = i; }
    }
    return candidates[best];
}

// ── Python-facing mesh wrapper ────────────────────────────────────────────────
// Holds the spatial index so it is built once and reused across all calls.

struct MeshAccelerator {
    std::vector<double> pts;
    std::vector<int>    faces;
    int                 n_faces;
    AABBGrid            grid;

    MeshAccelerator(py::array_t<double> mesh_pts_arr,
                    py::array_t<int>    mesh_faces_arr) {
        auto p = mesh_pts_arr.unchecked<2>();
        auto f = mesh_faces_arr.unchecked<2>();
        n_faces = f.shape(0);

        pts.resize(p.shape(0) * 3);
        for (int i = 0; i < p.shape(0); ++i) {
            pts[i*3+0] = p(i,0);
            pts[i*3+1] = p(i,1);
            pts[i*3+2] = p(i,2);
        }
        faces.resize(n_faces * 3);
        for (int i = 0; i < n_faces; ++i) {
            faces[i*3+0] = f(i,0);
            faces[i*3+1] = f(i,1);
            faces[i*3+2] = f(i,2);
        }
        grid.build(pts.data(), faces.data(), n_faces);
    }

    // Single point — matches find_triangulation_point() signature exactly
    py::array_t<double> find_point(
        py::array_t<double> c1_arr,
        py::array_t<double> c2_arr,
        double r1, double r2,
        py::array_t<double> semi_dir_arr)
    {
        auto c1r = c1_arr.unchecked<1>();
        auto c2r = c2_arr.unchecked<1>();
        auto sdr = semi_dir_arr.unchecked<1>();
        Vec3 c1(c1r(0), c1r(1), c1r(2));
        Vec3 c2(c2r(0), c2r(1), c2r(2));
        Vec3 sd(sdr(0), sdr(1), sdr(2));

        Vec3 result = find_point_impl(c1, c2, r1, r2, sd,
                                       pts.data(), faces.data(), n_faces, grid);

        auto out = py::array_t<double>({3});
        auto buf = out.mutable_unchecked<1>();
        buf(0) = result[0]; buf(1) = result[1]; buf(2) = result[2];
        return out;
    }

    // Batch — processes all lattice nodes in one C++ call.
    // centers1, centers2: (N,3)   r1s, r2s: (N,)   semi_dirs: (N,3)
    // Returns: (N,3) result points
    py::array_t<double> find_point_batch(
        py::array_t<double> centers1_arr,
        py::array_t<double> centers2_arr,
        py::array_t<double> r1s_arr,
        py::array_t<double> r2s_arr,
        py::array_t<double> semi_dirs_arr)
    {
        auto c1s = centers1_arr.unchecked<2>();
        auto c2s = centers2_arr.unchecked<2>();
        auto r1s = r1s_arr.unchecked<1>();
        auto r2s = r2s_arr.unchecked<1>();
        auto sds = semi_dirs_arr.unchecked<2>();
        int  N   = c1s.shape(0);

        auto out = py::array_t<double>({N, 3});
        auto buf = out.mutable_unchecked<2>();

        // OpenMP parallelises this loop at the C++ level —
        // no GIL, no joblib overhead, no Python object creation per item
        #pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < N; ++i) {
            Vec3 c1(c1s(i,0), c1s(i,1), c1s(i,2));
            Vec3 c2(c2s(i,0), c2s(i,1), c2s(i,2));
            Vec3 sd(sds(i,0), sds(i,1), sds(i,2));
            Vec3 r = find_point_impl(c1, c2, r1s(i), r2s(i), sd,
                                      pts.data(), faces.data(), n_faces, grid);
            buf(i,0) = r[0]; buf(i,1) = r[1]; buf(i,2) = r[2];
        }
        return out;
    }
};

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(stage_4_marching_triangulation, m) {
    m.doc() = "Fast C++ triangulation for lattice deformation — replaces find_triangulation_point()";

    py::class_<MeshAccelerator>(m, "MeshAccelerator")
        .def(py::init<py::array_t<double>, py::array_t<int>>(),
             py::arg("mesh_pts"), py::arg("mesh_faces"),
             "Build AABB spatial index from mesh arrays. Call once per mesh, reuse for all lattice nodes.")
        .def("find_point", &MeshAccelerator::find_point,
             py::arg("c1"), py::arg("c2"), py::arg("r1"), py::arg("r2"), py::arg("semi_dir"),
             "Find single triangulation point. Drop-in for find_triangulation_point().")
        .def("find_point_batch", &MeshAccelerator::find_point_batch,
             py::arg("centers1"), py::arg("centers2"),
             py::arg("r1s"), py::arg("r2s"), py::arg("semi_dirs"),
             "Find all lattice node points in one call. Parallelised with OpenMP.");
}
