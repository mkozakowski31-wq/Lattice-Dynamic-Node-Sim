import numpy as np 
import pyvista as pv
from pyvista import examples
from collections import Counter

edges = []
mesh = pv.read('/Users/marko/Documents/GitHub/marko/Models/LatticeTestBenchComplicatedGeo.obj')
mesh = mesh.triangulate()
points = mesh.points

#Bounding box
print(points.shape)
print("Z min:", points[:,2].min())
print("Z max:", points[:,2].max())
print("Y min:", points[:,1].min())
print("Y max:", points[:,1].max())
print("x min:", points[:,0].min())
print("x max:", points[:,0].max())

Span = input("Enter Span axis: ")
Cord = input("Enter Cord axis: ")

print("You entered:", Span, Cord)

input("Press Enter to continue...")

#Source vertices
def extract_face_vertices(obj_file_path):
    faces = []

    with open(obj_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('f '):  # Only process face lines
                parts = line.split()[1:]  # Skip the 'f'
                face_vertices = []
                for part in parts:
                    # Each part can be v/vt/vn or v//vn or v
                    vertex_index = part.split('/')[0]  # Take only the vertex index
                    face_vertices.append(int(vertex_index)-1)
                faces.append(face_vertices)
    return faces

# Example usage
if __name__ == "__main__":
    obj_file_path = "/Users/marko/Documents/GitHub/marko/Models/LatticeTestBenchComplicatedGeo.obj" 
    face_array = extract_face_vertices(obj_file_path)
    print(face_array)

print(len(face_array))
input("Press Enter to continue...")
def triangle_to_edge_pairs(triangle):
    a, b, c = triangle
    return [[a, b], [b, c], [a, c]]

def faces_to_edges(faces):
    edges = []
    for face in faces:
        edges.extend(triangle_to_edge_pairs(face))
    return edges

# Example usage
edges = faces_to_edges(face_array)
edges = np.sort(edges, axis=1)
for e in edges:
    print(e)
input("Press Enter to continue...")

def remove_all_duplicate_edges(edges):
    # Count occurrences
    counts = Counter(tuple(edge) for edge in edges)
    
    # Keep only edges that occur exactly once
    unique_edges = [list(edge) for edge, count in counts.items() if count == 1]
    
    return unique_edges

# Example usage
filtered_edges = remove_all_duplicate_edges(edges)
filtered_edges = np.array(filtered_edges)
filtered_edges = filtered_edges.reshape(-1)
filtered_edges = np.unique(filtered_edges)

for e in filtered_edges:
    print(e)
print("Total edges:", len(edges))
print("Unique boundary edges:", len(filtered_edges))

input("press Enter to continue...") 

vertex_coords = mesh.points[filtered_edges]  # Example vertex indices
point = pv.PolyData(vertex_coords)


edges = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)

# Plot mesh with highlighted boundary edges
p = pv.Plotter(shape=(1,2))
p.subplot(0, 0)
p.show_axes()
p.show_bounds(grid='back', location='all', show_xlabels=True, show_ylabels=True, show_zlabels=True)
p.add_mesh(mesh, color='red', show_edges=False)
p.add_mesh(edges, color='blue', line_width=1)
p.add_mesh(point, color='red', point_size=15, render_points_as_spheres=True)  # highlight vertex

#p.add_points(LE_points, color="yellow", point_size=10, render_points_as_spheres=True)
# p.add_points(TE_points, color="green", point_size=10, render_points_as_spheres=True)
# p.add_points(RT_points, color="purple", point_size=10, render_points_as_spheres=True)
# p.add_points(TP_points, color="orange", point_size=10, render_points_as_spheres=True)
# UV map
p.subplot(0, 1)
p.add_text('UV Map', font_size=20, font='times')


p.show()

