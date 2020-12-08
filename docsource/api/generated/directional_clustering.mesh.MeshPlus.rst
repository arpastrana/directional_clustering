.. rst-class:: detail

MeshPlus
====================================

.. currentmodule:: directional_clustering.mesh

.. autoclass:: MeshPlus

    
    

    .. rubric:: Attributes

    .. autosummary::
    

    .. rubric:: Inherited Attributes

    .. autosummary::
    
        ~MeshPlus.DATASCHEMA
        ~MeshPlus.JSONSCHEMA
        ~MeshPlus.adjacency
        ~MeshPlus.data
        ~MeshPlus.dtype
        ~MeshPlus.guid
        ~MeshPlus.name

    
    

    
    

    .. rubric:: Methods

    .. autosummary::
        :toctree:

    
        ~MeshPlus.clustering_label
        ~MeshPlus.vector_field
        ~MeshPlus.vector_fields

    .. rubric:: Inherited Methods

    .. autosummary::
        :toctree:

    
        ~MeshPlus.__init__
        ~MeshPlus.add_face
        ~MeshPlus.add_vertex
        ~MeshPlus.area
        ~MeshPlus.bounding_box
        ~MeshPlus.bounding_box_xy
        ~MeshPlus.centroid
        ~MeshPlus.clear
        ~MeshPlus.collapse_edge
        ~MeshPlus.connected_components
        ~MeshPlus.copy
        ~MeshPlus.cull_vertices
        ~MeshPlus.cut
        ~MeshPlus.delete_face
        ~MeshPlus.delete_vertex
        ~MeshPlus.dual
        ~MeshPlus.edge_attribute
        ~MeshPlus.edge_attributes
        ~MeshPlus.edge_coordinates
        ~MeshPlus.edge_direction
        ~MeshPlus.edge_faces
        ~MeshPlus.edge_length
        ~MeshPlus.edge_loop
        ~MeshPlus.edge_midpoint
        ~MeshPlus.edge_point
        ~MeshPlus.edge_strip
        ~MeshPlus.edge_vector
        ~MeshPlus.edges
        ~MeshPlus.edges_attribute
        ~MeshPlus.edges_attributes
        ~MeshPlus.edges_on_boundaries
        ~MeshPlus.edges_on_boundary
        ~MeshPlus.edges_where
        ~MeshPlus.edges_where_predicate
        ~MeshPlus.euler
        ~MeshPlus.face_adjacency
        ~MeshPlus.face_adjacency_halfedge
        ~MeshPlus.face_adjacency_vertices
        ~MeshPlus.face_area
        ~MeshPlus.face_aspect_ratio
        ~MeshPlus.face_attribute
        ~MeshPlus.face_attributes
        ~MeshPlus.face_center
        ~MeshPlus.face_centroid
        ~MeshPlus.face_coordinates
        ~MeshPlus.face_corners
        ~MeshPlus.face_curvature
        ~MeshPlus.face_degree
        ~MeshPlus.face_flatness
        ~MeshPlus.face_halfedges
        ~MeshPlus.face_max_degree
        ~MeshPlus.face_min_degree
        ~MeshPlus.face_neighborhood
        ~MeshPlus.face_neighbors
        ~MeshPlus.face_normal
        ~MeshPlus.face_plane
        ~MeshPlus.face_skewness
        ~MeshPlus.face_vertex_ancestor
        ~MeshPlus.face_vertex_descendant
        ~MeshPlus.face_vertices
        ~MeshPlus.faces
        ~MeshPlus.faces_attribute
        ~MeshPlus.faces_attributes
        ~MeshPlus.faces_on_boundary
        ~MeshPlus.faces_where
        ~MeshPlus.faces_where_predicate
        ~MeshPlus.flip_cycles
        ~MeshPlus.from_data
        ~MeshPlus.from_json
        ~MeshPlus.from_lines
        ~MeshPlus.from_obj
        ~MeshPlus.from_off
        ~MeshPlus.from_ply
        ~MeshPlus.from_points
        ~MeshPlus.from_polygons
        ~MeshPlus.from_polyhedron
        ~MeshPlus.from_polylines
        ~MeshPlus.from_shape
        ~MeshPlus.from_stl
        ~MeshPlus.from_vertices_and_faces
        ~MeshPlus.genus
        ~MeshPlus.get_any_face
        ~MeshPlus.get_any_face_vertex
        ~MeshPlus.get_any_vertex
        ~MeshPlus.get_any_vertices
        ~MeshPlus.gkey_key
        ~MeshPlus.halfedge_face
        ~MeshPlus.has_edge
        ~MeshPlus.has_face
        ~MeshPlus.has_halfedge
        ~MeshPlus.has_vertex
        ~MeshPlus.index_key
        ~MeshPlus.index_vertex
        ~MeshPlus.insert_vertex
        ~MeshPlus.is_connected
        ~MeshPlus.is_edge_on_boundary
        ~MeshPlus.is_empty
        ~MeshPlus.is_face_on_boundary
        ~MeshPlus.is_manifold
        ~MeshPlus.is_orientable
        ~MeshPlus.is_quadmesh
        ~MeshPlus.is_regular
        ~MeshPlus.is_trimesh
        ~MeshPlus.is_valid
        ~MeshPlus.is_vertex_connected
        ~MeshPlus.is_vertex_on_boundary
        ~MeshPlus.join
        ~MeshPlus.key_gkey
        ~MeshPlus.key_index
        ~MeshPlus.normal
        ~MeshPlus.number_of_edges
        ~MeshPlus.number_of_faces
        ~MeshPlus.number_of_vertices
        ~MeshPlus.quads_to_triangles
        ~MeshPlus.remove_unused_vertices
        ~MeshPlus.smooth_area
        ~MeshPlus.smooth_centroid
        ~MeshPlus.split_edge
        ~MeshPlus.split_face
        ~MeshPlus.summary
        ~MeshPlus.to_data
        ~MeshPlus.to_json
        ~MeshPlus.to_lines
        ~MeshPlus.to_obj
        ~MeshPlus.to_off
        ~MeshPlus.to_ply
        ~MeshPlus.to_points
        ~MeshPlus.to_polygons
        ~MeshPlus.to_polylines
        ~MeshPlus.to_quadmesh
        ~MeshPlus.to_stl
        ~MeshPlus.to_trimesh
        ~MeshPlus.to_vertices_and_faces
        ~MeshPlus.transform
        ~MeshPlus.transform_numpy
        ~MeshPlus.transformed
        ~MeshPlus.unify_cycles
        ~MeshPlus.unset_edge_attribute
        ~MeshPlus.unset_face_attribute
        ~MeshPlus.unset_vertex_attribute
        ~MeshPlus.update_default_edge_attributes
        ~MeshPlus.update_default_face_attributes
        ~MeshPlus.update_default_vertex_attributes
        ~MeshPlus.validate_data
        ~MeshPlus.validate_json
        ~MeshPlus.vertex_area
        ~MeshPlus.vertex_attribute
        ~MeshPlus.vertex_attributes
        ~MeshPlus.vertex_coordinates
        ~MeshPlus.vertex_curvature
        ~MeshPlus.vertex_degree
        ~MeshPlus.vertex_faces
        ~MeshPlus.vertex_index
        ~MeshPlus.vertex_laplacian
        ~MeshPlus.vertex_max_degree
        ~MeshPlus.vertex_min_degree
        ~MeshPlus.vertex_neighborhood
        ~MeshPlus.vertex_neighborhood_centroid
        ~MeshPlus.vertex_neighbors
        ~MeshPlus.vertex_normal
        ~MeshPlus.vertices
        ~MeshPlus.vertices_attribute
        ~MeshPlus.vertices_attributes
        ~MeshPlus.vertices_on_boundaries
        ~MeshPlus.vertices_on_boundary
        ~MeshPlus.vertices_where
        ~MeshPlus.vertices_where_predicate

    
    