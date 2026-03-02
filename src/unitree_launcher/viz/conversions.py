"""MuJoCo geometry to trimesh conversion for the viser web viewer.

Converts MuJoCo visual geoms (meshes and primitives) into trimesh objects
grouped by body, suitable for uploading to viser as per-body scene nodes.
"""
from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import viser.transforms as vtf


# MuJoCo geom type constants.
_GEOM_PLANE = 0
_GEOM_SPHERE = 2
_GEOM_CAPSULE = 3
_GEOM_CYLINDER = 5
_GEOM_BOX = 6
_GEOM_MESH = 7

# Visual geom groups (0 = default visual, 2 = visual mesh in G1 model).
_VISUAL_GROUPS = {0, 2}


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to wxyz quaternion via viser's SO3."""
    return vtf.SO3.from_matrix(R).wxyz


def mujoco_mesh_to_trimesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    """Convert a MuJoCo mesh geom to a trimesh object.

    Reads vertices/faces from the model's mesh data arrays using the geom's
    mesh data ID.  Applies geom_rgba as a uniform vertex color.

    Args:
        mj_model: MuJoCo model.
        geom_id: Index of the geom (must be type mesh with valid dataid).

    Returns:
        A trimesh.Trimesh with vertices, faces, and vertex colors.
    """
    mesh_id = mj_model.geom_dataid[geom_id]
    assert mesh_id >= 0, f"geom {geom_id} has no mesh data"

    vert_adr = mj_model.mesh_vertadr[mesh_id]
    face_adr = mj_model.mesh_faceadr[mesh_id]
    n_vert = mj_model.mesh_vertnum[mesh_id]
    n_face = mj_model.mesh_facenum[mesh_id]

    vertices = mj_model.mesh_vert[vert_adr:vert_adr + n_vert].copy()
    faces = mj_model.mesh_face[face_adr:face_adr + n_face].copy()

    rgba = mj_model.geom_rgba[geom_id]
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(rgba_uint8, (len(vertices), 1)),
    )
    return mesh


def create_primitive_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    """Create a trimesh primitive (capsule, sphere, box, cylinder) for a geom.

    Args:
        mj_model: MuJoCo model.
        geom_id: Index of the geom (must be a primitive type).

    Returns:
        A trimesh.Trimesh for the primitive geometry.
    """
    gtype = mj_model.geom_type[geom_id]
    size = mj_model.geom_size[geom_id]
    rgba = mj_model.geom_rgba[geom_id]
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    if gtype == _GEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=size[0])
    elif gtype == _GEOM_CAPSULE:
        # MuJoCo capsule: size[0]=radius, size[1]=half-length
        mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    elif gtype == _GEOM_BOX:
        # MuJoCo box: size = half-extents
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif gtype == _GEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    else:
        raise ValueError(f"Unsupported primitive geom type {gtype}")

    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(rgba_uint8, (len(mesh.vertices), 1)),
    )
    return mesh


def _geom_local_transform(mj_model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Get the 4x4 transform of a geom relative to its parent body."""
    T = np.eye(4)
    T[:3, 3] = mj_model.geom_pos[geom_id]

    # MuJoCo stores geom quaternion as wxyz.
    quat_wxyz = mj_model.geom_quat[geom_id]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat_wxyz)
    T[:3, :3] = R.reshape(3, 3)
    return T


def build_body_meshes(
    mj_model: mujoco.MjModel,
    visual_only: bool = True,
) -> dict[int, trimesh.Trimesh]:
    """Build a merged trimesh per dynamic body from the MuJoCo model.

    For each body (excluding the world body), collects its geoms, converts
    each to trimesh with the geom-local transform baked into vertices, and
    merges all geoms per body into a single mesh.

    Args:
        mj_model: MuJoCo model.
        visual_only: If True, only include geoms in visual groups (0, 2).
            If False, also include collision geoms (group 3).

    Returns:
        Dict mapping body_id -> merged trimesh.Trimesh.
        Bodies with no eligible geoms are omitted.
    """
    body_geoms: dict[int, list[trimesh.Trimesh]] = {}

    for geom_id in range(mj_model.ngeom):
        body_id = mj_model.geom_bodyid[geom_id]
        if body_id == 0:
            continue  # Skip world body (floor plane, etc.)

        gtype = mj_model.geom_type[geom_id]
        group = mj_model.geom_group[geom_id]

        if gtype == _GEOM_PLANE:
            continue  # Skip plane geoms

        if visual_only and group not in _VISUAL_GROUPS:
            continue

        # Convert to trimesh.
        if gtype == _GEOM_MESH:
            mesh = mujoco_mesh_to_trimesh(mj_model, geom_id)
        else:
            mesh = create_primitive_mesh(mj_model, geom_id)

        # Apply geom-local transform (position + rotation relative to body).
        T = _geom_local_transform(mj_model, geom_id)
        mesh.apply_transform(T)

        body_geoms.setdefault(body_id, []).append(mesh)

    # Merge geoms per body.
    result: dict[int, trimesh.Trimesh] = {}
    for body_id, meshes in body_geoms.items():
        if len(meshes) == 1:
            result[body_id] = meshes[0]
        else:
            result[body_id] = trimesh.util.concatenate(meshes)

    return result
