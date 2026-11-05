# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""3D camera, interpolation, coordinate conversion, and shear/perspective projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyhexen.core.fixed import FRACUNIT

BAM_TO_RAD: float = (2.0 * math.pi) / float(1 << 32)
RAD_TO_BAM: float = float(1 << 32) / (2.0 * math.pi)
AUTHENTIC_PIXEL_ASPECT: float = 1.2
MAX_PITCH_RAD: float = math.radians(89.0)


def bam_to_rad(bam_angle: int) -> float:
    """Convert a 32-bit unsigned BAM angle to radians in [0, 2*pi)."""
    return (bam_angle & 0xFFFFFFFF) * BAM_TO_RAD


def rad_to_bam(radians: float) -> int:
    """Convert radians to a 32-bit unsigned BAM angle."""
    return round(radians * RAD_TO_BAM) & 0xFFFFFFFF


def lookdir_to_pitch_rad(lookdir: float) -> float:
    """Convert Hexen's lookdir (-110..90) to pitch angle in radians."""
    # In Hexen (p_user.c), lookdir ranges from -110 to +90, roughly corresponding
    # to vertical y-shear tangent lookdir / 160.0.
    return math.atan(lookdir / 160.0)


def map_to_gl_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert map coordinates (x, y, z) to OpenGL coordinates (x, z, -y)."""
    return (float(x), float(z), float(-y))


def _lerp_angle_rad(a0: float, a1: float, alpha: float) -> float:
    """Shortest-arc linear interpolation between two radian angles."""
    diff = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    return (a0 + diff * alpha) % (2.0 * math.pi)


@dataclass(slots=True)
class Camera:
    """3D camera supporting tic interpolation, free-fly movement, and y-shearing (ADR-006).

    Map coordinate convention:
      +X = East (yaw = 0)
      +Y = North (yaw = pi/2)
      +Z = Up
    OpenGL world convention (§3.4):
      (X_gl, Y_gl, Z_gl) = (x, z, -y)
    Matrices returned by `view_matrix` and `projection_matrix` are 4x4 `float32`
    numpy arrays in standard mathematical row-major layout (`v_clip = proj @ view @ v_world`);
    pass `matrix.T.tobytes()` (column-major) when uploading to OpenGL / moderngl uniforms.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 41.0
    yaw: float = 0.0
    pitch: float = 0.0

    prev_x: float = 0.0
    prev_y: float = 0.0
    prev_z: float = 41.0
    prev_yaw: float = 0.0
    prev_pitch: float = 0.0

    true_pitch: bool = False
    authentic_aspect: bool = True
    near: float = 1.0
    far: float = 16384.0

    def save_prev_state(self) -> None:
        """Snapshot current transform as previous-tic state for frame interpolation."""
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
        self.prev_yaw = self.yaw
        self.prev_pitch = self.pitch

    def snap_interpolation(self) -> None:
        """Alias for save_prev_state (used after teleports or map loads)."""
        self.save_prev_state()

    def set_from_fixed(
        self,
        x_fixed: int,
        y_fixed: int,
        z_fixed: int,
        angle_bam: int,
        lookdir: int = 0,
        *,
        snap: bool = False,
    ) -> None:
        """Set camera state from 16.16 fixed-point map coordinates and 32-bit BAM angle."""
        self.x = x_fixed / float(FRACUNIT)
        self.y = y_fixed / float(FRACUNIT)
        self.z = z_fixed / float(FRACUNIT)
        self.yaw = bam_to_rad(angle_bam)
        self.pitch = lookdir_to_pitch_rad(lookdir)
        if snap:
            self.save_prev_state()

    def interpolated_state(self, alpha: float = 1.0) -> tuple[float, float, float, float, float]:
        """Return interpolated (x, y, z, yaw, pitch) in map coordinates for alpha in [0, 1]."""
        a = max(0.0, min(1.0, float(alpha)))
        ix = self.prev_x + (self.x - self.prev_x) * a
        iy = self.prev_y + (self.y - self.prev_y) * a
        iz = self.prev_z + (self.z - self.prev_z) * a
        iyaw = _lerp_angle_rad(self.prev_yaw, self.yaw, a)
        ipitch = self.prev_pitch + (self.pitch - self.prev_pitch) * a
        return ix, iy, iz, iyaw, ipitch

    def move_free_fly(
        self,
        dt: float,
        forward: float = 0.0,
        right: float = 0.0,
        up: float = 0.0,
        yaw_delta: float = 0.0,
        pitch_delta: float = 0.0,
        speed: float = 400.0,
    ) -> None:
        """Update camera position and orientation in free-fly mode."""
        self.yaw = (self.yaw + yaw_delta) % (2.0 * math.pi)
        self.pitch = max(-MAX_PITCH_RAD, min(MAX_PITCH_RAD, self.pitch + pitch_delta))

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        # Horizontal forward = (cos(yaw), sin(yaw)), right = (sin(yaw), -cos(yaw)) in map coords
        step = speed * dt
        self.x += (forward * cos_yaw + right * sin_yaw) * step
        self.y += (forward * sin_yaw - right * cos_yaw) * step
        self.z += up * step

    def view_matrix(self, alpha: float = 1.0) -> np.ndarray:
        """Build 4x4 float32 view matrix transforming GL world (x, z, -y) to eye space."""
        ix, iy, iz, iyaw, ipitch = self.interpolated_state(alpha)
        eye_x, eye_y, eye_z = map_to_gl_coords(ix, iy, iz)

        cos_yaw = math.cos(iyaw)
        sin_yaw = math.sin(iyaw)

        # In GL space (x, z, -y):
        # Horizontal forward = (cos_yaw, 0, -sin_yaw)
        # Right              = (sin_yaw, 0,  cos_yaw)
        # World up           = (0,       1,  0)
        rx, ry, rz = sin_yaw, 0.0, cos_yaw
        if self.true_pitch:
            cos_p = math.cos(ipitch)
            sin_p = math.sin(ipitch)
            fx, fy, fz = cos_yaw * cos_p, sin_p, -sin_yaw * cos_p
            ux, uy, uz = -cos_yaw * sin_p, cos_p, sin_yaw * sin_p
        else:
            fx, fy, fz = cos_yaw, 0.0, -sin_yaw
            ux, uy, uz = 0.0, 1.0, 0.0

        # In OpenGL eye space, camera looks down -Z, so Z_eye = -forward
        zx, zy, zz = -fx, -fy, -fz

        mat = np.array(
            [
                [rx, ry, rz, -(rx * eye_x + ry * eye_y + rz * eye_z)],
                [ux, uy, uz, -(ux * eye_x + uy * eye_y + uz * eye_z)],
                [zx, zy, zz, -(zx * eye_x + zy * eye_y + zz * eye_z)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        if self.authentic_aspect:
            # Scale world Y (map Z) by 1.2x for authentic 320x200 non-square pixel aspect (§3.4)
            scale = np.diag([1.0, AUTHENTIC_PIXEL_ASPECT, 1.0, 1.0]).astype(np.float32)
            mat = mat @ scale
        return mat

    def projection_matrix(
        self,
        aspect: float,
        fov_deg: float = 90.0,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Build 4x4 float32 perspective projection matrix (with y-shear when true_pitch=False)."""
        _, _, _, _, ipitch = self.interpolated_state(alpha)

        # Horizontal FOV -> vertical focal length
        half_fov_x = math.radians(fov_deg) * 0.5
        tan_half_fov_x = math.tan(half_fov_x)
        tan_half_fov_y = tan_half_fov_x / max(aspect, 1e-6)

        f_x = 1.0 / tan_half_fov_x
        f_y = 1.0 / tan_half_fov_y

        nf = 1.0 / (self.near - self.far)
        a = (self.far + self.near) * nf
        b = (2.0 * self.far * self.near) * nf

        # In y-shearing mode (ADR-006), shift eye-space Y by z_eye * tan(pitch)
        y_shear = 0.0 if self.true_pitch else (f_y * math.tan(ipitch))

        return np.array(
            [
                [f_x, 0.0, 0.0, 0.0],
                [0.0, f_y, y_shear, 0.0],
                [0.0, 0.0, a, b],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
