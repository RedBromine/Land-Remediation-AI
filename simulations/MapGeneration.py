#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Groundwater Geology Generator — Multi-Stratum 3D Terrain Modeling System

Display order (top→bottom): soil cap → top stratum → ... → bottom stratum
Generation order (bottom→top): bottom stratum → ... → top stratum → soil cap
"""

import sys
import traceback
import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QColorDialog,
    QFormLayout,
    QTabWidget,
    QComboBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap, QFont

# Post-processing window
from postprocess_window import PostProcessWindow


# =========================================================
# Pure NumPy Perlin Noise
# =========================================================

class _PerlinNoise:
    _PERM = np.array([
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
        140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
        247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
        57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
        74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
        60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
        65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
        200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
        52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
        207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
        119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
        129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
        218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
        81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
        184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
        222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    ], dtype=np.int32)

    def __init__(self, base=0):
        self.base = int(base) & 0xFF
        rng = np.random.RandomState(self.base)
        perm = self._PERM.copy()
        rng.shuffle(perm)
        self._perm512 = np.concatenate([perm, perm]).astype(np.int32)

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _grad(hash_arr, x, y):
        h = hash_arr & 15
        grad_x = np.where((h & 7) < 4, x, y)
        grad_y = np.where((h & 7) < 4, y, x)
        return (np.where((h & 1) == 0, grad_x, -grad_x)
                + np.where((h & 2) == 0, grad_y, -grad_y))

    def noise2(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xi = np.floor(x).astype(np.int32) & 255
        yi = np.floor(y).astype(np.int32) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = self._fade(xf)
        v = self._fade(yf)
        p = self._perm512
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]
        g1 = self._grad(aa, xf,     yf)
        g2 = self._grad(ba, xf - 1, yf)
        g3 = self._grad(ab, xf,     yf - 1)
        g4 = self._grad(bb, xf - 1, yf - 1)
        x1 = g1 + u * (g2 - g1)
        x2 = g3 + u * (g4 - g3)
        return x1 + v * (x2 - x1)

    def fbm2(self, x, y, octaves=1, persistence=0.5, lacunarity=2.0):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        value = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        for _ in range(octaves):
            value += amplitude * self.noise2(x * frequency, y * frequency)
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return value / max(max_value, 1e-10)


# =========================================================
# Stratum Configuration
# =========================================================

class StratumConfig:
    PRESETS = {
        "custom":          {"color": [0.50, 0.50, 0.50], "permeability": 1.0},
        "黏土层":          {"color": [0.55, 0.40, 0.25], "permeability": 0.001},
        "砂岩层":          {"color": [0.76, 0.70, 0.50], "permeability": 1.0},
        "泥岩层":          {"color": [0.45, 0.35, 0.30], "permeability": 0.01},
        "石灰岩层":        {"color": [0.65, 0.60, 0.55], "permeability": 0.1},
        "页岩层":          {"color": [0.35, 0.35, 0.38], "permeability": 0.0001},
        "砾岩层":          {"color": [0.60, 0.55, 0.45], "permeability": 5.0},
        "玄武岩层":        {"color": [0.25, 0.25, 0.28], "permeability": 0.00001},
        "花岗岩层":        {"color": [0.55, 0.50, 0.48], "permeability": 0.000001},
    }

    def __init__(self, name="砂岩层", thickness=10.0, color=None, permeability=1.0, preset_name=None):
        self.name = str(name)
        self.thickness = float(thickness)
        if color is None:
            color = self.PRESETS.get(name, self.PRESETS["custom"])["color"]
        self.color = list(color)
        self.permeability = float(permeability)
        if preset_name is None:
            preset_name = name if name in self.PRESETS else "custom"
        self.preset_name = str(preset_name)

    def to_dict(self):
        return {
            "name": self.name,
            "thickness": self.thickness,
            "color": self.color.copy(),
            "permeability": self.permeability,
            "preset_name": self.preset_name,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.get("name", "地层"),
            thickness=d.get("thickness", 10.0),
            color=d.get("color", [0.5, 0.5, 0.5]),
            permeability=d.get("permeability", 1.0),
            preset_name=d.get("preset_name"),
        )


# =========================================================
# Geology Generator — Core Engine
# =========================================================

class GeologyGenerator:
    """
    self.strata[0]  = physical bottom stratum
    self.strata[-1] = physical top stratum
    Generation order: strata[0] → strata[1] → ... → strata[-1] → soil cap
    """

    def __init__(
        self,
        radius=100,
        irregularity=0.3,
        depth=50,
        max_height=15,
        resolution_xy=100,
        resolution_z=60,
        scale=60,
        octaves=5,
        persistence=0.5,
        lacunarity=2.0,
        soil_thickness=5.0,
        complexity=0.0,
        strata=None,
        seed=None,
        soil_threshold_min=0.15,
        soil_threshold_max=0.35,
    ):
        self.radius = radius
        self.irregularity = irregularity
        self.depth = depth
        self.max_height = max_height
        self.resolution_xy = resolution_xy
        self.resolution_z = resolution_z
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.soil_thickness = max(0.0, soil_thickness)
        self.complexity = max(0.0, min(1.0, complexity))
        self.soil_threshold_min = max(0.0, min(1.0, soil_threshold_min))
        self.soil_threshold_max = max(self.soil_threshold_min, min(1.0, soil_threshold_max))
        self.strata = []
        if strata:
            for s in strata:
                if isinstance(s, dict):
                    self.strata.append(StratumConfig.from_dict(s))
                elif isinstance(s, StratumConfig):
                    self.strata.append(s)
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.rng = np.random.RandomState(self.seed)

    def _generate_boundary_mask(self, x, y):
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        p1, p2, p3 = [self.rng.uniform(0, 2*np.pi) for _ in range(3)]
        f1, f2, f3 = self.rng.randint(2, 6), self.rng.randint(3, 9), self.rng.randint(5, 12)
        a1, a2, a3 = self.rng.uniform(0.3, 0.7), self.rng.uniform(0.2, 0.5), self.rng.uniform(0.1, 0.4)
        boundary_noise = (a1*np.sin(f1*theta+p1) + a2*np.cos(f2*theta+p2) + a3*np.sin(f3*theta+p3))
        boundary = self.radius * (1 + self.irregularity * boundary_noise)
        return rr <= boundary

    def _generate_terrain_surface(self, x, y):
        xx, yy = np.meshgrid(x, y)
        pn = _PerlinNoise(base=self.rng.randint(0, 256))
        raw = pn.fbm2(xx / self.scale, yy / self.scale, octaves=self.octaves,
                      persistence=self.persistence, lacunarity=self.lacunarity)
        terrain = 1.0 - np.abs(raw)
        terrain = terrain - terrain.min()
        terrain = terrain / terrain.max()
        return terrain * self.max_height

    def _generate_stratum_tops(self, x, y):
        """Bottom-up: strata[0] starts at -depth, stack upward."""
        if not self.strata:
            return []

        xx, yy = np.meshgrid(x, y)
        current_bottom = np.full_like(xx, -float(self.depth))
        stratum_tops = []
        c = self.complexity

        for stratum in self.strata:
            base_thick = float(stratum.thickness)

            if c > 1e-6:
                max_tilt = np.radians(30.0)
                tilt_angle = c * self.rng.uniform(0, max_tilt)
                tilt_azimuth = self.rng.uniform(0, 2*np.pi)
                tilt = np.tan(tilt_angle) * (xx*np.cos(tilt_azimuth) + yy*np.sin(tilt_azimuth))

                fold = np.zeros_like(xx)
                n_folds = max(1, int(c * 5))
                for _ in range(n_folds):
                    max_amp = min(base_thick * 0.4, 8.0)
                    amp = c * self.rng.uniform(0.5, max_amp)
                    freq = self.rng.uniform(0.003, 0.025)
                    direction = self.rng.uniform(0, 2*np.pi)
                    phase = self.rng.uniform(0, 2*np.pi)
                    fold += amp * np.sin(freq*(xx*np.cos(direction)+yy*np.sin(direction)) + phase)

                fault = np.zeros_like(xx)
                if c > 0.25:
                    n_faults = max(1, int((c - 0.25) / 0.75 * 4))
                    for _ in range(n_faults):
                        cx = self.rng.uniform(-self.radius*0.6, self.radius*0.6)
                        cy = self.rng.uniform(-self.radius*0.6, self.radius*0.6)
                        fault_dir = self.rng.uniform(0, 2*np.pi)
                        max_throw = min(base_thick * 0.6, 12.0)
                        throw = c * self.rng.uniform(base_thick*0.2, max_throw)
                        dx, dy = xx - cx, yy - cy
                        dist = dx*np.cos(fault_dir) + dy*np.sin(fault_dir)
                        width = self.rng.uniform(1.0, 3.0)
                        fault += throw * np.tanh(dist / width)

                thickness_field = base_thick + tilt + fold + fault
            else:
                thickness_field = np.full_like(xx, base_thick)

            thickness_field = np.maximum(thickness_field, 0.5)
            top = current_bottom + thickness_field
            stratum_tops.append(top)
            current_bottom = top.copy()

        return stratum_tops

    def generate_voxel(self):
        """
        Bottom-up stacking: bottom stratum starts at -depth,
        each stratum sits directly on top of the previous one,
        soil cap directly on top of the topmost stratum.
        No gaps, no bedrock.
        """
        x = np.linspace(-self.radius, self.radius, self.resolution_xy)
        y = np.linspace(-self.radius, self.radius, self.resolution_xy)
        z = np.linspace(-self.depth, self.max_height, self.resolution_z)

        inside_mask = self._generate_boundary_mask(x, y)
        terrain_surface = self._generate_terrain_surface(x, y)

        # ---- Patchy soil cover: ridged noise + random threshold ----
        # Noise below threshold → bare rock (soil thickness = 0).
        # Noise above threshold → normal soil thickness.
        # This creates clear patches of exposed rock and soil-covered areas.
        xx, yy = np.meshgrid(x, y)
        soil_pn = _PerlinNoise(base=self.rng.randint(0, 256))
        soil_raw = soil_pn.fbm2(
            xx / (self.scale * 0.5), yy / (self.scale * 0.5),
            octaves=3, persistence=0.5, lacunarity=2.0)
        soil_noise = 1.0 - np.abs(soil_raw)  # ridged → [0,1]
        soil_noise = soil_noise - soil_noise.min()
        soil_noise = soil_noise / soil_noise.max()

        # Random threshold within user-defined range
        t_min = self.soil_threshold_min
        t_max = self.soil_threshold_max
        threshold = self.rng.uniform(t_min, t_max)
        soil_thickness_field = np.where(
            soil_noise > threshold,
            (soil_noise - threshold) / (1.0 - threshold) * self.soil_thickness,
            0.0
        )

        # Per-point soil bottom (varies across the terrain)
        soil_bottom_field = terrain_surface - soil_thickness_field

        stratum_tops = self._generate_stratum_tops(x, y)

        # Force top stratum to extend up to soil_bottom_field point-by-point —
        # top rock layer follows the soil base everywhere, no gap, no flattening.
        if stratum_tops:
            if len(stratum_tops) >= 2:
                stratum_tops[-1] = np.maximum(soil_bottom_field, stratum_tops[-2] + 0.1)
            else:
                stratum_tops[-1] = soil_bottom_field

        nx, ny, nz = self.resolution_xy, self.resolution_xy, self.resolution_z
        n_strata = len(self.strata)
        SOIL_ID = n_strata + 1

        solid = np.zeros((nx, ny, nz), dtype=np.int32)

        for i in range(nx):
            for j in range(ny):
                if not inside_mask[i, j]:
                    continue

                surf_z = terrain_surface[i, j]
                sb_z = soil_bottom_field[i, j]  # per-point, non-uniform

                for k in range(nz):
                    cz = z[k]

                    if cz > surf_z:
                        solid[i, j, k] = 0  # air above terrain
                    elif self.soil_thickness > 0 and cz >= sb_z:
                        solid[i, j, k] = SOIL_ID  # soil cap (patchy)
                    else:
                        # Bottom-up: strata[0] at bottom, strata[-1] at top
                        for s_idx in range(n_strata):
                            if cz <= stratum_tops[s_idx][i, j]:
                                solid[i, j, k] = s_idx + 1
                                break
                        # Below deepest stratum → remains 0 (air)

        return solid, x, y, z, stratum_tops, soil_bottom_field, terrain_surface, inside_mask


# =========================================================
# Fracture / Lens Config
# =========================================================

class FractureConfig:
    """Configuration for a single fracture feature."""

    def __init__(self, name="裂隙", length=15.0, width_cells=1,
                 permeability=1000.0, color=None):
        self.name = str(name)
        self.length = float(length)
        self.width_cells = max(1, width_cells)
        self.permeability = float(permeability)
        self.color = list(color) if color else [0.95, 0.15, 0.15]

    def to_dict(self):
        return {"name": self.name, "length": self.length,
                "width_cells": self.width_cells,
                "permeability": self.permeability, "color": self.color.copy()}

    @classmethod
    def from_dict(cls, d):
        return cls(name=d.get("name", "裂隙"), length=d.get("length", 15.0),
                   width_cells=d.get("width_cells", 1),
                   permeability=d.get("permeability", 1000.0),
                   color=d.get("color", [0.95, 0.15, 0.15]))


class LensConfig:
    """Configuration for a single lens feature."""

    def __init__(self, name="透镜体", radius=5.0,
                 permeability=100.0, color=None):
        self.name = str(name)
        self.radius = float(radius)
        self.permeability = float(permeability)
        self.color = list(color) if color else [0.2, 0.8, 0.9]

    def to_dict(self):
        return {"name": self.name, "radius": self.radius,
                "permeability": self.permeability, "color": self.color.copy()}

    @classmethod
    def from_dict(cls, d):
        return cls(name=d.get("name", "透镜体"), radius=d.get("radius", 5.0),
                   permeability=d.get("permeability", 100.0),
                   color=d.get("color", [0.2, 0.8, 0.9]))


# =========================================================
# Fracture Generator — Post-processing (config-list driven)
# =========================================================

class FractureGenerator:
    """
    Generate fractures from a FractureConfig list.
    Each config produces one fracture line segment.
    Fractures do NOT penetrate soil cap or ground surface.
    """

    def __init__(self, configs, seed=None):
        # Accept FractureConfig objects or dicts
        self.configs = []
        for c in (configs or []):
            if isinstance(c, dict):
                self.configs.append(FractureConfig.from_dict(c))
            else:
                self.configs.append(c)
        self.seed = seed

    def generate(self, solid, x, y, z, stratum_tops, terrain_surface, inside_mask):
        nx, ny, nz = solid.shape
        if not self.configs or not stratum_tops:
            return np.zeros((nx, ny, nz), dtype=np.int32)

        rng = np.random.RandomState(self.seed)
        fracture_ids = np.zeros((nx, ny, nz), dtype=np.int32)
        n_strata = len(stratum_tops)

        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
        avg_d = (dx + dy + dz) / 3.0

        rock_mask = (solid > 0) & (solid <= n_strata)
        rock_indices = np.argwhere(rock_mask)
        if len(rock_indices) == 0:
            return fracture_ids

        for fid, cfg in enumerate(self.configs, 1):
            idx = rng.randint(0, len(rock_indices))
            si, sj, sk = rock_indices[idx]

            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            dir_x = np.sin(phi) * np.cos(theta)
            dir_y = np.sin(phi) * np.sin(theta)
            dir_z = np.cos(phi)

            current_len = 0.0
            ci, cj, ck = float(si), float(sj), float(sk)
            path_cells = []

            while current_len < cfg.length:
                ii, jj, kk = int(round(ci)), int(round(cj)), int(round(ck))
                if not (0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz):
                    break
                if not inside_mask[ii, jj]:
                    break
                cell_marker = solid[ii, jj, kk]
                if cell_marker == 0 or cell_marker == n_strata + 1:
                    break
                if z[kk] > terrain_surface[ii, jj]:
                    break
                path_cells.append((ii, jj, kk))
                current_len += avg_d
                ci += dir_x * avg_d / dx
                cj += dir_y * avg_d / dy
                ck += dir_z * avg_d / dz

            # Mark path + width
            wc = cfg.width_cells
            for ii, jj, kk in path_cells:
                fracture_ids[ii, jj, kk] = fid
                if wc > 1:
                    half = wc // 2
                    for wi in range(-half, wc - half):
                        for wj in range(-half, wc - half):
                            ni_idx, nj_idx = ii + wi, jj + wj
                            if (0 <= ni_idx < nx and 0 <= nj_idx < ny and
                                    solid[ni_idx, nj_idx, kk] > 0 and
                                    solid[ni_idx, nj_idx, kk] <= n_strata):
                                fracture_ids[ni_idx, nj_idx, kk] = fid

        return fracture_ids


# =========================================================
# Lens Generator — Post-processing (config-list driven)
# =========================================================

class LensGenerator:
    """
    Generate lenses from a LensConfig list.
    Each config produces one ellipsoidal lens.
    Lenses do NOT penetrate soil cap or ground surface.
    """

    def __init__(self, configs, seed=None):
        # Accept LensConfig objects or dicts
        self.configs = []
        for c in (configs or []):
            if isinstance(c, dict):
                self.configs.append(LensConfig.from_dict(c))
            else:
                self.configs.append(c)
        self.seed = seed

    def generate(self, solid, x, y, z, stratum_tops, terrain_surface, inside_mask):
        nx, ny, nz = solid.shape
        if not self.configs or not stratum_tops:
            return np.zeros((nx, ny, nz), dtype=np.int32)

        rng = np.random.RandomState(self.seed)
        lens_ids = np.zeros((nx, ny, nz), dtype=np.int32)
        n_strata = len(stratum_tops)

        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        dz = float(z[1] - z[0]) if len(z) > 1 else 1.0

        # Pre-filter: rock cells that are inside the boundary
        rock_mask = (solid > 0) & (solid <= n_strata) & inside_mask[:, :, np.newaxis]
        rock_indices = np.argwhere(rock_mask)
        if len(rock_indices) == 0:
            return lens_ids

        # Top surface of rock = stratum_tops[-1] (contact with soil cap)
        rock_top = stratum_tops[-1] if stratum_tops else terrain_surface

        for lid, cfg in enumerate(self.configs, 1):
            max_retries = 50
            placed = False

            for _attempt in range(max_retries):
                # Pick center from rock cells inside boundary
                idx = rng.randint(0, len(rock_indices))
                ci, cj, ck = rock_indices[idx]

                # Random ellipsoid axes (natural variation)
                rx = max(1, int(cfg.radius / dx * rng.uniform(0.7, 1.3)))
                ry = max(1, int(cfg.radius / dy * rng.uniform(0.7, 1.3)))
                rz = max(1, int(cfg.radius / dz * rng.uniform(0.7, 1.3)))

                # Check: ellipsoid top must not exceed rock top surface
                # (so it won't penetrate soil cap)
                top_z = z[ck] + rz * dz
                if top_z > rock_top[ci, cj]:
                    continue  # too close to surface, try again

                # Generate ellipsoid — cells outside boundary are naturally
                # clipped by the inside_mask check, so boundary truncation is OK
                for ii in range(max(0, ci - rx), min(nx, ci + rx + 1)):
                    for jj in range(max(0, cj - ry), min(ny, cj + ry + 1)):
                        for kk in range(max(0, ck - rz), min(nz, ck + rz + 1)):
                            if not inside_mask[ii, jj]:
                                continue
                            dx_n = (ii - ci) / max(rx, 1)
                            dy_n = (jj - cj) / max(ry, 1)
                            dz_n = (kk - ck) / max(rz, 1)
                            if dx_n**2 + dy_n**2 + dz_n**2 > 1.0:
                                continue
                            if solid[ii, jj, kk] > 0 and solid[ii, jj, kk] <= n_strata:
                                lens_ids[ii, jj, kk] = lid

                placed = True
                break  # successfully placed

        return lens_ids


# =========================================================
# Stratum Edit Dialog
# =========================================================

class StratumEditDialog(QDialog):
    def __init__(self, config=None, parent=None, is_soil_cap=False, hide_thickness=False):
        super().__init__(parent)
        self.config = config or StratumConfig()
        self.is_soil_cap = is_soil_cap
        self.hide_thickness = hide_thickness
        self.setWindowTitle("编辑覆土层" if is_soil_cap else "编辑地层")
        self.setMinimumWidth(350)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        form = QFormLayout()

        # Name
        self.name_edit = QLineEdit(self.config.name)
        form.addRow("地层名称:", self.name_edit)

        # Preset combo
        self.preset_combo = QComboBox()
        presets = list(StratumConfig.PRESETS.keys())
        self.preset_combo.addItems(presets)
        idx = self.preset_combo.findText(self.config.preset_name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        form.addRow("预设类型:", self.preset_combo)

        # Thickness (hidden for soil cap and top stratum — system-calculated)
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(1.0, 500.0)
        self.thickness_spin.setValue(self.config.thickness)
        self.thickness_spin.setSingleStep(1.0)
        self.thickness_spin.setDecimals(1)
        if not self.is_soil_cap and not self.hide_thickness:
            form.addRow("平均厚度 (m):", self.thickness_spin)

        # Permeability
        self.perm_spin = QDoubleSpinBox()
        self.perm_spin.setRange(0.0, 10000.0)
        self.perm_spin.setDecimals(8)
        self.perm_spin.setSingleStep(0.0001)
        self.perm_spin.setValue(round(self.config.permeability, 8))
        form.addRow("渗透系数 K:", self.perm_spin)

        # Color
        color_layout = QHBoxLayout()
        self.color_preview = QPushButton()
        self.color_preview.setFixedSize(50, 28)
        self.current_color = QColor(
            int(self.config.color[0]*255), int(self.config.color[1]*255), int(self.config.color[2]*255))
        self._update_color_preview()
        self.color_preview.clicked.connect(self.choose_color)
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(QLabel("点击选择颜色"))
        color_layout.addStretch()
        form.addRow("地层颜色:", color_layout)

        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def on_preset_changed(self, text):
        preset = StratumConfig.PRESETS.get(text)
        if preset:
            c = preset["color"]
            self.current_color = QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))
            self._update_color_preview()
            self.perm_spin.setValue(round(preset.get("permeability", 1.0), 8))

    def choose_color(self):
        color = QColorDialog.getColor(self.current_color, self, "选择地层颜色")
        if color.isValid():
            self.current_color = color
            self._update_color_preview()

    def _update_color_preview(self):
        self.color_preview.setStyleSheet(
            f"background-color: {self.current_color.name()}; border: 1px solid #999; border-radius: 3px;")

    def get_config(self):
        return StratumConfig(
            name=self.name_edit.text().strip() or "地层",
            thickness=self.thickness_spin.value(),
            color=[self.current_color.redF(), self.current_color.greenF(), self.current_color.blueF()],
            permeability=round(self.perm_spin.value(), 8),
            preset_name=self.preset_combo.currentText(),
        )


# =========================================================
# Fracture Edit Dialog
# =========================================================

class FractureEditDialog(QDialog):
    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.config = config or FractureConfig()
        self.setWindowTitle("编辑裂隙")
        self.setMinimumWidth(350)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        form = QFormLayout()

        self.name_edit = QLineEdit(self.config.name)
        form.addRow("裂隙名称:", self.name_edit)

        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(1.0, 200.0)
        self.length_spin.setValue(self.config.length)
        self.length_spin.setSingleStep(1.0)
        self.length_spin.setDecimals(1)
        form.addRow("长度 (m):", self.length_spin)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10)
        self.width_spin.setValue(self.config.width_cells)
        form.addRow("宽度 (cell数):", self.width_spin)

        self.perm_spin = QDoubleSpinBox()
        self.perm_spin.setRange(0.0, 100000.0)
        self.perm_spin.setDecimals(8)
        self.perm_spin.setSingleStep(1.0)
        self.perm_spin.setValue(round(self.config.permeability, 8))
        form.addRow("渗透系数 K:", self.perm_spin)

        color_layout = QHBoxLayout()
        self.color_preview = QPushButton()
        self.color_preview.setFixedSize(50, 28)
        self.current_color = QColor(
            int(self.config.color[0]*255), int(self.config.color[1]*255), int(self.config.color[2]*255))
        self._update_color_preview()
        self.color_preview.clicked.connect(self.choose_color)
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(QLabel("点击选择颜色"))
        color_layout.addStretch()
        form.addRow("裂隙颜色:", color_layout)

        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def choose_color(self):
        color = QColorDialog.getColor(self.current_color, self, "选择裂隙颜色")
        if color.isValid():
            self.current_color = color
            self._update_color_preview()

    def _update_color_preview(self):
        self.color_preview.setStyleSheet(
            f"background-color: {self.current_color.name()}; border: 1px solid #999; border-radius: 3px;")

    def get_config(self):
        return FractureConfig(
            name=self.name_edit.text().strip() or "裂隙",
            length=self.length_spin.value(),
            width_cells=self.width_spin.value(),
            permeability=round(self.perm_spin.value(), 8),
            color=[self.current_color.redF(), self.current_color.greenF(), self.current_color.blueF()],
        )


# =========================================================
# Lens Edit Dialog
# =========================================================

class LensEditDialog(QDialog):
    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.config = config or LensConfig()
        self.setWindowTitle("编辑透镜体")
        self.setMinimumWidth(350)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        form = QFormLayout()

        self.name_edit = QLineEdit(self.config.name)
        form.addRow("透镜体名称:", self.name_edit)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.5, 100.0)
        self.radius_spin.setValue(self.config.radius)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setDecimals(1)
        form.addRow("半径 (m):", self.radius_spin)

        self.perm_spin = QDoubleSpinBox()
        self.perm_spin.setRange(0.0, 100000.0)
        self.perm_spin.setDecimals(8)
        self.perm_spin.setSingleStep(0.1)
        self.perm_spin.setValue(round(self.config.permeability, 8))
        form.addRow("渗透系数 K:", self.perm_spin)

        color_layout = QHBoxLayout()
        self.color_preview = QPushButton()
        self.color_preview.setFixedSize(50, 28)
        self.current_color = QColor(
            int(self.config.color[0]*255), int(self.config.color[1]*255), int(self.config.color[2]*255))
        self._update_color_preview()
        self.color_preview.clicked.connect(self.choose_color)
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(QLabel("点击选择颜色"))
        color_layout.addStretch()
        form.addRow("透镜体颜色:", color_layout)

        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def choose_color(self):
        color = QColorDialog.getColor(self.current_color, self, "选择透镜体颜色")
        if color.isValid():
            self.current_color = color
            self._update_color_preview()

    def _update_color_preview(self):
        self.color_preview.setStyleSheet(
            f"background-color: {self.current_color.name()}; border: 1px solid #999; border-radius: 3px;")

    def get_config(self):
        return LensConfig(
            name=self.name_edit.text().strip() or "透镜体",
            radius=self.radius_spin.value(),
            permeability=round(self.perm_spin.value(), 8),
            color=[self.current_color.redF(), self.current_color.greenF(), self.current_color.blueF()],
        )


# =========================================================
# Main Window
# =========================================================

class MainWindow(QMainWindow):
    """
    Display order (top→bottom): soil cap → top stratum → ... → bottom stratum
    Internal order (bottom→top): strata[0] = bottom, strata[-1] = top
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Groundwater Geology Generator — 多层地质建模系统")
        self.resize(1700, 1000)

        self.current_meshes = {}
        self._tracked_actors = []

        # Soil cap (fixed, displayed at top of list)
        self.soil_config = StratumConfig(
            name="覆土层", thickness=5.0,
            color=[0.76, 0.70, 0.50], permeability=0.01, preset_name="custom")

        # User strata: strata[0] = physical bottom, strata[-1] = physical top
        self.strata = [
            StratumConfig("页岩层", thickness=12.0, color=[0.35, 0.35, 0.38], permeability=0.0001, preset_name="页岩层"),
            StratumConfig("砂岩层", thickness=15.0, color=[0.76, 0.70, 0.50], permeability=1.0, preset_name="砂岩层"),
            StratumConfig("黏土层", thickness=10.0, color=[0.55, 0.40, 0.25], permeability=0.001, preset_name="黏土层"),
        ]

        # Fracture configs (each produces one fracture)
        self.fracture_configs = [
            FractureConfig("主裂隙", length=15.0, width_cells=1, permeability=1000.0, color=[0.95, 0.15, 0.15]),
            FractureConfig("次裂隙", length=10.0, width_cells=1, permeability=500.0, color=[0.85, 0.25, 0.25]),
        ]

        # Lens configs (each produces one lens)
        self.lens_configs = [
            LensConfig("砂透镜", radius=5.0, permeability=50.0, color=[0.2, 0.8, 0.9]),
            LensConfig("砾透镜", radius=3.0, permeability=200.0, color=[0.3, 0.7, 0.85]),
        ]

        self.setup_ui()
        self.refresh_stratum_list()
        self.refresh_fracture_list()
        self.refresh_lens_list()

    # ---- UI Setup ----

    def setup_ui(self):
        self.central = QWidget()
        self.setCentralWidget(self.central)
        main_layout = QHBoxLayout()
        self.central.setLayout(main_layout)

        # Left panel
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(360)
        control_layout = QVBoxLayout()
        self.control_panel.setLayout(control_layout)
        main_layout.addWidget(self.control_panel)

        self.tabs = QTabWidget()
        control_layout.addWidget(self.tabs)

        self.tab_basic = QWidget()
        self.tabs.addTab(self.tab_basic, "基础参数")
        self._setup_basic_tab()

        self.tab_noise = QWidget()
        self.tabs.addTab(self.tab_noise, "噪声参数")
        self._setup_noise_tab()

        self.tab_strata = QWidget()
        self.tabs.addTab(self.tab_strata, "地层管理")
        self._setup_strata_tab()

        self.tab_hetero = QWidget()
        self.tabs.addTab(self.tab_hetero, "非均质构造管理")
        self._setup_hetero_tab()

        self.status_label = QLabel(" 就绪，可生成地质模型 ")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)

        self.generate_button = QPushButton("生成 3D 地质模型")
        self.generate_button.setMinimumHeight(42)
        self.generate_button.clicked.connect(self.generate_model)
        control_layout.addWidget(self.generate_button)

        export_layout = QHBoxLayout()
        self.export_data_btn = QPushButton("导出网格数据")
        self.export_data_btn.clicked.connect(self.export_mesh_data)
        self.export_data_btn.setToolTip("导出 MODFLOW 友好的 .npz 格式")
        self.export_vtk_btn = QPushButton("导出 VTK")
        self.export_vtk_btn.clicked.connect(self.export_mesh_vtk)
        self.export_vtk_btn.setToolTip("导出 VTK 体素网格用于可视化")
        export_layout.addWidget(self.export_data_btn)
        export_layout.addWidget(self.export_vtk_btn)
        control_layout.addLayout(export_layout)

        # Post-processing button
        self.postprocess_btn = QPushButton("网格参数调整")
        self.postprocess_btn.setMinimumHeight(36)
        self.postprocess_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white; font-weight: bold;"
            "  border: none; padding: 8px; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { background-color: #3498db; }")
        self.postprocess_btn.clicked.connect(self.open_postprocess_window)
        self.postprocess_btn.setToolTip("K随机化 / 水头模拟 / 污染物输运")
        control_layout.addWidget(self.postprocess_btn)

        control_layout.addStretch()
        self.set_button_ready()

        # Right: 3D viewer
        self.viewer_widget = QWidget()
        viewer_layout = QVBoxLayout()
        self.viewer_widget.setLayout(viewer_layout)
        main_layout.addWidget(self.viewer_widget)
        self.plotter = QtInteractor(self.viewer_widget)
        viewer_layout.addWidget(self.plotter.interactor)

    def _setup_basic_tab(self):
        layout = QVBoxLayout()
        self.tab_basic.setLayout(layout)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        self.radius_spin = self._add_spin(layout, "区域半径 (m)", 10, 1000, 100)
        self.irregularity_spin = self._add_double(layout, "边界不规则度", 0.0, 1.0, 0.3, 0.05)
        self.depth_spin = self._add_spin(layout, "地下深度 (m)", 1, 500, 50)
        self.height_spin = self._add_spin(layout, "最大地表高度 (m)", 1, 500, 15)
        self.grid_spin = self._add_spin(layout, "XY 网格分辨率", 20, 300, 80)
        self.soil_spin = self._add_double(layout, "覆土厚度 (m)", 0.0, 50.0, 5.0, 0.5)
        self.soil_th_min_spin = self._add_double(layout, "覆土阈值下限", 0.0, 1.0, 0.15, 0.05)
        self.soil_th_max_spin = self._add_double(layout, "覆土阈值上限", 0.0, 1.0, 0.35, 0.05)
        self.complexity_spin = self._add_double(layout, "地层复杂度", 0.0, 1.0, 0.3, 0.05)

        info = QLabel("复杂度参考:\n  0.0=完全水平  0.4=倾斜+褶皱  0.8=强变形+断层\n覆土阈值: 下限越低裸露越少，上限越高裸露越多")
        info.setStyleSheet("color: #555; font-size: 11px; padding: 6px; background: #f5f5f5; border-radius: 4px;")
        layout.addWidget(info)
        layout.addStretch()

    def _setup_strata_tab(self):
        layout = QVBoxLayout()
        self.tab_strata.setLayout(layout)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(QLabel("地层列表（从上到下 = 物理从上到下）"))

        self.strata_list = QListWidget()
        self.strata_list.setMinimumHeight(220)
        self.strata_list.itemDoubleClicked.connect(self.edit_selected_stratum)
        layout.addWidget(self.strata_list)

        btn_grid = QGridLayout()
        self.add_btn = QPushButton("+ 添加")
        self.add_btn.clicked.connect(self.add_stratum)
        self.del_btn = QPushButton("- 删除")
        self.del_btn.clicked.connect(self.delete_selected_stratum)
        self.edit_btn = QPushButton("编辑")
        self.edit_btn.clicked.connect(self.edit_selected_stratum)
        self.up_btn = QPushButton("上移")
        self.up_btn.clicked.connect(self.move_stratum_up)
        self.down_btn = QPushButton("下移")
        self.down_btn.clicked.connect(self.move_stratum_down)
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.clicked.connect(self.reset_strata)

        btn_grid.addWidget(self.add_btn, 0, 0)
        btn_grid.addWidget(self.del_btn, 0, 1)
        btn_grid.addWidget(self.edit_btn, 0, 2)
        btn_grid.addWidget(self.up_btn, 1, 0)
        btn_grid.addWidget(self.down_btn, 1, 1)
        btn_grid.addWidget(self.reset_btn, 1, 2)
        layout.addLayout(btn_grid)

        self.strata_info = QLabel("覆土层(固定) + 3 个地层")
        self.strata_info.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.strata_info)
        layout.addStretch()

    def _setup_noise_tab(self):
        layout = QVBoxLayout()
        self.tab_noise.setLayout(layout)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        self.scale_spin = self._add_spin(layout, "地形起伏尺度", 5, 300, 60)
        self.octave_spin = self._add_spin(layout, "噪声叠加层数", 1, 10, 5)
        self.persistence_spin = self._add_double(layout, "Persistence", 0.1, 1.0, 0.5, 0.05)
        self.lacunarity_spin = self._add_double(layout, "Lacunarity", 1.0, 5.0, 2.0, 0.1)
        layout.addStretch()

    def _setup_hetero_tab(self):
        layout = QVBoxLayout()
        self.tab_hetero.setLayout(layout)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # ---- Fracture list ----
        frac_title = QLabel("— 裂隙列表 —")
        frac_title.setStyleSheet("font-weight: bold; color: #c0392b;")
        frac_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(frac_title)

        self.fracture_list = QListWidget()
        self.fracture_list.setMinimumHeight(140)
        self.fracture_list.itemDoubleClicked.connect(self.edit_selected_fracture)
        layout.addWidget(self.fracture_list)

        frac_btn = QGridLayout()
        self.frac_add_btn = QPushButton("+ 添加")
        self.frac_add_btn.clicked.connect(self.add_fracture)
        self.frac_del_btn = QPushButton("- 删除")
        self.frac_del_btn.clicked.connect(self.delete_selected_fracture)
        self.frac_edit_btn = QPushButton("编辑")
        self.frac_edit_btn.clicked.connect(self.edit_selected_fracture)
        frac_btn.addWidget(self.frac_add_btn, 0, 0)
        frac_btn.addWidget(self.frac_del_btn, 0, 1)
        frac_btn.addWidget(self.frac_edit_btn, 0, 2)
        layout.addLayout(frac_btn)

        self.fracture_info = QLabel("2 个裂隙")
        self.fracture_info.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.fracture_info)

        # ---- Lens list ----
        lens_title = QLabel("— 透镜体列表 —")
        lens_title.setStyleSheet("font-weight: bold; color: #2980b9; margin-top: 8px;")
        lens_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lens_title)

        self.lens_list = QListWidget()
        self.lens_list.setMinimumHeight(140)
        self.lens_list.itemDoubleClicked.connect(self.edit_selected_lens)
        layout.addWidget(self.lens_list)

        lens_btn = QGridLayout()
        self.lens_add_btn = QPushButton("+ 添加")
        self.lens_add_btn.clicked.connect(self.add_lens)
        self.lens_del_btn = QPushButton("- 删除")
        self.lens_del_btn.clicked.connect(self.delete_selected_lens)
        self.lens_edit_btn = QPushButton("编辑")
        self.lens_edit_btn.clicked.connect(self.edit_selected_lens)
        lens_btn.addWidget(self.lens_add_btn, 0, 0)
        lens_btn.addWidget(self.lens_del_btn, 0, 1)
        lens_btn.addWidget(self.lens_edit_btn, 0, 2)
        layout.addLayout(lens_btn)

        self.lens_info = QLabel("2 个透镜体")
        self.lens_info.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.lens_info)

        layout.addStretch()

    def _add_spin(self, parent_layout, label, min_v, max_v, default):
        lbl = QLabel(label)
        lbl.setWordWrap(True)
        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        spin.setKeyboardTracking(False)
        parent_layout.addWidget(lbl)
        parent_layout.addWidget(spin)
        return spin

    def _add_double(self, parent_layout, label, min_v, max_v, default, step):
        lbl = QLabel(label)
        lbl.setWordWrap(True)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)
        spin.setValue(default)
        spin.setDecimals(3 if step < 0.1 else 2)
        spin.setKeyboardTracking(False)
        parent_layout.addWidget(lbl)
        parent_layout.addWidget(spin)
        return spin

    # ================================================================
    # Row <-> Index Mapping
    # ================================================================
    # List display (top→bottom): Row 0=soil, Row 1=top stratum, ..., Row N=bottom stratum
    # Internal storage (bottom→top): strata[0]=bottom, strata[-1]=top
    #
    # Row 0 → soil cap (special)
    # Row k (k>=1) → strata[-k]  (top stratum at row 1, bottom stratum at row len(strata))
    #
    # strata[i] → Row = len(strata) - i
    # ================================================================

    def _row_to_index(self, row):
        """Convert list row to strata array index. Row 0 = soil cap."""
        if row <= 0:
            return None  # soil cap
        idx = len(self.strata) - row
        if 0 <= idx < len(self.strata):
            return idx
        return None

    def _index_to_row(self, idx):
        """Convert strata array index to list row."""
        return len(self.strata) - idx

    # ---- Strata List (top-to-bottom display) ----

    def refresh_stratum_list(self):
        self.strata_list.clear()

        # Row 0: Soil cap (always on top)
        sc = self.soil_config
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(sc.color[0]*255), int(sc.color[1]*255), int(sc.color[2]*255))
        k_str = f"{sc.permeability:.6g}"
        text = f"[覆土] {sc.name}  |  厚={sc.thickness:.1f}m  |  K={k_str}"
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, -1)
        item.setBackground(QColor(240, 248, 240))
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(color_hex))
        item.setIcon(QIcon(pixmap))
        self.strata_list.addItem(item)

        # Rows 1..N: strata from top to bottom
        # strata[-1] (physical top) → Row 1
        # strata[0] (physical bottom) → Row len(strata)
        for display_idx in range(len(self.strata)):
            arr_idx = len(self.strata) - 1 - display_idx  # -1, -2, ..., 0
            s = self.strata[arr_idx]
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(s.color[0]*255), int(s.color[1]*255), int(s.color[2]*255))
            k_str = f"{s.permeability:.6g}"
            # Top stratum (display_idx==0) has auto-calculated thickness
            if display_idx == 0:
                thick_str = "系统自动计算"
            else:
                thick_str = f"平均厚={s.thickness:.1f}m"
            text = f"[{display_idx+1}] {s.name}  |  {thick_str}  |  K={k_str}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, arr_idx)
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(color_hex))
            item.setIcon(QIcon(pixmap))
            self.strata_list.addItem(item)

        self.strata_info.setText(f"覆土层(固定) + {len(self.strata)} 个地层")

    # ---- Strata CRUD ----

    def add_stratum(self):
        """Add new stratum at the top (just below soil cap)."""
        dialog = StratumEditDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            # Insert at end of array = physical top (displayed just below soil)
            self.strata.append(dialog.get_config())
            self.refresh_stratum_list()

    def edit_selected_stratum(self):
        row = self.strata_list.currentRow()
        if row < 0:
            return
        if row == 0:
            # Edit soil cap
            dialog = StratumEditDialog(self.soil_config, parent=self, is_soil_cap=True)
            if dialog.exec_() == QDialog.Accepted:
                new_cfg = dialog.get_config()
                self.soil_config.name = new_cfg.name
                self.soil_config.color = new_cfg.color
                self.soil_config.permeability = round(new_cfg.permeability, 6)
                self.soil_config.preset_name = new_cfg.preset_name
                self.refresh_stratum_list()
            return

        idx = self._row_to_index(row)
        if idx is None:
            return
        # Top stratum (physical top, just below soil) has auto-calculated thickness
        is_top = (idx == len(self.strata) - 1)
        dialog = StratumEditDialog(self.strata[idx], parent=self, hide_thickness=is_top)
        if dialog.exec_() == QDialog.Accepted:
            self.strata[idx] = dialog.get_config()
            self.refresh_stratum_list()

    def delete_selected_stratum(self):
        row = self.strata_list.currentRow()
        if row <= 0:
            QMessageBox.information(self, "提示", "覆土层不可删除")
            return
        idx = self._row_to_index(row)
        if idx is None:
            return
        reply = QMessageBox.question(self, "确认删除",
                                     f"删除地层 '{self.strata[idx].name}'？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.strata.pop(idx)
            self.refresh_stratum_list()

    def move_stratum_up(self):
        """Move selected stratum UP in the list = toward physical top."""
        row = self.strata_list.currentRow()
        if row <= 1:  # soil cap (0) or topmost stratum (1) cannot move up
            return
        idx = self._row_to_index(row)
        if idx is None or idx >= len(self.strata) - 1:
            return
        # Moving UP in display = moving toward end of array (physical top)
        self.strata[idx], self.strata[idx + 1] = self.strata[idx + 1], self.strata[idx]
        self.refresh_stratum_list()
        self.strata_list.setCurrentRow(row - 1)  # follows the item up

    def move_stratum_down(self):
        """Move selected stratum DOWN in the list = toward physical bottom."""
        row = self.strata_list.currentRow()
        if row <= 0:
            return
        idx = self._row_to_index(row)
        if idx is None or idx <= 0:
            return
        # Moving DOWN in display = moving toward start of array (physical bottom)
        self.strata[idx], self.strata[idx - 1] = self.strata[idx - 1], self.strata[idx]
        self.refresh_stratum_list()
        self.strata_list.setCurrentRow(row + 1)  # follows the item down

    def reset_strata(self):
        self.soil_config = StratumConfig(
            name="覆土层", thickness=self.soil_spin.value(),
            color=[0.76, 0.70, 0.50], permeability=0.01, preset_name="custom")
        self.strata = [
            StratumConfig("页岩层", thickness=12.0, color=[0.35, 0.35, 0.38], permeability=0.0001, preset_name="页岩层"),
            StratumConfig("砂岩层", thickness=15.0, color=[0.76, 0.70, 0.50], permeability=1.0, preset_name="砂岩层"),
            StratumConfig("黏土层", thickness=10.0, color=[0.55, 0.40, 0.25], permeability=0.001, preset_name="黏土层"),
        ]
        self.refresh_stratum_list()

    # ---- Fracture CRUD ----

    def refresh_fracture_list(self):
        self.fracture_list.clear()
        for idx, f in enumerate(self.fracture_configs):
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(f.color[0]*255), int(f.color[1]*255), int(f.color[2]*255))
            k_str = f"{f.permeability:.4g}"
            text = f"[{idx+1}] {f.name}  |  长={f.length:.1f}m  |  宽={f.width_cells}cell  |  K={k_str}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(color_hex))
            item.setIcon(QIcon(pixmap))
            self.fracture_list.addItem(item)
        self.fracture_info.setText(f"{len(self.fracture_configs)} 个裂隙")

    def add_fracture(self):
        dialog = FractureEditDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.fracture_configs.append(dialog.get_config())
            self.refresh_fracture_list()

    def edit_selected_fracture(self):
        row = self.fracture_list.currentRow()
        if row < 0 or row >= len(self.fracture_configs):
            return
        dialog = FractureEditDialog(self.fracture_configs[row], parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.fracture_configs[row] = dialog.get_config()
            self.refresh_fracture_list()

    def delete_selected_fracture(self):
        row = self.fracture_list.currentRow()
        if row < 0 or row >= len(self.fracture_configs):
            return
        reply = QMessageBox.question(self, "确认删除",
                                     f"删除裂隙 '{self.fracture_configs[row].name}'？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.fracture_configs.pop(row)
            self.refresh_fracture_list()

    # ---- Lens CRUD ----

    def refresh_lens_list(self):
        self.lens_list.clear()
        for idx, l in enumerate(self.lens_configs):
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(l.color[0]*255), int(l.color[1]*255), int(l.color[2]*255))
            k_str = f"{l.permeability:.4g}"
            text = f"[{idx+1}] {l.name}  |  r={l.radius:.1f}m  |  K={k_str}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(color_hex))
            item.setIcon(QIcon(pixmap))
            self.lens_list.addItem(item)
        self.lens_info.setText(f"{len(self.lens_configs)} 个透镜体")

    def add_lens(self):
        dialog = LensEditDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.lens_configs.append(dialog.get_config())
            self.refresh_lens_list()

    def edit_selected_lens(self):
        row = self.lens_list.currentRow()
        if row < 0 or row >= len(self.lens_configs):
            return
        dialog = LensEditDialog(self.lens_configs[row], parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.lens_configs[row] = dialog.get_config()
            self.refresh_lens_list()

    def delete_selected_lens(self):
        row = self.lens_list.currentRow()
        if row < 0 or row >= len(self.lens_configs):
            return
        reply = QMessageBox.question(self, "确认删除",
                                     f"删除透镜体 '{self.lens_configs[row].name}'？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.lens_configs.pop(row)
            self.refresh_lens_list()

    # ---- Button State ----

    def set_button_busy(self):
        self.generate_button.setEnabled(False)
        self.generate_button.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold;"
            "  border: none; padding: 10px; border-radius: 6px; font-size: 14px; }"
            "QPushButton:disabled { background-color: #c0392b; color: #cccccc; }")
        self.status_label.setText(" 正在生成地质模型... ")
        self.status_label.setStyleSheet(
            "QLabel { background-color: #fdf2f2; color: #c0392b; font-weight: bold;"
            "  padding: 8px; border-radius: 4px; }")
        QApplication.processEvents()

    def set_button_ready(self):
        self.generate_button.setEnabled(True)
        self.generate_button.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold;"
            "  border: none; padding: 10px; border-radius: 6px; font-size: 14px; }"
            "QPushButton:hover { background-color: #2ecc71; }")
        self.status_label.setText(" 就绪，可生成地质模型 ")
        self.status_label.setStyleSheet(
            "QLabel { background-color: #eafaf1; color: #27ae60; font-weight: bold;"
            "  padding: 8px; border-radius: 4px; }")
        QApplication.processEvents()

    # ---- Safe Cleanup ----

    def _safe_remove_all_actors(self):
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        try:
            if hasattr(self.plotter, 'axes_widget') and self.plotter.axes_widget is not None:
                try:
                    self.plotter.axes_widget.SetEnabled(0)
                    self.plotter.axes_widget.SetInteractor(None)
                except Exception:
                    pass
                self.plotter.axes_widget = None
        except Exception:
            pass
        try:
            if hasattr(self.plotter, 'cube_axes_actor'):
                self.plotter.cube_axes_actor = None
        except Exception:
            pass
        try:
            self.plotter.renderer.RemoveAllViewProps()
        except Exception:
            pass
        try:
            self.plotter.renderer._actors = {}
        except Exception:
            pass
        self._tracked_actors = []

    # ---- Core Generation ----

    def generate_model(self):
        try:
            self.set_button_busy()

            if self.depth_spin.value() <= 0:
                QMessageBox.warning(self, "输入错误", "深度必须大于 0")
                self.set_button_ready()
                return
            if len(self.strata) == 0:
                QMessageBox.warning(self, "输入错误", "请至少添加一个地层")
                self.set_button_ready()
                return

            self._safe_remove_all_actors()

            radius = self.radius_spin.value()
            irregularity = self.irregularity_spin.value()
            depth = self.depth_spin.value()
            max_height = self.height_spin.value()
            resolution_xy = self.grid_spin.value()
            resolution_z = max(30, int(depth) + int(max_height))

            soil_thickness = self.soil_spin.value()
            self.soil_config.thickness = soil_thickness
            complexity = self.complexity_spin.value()

            generator = GeologyGenerator(
                radius=radius,
                irregularity=irregularity,
                depth=depth,
                max_height=max_height,
                resolution_xy=resolution_xy,
                resolution_z=resolution_z,
                scale=self.scale_spin.value(),
                octaves=self.octave_spin.value(),
                persistence=self.persistence_spin.value(),
                lacunarity=self.lacunarity_spin.value(),
                soil_thickness=soil_thickness,
                complexity=complexity,
                strata=[s.to_dict() for s in self.strata],
                soil_threshold_min=self.soil_th_min_spin.value(),
                soil_threshold_max=self.soil_th_max_spin.value(),
            )

            solid, x, y, z, stratum_tops, soil_bottom, terrain_surface, inside_mask = \
                generator.generate_voxel()

            self.last_solid = solid
            self.last_x, self.last_y, self.last_z = x, y, z
            self.last_stratum_tops = stratum_tops
            self.last_soil_bottom = soil_bottom
            self.last_terrain_surface = terrain_surface
            self.last_inside_mask = inside_mask

            # ---- Post-processing: Fractures (config-list driven) ----
            frac_gen = FractureGenerator(
                configs=self.fracture_configs,
                seed=generator.seed + 1000,
            )
            fracture_ids = frac_gen.generate(
                solid, x, y, z, stratum_tops, terrain_surface, inside_mask)
            self.last_fracture_ids = fracture_ids

            # ---- Post-processing: Lenses (config-list driven) ----
            lens_gen = LensGenerator(
                configs=self.lens_configs,
                seed=generator.seed + 2000,
            )
            lens_ids = lens_gen.generate(
                solid, x, y, z, stratum_tops, terrain_surface, inside_mask)
            self.last_lens_ids = lens_ids

            grid = pv.ImageData()
            grid.dimensions = np.array(solid.shape) + 1
            grid.origin = (float(x.min()), float(y.min()), float(z.min()))
            grid.spacing = (
                float(x[1] - x[0]) if len(x) > 1 else 1.0,
                float(y[1] - y[0]) if len(y) > 1 else 1.0,
                float(z[1] - z[0]) if len(z) > 1 else 1.0,
            )
            grid.cell_data["stratum_id"] = solid.flatten(order="F")
            grid.cell_data["fracture_id"] = fracture_ids.flatten(order="F")
            grid.cell_data["lens_id"] = lens_ids.flatten(order="F")
            # Build and save permeability array
            n_strata = len(self.strata)
            K_arr = np.zeros_like(solid, dtype=np.float64)
            for s_idx, s in enumerate(self.strata):
                K_arr[solid == (s_idx + 1)] = s.permeability
            if self.soil_config.thickness > 0:
                K_arr[solid == (n_strata + 1)] = self.soil_config.permeability
            for fid, fcfg in enumerate(self.fracture_configs, 1):
                K_arr[fracture_ids == fid] = fcfg.permeability
            for lid, lcfg in enumerate(self.lens_configs, 1):
                K_arr[lens_ids == lid] = lcfg.permeability
            grid.cell_data["permeability"] = K_arr.flatten(order="F")
            self.last_grid = grid

            # Layered rendering: bottom stratum → ... → top stratum → soil cap
            n_strata = len(self.strata)
            SOIL_ID = n_strata + 1
            OPACITY = 0.9
            self.current_meshes = {}

            render_order = []
            for s_idx, s in enumerate(self.strata):
                render_order.append((s_idx + 1, s.name, s.color))
            if soil_thickness > 0:
                render_order.append((SOIL_ID, self.soil_config.name, self.soil_config.color))

            for marker_id, name, color in render_order:
                layer_grid = grid.threshold(
                    [marker_id - 0.5, marker_id + 0.5],
                    scalars="stratum_id"
                )
                if layer_grid.n_cells == 0:
                    continue
                surface = layer_grid.extract_surface(algorithm='dataset_surface')
                if surface.n_points == 0:
                    continue
                self.current_meshes[marker_id] = surface
                actor = self.plotter.add_mesh(
                    surface, color=color, opacity=OPACITY,
                    smooth_shading=True, show_edges=False, lighting=True,
                    specular=0.3, specular_power=20)
                self._tracked_actors.append(actor)

            # ---- Render fractures (per-config color) ----
            for fid, fcfg in enumerate(self.fracture_configs, 1):
                mask = (fracture_ids == fid)
                if not np.any(mask):
                    continue
                frac_grid = grid.threshold([fid - 0.5, fid + 0.5], scalars="fracture_id")
                if frac_grid.n_cells > 0:
                    frac_surf = frac_grid.extract_surface(algorithm='dataset_surface')
                    if frac_surf.n_points > 0:
                        frac_actor = self.plotter.add_mesh(
                            frac_surf, color=fcfg.color, opacity=0.95,
                            smooth_shading=True, show_edges=False, lighting=True,
                            specular=0.5, specular_power=30)
                        self._tracked_actors.append(frac_actor)

            # ---- Render lenses (per-config color) ----
            for lid, lcfg in enumerate(self.lens_configs, 1):
                mask = (lens_ids == lid)
                if not np.any(mask):
                    continue
                lens_grid = grid.threshold([lid - 0.5, lid + 0.5], scalars="lens_id")
                if lens_grid.n_cells > 0:
                    lens_surf = lens_grid.extract_surface(algorithm='dataset_surface')
                    if lens_surf.n_points > 0:
                        lens_actor = self.plotter.add_mesh(
                            lens_surf, color=lcfg.color, opacity=0.85,
                            smooth_shading=True, show_edges=False, lighting=True,
                            specular=0.4, specular_power=25)
                        self._tracked_actors.append(lens_actor)

            axes_actor = self.plotter.add_axes()
            self._tracked_actors.append(axes_actor)
            self._add_reference_grid(radius, depth, max_height)
            self._add_bounds_labels(radius, depth, max_height)

            self.plotter.camera_position = 'iso'
            self.plotter.reset_camera()
            self.plotter.render()
            self.set_button_ready()

        except Exception as e:
            QMessageBox.critical(self, "程序错误",
                                 f"生成失败: {str(e)}\n\n{traceback.format_exc()}")
            self.set_button_ready()

    def _add_reference_grid(self, radius, depth, max_height):
        xy_step = 20
        xy_ext = ((radius // xy_step) + 1) * xy_step
        x_min, x_max = -xy_ext, xy_ext
        y_min, y_max = -xy_ext, xy_ext
        z_step = 10
        z_min_ext = ((depth // z_step) + 1) * z_step
        z_max_ext = ((max_height // z_step) + 1) * z_step
        z_min, z_max = -z_min_ext, z_max_ext

        points, conn, pt_idx = [], [], 0
        for gx in range(x_min, x_max + 1, xy_step):
            points.extend([[gx, y_min, z_min], [gx, y_max, z_min]])
            conn.extend([2, pt_idx, pt_idx + 1])
            pt_idx += 2
        for gy in range(y_min, y_max + 1, xy_step):
            points.extend([[x_min, gy, z_min], [x_max, gy, z_min]])
            conn.extend([2, pt_idx, pt_idx + 1])
            pt_idx += 2
        if points:
            poly = pv.PolyData(np.array(points, dtype=np.float32), np.array(conn, dtype=np.int64))
            actor = self.plotter.add_mesh(poly, color='gray', line_width=1, opacity=0.4)
            self._tracked_actors.append(actor)

        points, conn, pt_idx = [], [], 0
        for gz in range(z_min, z_max + 1, z_step):
            points.extend([[x_min, y_min, gz], [x_min, y_max, gz]])
            conn.extend([2, pt_idx, pt_idx + 1])
            pt_idx += 2
        if points:
            poly = pv.PolyData(np.array(points, dtype=np.float32), np.array(conn, dtype=np.int64))
            actor = self.plotter.add_mesh(poly, color='darkgray', line_width=1, opacity=0.4)
            self._tracked_actors.append(actor)

    def _add_bounds_labels(self, radius, depth, max_height):
        xy_step = 20
        xy_ext = ((radius // xy_step) + 1) * xy_step
        x_min, x_max = -xy_ext, xy_ext
        y_min, y_max = -xy_ext, xy_ext
        z_step = 10
        z_min_ext = ((depth // z_step) + 1) * z_step
        z_max_ext = ((max_height // z_step) + 1) * z_step
        z_min, z_max = -z_min_ext, z_max_ext
        bounds = [float(x_min), float(x_max), float(y_min), float(y_max), float(z_min), float(z_max)]
        n_x = int((x_max - x_min) / xy_step) + 1
        n_y = int((y_max - y_min) / xy_step) + 1
        n_z = int((z_max - z_min) / z_step) + 1
        bounds_actor = self.plotter.show_bounds(
            bounds=bounds, grid='front', location='outer', ticks='outside',
            all_edges=True, xtitle='X (m)', ytitle='Y (m)', ztitle='Z (m)',
            fmt='%.0f', n_xlabels=n_x, n_ylabels=n_y, n_zlabels=n_z)
        try:
            bounds_actor.SetXAxisRange(x_min, x_max)
            bounds_actor.SetYAxisRange(y_min, y_max)
            bounds_actor.SetZAxisRange(z_min, z_max)
        except Exception:
            pass
        self._tracked_actors.append(bounds_actor)

    # ---- Export ----

    def open_postprocess_window(self):
        """Open post-processing window with current model data."""
        if not hasattr(self, 'last_solid') or self.last_solid is None:
            QMessageBox.information(self, "提示", "请先生成地质模型")
            return
        try:
            grid_data = {
                'stratum_id': self.last_solid,
                'permeability': self.last_grid.cell_data.get("permeability", np.zeros_like(self.last_solid, dtype=np.float64)),
                'idomain': (self.last_solid > 0).astype(np.int32),
                'fracture_id': getattr(self, 'last_fracture_ids', np.zeros_like(self.last_solid)),
                'lens_id': getattr(self, 'last_lens_ids', np.zeros_like(self.last_solid)),
                'terrain_surface': getattr(self, 'last_terrain_surface', None),
                'soil_bottom': getattr(self, 'last_soil_bottom', None),
                'inside_mask': getattr(self, 'last_inside_mask', None),
            }
            self.pp_window = PostProcessWindow(
                parent=self,
                grid_data=grid_data,
                x=self.last_x, y=self.last_y, z=self.last_z
            )
            self.pp_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开后处理窗口失败:\n{str(e)}\n\n{traceback.format_exc()}")

    def export_mesh_data(self):
        """Export MODFLOW-friendly .npz with structured grid data."""
        if not hasattr(self, 'last_solid') or self.last_solid is None:
            QMessageBox.warning(self, "导出失败", "请先生成模型")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存网格数据", "geology_data.npz", "NumPy Files (*.npz)")
        if not filename:
            return
        try:
            solid = self.last_solid
            x, y, z = self.last_x, self.last_y, self.last_z
            stratum_tops = self.last_stratum_tops
            soil_bottom = self.last_soil_bottom
            terrain_surface = self.last_terrain_surface
            inside_mask = self.last_inside_mask
            n_strata = len(self.strata)

            # ---- Build permeability array (K per cell) ----
            # Priority: lens > fracture > stratum/soil
            K = np.zeros_like(solid, dtype=np.float64)
            for s_idx, s in enumerate(self.strata):
                K[solid == (s_idx + 1)] = s.permeability
            if self.soil_config.thickness > 0:
                K[solid == (n_strata + 1)] = self.soil_config.permeability

            # Override with per-fracture K (each config has its own permeability)
            fracture_ids = getattr(self, 'last_fracture_ids', np.zeros_like(solid))
            for fid, fcfg in enumerate(self.fracture_configs, 1):
                K[fracture_ids == fid] = fcfg.permeability

            # Override with per-lens K (each config has its own permeability)
            lens_ids = getattr(self, 'last_lens_ids', np.zeros_like(solid))
            for lid, lcfg in enumerate(self.lens_configs, 1):
                K[lens_ids == lid] = lcfg.permeability

            # ---- Build IDOMAIN-like active mask ----
            idomain = (solid > 0).astype(np.int32)

            # ---- Grid spacings ----
            dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
            dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
            dz = float(z[1] - z[0]) if len(z) > 1 else 1.0

            # ---- Save ----
            np.savez(
                filename,
                # Core voxel data
                stratum_id=solid,              # (nx, ny, nz) int — stratum markers
                permeability=K,                # (nx, ny, nz) float — K per cell
                idomain=idomain,               # (nx, ny, nz) int — 1=active, 0=inactive
                fracture_id=fracture_ids,      # (nx, ny, nz) int — 0=none, 1..M=fracture
                lens_id=lens_ids,              # (nx, ny, nz) int — 0=none, 1..P=lens
                # Geometry
                x=x, y=y, z=z,                 # 1D coordinate axes
                dx=dx, dy=dy, dz=dz,           # grid spacings
                # Surfaces
                top_elevation=terrain_surface, # (nx, ny) — ground surface
                soil_bottom=soil_bottom,       # (nx, ny) — soil cap bottom (non-uniform)
                inside_mask=inside_mask,       # (nx, ny) bool — parcel boundary
                # Metadata
                n_strata=n_strata,
                soil_thickness_avg=self.soil_config.thickness,
            )

            QMessageBox.information(
                self, "导出成功",
                f"已保存: {filename}\n\n"
                f"包含数组:\n"
                f"  stratum_id    {solid.shape}   int32   地层标记\n"
                f"  permeability  {K.shape}   float64 渗透系数(裂隙/透镜体已覆盖)\n"
                f"  idomain       {idomain.shape}   int32   1=活跃 0=空白\n"
                f"  fracture_id   {fracture_ids.shape}   int32   裂隙标记\n"
                f"  lens_id       {lens_ids.shape}   int32   透镜体标记\n"
                f"  top_elevation {terrain_surface.shape}   float64 地表高程\n"
                f"  soil_bottom   {soil_bottom.shape}   float64 覆土底面\n"
                f"  x, y, z       坐标轴 + 网格间距 dx, dy, dz")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"{str(e)}\n\n{traceback.format_exc()}")

    def export_mesh_vtk(self):
        if not hasattr(self, 'last_grid') or self.last_grid is None:
            QMessageBox.warning(self, "导出失败", "请先生成模型")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存 VTK", "geology_model.vtu", "VTK Files (*.vtu *.vtk)")
        if not filename:
            return
        try:
            self.last_grid.save(filename)
            QMessageBox.information(self, "导出成功", f"已保存:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Microsoft YaHei", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
