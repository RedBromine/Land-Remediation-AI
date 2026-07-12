#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing Window — K Randomization, Steady-State Head, Contaminant Transport

Features:
  - K randomization via Perlin noise (precision 1e-8)
  - Steady-state head: iterative solver from initial guess
  - Contaminant: 3D slider positioning + Kriging interpolation + advection-dispersion
  - Three display tabs: K (log colorbar) / Head / Concentration
"""

import sys
import traceback
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTabWidget, QRadioButton, QButtonGroup,
    QGroupBox, QScrollArea, QFrame, QLineEdit, QTextEdit, QListWidget,
    QComboBox, QCheckBox, QAbstractSpinBox,
    QListWidgetItem, QSlider, QProgressDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap

# Import Perlin noise from main module (lazy import inside functions)


# =========================================================
# Kriging Interpolator (pure NumPy)
# =========================================================

class OrdinaryKriging:
    """Simplified Ordinary Kriging using exponential variogram."""

    def __init__(self, x, y, z, values, nugget=0.0, sill=1.0, range_=10.0):
        self.coords = np.column_stack([x, y, z]).astype(np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.nugget = nugget
        self.sill = sill
        self.range_ = range_
        self.n = len(values)
        self._build_kriging_matrix()

    def _variogram(self, h):
        h = np.asarray(h, dtype=np.float64)
        return self.nugget + (self.sill - self.nugget) * (1.0 - np.exp(-3.0 * h / self.range_))

    def _build_kriging_matrix(self):
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        K = self._variogram(dist)
        K_ext = np.zeros((self.n + 1, self.n + 1), dtype=np.float64)
        K_ext[:self.n, :self.n] = K
        K_ext[:self.n, self.n] = 1.0
        K_ext[self.n, :self.n] = 1.0
        K_ext[self.n, self.n] = 0.0
        rhs = np.zeros(self.n + 1, dtype=np.float64)
        rhs[:self.n] = self.values
        try:
            self.weights = np.linalg.solve(K_ext, rhs)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(K_ext, rhs, rcond=None)[0]

    def predict(self, x, y, z):
        new_coords = np.column_stack([x, y, z]).astype(np.float64)
        n_new = len(x)
        results = np.zeros(n_new, dtype=np.float64)
        for i in range(n_new):
            diff = self.coords - new_coords[i]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            gamma = self._variogram(dist)
            results[i] = np.dot(gamma, self.weights[:self.n]) + self.weights[self.n]
        return results


# =========================================================
# Angle Dial (Circular Knob) Widget
# =========================================================

class AngleDial(QWidget):
    """Circular dial for angle input.
    0 rad = +X axis (right), increases counter-clockwise (math convention).
    Drag the knob handle or click on the circle to set angle."""

    angleChanged = pyqtSignal(float)

    def __init__(self, parent=None, size=100):
        super().__init__(parent)
        self._angle = 0.0  # radians
        self._size = size
        self._dragging = False
        self.setMinimumSize(size, size)
        self.setMaximumSize(size, size)
        self.setCursor(Qt.CrossCursor)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = float(value) % (2 * np.pi)
        self.update()
        self.angleChanged.emit(self._angle)

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QPen, QBrush, QFont
        from PyQt5.QtCore import QRectF, QPointF
        from PyQt5.QtGui import QColor

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        radius = min(w, h) / 2.0 - 8

        # Background circle
        painter.setPen(QPen(QColor('#cccccc'), 2))
        painter.setBrush(QBrush(QColor('#f5f5f5')))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Tick marks every 30 degrees
        painter.setPen(QPen(QColor('#999999'), 1))
        for deg in range(0, 360, 30):
            rad = np.radians(deg)
            x1 = cx + (radius - 6) * np.cos(rad)
            y1 = cy - (radius - 6) * np.sin(rad)
            x2 = cx + (radius - 2) * np.cos(rad)
            y2 = cy - (radius - 2) * np.sin(rad)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # 0 label (+X direction)
        painter.setPen(QPen(QColor('#333333'), 1))
        font = QFont('Arial', 8)
        painter.setFont(font)
        painter.drawText(QRectF(cx + radius - 4, cy - 12, 20, 12), '0')
        painter.drawText(QRectF(cx - 6, cy - radius - 2, 20, 12), 'π/2')
        painter.drawText(QRectF(cx - radius - 18, cy - 12, 20, 12), 'π')
        painter.drawText(QRectF(cx - 8, cy + radius - 10, 30, 12), '3π/2')

        # Direction arrow (from center along angle)
        painter.setPen(QPen(QColor('#2196F3'), 3))
        end_x = cx + (radius - 12) * np.cos(self._angle)
        end_y = cy - (radius - 12) * np.sin(self._angle)
        painter.drawLine(QPointF(cx, cy), QPointF(end_x, end_y))

        # Arrow head
        arrow_size = 6
        dx, dy = end_x - cx, end_y - cy
        line_len = np.sqrt(dx**2 + dy**2) + 1e-10
        ux, uy = dx / line_len, dy / line_len
        # Perpendicular
        px, py = -uy, ux
        tip_x = end_x
        tip_y = end_y
        back_x = tip_x - arrow_size * ux
        back_y = tip_y - arrow_size * uy
        left_x = back_x + arrow_size * 0.5 * px
        left_y = back_y + arrow_size * 0.5 * py
        right_x = back_x - arrow_size * 0.5 * px
        right_y = back_y - arrow_size * 0.5 * py
        painter.setBrush(QBrush(QColor('#2196F3')))
        painter.drawPolygon(
            QPointF(tip_x, tip_y),
            QPointF(left_x, left_y),
            QPointF(right_x, right_y))

        # Center dot
        painter.setBrush(QBrush(QColor('#1976D2')))
        painter.setPen(QPen(QColor('#1976D2'), 1))
        painter.drawEllipse(QPointF(cx, cy), 4, 4)

        # Current angle text
        painter.setPen(QPen(QColor('#333333'), 1))
        painter.setFont(QFont('Arial', 9, QFont.Bold))
        deg = np.degrees(self._angle)
        painter.drawText(QRectF(cx - 30, cy + radius + 2, 60, 16),
                         Qt.AlignCenter, f"{deg:.1f}°")

        painter.end()

    def mousePressEvent(self, event):
        self._dragging = True
        self._update_from_pos(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_from_pos(event.pos())

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def _update_from_pos(self, pos):
        cx, cy = self.width() / 2.0, self.height() / 2.0
        dx = pos.x() - cx
        dy = cy - pos.y()  # flip Y for math convention
        if abs(dx) < 1 and abs(dy) < 1:
            return
        self.angle = np.arctan2(dy, dx) % (2 * np.pi)


# =========================================================
# Post-processing Engine
# =========================================================

class PostProcessEngine:

    def __init__(self, grid_data, x, y, z):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)
        self.dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        self.dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        self.dz = float(z[1] - z[0]) if len(z) > 1 else 1.0

        perm = grid_data.get('permeability', np.zeros((self.nx, self.ny, self.nz), dtype=np.float64))
        if perm.ndim == 1 and perm.size == self.nx * self.ny * self.nz:
            perm = perm.reshape((self.nx, self.ny, self.nz), order='F')
        self.stratum_id = grid_data.get('stratum_id', np.zeros((self.nx, self.ny, self.nz), dtype=np.int32))
        self.permeability = perm
        self.idomain = grid_data.get('idomain', (self.stratum_id > 0).astype(np.int32))
        self.fracture_id = grid_data.get('fracture_id', np.zeros((self.nx, self.ny, self.nz), dtype=np.int32))
        self.lens_id = grid_data.get('lens_id', np.zeros((self.nx, self.ny, self.nz), dtype=np.int32))
        self.terrain_surface = grid_data.get('terrain_surface', None)
        self.soil_bottom = grid_data.get('soil_bottom', None)
        self.inside_mask = grid_data.get('inside_mask', None)

        self.K_original = self.permeability.copy()
        self.K_randomized = None

        # Restore head fields if present in loaded data
        hi = grid_data.get('head_initial', None)
        hs = grid_data.get('head_steady', None)
        self.head_initial = np.asarray(hi, dtype=np.float64) if hi is not None else None
        self.head_steady = np.asarray(hs, dtype=np.float64) if hs is not None else None

        # Restore concentration fields if present
        ci = grid_data.get('concentration_initial', None)
        cf = grid_data.get('concentration_final', None)
        self.concentration_initial = np.asarray(ci, dtype=np.float64) if ci is not None else None
        self.concentration = np.asarray(cf, dtype=np.float64) if cf is not None else None

        # Restore source info if present
        src_idx = grid_data.get('source_idx', None)
        src_c = grid_data.get('source_c', None)
        if src_idx is not None and src_c is not None:
            self.source_idx = [tuple(int(x) for x in idx) for idx in src_idx]
            self.source_c = np.asarray(src_c, dtype=np.float64)
        else:
            self.source_idx = []
            self.source_c = np.array([], dtype=np.float64)

        self.seed = 42

    # ---- K Randomization ----

    def randomize_K(self, noise_scale=20.0, amplitude_pct=20.0, seed=None):
        """Strict layered percentage noise with log-domain boundary smoothing.
        Each stratum gets 3D Perlin noise perturbation limited to ±amplitude_pct.
        K_new = K_base × (1 ± amplitude_pct/100), strictly no order-of-magnitude changes.
        Boundary cells (mixed stratum_id) get log-linear interpolation between adjacent layers."""
        from MapGeneration import _PerlinNoise
        if seed is not None:
            self.seed = seed
        rng = np.random.RandomState(self.seed)
        pn = _PerlinNoise(base=rng.randint(0, 256))
        xx, yy, zz = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        # ---- Step 1: Generate 3D noise with Z-independence ----
        noise_3d = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        for k in range(self.nz):
            z_offset = k * 97.3
            noise_2d = pn.fbm2(
                xx[:, :, k] / noise_scale,
                yy[:, :, k] / noise_scale + z_offset,
                octaves=2, persistence=0.5, lacunarity=2.0)
            noise_3d[:, :, k] = np.clip(noise_2d, -5.0, 5.0)

        # ---- Step 2: Normalize per stratum to strict [-1, 1] ----
        active = self.idomain > 0
        sids = np.unique(self.stratum_id[active])
        sid_list = sorted([int(s) for s in sids if s == int(s)])  # Only integer stratum IDs
        processed_mask = np.zeros_like(active)
        for sid in sid_list:
            mask = (self.stratum_id == sid) & active
            processed_mask |= mask
            if not np.any(mask):
                continue
            vals = noise_3d[mask]
            v_min, v_max = float(vals.min()), float(vals.max())
            if v_max > v_min:
                noise_3d[mask] = np.clip(
                    2.0 * (vals - v_min) / (v_max - v_min) - 1.0, -1.0, 1.0)
            else:
                noise_3d[mask] = 0.0

        # Transition zone cells (non-integer stratum_id): set noise=0 (no perturbation)
        transition_mask = active & ~processed_mask
        noise_3d[transition_mask] = 0.0

        # ---- Step 3: Apply strict percentage perturbation ----
        K_new = self.K_original.copy()
        frac = amplitude_pct / 100.0
        K_new[active] = K_new[active] * (1.0 + frac * noise_3d[active])

        # Sanity check: per-cell clip to [K_base_cell*(1-frac), K_base_cell*(1+frac)]
        # Each cell uses its OWN K_original as base, not the layer's first cell.
        # This preserves heterogeneous structures (fractures K=1000, sandstone K=1) correctly.
        K_min_cell = self.K_original * (1.0 - frac)
        K_max_cell = self.K_original * (1.0 + frac)
        K_new[active] = np.clip(K_new[active], K_min_cell[active], K_max_cell[active])

        # ---- Step 4: Log-domain smoothing at boundaries (±1 cell) ----
        # Find cells whose stratum_id is non-integer (transition zones)
        logK_new = np.log10(np.maximum(K_new, 1e-12))
        non_integer_mask = active & (np.abs(self.stratum_id - np.round(self.stratum_id)) > 0.01)
        if np.any(non_integer_mask):
            # For transition cells, smooth logK with neighbors from adjacent strata
            for _ in range(3):  # 3 iterations of neighbor smoothing
                logK_avg = logK_new.copy()
                logK_avg[1:-1, :, :] += logK_new[:-2, :, :] + logK_new[2:, :, :]
                logK_avg[:, 1:-1, :] += logK_new[:, :-2, :] + logK_new[:, 2:, :]
                logK_avg[:, :, 1:-1] += logK_new[:, :, :-2] + logK_new[:, :, 2:]
                logK_avg[1:-1, :, :] /= 3.0
                logK_avg[:, 1:-1, :] /= 3.0
                logK_avg[:, :, 1:-1] /= 3.0
                logK_new[non_integer_mask] = logK_avg[non_integer_mask]
            K_new[non_integer_mask] = 10.0 ** logK_new[non_integer_mask]

        K_new = np.maximum(K_new, 1e-12)
        self.K_randomized = K_new
        return K_new

    # ---- Initial Head ----

    def generate_initial_head(self, gradient_slope=0.01, gradient_direction=0.0,
                              depth_weight=0.1, min_head=-10.0,
                              unsaturated_mode=False, water_level=None):
        """Generate initial head field.
        Saturated mode: h = surf + grad_term + depth_corr, all boundaries Neumann.
        Unsaturated mode: upstream=Dirichlet(high h), downstream=Dirichlet(low h),
                          solve for phreatic surface. water_level is initial estimate.
        """
        xx, yy, zz = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        active = self.idomain > 0

        if self.terrain_surface is not None:
            surf = self.terrain_surface[:, :, np.newaxis]
        else:
            surf = np.zeros((self.nx, self.ny, 1), dtype=np.float64)
            for i in range(self.nx):
                for j in range(self.ny):
                    active_z = self.z[self.stratum_id[i, j, :] > 0]
                    surf[i, j, 0] = active_z.max() if len(active_z) > 0 else self.z.max()
            surf = np.repeat(surf, self.nz, axis=2)

        dist = xx * np.cos(gradient_direction) + yy * np.sin(gradient_direction)
        grad_term = gradient_slope * dist

        if unsaturated_mode:
            # Unsaturated: upstream=Dirichlet(water_level), downstream=Dirichlet(lower)
            # water_level is user-specified upstream water table elevation
            dist_range = dist.max() - dist.min()
            head_loss = gradient_slope * dist_range if dist_range > 0 else 1.0
            # Upstream head = user-specified water level (NOT terrain top)
            wl = water_level if water_level is not None else float(surf.mean())
            # min_head = model bottom so water_level is never clipped
            min_head = float(self.z.min())
            h_upstream = wl
            h_downstream = wl - head_loss
            # Linear interpolation from upstream to downstream
            dist_norm = (dist - dist.min()) / (dist_range + 1e-20)
            h = h_upstream * (1 - dist_norm) + h_downstream * dist_norm
            # Below water table: hydrostatic pressure increases with depth
            below_wt = zz <= wl
            h[below_wt] += depth_weight * np.maximum(0, wl - zz[below_wt])
            # Above water table: h = water_level (vadose zone, constant)
            above_wt = zz > wl
            h[above_wt] = wl
            # Store boundary info for SOR solver
            self._upstream_dir = gradient_direction + np.pi
            self._downstream_dir = gradient_direction
            self._h_upstream = h_upstream
            self._h_downstream = h_downstream
        else:
            # Saturated mode (original)
            depth_corr = depth_weight * np.maximum(0, surf - zz)
            h = surf + grad_term + depth_corr

        h = np.maximum(h, min_head)
        h_init = np.full((self.nx, self.ny, self.nz), min_head, dtype=np.float64)
        h_init[active] = h[active]
        self.head_initial = h_init
        self.gradient_direction = gradient_direction
        self._water_table_level = water_level if unsaturated_mode else None
        return h_init

    # ---- Steady-State Head (Iterative Solver) ----

    def solve_steady_state(self, max_iterations=5000, tolerance=1e-6,
                           relaxation=1.0, use_randomized_K=True,
                           progress_callback=None,
                           early_stop_patience=2, min_relative_change=0.01):
        """
        Vectorized SOR solver for steady-state: ∇·(K∇h) = 0.
        Saturated: all boundaries Neumann (open system).
        Unsaturated: upstream/downstream Dirichlet, lateral Neumann.
        """
        if self.head_initial is None:
            raise ValueError("Must generate initial head first")

        K = self.K_randomized if (use_randomized_K and self.K_randomized is not None) else self.K_original
        h = self.head_initial.copy()
        active = self.idomain > 0

        # Detect unsaturated mode
        unsat = hasattr(self, '_upstream_dir') and self._upstream_dir is not None

        # Dirichlet mask for unsaturated mode (upstream/downstream boundaries)
        dirichlet_mask = np.zeros(active.shape, dtype=bool)
        if unsat:
            up_dir = self._upstream_dir
            dn_dir = self._downstream_dir
            cos_up = np.cos(up_dir)
            sin_up = np.sin(up_dir)
            cos_dn = np.cos(dn_dir)
            sin_dn = np.sin(dn_dir)
            # Upstream boundary (inflow)
            if abs(cos_up) > 0.5:
                if cos_up > 0:
                    dirichlet_mask[0, :, :] |= active[0, :, :]
                else:
                    dirichlet_mask[-1, :, :] |= active[-1, :, :]
            if abs(sin_up) > 0.5:
                if sin_up > 0:
                    dirichlet_mask[:, 0, :] |= active[:, 0, :]
                else:
                    dirichlet_mask[:, -1, :] |= active[:, -1, :]
            # Downstream boundary (outflow)
            if abs(cos_dn) > 0.5:
                if cos_dn > 0:
                    dirichlet_mask[0, :, :] |= active[0, :, :]
                else:
                    dirichlet_mask[-1, :, :] |= active[-1, :, :]
            if abs(sin_dn) > 0.5:
                if sin_dn > 0:
                    dirichlet_mask[:, 0, :] |= active[:, 0, :]
                else:
                    dirichlet_mask[:, -1, :] |= active[:, -1, :]

        # Iterate mask = active - Dirichlet
        iterate_mask = active & ~dirichlet_mask
        iterate_sub = iterate_mask[1:-1, 1:-1, 1:-1]

        # Harmonic mean K at 6 faces
        Kxp = 2.0 * K[1:-1, 1:-1, 1:-1] * K[2:, 1:-1, 1:-1] / (K[1:-1, 1:-1, 1:-1] + K[2:, 1:-1, 1:-1] + 1e-20)
        Kxm = 2.0 * K[1:-1, 1:-1, 1:-1] * K[:-2, 1:-1, 1:-1] / (K[1:-1, 1:-1, 1:-1] + K[:-2, 1:-1, 1:-1] + 1e-20)
        Kyp = 2.0 * K[1:-1, 1:-1, 1:-1] * K[1:-1, 2:, 1:-1] / (K[1:-1, 1:-1, 1:-1] + K[1:-1, 2:, 1:-1] + 1e-20)
        Kym = 2.0 * K[1:-1, 1:-1, 1:-1] * K[1:-1, :-2, 1:-1] / (K[1:-1, 1:-1, 1:-1] + K[1:-1, :-2, 1:-1] + 1e-20)
        Kzp = 2.0 * K[1:-1, 1:-1, 1:-1] * K[1:-1, 1:-1, 2:] / (K[1:-1, 1:-1, 1:-1] + K[1:-1, 1:-1, 2:] + 1e-20)
        Kzm = 2.0 * K[1:-1, 1:-1, 1:-1] * K[1:-1, 1:-1, :-2] / (K[1:-1, 1:-1, 1:-1] + K[1:-1, 1:-1, :-2] + 1e-20)

        denom = (Kxp + Kxm) / self.dx**2 + (Kyp + Kym) / self.dy**2 + (Kzp + Kzm) / self.dz**2 + 1e-20

        omega = relaxation
        CHECK_INTERVAL = 50
        last_err_at_check = None
        patience_counter = 0
        early_stopped = False
        diverged_counter = 0
        prev_max_change = None

        for it in range(1, max_iterations + 1):
            h_old = h.copy()

            # Neumann BC
            h[0, :, :] = h[1, :, :]
            h[-1, :, :] = h[-2, :, :]
            h[:, 0, :] = h[:, 1, :]
            h[:, -1, :] = h[:, -2, :]
            h[:, :, 0] = h[:, :, 1]
            h[:, :, -1] = h[:, :, -2]

            # Restore Dirichlet values
            if unsat:
                h[dirichlet_mask] = self.head_initial[dirichlet_mask]

            h_sub = h[1:-1, 1:-1, 1:-1]

            rhs = (Kxp * h[2:, 1:-1, 1:-1] + Kxm * h[:-2, 1:-1, 1:-1]) / self.dx**2 + \
                  (Kyp * h[1:-1, 2:, 1:-1] + Kym * h[1:-1, :-2, 1:-1]) / self.dy**2 + \
                  (Kzp * h[1:-1, 1:-1, 2:] + Kzm * h[1:-1, 1:-1, :-2]) / self.dz**2

            h_new = rhs / denom
            h_sub[iterate_sub] = h_sub[iterate_sub] + omega * (h_new[iterate_sub] - h_sub[iterate_sub])
            h[1:-1, 1:-1, 1:-1] = h_sub

            max_change = np.max(np.abs(h - h_old))

            if prev_max_change is not None and max_change > prev_max_change * 1.1:
                diverged_counter += 1
                if diverged_counter >= 2:
                    omega = max(0.5, omega * 0.8)
                    h = np.clip(h, h_old.min() - 10, h_old.max() + 10)
                    diverged_counter = 0
            else:
                diverged_counter = max(0, diverged_counter - 1)

            prev_max_change = max_change

            if it % CHECK_INTERVAL == 0:
                canceled = False
                if progress_callback is not None:
                    keep_going = progress_callback(it, max_iterations, max_change)
                    if not keep_going:
                        canceled = True
                if canceled:
                    self._canceled = True
                    early_stopped = True
                    break
                if last_err_at_check is not None and last_err_at_check > 0:
                    relative_drop = (last_err_at_check - max_change) / last_err_at_check
                    if relative_drop < min_relative_change:
                        patience_counter += 1
                        if patience_counter >= early_stop_patience:
                            early_stopped = True
                            break
                    else:
                        patience_counter = 0
                last_err_at_check = max_change

            if max_change < tolerance:
                break

        h_range = float(h.max()) - float(h.min())
        has_nan = not np.all(np.isfinite(h))
        h_init_range = float(self.head_initial.max()) - float(self.head_initial.min())
        if has_nan or (h_range < h_init_range * 0.1 and h_init_range > 0.1):
            h = self.head_initial.copy()
            early_stopped = True

        # Extract water table for unsaturated mode
        if unsat:
            self._water_table = self._extract_water_table_from_head(h)

        self.head_steady = h
        return h, it, max_change, early_stopped, omega

    def _extract_water_table_from_head(self, h):
        """Extract phreatic surface from solved head.
        For each (x,y) column, find the cell where h crosses z."""
        active = self.idomain > 0
        nx, ny, nz = self.nx, self.ny, self.nz
        water_table = np.full((nx, ny), np.nan)
        for i in range(nx):
            for j in range(ny):
                k_list = np.where(active[i, j, :])[0]
                if len(k_list) == 0:
                    continue
                diff = h[i, j, k_list] - self.z[k_list]
                above = k_list[diff >= 0]
                if len(above) > 0:
                    water_table[i, j] = self.z[above[-1]]
                elif len(k_list) > 0:
                    water_table[i, j] = self.z[k_list[0]]
        return water_table

    def _auto_porosity(self, K, active):
        """Compute effective porosity from K values using empirical relationship.
        K range [1e-12, 1e4] maps to porosity range [0.01, 0.5].
        """
        logK = np.log10(K[active] + 1e-20)
        logK_min, logK_max = logK.min(), logK.max()
        porosity = np.zeros_like(K)
        if logK_max > logK_min:
            phi = 0.1 + 0.3 * (logK - logK_min) / (logK_max - logK_min)
        else:
            phi = np.full_like(logK, 0.2)
        phi = np.clip(phi, 0.01, 0.5)
        porosity[active] = phi
        return porosity

    def _build_source_mask(self, source_points, source_concentrations):
        """Convert source points to grid indices and Dirichlet mask.
        Returns (src_idx, src_c) where src_idx is (i,j,k) list, src_c is concentration list."""
        src_idx = []
        src_c = []
        for (px, py, pz), conc in zip(source_points, source_concentrations):
            i = int(np.clip(np.argmin(np.abs(self.x - px)), 1, self.nx - 2))
            j = int(np.clip(np.argmin(np.abs(self.y - py)), 1, self.ny - 2))
            k = int(np.clip(np.argmin(np.abs(self.z - pz)), 1, self.nz - 2))
            src_idx.append((i, j, k))
            src_c.append(conc)
        return src_idx, np.array(src_c, dtype=np.float64)

    def _apply_dirichlet(self, C, src_idx, src_c):
        """Enforce Dirichlet BC: set source cells to fixed concentration."""
        for (i, j, k), conc in zip(src_idx, src_c):
            C[i, j, k] = conc
        return C

    def _transport_step(self, C, vx, vy, vz, Dxx, Dyy, Dzz,
                        active, dt_eff, src_idx, src_c):
        """Optimized transport: Semi-Lagrangian advection + ANISOTROPIC Gaussian dispersion.

        Directional dispersion: sigma is computed PER AXIS from Dxx/Dyy/Dzz,
        so the Gaussian spreading is stronger along the flow direction
        (where D is larger) and weaker transverse — physically correct.
        Gaussian filter (scipy) is unconditionally stable, 10-50x faster."""
        from scipy.ndimage import map_coordinates, gaussian_filter

        # Precompute grid coords (once)
        if not hasattr(self, '_xx_grid'):
            self._xx_grid, self._yy_grid, self._zz_grid = np.meshgrid(
                self.x, self.y, self.z, indexing='ij')
            self._x_scale = (self.nx - 1) / (self.x[-1] - self.x[0]) if len(self.x) > 1 else 1.0
            self._y_scale = (self.ny - 1) / (self.y[-1] - self.y[0]) if len(self.y) > 1 else 1.0
            self._z_scale = (self.nz - 1) / (self.z[-1] - self.z[0]) if len(self.z) > 1 else 1.0

        # ---- 1. Semi-Lagrangian advection (one big step, unconditional) ----
        xi = (self._xx_grid - vx*dt_eff - self.x[0]) * self._x_scale
        yi = (self._yy_grid - vy*dt_eff - self.y[0]) * self._y_scale
        zi = (self._zz_grid - vz*dt_eff - self.z[0]) * self._z_scale

        coords = np.array([xi.ravel(), yi.ravel(), zi.ravel()])
        C[:] = map_coordinates(C, coords, order=1, mode='constant', cval=0.0).reshape(self.nx, self.ny, self.nz)
        C[~active] = 0.0
        C = self._apply_dirichlet(C, src_idx, src_c)
        np.maximum(C, 0.0, out=C)

        # ---- 2. ANISOTROPIC Gaussian dispersion (directional along flow) ----
        # sigma_i = sqrt(2 * D_ii * dt) / d_i  (grid units)
        # Dxx/Dyy/Dzz already encode directionality: larger along flow velocity
        # Using median per axis avoids extreme values from K_randomized
        Dxx_med = np.median(Dxx[Dxx > 0]) if np.any(Dxx > 0) else 1e-3
        Dyy_med = np.median(Dyy[Dyy > 0]) if np.any(Dyy > 0) else 1e-3
        Dzz_med = np.median(Dzz[Dzz > 0]) if np.any(Dzz > 0) else 1e-3
        sigma_x = np.sqrt(2.0 * Dxx_med * dt_eff) / self.dx
        sigma_y = np.sqrt(2.0 * Dyy_med * dt_eff) / self.dy
        sigma_z = np.sqrt(2.0 * Dzz_med * dt_eff) / self.dz
        sigma = (sigma_x, sigma_y, sigma_z)

        if max(sigma) > 1e-6:
            C[:] = gaussian_filter(C, sigma=sigma, mode='constant', cval=0.0)
            np.maximum(C, 0.0, out=C)
            C[~active] = 0.0
            C = self._apply_dirichlet(C, src_idx, src_c)

        # ---- 3. Decay (analytical) ----
        if self.decay_rate > 0:
            C[active] *= np.exp(-self.decay_rate * dt_eff / self.retardation)

        return C

    def _compute_dispersion_tensor(self, vx, vy, vz, alpha_L_3d, alpha_T_3d, Dm,
                                    D_min=1e-3):
        """Compute Dxx/Dyy/Dzz from dispersivity tensor and velocity field.
        D_min: floor value (m²/d) to ensure visible dispersion even at low velocity.
        Typical D_min = 1e-3 ~ 1e-2 m²/d for groundwater."""
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-20
        Dxx = np.maximum(Dm + alpha_L_3d * vx**2 / v_mag + alpha_T_3d * (vy**2 + vz**2) / v_mag, D_min)
        Dyy = np.maximum(Dm + alpha_L_3d * vy**2 / v_mag + alpha_T_3d * (vx**2 + vz**2) / v_mag, D_min)
        Dzz = np.maximum(Dm + alpha_L_3d * vz**2 / v_mag + alpha_T_3d * (vx**2 + vy**2) / v_mag, D_min)
        return Dxx, Dyy, Dzz

    def _adaptive_alpha(self, K, long_factor, trans_factor):
        """Compute cell-wise adaptive dispersivity from K values.
        alpha_L = long_factor * dx * max(0.1, 1 + 0.1 * log10(K / K_median + 1))
        alpha_T = trans_factor * alpha_L
        Clipped to ensure positivity even for very small K.
        """
        K_active = K[K > 0]
        K_median = np.median(K_active) if len(K_active) > 0 else 1.0
        log_term = np.log10(K / (K_median + 1e-20) + 1.0)
        # Clip: minimum 0.1 ensures positivity; max 50 prevents overflow
        scale = np.clip(1.0 + 0.1 * log_term, 0.1, 50.0)
        alpha_L = long_factor * self.dx * scale
        alpha_T = trans_factor * alpha_L
        return alpha_L, alpha_T

    # ---- Contaminant: Initialize (Dirichlet BC + 1-day pre-run) ----

    def initialize_contaminant(self, source_points, source_concentrations,
                               dispersivity_long=20.0, dispersivity_trans=0.25,
                               diffusion_coeff=1e-5, retardation=1.0,
                               decay_rate=0.0, use_randomized_K=True,
                               adaptive_alpha=False,
                               solubility=None, density=None,
                               progress_callback=None):
        """Dirichlet point-source model:
        1. Reset field to zero everywhere
        2. Set source cells to fixed concentration (Dirichlet BC)
        3. Advance 1 day to get initial concentration distribution

        Parameters:
            adaptive_alpha: if True, compute dispersivity from K values
                           (dispersivity_long used as longitudinal_factor,
                            dispersivity_trans used as transverse_factor)
            solubility: contaminant solubility in mg/L (for display/reference)
            density: contaminant density in g/cm3 (for display/reference)
            progress_callback(msg, pct): called with message and 0-100 percent
        """
        active = self.idomain > 0
        K = self.K_randomized if (use_randomized_K and self.K_randomized is not None) else self.K_original

        if progress_callback:
            progress_callback("预处理: 构建源点掩膜...", 5)

        # Save source info
        self.source_idx, self.source_c = self._build_source_mask(
            source_points, source_concentrations)
        self.adaptive_alpha = adaptive_alpha
        self.dispersivity_long = dispersivity_long
        self.dispersivity_trans = dispersivity_trans
        self.diffusion_coeff = diffusion_coeff
        self.retardation = retardation
        self.decay_rate = decay_rate
        self.solubility = solubility
        self.density = density

        if progress_callback:
            progress_callback("预处理: 计算流速场...", 15)

        # Precompute velocity field
        porosity = self._auto_porosity(K, active)
        h = self.head_steady
        self._vx = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self._vy = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self._vz = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        self._vx[1:-1, 1:-1, 1:-1] = -K[1:-1, 1:-1, 1:-1] / (porosity[1:-1, 1:-1, 1:-1] + 1e-20) * \
            (h[2:, 1:-1, 1:-1] - h[:-2, 1:-1, 1:-1]) / (2 * self.dx)
        self._vy[1:-1, 1:-1, 1:-1] = -K[1:-1, 1:-1, 1:-1] / (porosity[1:-1, 1:-1, 1:-1] + 1e-20) * \
            (h[1:-1, 2:, 1:-1] - h[1:-1, :-2, 1:-1]) / (2 * self.dy)
        self._vz[1:-1, 1:-1, 1:-1] = -K[1:-1, 1:-1, 1:-1] / (porosity[1:-1, 1:-1, 1:-1] + 1e-20) * \
            (h[1:-1, 1:-1, 2:] - h[1:-1, 1:-1, :-2]) / (2 * self.dz)

        if progress_callback:
            progress_callback("预处理: 计算弥散张量...", 30)

        # Precompute dispersivity (adaptive or fixed)
        if self.adaptive_alpha:
            alpha_L, alpha_T = self._adaptive_alpha(
                K, self.dispersivity_long, self.dispersivity_trans)
        else:
            alpha_L = np.full_like(K, self.dispersivity_long, dtype=np.float64)
            alpha_T = np.full_like(K, self.dispersivity_trans, dtype=np.float64)

        v_mag = np.sqrt(self._vx**2 + self._vy**2 + self._vz**2) + 1e-20
        # D_min floor: ensures visible dispersion even when v→0
        self._Dxx, self._Dyy, self._Dzz = self._compute_dispersion_tensor(
            self._vx, self._vy, self._vz, alpha_L, alpha_T,
            self.diffusion_coeff, D_min=0.5)

        # Step 1: Zero everywhere, set Dirichlet BC
        C = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        C = self._apply_dirichlet(C, self.source_idx, self.source_c)

        if progress_callback:
            progress_callback("推进: 1天预模拟...", 40)

        # Step 2: Advance 1 day to get initial distribution
        dt_init = 1.0  # 1 day
        n_init_steps = max(1, int(np.ceil(dt_init / 0.5)))  # sub-step for safety
        dt_sub = dt_init / n_init_steps
        for step in range(n_init_steps):
            C = self._transport_step(C, self._vx, self._vy, self._vz,
                                     self._Dxx, self._Dyy, self._Dzz,
                                     active, dt_sub,
                                     self.source_idx, self.source_c)
            if progress_callback:
                pct = 40 + int(60 * (step + 1) / n_init_steps)
                progress_callback(f"推进: 子步 {step+1}/{n_init_steps}...", pct)

        self.concentration_initial = C.copy()
        self.concentration = C.copy()
        return self.concentration_initial

    # ---- Contaminant: Transient Transport ----

    def simulate_contaminant(self, duration_days=30.0, dt_days=1.0,
                             use_randomized_K=True,
                             progress_callback=None):
        """Semi-Lagrangian advection + explicit dispersion with Dirichlet BC.
        Starts from concentration_initial, advances for duration_days."""
        if not hasattr(self, 'concentration_initial'):
            raise ValueError("Must initialize contaminant first")
        if self.head_steady is None:
            raise ValueError("Must solve steady-state head first")
        if not hasattr(self, 'source_idx') or not self.source_idx:
            raise ValueError("No source points defined")

        C = self.concentration_initial.copy()
        active = self.idomain > 0

        n_steps = max(1, int(np.ceil(duration_days / dt_days)))
        t_current = 0.0
        REPORT_INTERVAL = max(1, n_steps // 100)

        canceled = False
        for step in range(n_steps):
            dt_eff = min(dt_days, duration_days - t_current)
            C = self._transport_step(C, self._vx, self._vy, self._vz,
                                     self._Dxx, self._Dyy, self._Dzz,
                                     active, dt_eff,
                                     self.source_idx, self.source_c)
            t_current += dt_eff

            if progress_callback is not None and step % REPORT_INTERVAL == 0:
                keep_going = progress_callback(step, n_steps, t_current)
                if not keep_going:
                    canceled = True
                    break

        self.concentration = C.copy()
        self._canceled = canceled
        return self.concentration


# =========================================================
# Post-processing Window (UI)
# =========================================================

class PostProcessWindow(QMainWindow):

    def __init__(self, parent=None, grid_data=None, x=None, y=None, z=None):
        super().__init__(parent)
        self.setWindowTitle("网格参数调整 — K随机化 / 水头模拟 / 污染物运移")
        self.resize(1700, 1000)

        self.engine = None
        self.current_time_idx = 0
        self._pending_grid_data = grid_data
        self._pending_x = x
        self._pending_y = y
        self._pending_z = z
        self._marker_actor = None

        # Contaminant source list
        self.contaminant_sources = []  # list of (x, y, z, conc)

        self.setup_ui()
        if grid_data is not None and x is not None:
            self.load_data(grid_data, x, y, z)

    def setup_ui(self):
        self.central = QWidget()
        self.setCentralWidget(self.central)
        main_layout = QHBoxLayout()
        self.central.setLayout(main_layout)

        # ---- Left: Tab widget with 3 tabs ----
        self.left_tabs = QTabWidget()
        self.left_tabs.setFixedWidth(380)
        main_layout.addWidget(self.left_tabs)

        # Tab 0: Data source + K randomization
        self.tab_k = QWidget()
        self.left_tabs.addTab(self.tab_k, "按 K 显示")
        self._setup_tab_k()

        # Tab 1: Head
        self.tab_head = QWidget()
        self.left_tabs.addTab(self.tab_head, "按水头显示")
        self._setup_tab_head()

        # Tab 2: Contaminant
        self.tab_conc = QWidget()
        self.left_tabs.addTab(self.tab_conc, "按污染物浓度显示")
        self._setup_tab_conc()

        # Tab switch handler
        self.left_tabs.currentChanged.connect(self.on_tab_changed)

        # ---- Right: 3D viewer ----
        self.viewer_widget = QWidget()
        viewer_layout = QVBoxLayout()
        self.viewer_widget.setLayout(viewer_layout)
        main_layout.addWidget(self.viewer_widget)
        self.plotter = QtInteractor(self.viewer_widget)
        viewer_layout.addWidget(self.plotter.interactor)

        # Initialize dispersivity mode visibility
        self.on_disp_mode_changed(self.disp_mode_combo.currentText())

        # Initialize unsaturated mode UI (default: saturated, hide water level input)
        self.on_unsat_mode_changed(Qt.Unchecked)

        self.statusBar().showMessage("就绪 — 请加载数据或接收主窗口数据")

    # ---- Tab 0: K Display ----

    def _setup_tab_k(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        scroll.setWidget(content)
        tab_layout = QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_k.setLayout(tab_layout)
        tab_layout.addWidget(scroll)

        # Data source
        src = QGroupBox("数据源")
        src_l = QVBoxLayout()
        src.setLayout(src_l)
        self.load_npz_btn = QPushButton("加载 .npz")
        self.load_npz_btn.clicked.connect(self.load_npz_file)
        src_l.addWidget(self.load_npz_btn)
        self.load_vtk_btn = QPushButton("加载 .vtk/.vtu")
        self.load_vtk_btn.clicked.connect(self.load_vtk_file)
        src_l.addWidget(self.load_vtk_btn)
        self.export_npz_btn_k = QPushButton("导出 .npz")
        self.export_npz_btn_k.clicked.connect(self.export_npz)
        src_l.addWidget(self.export_npz_btn_k)
        self.export_vtk_btn_k = QPushButton("导出 .vtk")
        self.export_vtk_btn_k.clicked.connect(self.export_vtk)
        src_l.addWidget(self.export_vtk_btn_k)
        layout.addWidget(src)

        # K randomization
        k_grp = QGroupBox("K 随机化（分层百分比噪声）")
        k_l = QVBoxLayout()
        k_grp.setLayout(k_l)
        info = QLabel("每层独立做百分比浮动，保持层内连续性")
        info.setStyleSheet("color: #888; font-size: 11px;")
        k_l.addWidget(info)
        self.k_noise_scale = self._add_double_spin(k_l, "噪声尺度 (m)", 1.0, 200.0, 20.0, 1.0, 1)
        self.k_amplitude = self._add_double_spin(k_l, "扰动幅度 (%)", 0.0, 100.0, 20.0, 1.0, 1)
        self.k_seed = self._add_spin(k_l, "随机种子", 0, 999999, 42)
        self.k_btn = QPushButton("执行 K 随机化")
        self.k_btn.clicked.connect(self.do_k_randomization)
        k_l.addWidget(self.k_btn)
        layout.addWidget(k_grp)

        # K display range filter
        range_grp = QGroupBox("K 显示范围过滤")
        range_l = QVBoxLayout()
        range_grp.setLayout(range_l)
        self.k_range_enable = QCheckBox("启用范围过滤")
        self.k_range_enable.setChecked(False)
        self.k_range_enable.stateChanged.connect(self.render_k)
        range_l.addWidget(self.k_range_enable)
        self.k_range_min = self._add_double_spin(range_l, "logK 下限", -10.0, 10.0, -4.0, 0.5, 2)
        self.k_range_max = self._add_double_spin(range_l, "logK 上限", -10.0, 10.0, 0.0, 0.5, 2)
        layout.addWidget(range_grp)

        self.k_render_btn = QPushButton("刷新 K 渲染")
        self.k_render_btn.clicked.connect(self.render_k)
        layout.addWidget(self.k_render_btn)

        # Observation point
        self._add_observation_sliders(layout, "k")
        layout.addStretch()

    # ---- Tab 1: Head Display ----

    def _setup_tab_head(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        scroll.setWidget(content)
        tab_layout = QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_head.setLayout(tab_layout)
        tab_layout.addWidget(scroll)

        init_grp = QGroupBox("初始水头生成")
        init_l = QVBoxLayout()
        init_grp.setLayout(init_l)

        # Unsaturated flow mode
        self.unsat_check = QCheckBox("启用非饱和流模式")
        self.unsat_check.setChecked(False)
        self.unsat_check.stateChanged.connect(self.on_unsat_mode_changed)
        init_l.addWidget(self.unsat_check)

        # Water level: manual label+spin so label can be hidden with checkbox
        self.wl_value_label = QLabel("地下水位高程 (m)")
        init_l.addWidget(self.wl_value_label)
        self.unsat_water_level = QDoubleSpinBox()
        self.unsat_water_level.setRange(-100.0, 100.0)
        self.unsat_water_level.setSingleStep(0.5)
        self.unsat_water_level.setDecimals(2)
        self.unsat_water_level.setValue(0.0)
        try:
            self.unsat_water_level.setStepType(QAbstractSpinBox.DefaultStepType)
        except Exception:
            pass
        init_l.addWidget(self.unsat_water_level)

        # Gradient controls
        self.head_grad_slope = self._add_double_spin(init_l, "梯度斜率", -0.1, 0.1, 0.05, 0.001, 4)

        # Circular angle dial for gradient direction
        dial_l = QHBoxLayout()
        self.angle_dial = AngleDial(size=100)
        self.angle_dial.angleChanged.connect(self.on_angle_dial_changed)
        dial_l.addWidget(self.angle_dial)
        dial_info = QVBoxLayout()
        dial_info.addWidget(QLabel("梯度方向"))
        self.head_grad_dir = QDoubleSpinBox()
        self.head_grad_dir.setRange(0.0, 6.2832)
        self.head_grad_dir.setDecimals(3)
        self.head_grad_dir.setSingleStep(0.1)
        self.head_grad_dir.setValue(0.0)
        self.head_grad_dir.valueChanged.connect(self.on_grad_dir_spin_changed)
        dial_info.addWidget(self.head_grad_dir)
        info_lbl = QLabel("0 rad = +X (右)\nπ/2 = +Y (上)\nπ = -X (左)\n3π/2 = -Y (下)")
        info_lbl.setStyleSheet("color: #888; font-size: 10px;")
        dial_info.addWidget(info_lbl)
        dial_l.addLayout(dial_info)
        init_l.addLayout(dial_l)

        self.head_depth_w = self._add_double_spin(init_l, "静水压预加梯度 (0=关闭)", 0.0, 5.0, 0.0, 0.1, 2)
        self.head_init_btn = QPushButton("生成初始水头")
        self.head_init_btn.clicked.connect(self.do_generate_head)
        init_l.addWidget(self.head_init_btn)
        layout.addWidget(init_grp)

        ss_grp = QGroupBox("稳态求解 (∇·(K∇h)=0)")
        ss_l = QVBoxLayout()
        ss_grp.setLayout(ss_l)
        self.ss_max_iter = self._add_spin(ss_l, "最大迭代次数", 100, 20000, 5000)
        self.ss_tol = self._add_double_spin(ss_l, "收敛容差", 1e-8, 1e-2, 1e-6, 1e-8, 8)
        self.ss_relax = self._add_double_spin(ss_l, "松弛因子 ω", 0.5, 2.0, 1.0, 0.05, 2)

        es_grp = QGroupBox("早停策略")
        es_l = QVBoxLayout()
        es_grp.setLayout(es_l)
        self.ss_es_patience = self._add_spin(es_l, "连续耐心次数", 1, 10, 2)
        self.ss_es_relative = self._add_double_spin(es_l, "最小相对下降率", 0.001, 0.5, 0.01, 0.001, 3)
        self.ss_es_patience_label = QLabel("每50次检查一次残差下降率\n连续N次下降<阈值则早停")
        self.ss_es_patience_label.setStyleSheet("color: #888; font-size: 11px;")
        es_l.addWidget(self.ss_es_patience_label)
        ss_l.addWidget(es_grp)

        self.ss_btn = QPushButton("求解稳态水头")
        self.ss_btn.clicked.connect(self.do_steady_state)
        ss_l.addWidget(self.ss_btn)
        self.ss_info = QLabel("未求解")
        self.ss_info.setStyleSheet("color: #666; font-size: 12px;")
        ss_l.addWidget(self.ss_info)
        layout.addWidget(ss_grp)

        self.head_render_btn = QPushButton("刷新水头渲染")
        self.head_render_btn.clicked.connect(self.render_head)
        layout.addWidget(self.head_render_btn)

        # Observation point
        self._add_observation_sliders(layout, "head")
        layout.addStretch()

    # ---- Tab 2: Contaminant Display ----

    def _setup_tab_conc(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        scroll.setWidget(content)
        tab_layout = QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_conc.setLayout(tab_layout)
        tab_layout.addWidget(scroll)

        # Contaminant species selection + physical properties
        species_grp = QGroupBox("污染物种类与物理参数")
        species_l = QVBoxLayout()
        species_grp.setLayout(species_l)

        species_sel_l = QHBoxLayout()
        species_sel_l.addWidget(QLabel("选择污染物"))
        self.species_combo = QComboBox()
        self.species_combo.addItems([
            "自定义",
            "氨氮 (NH3-N)",
            "1,2-二氯乙烷 (DCE)",
            "二氯乙烯 (TCE)"
        ])
        self.species_combo.currentIndexChanged.connect(self.on_species_changed)
        species_sel_l.addWidget(self.species_combo)
        species_l.addLayout(species_sel_l)

        self.solubility_spin = self._add_double_spin(species_l, "溶解度 (mg/L)", 0.0, 1e9, 10000.0, 100.0, 2)
        self.density_spin = self._add_double_spin(species_l, "密度 (g/cm³)", 0.5, 3.0, 0.91, 0.01, 2)
        self.mol_diff_spin = self._add_double_spin(species_l, "分子扩散系数 (m²/d)", 1e-15, 1e-2, 1e-5, 1e-5, 10)
        species_l.addWidget(QLabel("参数自动填充后仍可手动修改"))
        species_l.itemAt(species_l.count()-1).widget().setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(species_grp)

        # 3D Sliders for point positioning
        slider_grp = QGroupBox("污染源点位定位 (滑块)")
        slider_l = QVBoxLayout()
        slider_grp.setLayout(slider_l)

        # X slider
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X"))
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(0, 100)
        self.slider_x.valueChanged.connect(self.on_slider_changed)
        x_layout.addWidget(self.slider_x)
        self.slider_x_val = QDoubleSpinBox()
        self.slider_x_val.setDecimals(1)
        self.slider_x_val.setRange(-9999, 9999)
        self.slider_x_val.valueChanged.connect(self.on_spin_val_changed)
        x_layout.addWidget(self.slider_x_val)
        slider_l.addLayout(x_layout)

        # Y slider
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y"))
        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_y.setRange(0, 100)
        self.slider_y.valueChanged.connect(self.on_slider_changed)
        y_layout.addWidget(self.slider_y)
        self.slider_y_val = QDoubleSpinBox()
        self.slider_y_val.setDecimals(1)
        self.slider_y_val.setRange(-9999, 9999)
        self.slider_y_val.valueChanged.connect(self.on_spin_val_changed)
        y_layout.addWidget(self.slider_y_val)
        slider_l.addLayout(y_layout)

        # Z slider
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z"))
        self.slider_z = QSlider(Qt.Horizontal)
        self.slider_z.setRange(0, 100)
        self.slider_z.valueChanged.connect(self.on_slider_changed)
        z_layout.addWidget(self.slider_z)
        self.slider_z_val = QDoubleSpinBox()
        self.slider_z_val.setDecimals(1)
        self.slider_z_val.setRange(-9999, 9999)
        self.slider_z_val.valueChanged.connect(self.on_spin_val_changed)
        z_layout.addWidget(self.slider_z_val)
        slider_l.addLayout(z_layout)

        # Concentration (mg/L)
        c_layout = QHBoxLayout()
        c_layout.addWidget(QLabel("点源浓度 (mg/L)"))
        self.source_conc_spin = QDoubleSpinBox()
        self.source_conc_spin.setRange(0.0, 10000000.0)
        self.source_conc_spin.setDecimals(2)
        self.source_conc_spin.setValue(1000.0)
        self.source_conc_spin.setSingleStep(10.0)
        c_layout.addWidget(self.source_conc_spin)
        slider_l.addLayout(c_layout)

        btn_l = QHBoxLayout()
        self.add_source_btn = QPushButton("+ 添加源点")
        self.add_source_btn.clicked.connect(self.add_contaminant_source)
        btn_l.addWidget(self.add_source_btn)
        self.preview_marker_btn = QPushButton("预览标记")
        self.preview_marker_btn.clicked.connect(self.preview_marker)
        btn_l.addWidget(self.preview_marker_btn)
        slider_l.addLayout(btn_l)

        layout.addWidget(slider_grp)

        # Source list (CRUD)
        list_grp = QGroupBox("污染源列表")
        list_l = QVBoxLayout()
        list_grp.setLayout(list_l)
        self.source_list = QListWidget()
        self.source_list.setMinimumHeight(80)
        self.source_list.setMaximumHeight(150)
        self.source_list.itemDoubleClicked.connect(self.edit_source)
        list_l.addWidget(self.source_list)
        list_btns = QHBoxLayout()
        self.del_source_btn = QPushButton("- 删除")
        self.del_source_btn.clicked.connect(self.del_source)
        self.clear_source_btn = QPushButton("清空")
        self.clear_source_btn.clicked.connect(self.clear_sources)
        list_btns.addWidget(self.del_source_btn)
        list_btns.addWidget(self.clear_source_btn)
        list_l.addLayout(list_btns)
        layout.addWidget(list_grp)

        # Parameters
        param_grp = QGroupBox("污染物参数")
        param_l = QVBoxLayout()
        param_grp.setLayout(param_l)

        # Dispersivity mode selector
        mode_l = QHBoxLayout()
        mode_l.addWidget(QLabel("弥散度模式"))
        self.disp_mode_combo = QComboBox()
        self.disp_mode_combo.addItems(["手动固定值", "自适应 (基于K)"])
        self.disp_mode_combo.setCurrentIndex(1)  # Default: adaptive
        self.disp_mode_combo.currentTextChanged.connect(self.on_disp_mode_changed)
        mode_l.addWidget(self.disp_mode_combo)
        param_l.addLayout(mode_l)

        # Manual mode: fixed alpha_L, alpha_T
        self.manual_alpha_l_label = QLabel("纵向弥散度 α_L (m)")
        param_l.addWidget(self.manual_alpha_l_label)
        self.manual_alpha_l = QDoubleSpinBox()
        self.manual_alpha_l.setRange(0.1, 500.0)
        self.manual_alpha_l.setSingleStep(1.0)
        self.manual_alpha_l.setDecimals(2)
        self.manual_alpha_l.setValue(20.0)
        param_l.addWidget(self.manual_alpha_l)

        self.manual_alpha_t_label = QLabel("横向弥散度 α_T (m)")
        param_l.addWidget(self.manual_alpha_t_label)
        self.manual_alpha_t = QDoubleSpinBox()
        self.manual_alpha_t.setRange(0.01, 100.0)
        self.manual_alpha_t.setSingleStep(0.1)
        self.manual_alpha_t.setDecimals(2)
        self.manual_alpha_t.setValue(5.0)
        param_l.addWidget(self.manual_alpha_t)

        # Adaptive mode: longitudinal factor, transverse factor
        self.adaptive_long_label = QLabel("纵向放大系数")
        param_l.addWidget(self.adaptive_long_label)
        self.adaptive_long_factor = QDoubleSpinBox()
        self.adaptive_long_factor.setRange(0.1, 200.0)
        self.adaptive_long_factor.setSingleStep(1.0)
        self.adaptive_long_factor.setDecimals(2)
        self.adaptive_long_factor.setValue(20.0)
        param_l.addWidget(self.adaptive_long_factor)

        self.adaptive_trans_label = QLabel("横向比例系数")
        param_l.addWidget(self.adaptive_trans_label)
        self.adaptive_trans_factor = QDoubleSpinBox()
        self.adaptive_trans_factor.setRange(0.01, 1.0)
        self.adaptive_trans_factor.setSingleStep(0.01)
        self.adaptive_trans_factor.setDecimals(2)
        self.adaptive_trans_factor.setValue(0.25)
        param_l.addWidget(self.adaptive_trans_factor)

        self.adaptive_info = QLabel("α_L = 系数 × dx × (1 + 0.1·log₁₀(K/Kₘ))")
        self.adaptive_info.setStyleSheet("color: #888; font-size: 11px;")
        param_l.addWidget(self.adaptive_info)

        self.conc_diff = self._add_double_spin(param_l, "扩散系数 (m²/d)", 1e-15, 1e-2, 1e-5, 1e-5, 10)
        self.conc_retard = self._add_double_spin(param_l, "阻滞因子 R", 1.0, 50.0, 1.0, 0.1, 2)
        self.conc_decay = self._add_double_spin(param_l, "衰减系数 λ (1/d)", 0.0, 1.0, 0.0, 0.001, 6)
        self.conc_duration = self._add_double_spin(param_l, "模拟时长 (天)", 1.0, 36500.0, 365.0, 1.0, 1)
        self.conc_dt = self._add_double_spin(param_l, "时间步长 (天)", 0.01, 365.0, 1.0, 0.1, 2)
        self.conc_init_btn = QPushButton("初始化污染物场 (Dirichlet BC)")
        self.conc_init_btn.clicked.connect(self.do_init_contaminant)
        param_l.addWidget(self.conc_init_btn)
        self.conc_run_btn = QPushButton("运行运移模拟")
        self.conc_run_btn.clicked.connect(self.do_contaminant_sim)
        param_l.addWidget(self.conc_run_btn)
        self.conc_reset_btn = QPushButton("重置浓度")
        self.conc_reset_btn.clicked.connect(self.do_reset_concentration)
        self.conc_reset_btn.setStyleSheet("QPushButton { color: #c0392b; }")
        param_l.addWidget(self.conc_reset_btn)
        layout.addWidget(param_grp)

        # Display options
        disp_grp = QGroupBox("显示选项")
        disp_l = QVBoxLayout()
        disp_grp.setLayout(disp_l)
        self.log_scale_check = QComboBox()
        self.log_scale_check.addItems(["线性色标", "对数色标 (log₁₀)"])
        disp_l.addWidget(QLabel("色标模式"))
        disp_l.addWidget(self.log_scale_check)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1000.0)
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setValue(0.0)
        self.threshold_spin.setSingleStep(0.001)
        disp_l.addWidget(QLabel("最小显示阈值"))
        disp_l.addWidget(self.threshold_spin)
        self.ncolors_spin = QSpinBox()
        self.ncolors_spin.setRange(2, 256)
        self.ncolors_spin.setValue(256)
        self.ncolors_spin.setSingleStep(1)
        disp_l.addWidget(QLabel("色标分级数"))
        disp_l.addWidget(self.ncolors_spin)
        self.clim_mode_combo = QComboBox()
        self.clim_mode_combo.addItems(["自动上限 (实际最大值取整)", "固定上限 (源头浓度)"])
        disp_l.addWidget(QLabel("色标上限模式"))
        disp_l.addWidget(self.clim_mode_combo)
        layout.addWidget(disp_grp)

        # Simulation info label
        self.time_label = QLabel("未初始化")
        self.time_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.time_label)

        self.conc_render_btn = QPushButton("刷新浓度渲染")
        self.conc_render_btn.clicked.connect(self.render_conc)
        layout.addWidget(self.conc_render_btn)

        # Observation point
        self._add_observation_sliders(layout, "conc")
        layout.addStretch()

    # ---- Slider & Marker ----

    def on_slider_changed(self):
        """Slider moved — update spin values and show marker."""
        if self.engine is None:
            return
        x = self._slider_to_coord(self.slider_x.value(), self.engine.x.min(), self.engine.x.max())
        y = self._slider_to_coord(self.slider_y.value(), self.engine.y.min(), self.engine.y.max())
        z = self._slider_to_coord(self.slider_z.value(), self.engine.z.min(), self.engine.z.max())
        self.slider_x_val.setValue(round(x, 1))
        self.slider_y_val.setValue(round(y, 1))
        self.slider_z_val.setValue(round(z, 1))
        self._show_marker(x, y, z)

    def on_spin_val_changed(self):
        """Spin value changed — update sliders."""
        if self.engine is None:
            return
        self.slider_x.setValue(self._coord_to_slider(self.slider_x_val.value(), self.engine.x.min(), self.engine.x.max()))
        self.slider_y.setValue(self._coord_to_slider(self.slider_y_val.value(), self.engine.y.min(), self.engine.y.max()))
        self.slider_z.setValue(self._coord_to_slider(self.slider_z_val.value(), self.engine.z.min(), self.engine.z.max()))
        self._show_marker(self.slider_x_val.value(), self.slider_y_val.value(), self.slider_z_val.value())

    def _slider_to_coord(self, val, cmin, cmax):
        return cmin + (cmax - cmin) * val / 100.0

    def _coord_to_slider(self, coord, cmin, cmax):
        if cmax == cmin:
            return 50
        return int(round(max(0, min(100, (coord - cmin) / (cmax - cmin) * 100))))

    def _show_marker(self, x, y, z):
        """Show a marker sphere in the 3D plot."""
        try:
            if self._marker_actor is not None:
                self.plotter.remove_actor(self._marker_actor)
                self._marker_actor = None
            sphere = pv.Sphere(radius=min(self.engine.dx, self.engine.dy, self.engine.dz) * 2, center=(x, y, z))
            self._marker_actor = self.plotter.add_mesh(sphere, color='red', opacity=0.9, show_edges=False)
            self.plotter.render()
        except Exception:
            pass

    def preview_marker(self):
        """Show marker at current slider position."""
        self._show_marker(self.slider_x_val.value(), self.slider_y_val.value(), self.slider_z_val.value())

    # ---- Contaminant Source CRUD ----

    def add_contaminant_source(self):
        x = self.slider_x_val.value()
        y = self.slider_y_val.value()
        z = self.slider_z_val.value()
        c = self.source_conc_spin.value()
        idx = len(self.contaminant_sources)
        self.contaminant_sources.append((x, y, z, c))
        item = QListWidgetItem(f"[{idx+1}] X={x:.1f} Y={y:.1f} Z={z:.1f} C={c:.2f} mg/L")
        item.setData(Qt.UserRole, idx)
        self.source_list.addItem(item)

    def edit_source(self):
        row = self.source_list.currentRow()
        if row < 0 or row >= len(self.contaminant_sources):
            return
        x, y, z, c = self.contaminant_sources[row]
        self.slider_x_val.setValue(x)
        self.slider_y_val.setValue(y)
        self.slider_z_val.setValue(z)
        self.source_conc_spin.setValue(c)
        self._show_marker(x, y, z)

    def del_source(self):
        row = self.source_list.currentRow()
        if row < 0 or row >= len(self.contaminant_sources):
            return
        self.contaminant_sources.pop(row)
        self.refresh_source_list()

    def clear_sources(self):
        self.contaminant_sources.clear()
        self.source_list.clear()

    def refresh_source_list(self):
        self.source_list.clear()
        for idx, (x, y, z, c) in enumerate(self.contaminant_sources):
            item = QListWidgetItem(f"[{idx+1}] X={x:.1f} Y={y:.1f} Z={z:.1f} C={c:.2f} mg/L")
            item.setData(Qt.UserRole, idx)
            self.source_list.addItem(item)

    # ---- UI Helpers ----

    def _add_spin(self, parent, label, min_v, max_v, default):
        lbl = QLabel(label)
        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        try:
            spin.setStepType(QAbstractSpinBox.DefaultStepType)
        except Exception:
            pass
        parent.addWidget(lbl)
        parent.addWidget(spin)
        return spin

    def _add_double_spin(self, parent, label, min_v, max_v, default, step, decimals):
        lbl = QLabel(label)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(default)
        # Disable adaptive step: always use fixed singleStep
        try:
            spin.setStepType(QAbstractSpinBox.DefaultStepType)
        except Exception:
            pass
        parent.addWidget(lbl)
        parent.addWidget(spin)
        return spin

    # ---- Data Loading ----

    def load_data(self, grid_data, x, y, z):
        try:
            self.engine = PostProcessEngine(grid_data, x, y, z)
            # Setup slider ranges
            self.slider_x_val.setRange(float(x.min()), float(x.max()))
            self.slider_y_val.setRange(float(y.min()), float(y.max()))
            self.slider_z_val.setRange(float(z.min()), float(z.max()))
            self.slider_x_val.setValue(0)
            self.slider_y_val.setValue(0)
            self.slider_z_val.setValue(float(z.mean()))
            self.on_spin_val_changed()

            # Initialize observation point at model center
            self._obs_x = float(x.mean())
            self._obs_y = float(y.mean())
            self._obs_z = float(z.mean())
            self._obs_initialized = True
            self._sync_all_obs_sliders()

            self.statusBar().showMessage(f"数据已加载: {self.engine.nx} x {self.engine.ny} x {self.engine.nz}")
            self.on_tab_changed(0)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def load_npz_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "加载 .npz", "", "NumPy Files (*.npz)")
        if not filename:
            return
        try:
            data = np.load(filename)
            gd = {
                'stratum_id': data.get('stratum_id', np.zeros((1,1,1), dtype=np.int32)),
                'permeability': data.get('permeability', np.zeros((1,1,1), dtype=np.float64)),
                'idomain': data.get('idomain', None),
                'fracture_id': data.get('fracture_id', np.zeros((1,1,1), dtype=np.int32)),
                'lens_id': data.get('lens_id', np.zeros((1,1,1), dtype=np.int32)),
                'terrain_surface': data.get('top_elevation', None) if 'top_elevation' in data else None,
                'soil_bottom': data.get('soil_bottom', None) if 'soil_bottom' in data else None,
                'inside_mask': data.get('inside_mask', None) if 'inside_mask' in data else None,
                # Head fields
                'head_initial': data.get('head_initial', None) if 'head_initial' in data else None,
                'head_steady': data.get('head_steady', None) if 'head_steady' in data else None,
                # Concentration fields
                'concentration_initial': data.get('concentration_initial', None) if 'concentration_initial' in data else None,
                'concentration_final': data.get('concentration_final', None) if 'concentration_final' in data else None,
                # Source info
                'source_idx': data.get('source_idx', None) if 'source_idx' in data else None,
                'source_c': data.get('source_c', None) if 'source_c' in data else None,
                # Physical parameters
                'solubility': float(data['solubility']) if 'solubility' in data else None,
                'density': float(data['density']) if 'density' in data else None,
            }
            x = data.get('x', np.arange(gd['stratum_id'].shape[0]))
            y = data.get('y', np.arange(gd['stratum_id'].shape[1]))
            z = data.get('z', np.arange(gd['stratum_id'].shape[2]))
            self.load_data(gd, x, y, z)

            # Restore contaminant sources to UI list
            if hasattr(self.engine, 'source_idx') and self.engine.source_idx:
                self.contaminant_sources = []
                for (i, j, k), c in zip(self.engine.source_idx, self.engine.source_c):
                    self.contaminant_sources.append((
                        float(self.engine.x[i]),
                        float(self.engine.y[j]),
                        float(self.engine.z[k]),
                        float(c)
                    ))
                self.refresh_source_list()

            # Restore physical parameters to UI
            if hasattr(self.engine, 'solubility') and self.engine.solubility is not None:
                self.solubility_spin.setValue(self.engine.solubility)
            if hasattr(self.engine, 'density') and self.engine.density is not None:
                self.density_spin.setValue(self.engine.density)

            msg = f"数据已加载: {self.engine.nx} x {self.engine.ny} x {self.engine.nz}"
            if self.engine.head_steady is not None:
                msg += " | 含稳态水头"
            if self.engine.concentration is not None:
                msg += " | 含浓度场"
            if self.engine.source_idx:
                msg += f" | {len(self.engine.source_idx)}个污染源"
            self.statusBar().showMessage(msg)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"{str(e)}\n\n{traceback.format_exc()}")

    def load_vtk_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "加载 VTK", "", "VTK Files (*.vtk *.vtu)")
        if not filename:
            return
        QMessageBox.information(self, "提示", "VTK 加载为简化模式，建议使用 .npz 格式")

    # ---- Actions ----

    def on_tab_changed(self, idx):
        if self.engine is None:
            return
        if idx == 0:
            self.render_k()
        elif idx == 1:
            self.render_head()
        elif idx == 2:
            self.render_conc()

    def do_k_randomization(self):
        if self.engine is None:
            QMessageBox.warning(self, "提示", "请先加载数据")
            return
        try:
            self.set_busy("正在随机化 K...")
            K_new = self.engine.randomize_K(
                noise_scale=self.k_noise_scale.value(),
                amplitude_pct=self.k_amplitude.value(),
                seed=self.k_seed.value(),
            )
            self.left_tabs.setCurrentIndex(0)
            self.render_k()
            self.statusBar().showMessage(f"K 随机化完成: 均值={K_new.mean():.2e}, 范围=[{K_new.min():.2e}, {K_new.max():.2e}]")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"{str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.set_ready()

    def do_generate_head(self):
        if self.engine is None:
            QMessageBox.warning(self, "提示", "请先加载数据")
            return
        try:
            self.set_busy("正在生成初始水头...")
            use_unsat = self.unsat_check.isChecked()
            h = self.engine.generate_initial_head(
                gradient_slope=self.head_grad_slope.value(),
                gradient_direction=self.head_grad_dir.value(),
                depth_weight=self.head_depth_w.value(),
                unsaturated_mode=use_unsat,
                water_level=self.unsat_water_level.value() if use_unsat else None,
            )
            # Auto-render if currently on head tab
            if self.left_tabs.currentIndex() == 1:
                self.render_head()
            mode = "非饱和流" if use_unsat else "饱和流"
            self.statusBar().showMessage(f"[{mode}] 初始水头: 均值={h.mean():.2f}m, 范围=[{h.min():.2f}, {h.max():.2f}]")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"{str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.set_ready()

    def do_steady_state(self):
        if self.engine is None or self.engine.head_initial is None:
            QMessageBox.warning(self, "提示", "请先生成初始水头")
            return
        progress = None
        try:
            max_iter = self.ss_max_iter.value()
            progress = QProgressDialog("求解稳态水头 (SOR迭代)...", "取消", 0, max_iter, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            def callback(it, max_it, err):
                progress.setValue(it)
                progress.setLabelText(f"SOR迭代: {it}/{max_it}  |  残差={err:.2e}")
                QApplication.processEvents()
                return not progress.wasCanceled()

            self.set_busy("正在求解稳态水头...")
            h, it, err, early_stopped, final_omega = self.engine.solve_steady_state(
                max_iterations=max_iter,
                tolerance=self.ss_tol.value(),
                relaxation=self.ss_relax.value(),
                progress_callback=callback,
                early_stop_patience=self.ss_es_patience.value(),
                min_relative_change=self.ss_es_relative.value(),
            )
            progress.setValue(max_iter)

            # Check if user canceled
            canceled = getattr(self.engine, '_canceled', False)
            if canceled:
                self.ss_info.setText("已取消")
                self.statusBar().showMessage("稳态求解已取消")
                return

            es_msg = " (早停)" if early_stopped else ""
            self.ss_info.setText(f"迭代 {it} 次{es_msg}, 残差={err:.2e}, ω={final_omega:.3f}")
            self.left_tabs.setCurrentIndex(1)
            self.render_head()
            self.statusBar().showMessage(
                f"稳态水头{es_msg}: 迭代{it}次, 残差={err:.2e}, 范围=[{h.min():.2f}, {h.max():.2f}]")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"{str(e)}\n\n{traceback.format_exc()}")
        finally:
            if progress is not None:
                progress.close()
            self.set_ready()

    def _add_observation_sliders(self, layout, prefix):
        """Add shared observation point sliders to a tab layout.
        All tabs share the same physical observation point."""
        obs_grp = QGroupBox("观察点")
        obs_l = QVBoxLayout()
        obs_grp.setLayout(obs_l)

        # Initialize shared observation point at model center if not set
        if not hasattr(self, '_obs_initialized'):
            self._obs_x = 0.0
            self._obs_y = 0.0
            self._obs_z = 0.0
            self._obs_initialized = False

        for axis, label, attr_name in [('x', 'X', '_obs_x'), ('y', 'Y', '_obs_y'), ('z', 'Z', '_obs_z')]:
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(f"{label}"))

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setObjectName(f"obs_{prefix}_{axis}")
            slider.valueChanged.connect(lambda val, ax=axis, p=prefix: self.on_obs_slider_changed(val, ax, p))
            slider_layout.addWidget(slider)

            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(-99999, 99999)
            spin.setSingleStep(1.0)
            spin.setObjectName(f"obs_{prefix}_{axis}_spin")
            spin.valueChanged.connect(lambda val, ax=axis, p=prefix: self.on_obs_spin_changed(val, ax, p))
            slider_layout.addWidget(spin)

            obs_l.addLayout(slider_layout)
            setattr(self, f"obs_{prefix}_{axis}_slider", slider)
            setattr(self, f"obs_{prefix}_{axis}_spin", spin)

        # Value display label
        self.obs_value_label = QLabel("观察点参数 --")
        self.obs_value_label.setStyleSheet("color: #333; font-size: 11px; font-family: monospace;")
        obs_l.addWidget(self.obs_value_label)

        layout.addWidget(obs_grp)

    def on_angle_dial_changed(self, angle_rad):
        """Angle dial changed -> sync spinbox."""
        self.head_grad_dir.blockSignals(True)
        self.head_grad_dir.setValue(angle_rad)
        self.head_grad_dir.blockSignals(False)

    def on_grad_dir_spin_changed(self, value):
        """Gradient direction spinbox changed -> sync dial."""
        self.angle_dial.angle = value

    def on_unsat_mode_changed(self, state):
        """Toggle unsaturated flow mode UI.
        Saturated mode: hide water level input.
        Unsaturated mode: show water level input."""
        enabled = (state == Qt.Checked)
        self.unsat_water_level.setVisible(enabled)
        if hasattr(self, 'wl_value_label'):
            self.wl_value_label.setVisible(enabled)

    def on_obs_slider_changed(self, value, axis, prefix):
        """Shared observation point slider changed — update all tabs."""
        if self.engine is None:
            return
        nx, ny, nz = self.engine.nx, self.engine.ny, self.engine.nz
        x_r = float(self.engine.x[-1] - self.engine.x[0]) if nx > 1 else 1.0
        y_r = float(self.engine.y[-1] - self.engine.y[0]) if ny > 1 else 1.0
        z_r = float(self.engine.z[-1] - self.engine.z[0]) if nz > 1 else 1.0

        x0, y0, z0 = float(self.engine.x[0]), float(self.engine.y[0]), float(self.engine.z[0])

        if axis == 'x':
            self._obs_x = x0 + (value / 1000.0) * x_r
            self._obs_initialized = True
        elif axis == 'y':
            self._obs_y = y0 + (value / 1000.0) * y_r
            self._obs_initialized = True
        elif axis == 'z':
            self._obs_z = z0 + (value / 1000.0) * z_r
            self._obs_initialized = True

        self._sync_all_obs_sliders()
        self._refresh_current_tab()

    def on_obs_spin_changed(self, value, axis, prefix):
        """Shared observation point spinbox changed — update all tabs."""
        if self.engine is None:
            return
        nx, ny, nz = self.engine.nx, self.engine.ny, self.engine.nz
        x_r = float(self.engine.x[-1] - self.engine.x[0]) if nx > 1 else 1.0
        y_r = float(self.engine.y[-1] - self.engine.y[0]) if ny > 1 else 1.0
        z_r = float(self.engine.z[-1] - self.engine.z[0]) if nz > 1 else 1.0

        x0, y0, z0 = float(self.engine.x[0]), float(self.engine.y[0]), float(self.engine.z[0])

        if axis == 'x' and x_r > 0:
            self._obs_x = value
            self._obs_initialized = True
        elif axis == 'y' and y_r > 0:
            self._obs_y = value
            self._obs_initialized = True
        elif axis == 'z' and z_r > 0:
            self._obs_z = value
            self._obs_initialized = True

        self._sync_all_obs_sliders()
        self._refresh_current_tab()

    def _sync_all_obs_sliders(self):
        """Sync all observation point sliders/spins across tabs (without triggering callbacks)."""
        for prefix in ['k', 'head', 'conc']:
            for axis in ['x', 'y', 'z']:
                slider = getattr(self, f"obs_{prefix}_{axis}_slider", None)
                spin = getattr(self, f"obs_{prefix}_{axis}_spin", None)
                if slider is None or spin is None or self.engine is None:
                    continue
                val = getattr(self, f'_obs_{axis}', 0.0)
                nx, ny, nz = self.engine.nx, self.engine.ny, self.engine.nz
                x_r = float(self.engine.x[-1] - self.engine.x[0]) if nx > 1 else 1.0
                y_r = float(self.engine.y[-1] - self.engine.y[0]) if ny > 1 else 1.0
                z_r = float(self.engine.z[-1] - self.engine.z[0]) if nz > 1 else 1.0
                x0, y0, z0 = float(self.engine.x[0]), float(self.engine.y[0]), float(self.engine.z[0])
                ranges = {'x': (x0, x_r), 'y': (y0, y_r), 'z': (z0, z_r)}
                origin, range_v = ranges[axis]
                if range_v > 0:
                    pct = int((val - origin) / range_v * 1000)
                    pct = max(0, min(1000, pct))
                else:
                    pct = 500
                slider.blockSignals(True)
                slider.setValue(pct)
                slider.blockSignals(False)
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)

    def _refresh_current_tab(self):
        """Refresh the currently visible tab's render."""
        idx = self.left_tabs.currentIndex()
        if idx == 0:
            self.render_k()
        elif idx == 1:
            self.render_head()
        elif idx == 2:
            self.render_conc()

    def _interp_value_at_obs(self, field_3d):
        """Trilinear interpolation of a 3D field at the observation point."""
        if self.engine is None or not self._obs_initialized:
            return None
        px, py, pz = self._obs_x, self._obs_y, self._obs_z

        # Find containing cell
        ix = int(np.clip(np.searchsorted(self.engine.x, px) - 1, 0, self.engine.nx - 2))
        iy = int(np.clip(np.searchsorted(self.engine.y, py) - 1, 0, self.engine.ny - 2))
        iz = int(np.clip(np.searchsorted(self.engine.z, pz) - 1, 0, self.engine.nz - 2))

        # Local coordinates
        dx = self.engine.x[ix+1] - self.engine.x[ix] if self.engine.nx > 1 else 1.0
        dy = self.engine.y[iy+1] - self.engine.y[iy] if self.engine.ny > 1 else 1.0
        dz = self.engine.z[iz+1] - self.engine.z[iz] if self.engine.nz > 1 else 1.0
        fx = (px - self.engine.x[ix]) / dx if dx > 0 else 0.0
        fy = (py - self.engine.y[iy]) / dy if dy > 0 else 0.0
        fz = (pz - self.engine.z[iz]) / dz if dz > 0 else 0.0
        fx = np.clip(fx, 0, 1); fy = np.clip(fy, 0, 1); fz = np.clip(fz, 0, 1)

        # Trilinear interpolation
        c000 = field_3d[ix, iy, iz]
        c001 = field_3d[ix, iy, iz+1]
        c010 = field_3d[ix, iy+1, iz]
        c011 = field_3d[ix, iy+1, iz+1]
        c100 = field_3d[ix+1, iy, iz]
        c101 = field_3d[ix+1, iy, iz+1]
        c110 = field_3d[ix+1, iy+1, iz]
        c111 = field_3d[ix+1, iy+1, iz+1]

        return (1-fx)*(1-fy)*(1-fz)*c000 + (1-fx)*(1-fy)*fz*c001 + \
               (1-fx)*fy*(1-fz)*c010 + (1-fx)*fy*fz*c011 + \
               fx*(1-fy)*(1-fz)*c100 + fx*(1-fy)*fz*c101 + \
               fx*fy*(1-fz)*c110 + fx*fy*fz*c111

    def _draw_observation_point(self, mode='k'):
        """Draw observation point marker and display value in corner."""
        if self.engine is None or not self._obs_initialized:
            return

        pos = (self._obs_x, self._obs_y, self._obs_z)
        r = min(self.engine.dx, self.engine.dy, self.engine.dz) * 0.8

        # Green sphere marker
        sphere = pv.Sphere(radius=r, center=pos, theta_resolution=16, phi_resolution=16)
        self.plotter.add_mesh(sphere, color='lime', opacity=0.9)

        # White crosshair for visibility
        for offset, color in [(-r*1.5, 'white'), (r*1.5, 'white')]:
            line_x = pv.Line((pos[0]+offset, pos[1], pos[2]), (pos[0]-offset, pos[1], pos[2]))
            line_y = pv.Line((pos[0], pos[1]+offset, pos[2]), (pos[0], pos[1]-offset, pos[2]))
            line_z = pv.Line((pos[0], pos[1], pos[2]+offset), (pos[0], pos[1], pos[2]-offset))
            for line in [line_x, line_y, line_z]:
                self.plotter.add_mesh(line, color=color, line_width=2)

        # Get interpolated value based on mode
        if mode == 'k':
            K = self.engine.K_randomized if self.engine.K_randomized is not None else self.engine.K_original
            val = self._interp_value_at_obs(K)
            label_text = f"({self._obs_x:.1f}, {self._obs_y:.1f}, {self._obs_z:.1f})\nK={val:.2e} m/s" if val is not None else ""
        elif mode == 'head':
            h = self.engine.head_steady if self.engine.head_steady is not None else self.engine.head_initial
            val = self._interp_value_at_obs(h) if h is not None else None
            label_text = f"({self._obs_x:.1f}, {self._obs_y:.1f}, {self._obs_z:.1f})\nHead={val:.2f}m" if val is not None else ""
        elif mode == 'conc':
            c = self.engine.concentration if self.engine.concentration is not None else \
                (self.engine.concentration_initial if hasattr(self.engine, 'concentration_initial') else None)
            val = self._interp_value_at_obs(c) if c is not None else None
            label_text = f"({self._obs_x:.1f}, {self._obs_y:.1f}, {self._obs_z:.1f})\nC={val:.4f}mg/L" if val is not None else ""
        else:
            label_text = ""

        if label_text:
            self.plotter.add_text(label_text, position='upper_left', font_size=10,
                                  color='black', shadow=True)
            # Also update the label in the sidebar
            if hasattr(self, 'obs_value_label'):
                short = label_text.replace('\n', ' | ')
                self.obs_value_label.setText(short)

    def on_species_changed(self, index):
        """Auto-fill physical parameters when a preset pollutant is selected."""
        presets = {
            1: (10000.0, 0.91, 1.0e-5),    # 氨氮
            2: (8000.0, 1.24, 8.6e-6),     # 二氯乙烷
            3: (1100.0, 1.22, 7.2e-6),     # 二氯乙烯
        }
        if index in presets:
            sol, dens, md = presets[index]
            self.solubility_spin.setValue(sol)
            self.density_spin.setValue(dens)
            self.mol_diff_spin.setValue(md)

    def on_disp_mode_changed(self, text):
        """Switch between manual and adaptive dispersivity inputs."""
        is_adaptive = (text == "自适应 (基于K)")
        # Manual mode controls
        self.manual_alpha_l_label.setVisible(not is_adaptive)
        self.manual_alpha_l.setVisible(not is_adaptive)
        self.manual_alpha_t_label.setVisible(not is_adaptive)
        self.manual_alpha_t.setVisible(not is_adaptive)
        # Adaptive mode controls
        self.adaptive_long_label.setVisible(is_adaptive)
        self.adaptive_long_factor.setVisible(is_adaptive)
        self.adaptive_trans_label.setVisible(is_adaptive)
        self.adaptive_trans_factor.setVisible(is_adaptive)
        self.adaptive_info.setVisible(is_adaptive)

    def do_init_contaminant(self):
        if self.engine is None:
            QMessageBox.warning(self, "提示", "请先加载数据")
            return
        if not self.contaminant_sources:
            QMessageBox.warning(self, "提示", "请至少添加一个污染源")
            return
        progress = None
        try:
            self.set_busy("正在初始化污染物场...")

            # Progress dialog for initialization
            progress = QProgressDialog("初始化污染物场...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            def init_callback(msg, pct):
                progress.setValue(pct)
                progress.setLabelText(msg)
                QApplication.processEvents()
                return not progress.wasCanceled()

            points = [(s[0], s[1], s[2]) for s in self.contaminant_sources]
            concs = [s[3] for s in self.contaminant_sources]
            use_adaptive = (self.disp_mode_combo.currentText() == "自适应 (基于K)")
            if use_adaptive:
                alpha_l = self.adaptive_long_factor.value()
                alpha_t = self.adaptive_trans_factor.value()
            else:
                alpha_l = self.manual_alpha_l.value()
                alpha_t = self.manual_alpha_t.value()
            c_init = self.engine.initialize_contaminant(
                source_points=points, source_concentrations=concs,
                dispersivity_long=alpha_l,
                dispersivity_trans=alpha_t,
                diffusion_coeff=self.mol_diff_spin.value(),
                retardation=self.conc_retard.value(),
                decay_rate=self.conc_decay.value(),
                adaptive_alpha=use_adaptive,
                solubility=self.solubility_spin.value(),
                density=self.density_spin.value(),
                progress_callback=init_callback,
            )
            progress.setValue(100)
            mode_str = "自适应K" if use_adaptive else "手动"
            self.left_tabs.setCurrentIndex(2)
            self.render_conc()
            self.statusBar().showMessage(f"污染物场 ({mode_str}): {len(points)}个源, 浓度范围=[{c_init.min():.2f}, {c_init.max():.2f}]")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"{str(e)}\n\n{traceback.format_exc()}")
        finally:
            if progress is not None:
                progress.close()
            self.set_ready()

    def do_contaminant_sim(self):
        if self.engine is None or not hasattr(self.engine, 'concentration_initial'):
            QMessageBox.warning(self, "提示", "请先初始化污染物场")
            return
        progress = None
        try:
            self.set_busy("正在运行污染物运移模拟...")

            # QProgressDialog for long-running simulation
            progress = QProgressDialog("污染物运移模拟中...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            def callback(step, total, t):
                pct = int(100 * step / max(total, 1))
                progress.setValue(pct)
                progress.setLabelText(f"运移模拟: 步 {step}/{total}  |  t={t:.2f}天")
                QApplication.processEvents()
                return not progress.wasCanceled()

            conc = self.engine.simulate_contaminant(
                duration_days=self.conc_duration.value(),
                dt_days=self.conc_dt.value(),
                progress_callback=callback,
            )
            progress.setValue(100)

            if getattr(self.engine, '_canceled', False):
                self.statusBar().showMessage("模拟已取消")
                QMessageBox.information(self, "取消", "运移模拟已被用户取消")
            else:
                self.current_time_idx = 0
                self.time_label.setText(f"最终结果 (t={self.conc_duration.value():.1f}天)")
                self.left_tabs.setCurrentIndex(2)
                self.render_conc()
                cmin, cmax = conc.min(), conc.max()
                self.statusBar().showMessage(f"污染物运移完成: 浓度范围=[{cmin:.4f}, {cmax:.4f}]")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"{str(e)}\n\n{traceback.format_exc()}")
        finally:
            if progress is not None:
                progress.close()
            self.set_ready()

    def do_reset_concentration(self):
        """Reset concentration: zero field + re-apply Dirichlet BC + 1-day pre-run."""
        if self.engine is None or not hasattr(self.engine, 'source_idx'):
            QMessageBox.warning(self, "提示", "无可重置的污染物场")
            return
        try:
            self.set_busy("正在重置浓度场...")
            active = self.engine.idomain > 0

            # Zero + Dirichlet + 1-day pre-run
            C = np.zeros((self.engine.nx, self.engine.ny, self.engine.nz), dtype=np.float64)
            C = self.engine._apply_dirichlet(C, self.engine.source_idx, self.engine.source_c)

            dt_init = 1.0
            n_init_steps = max(1, int(np.ceil(dt_init / 0.5)))
            dt_sub = dt_init / n_init_steps
            for _ in range(n_init_steps):
                C = self.engine._transport_step(C, self.engine._vx, self.engine._vy, self.engine._vz,
                                                self.engine._Dxx, self.engine._Dyy, self.engine._Dzz,
                                                active, dt_sub,
                                                self.engine.source_idx, self.engine.source_c)

            self.engine.concentration = C.copy()
            self.engine.concentration_initial = C.copy()
            self.time_label.setText("已重置 (Dirichlet BC + 1天预推进)")
            self.left_tabs.setCurrentIndex(2)
            self.render_conc()
            self.statusBar().showMessage("浓度已重置: Dirichlet源点 + 1天预推进")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"{str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.set_ready()

    def on_time_changed(self, value):
        """Placeholder — time slider removed, only final result stored."""
        self.current_time_idx = 0
        if self.engine is not None and self.engine.concentration is not None:
            self.time_label.setText(f"最终结果 (t={self.conc_duration.value():.1f}天)")
        self.render_conc()

    # ---- Rendering ----

    def _create_grid(self):
        grid = pv.ImageData()
        grid.dimensions = np.array([self.engine.nx, self.engine.ny, self.engine.nz]) + 1
        grid.origin = (float(self.engine.x.min()), float(self.engine.y.min()), float(self.engine.z.min()))
        grid.spacing = (self.engine.dx, self.engine.dy, self.engine.dz)
        return grid

    def _render_layered(self, grid, scalar_name, cmap, title, clim=None):
        """Per-cell (voxel) rendering: each cell rendered individually."""
        active = self.engine.idomain > 0
        self._render_layered_with_mask(grid, scalar_name, cmap, title, active, clim)

    def _render_layered_with_mask(self, grid, scalar_name, cmap, title, active_mask, clim=None):
        """Per-cell (voxel) rendering with custom active mask.
        Each cell is rendered individually so interior cells are visible."""
        n_colors = getattr(self, 'ncolors_spin', None)
        n_c = n_colors.value() if n_colors else 256

        # Add a temporary mask scalar to the grid for cell filtering
        grid.cell_data["_render_mask"] = active_mask.flatten(order="F").astype(np.uint8)

        # Low opacity + layered rendering so interior structures are visible
        OPACITY = 0.35

        # Step 1: filter by active_mask via threshold
        sids = sorted(np.unique(self.engine.stratum_id[active_mask]))
        for sid in sids:
            # Filter: stratum_id matches AND active_mask=True
            sid_mask = (self.engine.stratum_id == sid) & active_mask
            if not np.any(sid_mask):
                continue
            grid.cell_data["_sid_mask"] = sid_mask.flatten(order="F").astype(np.uint8)
            layer_grid = grid.threshold([0.5, 1.5], scalars="_sid_mask")
            if "_sid_mask" in grid.cell_data:
                del grid.cell_data["_sid_mask"]
            if layer_grid.n_cells == 0:
                continue
            # Extract surface (layer boundary) — gaps between layers let light through
            surface = layer_grid.extract_surface(algorithm='dataset_surface')
            if surface.n_points == 0:
                continue
            self.plotter.add_mesh(surface, scalars=scalar_name, cmap=cmap,
                opacity=OPACITY, show_scalar_bar=False, n_colors=n_c,
                smooth_shading=True, show_edges=False, lighting=True,
                clim=clim, specular=0.3, specular_power=20)

        # Scalar bar on a dummy invisible mesh for correct range
        active_flat = active_mask.flatten(order="F")
        if np.any(active_flat):
            active_grid = grid.extract_cells(np.where(active_flat)[0])
            if active_grid.n_cells > 0:
                self.plotter.add_mesh(active_grid, scalars=scalar_name, cmap=cmap,
                    opacity=0.0, show_scalar_bar=True,
                    scalar_bar_args={"title": title, "vertical": True},
                    clim=clim)

        # Clean up temporary scalar
        if "_render_mask" in grid.cell_data:
            del grid.cell_data["_render_mask"]

    def render_k(self):
        if self.engine is None:
            return
        self.plotter.clear()
        grid = self._create_grid()
        K = self.engine.K_randomized if (self.engine.K_randomized is not None) else self.engine.K_original
        K_orig = self.engine.K_original
        Kd = K.copy()
        Kd[Kd <= 0] = 1e-12
        logK = np.log10(Kd)
        grid.cell_data["log_K"] = logK.flatten(order="F")
        grid.cell_data["stratum_id"] = self.engine.stratum_id.flatten(order="F")

        # Check if range filter is enabled
        range_enabled = getattr(self, 'k_range_enable', None)
        use_filter = range_enabled.isChecked() if range_enabled is not None else False
        logK_min_spin = getattr(self, 'k_range_min', None)
        logK_max_spin = getattr(self, 'k_range_max', None)
        lo = logK_min_spin.value() if logK_min_spin is not None else -4.0
        hi = logK_max_spin.value() if logK_max_spin is not None else 0.0

        # Build active mask with optional range filter
        active = self.engine.idomain > 0
        if use_filter:
            active = active & (logK >= lo) & (logK <= hi)

        # Diagnostic
        sids = sorted(np.unique(self.engine.stratum_id[active]))
        lines = []
        for sid in sids:
            if sid != int(sid):
                continue
            mask = (self.engine.stratum_id == sid) & active
            if not np.any(mask):
                continue
            k_vals = K[mask]
            k_base = self.engine.K_original[mask][0]
            lines.append(f"[{int(sid)}] K_base={k_base:.2e} -> [{k_vals.min():.2e}, {k_vals.max():.2e}]")
        k_all_min, k_all_max = K[active].min(), K[active].max()
        lines.append(f"可见: [{k_all_min:.2e}, {k_all_max:.2e}]")
        if use_filter:
            lines.append(f"[过滤: logK={lo:.1f}~{hi:.1f}]")
        msg = " | ".join(lines)
        if len(msg) > 200:
            msg = msg[:197] + "..."
        self.statusBar().showMessage(msg)

        # Color bar always shows full range (all active cells), not filtered range
        all_active = self.engine.idomain > 0
        all_active_flat = all_active.flatten(order="F")
        full_logK = logK.flatten(order="F")
        full_clim = [float(full_logK[all_active_flat].min()), float(full_logK[all_active_flat].max())]

        self._render_layered_with_mask(grid, "log_K", "viridis",
            f"log₁₀(K [m/s])  [显示: {lo:.1f}~{hi:.1f}]" if use_filter else "log₁₀(K [m/s])",
            active, clim=full_clim)

        # ---- Direct render high-K heterogeneous structures (fractures, lenses) in RED ----
        # Use threshold on logK to find high-K cells, render them as individual red voxels
        # logK > 2.0 means K > 100 m/d — clearly heterogeneous
        grid.cell_data["_highK"] = (logK > 2.0).flatten(order="F").astype(np.uint8)
        highK_grid = grid.threshold([0.5, 1.5], scalars="_highK")
        if highK_grid.n_cells > 0:
            # Render as individual red voxels (not just surface outline)
            self.plotter.add_mesh(highK_grid, color='#ff0000', opacity=0.9,
                show_edges=True, edge_color='#cc0000', line_width=0.5,
                lighting=True, label='裂隙/透镜体')
        if "_highK" in grid.cell_data:
            del grid.cell_data["_highK"]

        self._draw_observation_point(mode='k')

    def render_head(self):
        if self.engine is None:
            return
        self.plotter.clear()
        grid = self._create_grid()
        if self.engine.head_steady is not None:
            h = self.engine.head_steady
        elif self.engine.head_initial is not None:
            h = self.engine.head_initial
        else:
            h = np.zeros((self.engine.nx, self.engine.ny, self.engine.nz), dtype=np.float64)
        grid.cell_data["head"] = h.flatten(order="F")
        grid.cell_data["stratum_id"] = self.engine.stratum_id.flatten(order="F")
        # Fixed colorbar range to model's z extent for consistent comparison
        z_min = float(self.engine.z.min())
        z_max = float(self.engine.z.max())
        self._render_layered(grid, "head", "coolwarm", "Head [m]", clim=[z_min, z_max])
        self._draw_observation_point(mode='head')

        # Draw gradient direction arrow at the bottom of the grid
        if hasattr(self.engine, 'gradient_direction'):
            self._draw_gradient_arrow(self.engine.gradient_direction)

        # Draw water table surface if unsaturated mode was used
        if hasattr(self.engine, '_water_table') and self.engine._water_table is not None:
            self._draw_water_table()

    def _draw_water_table(self):
        """Draw water table surface extracted from solved head.
        Shows the phreatic surface (free surface) from steady-state solution."""
        import pyvista as pv

        wt = getattr(self.engine, '_water_table', None)
        if wt is None:
            return

        # Build 3D points from water table grid
        nx, ny = wt.shape
        xx, yy = np.meshgrid(self.engine.x, self.engine.y, indexing='ij')
        points = np.column_stack((xx.ravel(), yy.ravel(), wt.ravel()))
        valid = ~np.isnan(points[:, 2])
        n_valid = np.count_nonzero(valid)
        if n_valid < 3:
            return
        points = points[valid]

        # Check z-variation: if all points are nearly coplanar in Z,
        # delaunay_2d may produce degenerate triangles. Add small jitter.
        z_range = points[:, 2].max() - points[:, 2].min()
        if z_range < 1e-6:
            points[:, 2] += np.random.RandomState(0).uniform(-1e-4, 1e-4, size=points.shape[0])

        # Create a structured grid from water table points
        surf = pv.PolyData(points)
        surf = surf.delaunay_2d()
        if surf.n_points == 0 or surf.n_cells == 0:
            return

        self.plotter.add_mesh(surf, color='#4FC3F7', opacity=0.5,
                              show_edges=True, edge_color='#0277BD', line_width=1,
                              lighting=False)

    def _draw_gradient_arrow(self, angle_rad):
        """Draw a large arrow at the bottom of the grid showing gradient direction.
        0 rad = +X (right), π/2 = +Y (up), etc."""
        import pyvista as pv

        x0 = (self.engine.x.min() + self.engine.x.max()) / 2.0
        y0 = (self.engine.y.min() + self.engine.y.max()) / 2.0
        z_bottom = self.engine.z.min() - self.engine.dz * 2

        # Arrow length = half the grid diagonal
        diag = np.sqrt((self.engine.x.max() - self.engine.x.min())**2 +
                       (self.engine.y.max() - self.engine.y.min())**2)
        arrow_len = diag * 0.25

        # End point
        x1 = x0 + arrow_len * np.cos(angle_rad)
        y1 = y0 + arrow_len * np.sin(angle_rad)

        shaft = pv.Line((x0, y0, z_bottom), (x1, y1, z_bottom))
        self.plotter.add_mesh(shaft, color='#2196F3', line_width=4)

        # Arrow head (cone)
        cone = pv.Cone(center=((x0+x1)/2, (y0+y1)/2, z_bottom),
                       direction=(np.cos(angle_rad), np.sin(angle_rad), 0),
                       height=arrow_len * 0.3, radius=arrow_len * 0.1,
                       resolution=16)
        cone.points[:, 0] += (x1 - x0) * 0.35
        cone.points[:, 1] += (y1 - y0) * 0.35
        self.plotter.add_mesh(cone, color='#2196F3', opacity=0.9)

        # No text label — arrow only

    def render_conc(self):
        if self.engine is None:
            return
        self.plotter.clear()
        grid = self._create_grid()
        if self.engine.concentration is not None:
            c = self.engine.concentration
        elif hasattr(self.engine, 'concentration_initial') and self.engine.concentration_initial is not None:
            c = self.engine.concentration_initial
        else:
            c = np.zeros((self.engine.nx, self.engine.ny, self.engine.nz), dtype=np.float64)

        c_safe = np.where(np.isfinite(c), c, 0.0)

        # ---- Compute color bar upper limit ----
        clim_mode = getattr(self, 'clim_mode_combo', None)
        use_auto_clim = (clim_mode.currentText() == "自动上限 (实际最大值取整)") if clim_mode else True

        source_mask = np.zeros(c_safe.shape, dtype=bool)
        if hasattr(self.engine, 'source_idx') and self.engine.source_idx:
            for (si, sj, sk) in self.engine.source_idx:
                source_mask[si, sj, sk] = True
        c_no_source = np.where(source_mask, 0.0, c_safe)
        c_actual_max = float(np.nanmax(c_no_source)) if np.any(c_no_source > 0) else float(np.nanmax(c_safe))

        def _round_up_1sig(x):
            if x <= 0: return 1.0
            d = 10.0 ** np.floor(np.log10(x))
            return np.ceil(x / d) * d

        if use_auto_clim:
            cmax_display = _round_up_1sig(c_actual_max * 1.2) if c_actual_max > 0 else 1.0
        else:
            src_c = getattr(self.engine, 'source_c', None)
            cmax_display = float(np.max(src_c)) if src_c is not None and len(src_c) > 0 else (float(np.nanmax(c_safe)) if np.any(c_safe > 0) else 1.0)

        use_log = getattr(self, 'log_scale_check', None)
        is_log = (use_log.currentText() == "对数色标 (log₁₀)") if use_log else False

        if is_log:
            c_display = np.log10(np.clip(c_safe, 1e-20, cmax_display))
            grid.cell_data["concentration"] = c_display.flatten(order="F")
            cmax = np.log10(cmax_display)
            cmin = np.log10(1e-20)
            title = f"log₁₀(C [mg/L]), max={cmax_display:.2f}"
            conc_cmap = self._get_conc_colormap(log=True)
        else:
            c_display = np.clip(c_safe, 0.0, cmax_display)
            grid.cell_data["concentration"] = c_display.flatten(order="F")
            cmax = cmax_display
            cmin = 0.0
            title = f"C [mg/L], max={cmax_display:.2f}"
            conc_cmap = self._get_conc_colormap(log=False)

        grid.cell_data["stratum_id"] = self.engine.stratum_id.flatten(order="F")
        self._render_layered(grid, "concentration", conc_cmap, title, clim=[cmin, cmax])

        # Source markers
        if hasattr(self.engine, 'source_idx') and self.engine.source_idx:
            for (si, sj, sk), sc in zip(self.engine.source_idx, self.engine.source_c):
                sphere = pv.Sphere(radius=min(self.engine.dx, self.engine.dy, self.engine.dz),
                                    center=(self.engine.x[si], self.engine.y[sj], self.engine.z[sk]))
                self.plotter.add_mesh(sphere, color='red', opacity=0.9)

        self._draw_observation_point(mode='conc')

    def _get_conc_colormap(self, log=False):
        """White -> Yellow -> Black colormap for concentration display.
        For linear: white(0) -> yellow(0.5) -> black(1.0)
        For log: uses matplotlib's 'afmhot' as base, inverted."""
        from matplotlib.colors import LinearSegmentedColormap
        if log:
            # For log scale: use afmhot_r (reversed, white->yellow->black)
            import matplotlib.cm as cm
            return cm.get_cmap('afmhot_r', 256)
        else:
            # Custom: light gray -> yellow -> black
            # 0 = very light gray (visible on white background) not pure white
            colors = [
                (0.94, 0.94, 0.94),  # 0.0: light gray (not pure white)
                (1.0, 1.0, 0.8),     # 0.1: very light yellow
                (1.0, 1.0, 0.5),     # 0.2: light yellow
                (1.0, 1.0, 0.0),     # 0.5: yellow
                (0.8, 0.8, 0.0),     # 0.6: olive
                (0.5, 0.5, 0.0),     # 0.7: dark olive
                (0.3, 0.3, 0.0),     # 0.8: very dark
                (0.1, 0.1, 0.1),     # 0.9: near black
                (0.0, 0.0, 0.0),     # 1.0: black
            ]
            positions = [0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            return LinearSegmentedColormap.from_list("gyb", list(zip(positions, colors)), N=256)

    # ---- Export ----

    def export_npz(self):
        if self.engine is None:
            QMessageBox.warning(self, "提示", "无数据可导出")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "导出 .npz", "postprocess_data.npz", "NumPy Files (*.npz)")
        if not filename:
            return
        try:
            data_dict = {
                'stratum_id': self.engine.stratum_id,
                'permeability': self.engine.K_original,
                'K_randomized': self.engine.K_randomized if self.engine.K_randomized is not None else self.engine.K_original,
                'idomain': self.engine.idomain,
                'x': self.engine.x, 'y': self.engine.y, 'z': self.engine.z,
            }
            if self.engine.head_steady is not None:
                data_dict['head_steady'] = self.engine.head_steady
            if self.engine.head_initial is not None:
                data_dict['head_initial'] = self.engine.head_initial
            if self.engine.concentration is not None:
                data_dict['concentration_final'] = self.engine.concentration
            if hasattr(self.engine, 'concentration_initial') and self.engine.concentration_initial is not None:
                data_dict['concentration_initial'] = self.engine.concentration_initial
            # Source info
            if hasattr(self.engine, 'source_idx') and self.engine.source_idx:
                data_dict['source_idx'] = np.array(self.engine.source_idx, dtype=np.int32)
                data_dict['source_c'] = self.engine.source_c
            # Physical parameters
            for attr, key in [('solubility', 'solubility'), ('density', 'density')]:
                if hasattr(self.engine, attr) and getattr(self.engine, attr) is not None:
                    data_dict[key] = getattr(self.engine, attr)
            np.savez(filename, **data_dict)
            QMessageBox.information(self, "导出成功", f"已保存:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def export_vtk(self):
        if self.engine is None:
            QMessageBox.warning(self, "提示", "无数据可导出")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "导出 VTK", "postprocess.vtu", "VTK Files (*.vtu)")
        if not filename:
            return
        try:
            grid = self._create_grid()
            K = self.engine.K_randomized if self.engine.K_randomized is not None else self.engine.K_original
            grid.cell_data["K"] = K.flatten(order="F")
            grid.cell_data["stratum_id"] = self.engine.stratum_id.flatten(order="F")
            # Head: steady > initial
            if self.engine.head_steady is not None:
                grid.cell_data["head"] = self.engine.head_steady.flatten(order="F")
            elif self.engine.head_initial is not None:
                grid.cell_data["head"] = self.engine.head_initial.flatten(order="F")
            # Concentration: final > initial
            if self.engine.concentration is not None:
                grid.cell_data["concentration"] = self.engine.concentration.flatten(order="F")
            elif hasattr(self.engine, 'concentration_initial') and self.engine.concentration_initial is not None:
                grid.cell_data["concentration"] = self.engine.concentration_initial.flatten(order="F")
            grid.save(filename)
            QMessageBox.information(self, "导出成功", f"已保存:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def set_busy(self, msg):
        self.statusBar().showMessage(msg)
        QApplication.processEvents()

    def set_ready(self):
        self.statusBar().showMessage("就绪")

    def closeEvent(self, event):
        try:
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.close()
        except Exception:
            pass
        event.accept()


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Microsoft YaHei", 9))
    window = PostProcessWindow()
    window.show()
    sys.exit(app.exec_())