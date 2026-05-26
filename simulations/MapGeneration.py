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

        # Random threshold: ~15-35% of area becomes bare rock
        threshold = self.rng.uniform(0.15, 0.35)
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
        self.perm_spin.setDecimals(6)
        self.perm_spin.setSingleStep(0.0001)
        self.perm_spin.setValue(round(self.config.permeability, 6))
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
            self.perm_spin.setValue(preset.get("permeability", 1.0))

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
            permeability=round(self.perm_spin.value(), 6),
            preset_name=self.preset_combo.currentText(),
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

        self.setup_ui()
        self.refresh_stratum_list()

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

        self.tab_strata = QWidget()
        self.tabs.addTab(self.tab_strata, "地层管理")
        self._setup_strata_tab()

        self.tab_noise = QWidget()
        self.tabs.addTab(self.tab_noise, "噪声参数")
        self._setup_noise_tab()

        self.status_label = QLabel(" 就绪，可生成地质模型 ")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)

        self.generate_button = QPushButton("生成 3D 地质模型")
        self.generate_button.setMinimumHeight(42)
        self.generate_button.clicked.connect(self.generate_model)
        control_layout.addWidget(self.generate_button)

        export_layout = QHBoxLayout()
        self.export_stl_btn = QPushButton("导出 STL")
        self.export_stl_btn.clicked.connect(self.export_mesh_stl)
        self.export_vtk_btn = QPushButton("导出 VTK")
        self.export_vtk_btn.clicked.connect(self.export_mesh_vtk)
        export_layout.addWidget(self.export_stl_btn)
        export_layout.addWidget(self.export_vtk_btn)
        control_layout.addLayout(export_layout)
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
        self.complexity_spin = self._add_double(layout, "地层复杂度", 0.0, 1.0, 0.3, 0.05)

        info = QLabel("复杂度参考:\n  0.0=完全水平  0.4=倾斜+褶皱  0.8=强变形+断层")
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
            )

            solid, x, y, z, stratum_tops, soil_bottom, terrain_surface, inside_mask = \
                generator.generate_voxel()

            self.last_solid = solid
            self.last_x, self.last_y, self.last_z = x, y, z
            self.last_stratum_tops = stratum_tops
            self.last_soil_bottom = soil_bottom
            self.last_terrain_surface = terrain_surface
            self.last_inside_mask = inside_mask

            grid = pv.ImageData()
            grid.dimensions = np.array(solid.shape) + 1
            grid.origin = (float(x.min()), float(y.min()), float(z.min()))
            grid.spacing = (
                float(x[1] - x[0]) if len(x) > 1 else 1.0,
                float(y[1] - y[0]) if len(y) > 1 else 1.0,
                float(z[1] - z[0]) if len(z) > 1 else 1.0,
            )
            grid.cell_data["stratum_id"] = solid.flatten(order="F")
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
        n_x = min(int((x_max - x_min) / xy_step) + 1, 15)
        n_y = min(int((y_max - y_min) / xy_step) + 1, 15)
        n_z = min(int((z_max - z_min) / z_step) + 1, 15)
        bounds_actor = self.plotter.show_bounds(
            bounds=bounds, grid='front', location='outer', ticks='outside',
            all_edges=True, xtitle='X (m)', ytitle='Y (m)', ztitle='高程 (m)',
            fmt='%.0f', n_xlabels=n_x, n_ylabels=n_y, n_zlabels=n_z)
        try:
            bounds_actor.SetXAxisRange(x_min, x_max)
            bounds_actor.SetYAxisRange(y_min, y_max)
            bounds_actor.SetZAxisRange(z_min, z_max)
        except Exception:
            pass
        self._tracked_actors.append(bounds_actor)

    # ---- Export ----

    def export_mesh_stl(self):
        if not self.current_meshes:
            QMessageBox.warning(self, "导出失败", "请先生成模型")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存 STL", "geology_model.stl", "STL Files (*.stl)")
        if not filename:
            return
        try:
            merged = pv.MultiBlock(list(self.current_meshes.values()))
            combined = merged.combine()
            combined.save(filename)
            QMessageBox.information(self, "导出成功", f"已保存:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

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