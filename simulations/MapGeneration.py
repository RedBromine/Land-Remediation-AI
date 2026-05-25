import sys
import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt


# =========================================================
# 纯 numpy Perlin Noise（替代 noise 库 C 扩展，永不 segfault）
# =========================================================

class _PerlinNoise:
    """纯 numpy 2D Perlin noise + FBM。向量化输入，无 C 扩展。"""

    # Ken Perlin 原始排列表
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
        """向量化 2D Perlin noise。x, y 可以是标量或 ndarray。"""
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
        """分形布朗运动。x, y 可以是标量或 ndarray。"""
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
# Terrain Generator
# =========================================================

class TerrainGenerator:

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
        lacunarity=2.0
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

    # =====================================================

    def generate_voxel(self):

        # =================================================
        # 坐标网格
        # =================================================

        x = np.linspace(
            -self.radius,
            self.radius,
            self.resolution_xy
        )

        y = np.linspace(
            -self.radius,
            self.radius,
            self.resolution_xy
        )

        z = np.linspace(
            -self.depth,
            self.max_height,
            self.resolution_z
        )

        xx, yy = np.meshgrid(x, y)

        rr = np.sqrt(xx**2 + yy**2)

        # =================================================
        # 随机不规则边界（3组正弦 — 原版）
        # =================================================

        theta = np.arctan2(yy, xx)

        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        phase3 = np.random.uniform(0, 2*np.pi)

        freq1 = np.random.randint(2, 6)
        freq2 = np.random.randint(3, 9)
        freq3 = np.random.randint(5, 12)

        amp1 = np.random.uniform(0.3, 0.7)
        amp2 = np.random.uniform(0.2, 0.5)
        amp3 = np.random.uniform(0.1, 0.4)

        boundary_noise = (

            amp1 * np.sin(freq1 * theta + phase1)

            + amp2 * np.cos(freq2 * theta + phase2)

            + amp3 * np.sin(freq3 * theta + phase3)
        )

        boundary = (

            self.radius

            * (

                1

                + self.irregularity

                * boundary_noise
            )
        )

        inside_mask = rr <= boundary

        # =================================================
        # 自然地形（Perlin Noise — 纯 numpy 实现）
        # =================================================

        seed = np.random.randint(0, 256)
        pn = _PerlinNoise(base=seed)

        raw = pn.fbm2(
            xx / self.scale,
            yy / self.scale,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity
        )

        # ridged noise
        terrain = 1.0 - np.abs(raw)

        # 归一化
        terrain = terrain - terrain.min()
        terrain = terrain / terrain.max()
        terrain_height = terrain * self.max_height

        # =================================================
        # 创建Voxel — 实心填充
        # =================================================

        solid = np.zeros(
            (
                self.resolution_xy,
                self.resolution_xy,
                self.resolution_z
            ),
            dtype=np.uint8
        )

        for i in range(self.resolution_xy):

            for j in range(self.resolution_xy):

                if not inside_mask[i, j]:
                    continue

                top = terrain_height[i, j]

                for k in range(self.resolution_z):

                    current_z = z[k]

                    if current_z <= top:

                        solid[i, j, k] = 1

        return solid, x, y, z


# =========================================================
# Main Window
# =========================================================

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Groundwater Terrain Generator"
        )

        self.resize(1600, 900)

        self.current_mesh = None

        # 追踪所有添加到 plotter 的 actor，用于安全清理
        self._tracked_actors = []

        self.setup_ui()

    # =====================================================
    # 按钮状态管理
    # =====================================================

    def set_button_busy(self):
        """生成中/清理中：红色，禁用"""
        self.generate_button.setEnabled(False)
        self.generate_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #e74c3c;"
            "  color: white;"
            "  font-weight: bold;"
            "  border: none;"
            "  padding: 8px;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #c0392b;"
            "  color: #cccccc;"
            "}"
        )
        self.status_label.setText(" 正在生成 / 清理中... ")
        self.status_label.setStyleSheet(
            "QLabel {"
            "  background-color: #fdf2f2;"
            "  color: #c0392b;"
            "  font-weight: bold;"
            "  padding: 6px;"
            "  border-radius: 4px;"
            "}"
        )
        QApplication.processEvents()

    def set_button_ready(self):
        """就绪：绿色，可用"""
        self.generate_button.setEnabled(True)
        self.generate_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #27ae60;"
            "  color: white;"
            "  font-weight: bold;"
            "  border: none;"
            "  padding: 8px;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #2ecc71;"
            "}"
        )
        self.status_label.setText(" 就绪，可生成地形 ")
        self.status_label.setStyleSheet(
            "QLabel {"
            "  background-color: #eafaf1;"
            "  color: #27ae60;"
            "  font-weight: bold;"
            "  padding: 6px;"
            "  border-radius: 4px;"
            "}"
        )
        QApplication.processEvents()

    # =====================================================

    def _safe_remove_all_actors(self):
        """
        安全移除所有 actor，不复建 Plotter。
        """
        if not hasattr(self, 'plotter') or self.plotter is None:
            return

        # 1. 禁用并释放 orientation marker widget
        if hasattr(self.plotter, 'axes_widget') and self.plotter.axes_widget is not None:
            try:
                self.plotter.axes_widget.SetEnabled(0)
                self.plotter.axes_widget.SetInteractor(None)
            except Exception:
                pass
            self.plotter.axes_widget = None

        # 2. 释放 cube_axes_actor 引用
        if hasattr(self.plotter, 'cube_axes_actor'):
            self.plotter.cube_axes_actor = None

        # 3. 最底层清空：直接移除 renderer 中所有 VTK props
        try:
            self.plotter.renderer.RemoveAllViewProps()
        except Exception:
            pass

        # 4. 清空 PyVista 内部的 actor 追踪字典
        try:
            self.plotter.renderer._actors = {}
        except Exception:
            pass

        self._tracked_actors = []

    # =====================================================

    def setup_ui(self):

        self.central = QWidget()

        self.setCentralWidget(self.central)

        self.layout = QHBoxLayout()

        self.central.setLayout(self.layout)

        # =================================================
        # 左侧控制栏
        # =================================================

        self.control_panel = QWidget()

        self.control_layout = QVBoxLayout()

        self.control_panel.setLayout(
            self.control_layout
        )

        self.control_panel.setFixedWidth(320)

        self.layout.addWidget(
            self.control_panel
        )

        # =================================================
        # 参数
        # =================================================

        self.radius_spin = self.create_spinbox(
            "区域尺度（平面半径，决定地块大小）",
            10,
            1000,
            100
        )

        self.irregularity_spin = self.create_double_spinbox(
            "边界不规则程度（0=圆形，1=极不规则）",
            0.0,
            1.0,
            0.3,
            0.05
        )

        self.depth_spin = self.create_spinbox(
            "深度（地下土层厚度）",
            1,
            500,
            50
        )

        self.height_spin = self.create_spinbox(
            "最大地表高度（地面最高点海拔）",
            1,
            500,
            15
        )

        self.grid_spin = self.create_spinbox(
            "XY分辨率（网格细度，越大越精细但越慢）",
            20,
            300,
            100
        )

        # =================================================
        # Perlin参数
        # =================================================

        self.scale_spin = self.create_spinbox(
            "地形起伏尺度（越大起伏越平缓，越小越陡峭）",
            5,
            300,
            60
        )

        self.octave_spin = self.create_spinbox(
            "噪声叠加层数（层数越多地形细节越丰富）",
            1,
            10,
            5
        )

        self.persistence_spin = self.create_double_spinbox(
            "Persistence—持续度（每层振幅衰减比例，越低细节越弱）",
            0.1,
            1.0,
            0.5,
            0.05
        )

        self.lacunarity_spin = self.create_double_spinbox(
            "Lacunarity—间隙度（每层频率增长比例，越高纹理越密）",
            1.0,
            5.0,
            2.0,
            0.1
        )

        # =================================================
        # 状态标签
        # =================================================

        self.status_label = QLabel(" 就绪，可生成地形 ")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(self.status_label)

        # =================================================
        # 按钮
        # =================================================

        self.generate_button = QPushButton(
            "生成3D结构"
        )

        self.generate_button.clicked.connect(
            self.generate_model
        )

        self.control_layout.addWidget(
            self.generate_button
        )

        self.export_button = QPushButton(
            "导出 STL"
        )

        self.export_button.clicked.connect(
            self.export_mesh
        )

        self.control_layout.addWidget(
            self.export_button
        )

        self.control_layout.addStretch()

        # 初始化为绿色就绪状态
        self.set_button_ready()

        # =================================================
        # 右侧3D窗口
        # =================================================

        self.viewer_widget = QWidget()

        self.viewer_layout = QVBoxLayout()

        self.viewer_widget.setLayout(
            self.viewer_layout
        )

        self.layout.addWidget(
            self.viewer_widget
        )

        # 只创建一次 Plotter，终生不复建
        self.plotter = QtInteractor(
            self.viewer_widget
        )

        self.viewer_layout.addWidget(
            self.plotter.interactor
        )

    # =====================================================

    def create_spinbox(
        self,
        label,
        min_v,
        max_v,
        default
    ):

        container = QWidget()

        layout = QVBoxLayout()

        container.setLayout(layout)

        lbl = QLabel(label)

        spin = QSpinBox()

        spin.setMinimum(min_v)

        spin.setMaximum(max_v)

        spin.setValue(default)

        spin.setKeyboardTracking(False)

        layout.addWidget(lbl)

        layout.addWidget(spin)

        self.control_layout.addWidget(
            container
        )

        return spin

    # =====================================================

    def create_double_spinbox(
        self,
        label,
        min_v,
        max_v,
        default,
        step
    ):

        container = QWidget()

        layout = QVBoxLayout()

        container.setLayout(layout)

        lbl = QLabel(label)

        spin = QDoubleSpinBox()

        spin.setMinimum(min_v)

        spin.setMaximum(max_v)

        spin.setSingleStep(step)

        spin.setValue(default)

        spin.setKeyboardTracking(False)

        layout.addWidget(lbl)

        layout.addWidget(spin)

        self.control_layout.addWidget(
            container
        )

        return spin

    # =====================================================

    def generate_model(self):

        try:

            # ==========================================
            # 置为红色忙碌状态，禁止再次点击
            # ==========================================

            self.set_button_busy()

            # ==========================================
            # 输入合法性检查
            # ==========================================

            if self.depth_spin.value() <= 0:

                QMessageBox.warning(
                    self,
                    "输入错误",
                    "深度必须大于0"
                )

                self.set_button_ready()

                return

            # ==========================================
            # 安全清理旧内容（不复建 Plotter）
            # ==========================================

            self._safe_remove_all_actors()

            # ==========================================
            # 读取参数
            # ==========================================

            radius = self.radius_spin.value()

            irregularity = self.irregularity_spin.value()

            depth = self.depth_spin.value()

            max_height = self.height_spin.value()

            resolution_xy = self.grid_spin.value()

            resolution_z = max(
                30,
                int(depth)
            )

            # ==========================================
            # 创建生成器
            # ==========================================

            generator = TerrainGenerator(

                radius=radius,

                irregularity=irregularity,

                depth=depth,

                max_height=max_height,

                resolution_xy=resolution_xy,

                resolution_z=resolution_z,

                scale=self.scale_spin.value(),

                octaves=self.octave_spin.value(),

                persistence=self.persistence_spin.value(),

                lacunarity=self.lacunarity_spin.value()
            )

            # ==========================================
            # 生成 voxel
            # ==========================================

            solid, x, y, z = generator.generate_voxel()

            # ==========================================
            # 创建 ImageData
            # ==========================================

            grid = pv.ImageData()

            grid.dimensions = np.array(
                solid.shape
            ) + 1

            grid.origin = (
                float(x.min()),
                float(y.min()),
                float(z.min())
            )

            grid.spacing = (
                float(x[1] - x[0]),
                float(y[1] - y[0]),
                float(z[1] - z[0])
            )

            grid.cell_data["solid"] = solid.flatten(order="F")

            # ==========================================
            # 提取体素
            # ==========================================

            threshold = grid.threshold(
                0.5,
                scalars="solid"
            )

            # 安全检查：如果 threshold 为空，避免 VTK 崩溃
            if threshold.n_cells == 0:

                QMessageBox.warning(
                    self,
                    "生成失败",
                    "未能提取到有效体素，请调整参数后重试。"
                )

                self.set_button_ready()

                return

            # ==========================================
            # 只提取表面
            # ==========================================

            surface = threshold.extract_surface(
                algorithm='dataset_surface'
            )

            self.current_mesh = surface

            # ==========================================
            # 高度颜色映射
            # ==========================================

            z_values = surface.points[:, 2]

            # ==========================================
            # 添加地形
            # ==========================================

            terrain_actor = self.plotter.add_mesh(

                surface,

                scalars=z_values,

                cmap="terrain",

                smooth_shading=False,

                show_edges=False,

                lighting=True
            )

            self._tracked_actors.append(terrain_actor)

            # ==========================================
            # 坐标轴
            # ==========================================

            axes_actor = self.plotter.add_axes()

            self._tracked_actors.append(axes_actor)

            # ==========================================
            # 计算对齐到固定刻度的坐标轴范围
            # X,Y 每 20 一个刻度 | Z 每 10 一个刻度
            # 都从 0 开始向两边延伸
            # ==========================================

            # X, Y 对齐到 20 的倍数
            xy_step = 20
            xy_ext = ((radius // xy_step) + 1) * xy_step
            x_min, x_max = -xy_ext, xy_ext
            y_min, y_max = -xy_ext, xy_ext

            # Z 对齐到 10 的倍数
            z_step = 10
            z_min_ext = ((depth // z_step) + 1) * z_step
            z_max_ext = ((max_height // z_step) + 1) * z_step
            z_min, z_max = -z_min_ext, z_max_ext

            # 刻度数量 = 区间长度 / 步长 + 1
            n_x = int((x_max - x_min) / xy_step) + 1
            n_y = int((y_max - y_min) / xy_step) + 1
            n_z = int((z_max - z_min) / z_step) + 1

            bounds = [float(x_min), float(x_max),
                      float(y_min), float(y_max),
                      float(z_min), float(z_max)]

            # ==========================================
            # 合并绘制参考网格线
            # XY底面步长 20 | Z方向步长 10
            # 使用对齐后的范围画线
            # ==========================================

            # --- XY底面网格 ---
            xy_points = []
            xy_conn = []
            pt_idx = 0

            for gx in range(x_min, x_max + 1, xy_step):
                xy_points.extend([
                    [gx, y_min, z_min],
                    [gx, y_max, z_min]
                ])
                xy_conn.extend([2, pt_idx, pt_idx + 1])
                pt_idx += 2

            for gy in range(y_min, y_max + 1, xy_step):
                xy_points.extend([
                    [x_min, gy, z_min],
                    [x_max, gy, z_min]
                ])
                xy_conn.extend([2, pt_idx, pt_idx + 1])
                pt_idx += 2

            if xy_points:
                poly_xy = pv.PolyData(
                    np.array(xy_points, dtype=np.float32),
                    np.array(xy_conn, dtype=np.int64)
                )
                actor_xy = self.plotter.add_mesh(
                    poly_xy, color='gray', line_width=1
                )
                self._tracked_actors.append(actor_xy)

            # --- Z方向线 ---
            z_points = []
            z_conn = []
            pt_idx = 0

            for gz in range(z_min, z_max + 1, z_step):
                z_points.extend([
                    [x_min, y_min, gz],
                    [x_min, y_max, gz]
                ])
                z_conn.extend([2, pt_idx, pt_idx + 1])
                pt_idx += 2

            if z_points:
                poly_z = pv.PolyData(
                    np.array(z_points, dtype=np.float32),
                    np.array(z_conn, dtype=np.int64)
                )
                actor_z = self.plotter.add_mesh(
                    poly_z, color='darkgray', line_width=1
                )
                self._tracked_actors.append(actor_z)

            # ==========================================
            # 中文坐标显示（固定刻度）
            # ==========================================

            bounds_actor = self.plotter.show_bounds(

                bounds=bounds,

                grid='front',

                location='outer',

                ticks='outside',

                all_edges=True,

                xtitle='X方向',

                ytitle='Y方向',

                ztitle='高程',

                fmt='%.0f',

                n_xlabels=n_x,

                n_ylabels=n_y,

                n_zlabels=n_z,
            )

            # 强制 VTK 底层对齐刻度范围
            bounds_actor.SetXAxisRange(x_min, x_max)
            bounds_actor.SetYAxisRange(y_min, y_max)
            bounds_actor.SetZAxisRange(z_min, z_max)

            self._tracked_actors.append(bounds_actor)

            # ==========================================
            # 相机
            # ==========================================

            self.plotter.camera_position = 'iso'

            self.plotter.reset_camera()

            self.plotter.render()

            # ==========================================
            # 渲染完成，恢复绿色就绪状态
            # ==========================================

            self.set_button_ready()

        except Exception as e:

            QMessageBox.critical(

                self,

                "程序错误",

                str(e)
            )

            self.set_button_ready()

    # =====================================================

    def export_mesh(self):

        if self.current_mesh is None:

            QMessageBox.warning(
                self,
                "Warning",
                "请先生成模型"
            )

            return

        filename, _ = QFileDialog.getSaveFileName(

            self,

            "保存 STL",

            "terrain.stl",

            "STL Files (*.stl)"
        )

        if filename:

            self.current_mesh.save(filename)

            QMessageBox.information(
                self,
                "Success",
                "导出成功"
            )


# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec_())