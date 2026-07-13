import os
import gc
import sys
import traceback
import numpy as np
import numbers
from dataclasses import dataclass
from typing import Optional
from PyQt6.QtGui import QPixmap, QImage, QGuiApplication, QShortcut, QKeySequence, QIcon
from PyQt6.QtCore import Qt, QUrl, QRect, QPoint, QSize, QThread, pyqtSignal, QTimer, QTime
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QRubberBand, QDoubleSpinBox, QRadioButton
from PyQt6.QtWidgets import QWidget, QLabel, QCheckBox, QSlider, QPushButton, QComboBox, QSpinBox, QFrame, QLineEdit
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QTabWidget, QScrollArea, QSizePolicy, QDialog
from PyQt6.QtWidgets import QProgressDialog
from matplotlib import pyplot as plt
import hyperlyse as hyper
from hyperlyse import cube_analyzer

SELECTION_COLORS = [
    [228,  26,  28],  # red
    [ 55, 126, 184],  # blue
    [ 77, 175,  74],  # green
    [152,  78, 163],  # purple
    [255, 127,   0],  # orange
    [255, 255,  51],  # yellow
    [166,  86,  40],  # brown
    [247, 129, 191],  # pink
    [153, 153, 153],  # grey
    [  0, 210, 213],  # cyan
]

HOVER_TOLERANCE = 4  # pixels in data coordinates for hit-testing


@dataclass
class Selection:
    sel_type: str                   # 'point' or 'rect'
    point: Optional[QPoint]         # set when sel_type == 'point'
    rect: Optional[QRect]           # set when sel_type == 'rect'
    spectrum_y: np.ndarray          # 1D (nbands,)
    color_rgb: list                 # [R,G,B] 0-255, for image overlay
    color_mpl: tuple                # (r,g,b) 0.0-1.0, for matplotlib
    label: str                      # e.g. "P1 (120,45)" or "R2 (10,20,30x40)"
    index: int                      # monotonic ID

hyper_quotes = ['"Hyper, hyper. We need the bass drum." - H.P. Baxxter',
                '"Travelling through hyper space ain\'t like dusting crops, boy!" - Han Solo']


class CubeAnalysisWorker(QThread):
    """Background worker for analyzing cubes."""
    discovered = pyqtSignal(int)  # total cube count (after discovery)
    progress = pyqtSignal(int, int, str, float, bool)  # current, total, name, avg_time, skipped
    finished = pyqtSignal(int, int)  # analyzed_count, skipped_count
    error = pyqtSignal(str)

    def __init__(self, cube_folder, sample_rate, include_subfolders):
        super().__init__()
        self.cube_folder = cube_folder
        self.sample_rate = sample_rate
        self.include_subfolders = include_subfolders
        self._analyzed = 0
        self._skipped = 0

    def run(self):
        try:
            def on_progress(i, total, name, avg_time, skipped=False):
                if skipped:
                    self._skipped += 1
                else:
                    self._analyzed += 1
                self.progress.emit(i + 1, total, name, avg_time, skipped)

            # Analyze sequentially (max_workers=1). analyze_cubes supports a
            # thread pool, but several cubes analyzed at once on background
            # threads contend for the GIL with the Qt GUI thread, freezing the
            # progress dialog (and multiply peak RAM). One cube at a time keeps
            # the UI responsive; per-cube analysis is still faster than before
            # thanks to the subsampled PCA fit.
            cube_analyzer.analyze_cubes(
                self.cube_folder,
                sample_rate=self.sample_rate,
                include_subfolders=self.include_subfolders,
                progress_callback=on_progress,
                discovered_callback=self.discovered.emit,
                max_workers=1)
            self.finished.emit(self._analyzed, self._skipped)
        except Exception as e:
            self.error.emit(traceback.format_exc())


class CubeSearchWorker(QThread):
    """Background worker for cross-cube spectrum search."""
    progress = pyqtSignal(int, int, str, float)  # current, total, cube_name, avg_time
    finished = pyqtSignal(list)  # results
    error = pyqtSignal(str)

    def __init__(self, cube_folder, x_query, y_query, sample_rate,
                 include_subfolders, custom_range, use_gradient,
                 squared_errs, num_hits, use_pca=False, exclude_cube_file=None,
                 include_cube_files=None):
        super().__init__()
        self.cube_folder = cube_folder
        self.x_query = x_query
        self.y_query = y_query
        self.sample_rate = sample_rate
        self.include_subfolders = include_subfolders
        self.custom_range = custom_range
        self.use_gradient = use_gradient
        self.squared_errs = squared_errs
        self.num_hits = num_hits
        self.use_pca = use_pca
        self.exclude_cube_file = exclude_cube_file
        self.include_cube_files = include_cube_files

    def run(self):
        try:
            def on_progress(current, total, cube_name, avg_time):
                self.progress.emit(current, total, cube_name, avg_time)

            results = cube_analyzer.search_in_cached_cubes(
                self.cube_folder,
                self.x_query,
                self.y_query,
                sample_rate=self.sample_rate,
                include_subfolders=self.include_subfolders,
                custom_range=self.custom_range,
                use_gradient=self.use_gradient,
                squared_errs=self.squared_errs,
                num_hits=self.num_hits,
                use_pca=self.use_pca,
                exclude_cube_file=self.exclude_cube_file,
                include_cube_files=self.include_cube_files,
                progress_callback=on_progress)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(traceback.format_exc())


class CubeLoadingWorker(QThread):
    """Background worker for loading cube data."""
    progress = pyqtSignal(str)  # status message
    finished = pyqtSignal(object)  # loaded cube object
    error = pyqtSignal(str)  # error message

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._cancelled = False

    def run(self):
        try:
            self.progress.emit("Loading cube metadata...")
            cube = hyper.Cube(self.filename)

            if self._cancelled:
                self.error.emit("Loading cancelled by user")
                return

            self.progress.emit("Generating RGB preview...")
            rgb = cube.to_rgb()

            if self._cancelled:
                self.error.emit("Loading cancelled by user")
                return

            # Emit cube object with the RGB already computed
            self.finished.emit(cube)
        except Exception as e:
            self.error.emit(traceback.format_exc())

    def cancel(self):
        """Cancel the loading operation."""
        self._cancelled = True


class CubeLoadingDialog(QDialog):
    """Modal dialog showing loading progress with cancel button."""
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading Cube Data")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMaximumHeight(180)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout()

        self.status_label = QLabel("Loading cube metadata...")
        layout.addWidget(self.status_label)

        # Elapsed time label
        self.elapsed_label = QLabel("Elapsed: 0s")
        layout.addWidget(self.elapsed_label)

        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Setup timer for elapsed time display
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_elapsed_time)
        self.start_time = None

    def showEvent(self, event):
        """Start the timer when dialog is shown."""
        super().showEvent(event)
        self.start_time = QTime.currentTime()
        self.timer.start(100)  # Update every 100ms

    def closeEvent(self, event):
        """Stop the timer when dialog is closed."""
        self.timer.stop()
        super().closeEvent(event)

    def _update_elapsed_time(self):
        """Update the elapsed time display."""
        if self.start_time is not None:
            elapsed_ms = self.start_time.msecsTo(QTime.currentTime())
            elapsed_s = elapsed_ms / 1000.0
            self.elapsed_label.setText(f"Elapsed: {elapsed_s:.1f}s")

    def on_cancel_clicked(self):
        """Emit cancel signal when cancel button is clicked."""
        self.cancel_requested.emit()
        self.reject()

    def update_status(self, message):
        """Update the status message."""
        self.status_label.setText(message)

    def set_error(self, message):
        """Show error message and change button to Close."""
        self.status_label.setText(f"Error while loading cube:\n{message}")
        self.cancel_button.setText("Close")
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.reject)



class MainWindow(QMainWindow):
    def __init__(self, config, rawfile=None):
        super(MainWindow, self).__init__(None)

        self.config=config

        # data members
        self.cube = None
        self.rgb = None
        self.pca = None
        self.error_map = None
        self.error_map_recompute_flag = True    # do we have to recompute the error map?
        self.pca_recompute_flag = True          # same for pca
        self.point_selection = None
        self.rect_selection = None
        self.spectrum_y = None
        self.selections = []              # list[Selection]
        self.selection_counter = 0        # monotonic, wraps color cycle
        self.hovered_selection_idx = None  # index into selections, or None
        self.rawfile = rawfile

        self.zoom = 1.0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_start_hs_v = 0
        self.drag_start_vs_v = 0
        self.rotation_quadrants = 0

        # Spotlight ("hold F") mode: dim+desaturate the image and emphasize
        # every marker/find with a pulsing ring. Momentary while F is held.
        self.spotlight_active = False
        self.spotlight_phase = 0.0
        self.spotlight_timer = QTimer(self)
        self.spotlight_timer.setInterval(40)  # ~25 fps pulse
        self.spotlight_timer.timeout.connect(self._on_spotlight_tick)

        self.db = hyper.Database(config.default_db_path)

        # Phase 3: cube search results and hit color cycle
        # Colors as (R,G,B) 0-255 for markers AND as hex strings for matplotlib
        self.hit_colors_rgb = [
            [230, 25, 75],    # red
            [60, 180, 75],    # green
            [0, 130, 200],    # blue
            [245, 130, 48],   # orange
            [145, 30, 180],   # purple
            [240, 50, 230],   # magenta
            [70, 240, 240],   # cyan
            [210, 245, 60],   # lime
            [250, 190, 212],  # pink
            [128, 0, 0],      # maroon
        ]
        self.hit_colors_hex = ['#%02x%02x%02x' % tuple(c) for c in self.hit_colors_rgb]
        self.cube_search_results = []  # list of hit dicts from cube_analyzer.search_in_cached_cubes
        self._search_worker = None  # CubeSearchWorker instance when search is running
        self._cube_tab_groups = []  # list of (cube_file, hits) ordered by best error
        self._cube_current_index = 0  # 0 = Source, 1..n = cube hit

        # Per-hit-cube render caches (keyed by cache_dir) so the mode views
        # (layers/similarity/PCA) work on matched cubes just like the source.
        # spectra survive setting changes; error-map/pca are invalidated by the
        # same recompute flags that invalidate the source cube's versions.
        self._hit_spectra_cache = {}
        self._hit_error_map_cache = {}
        self._hit_pca_cache = {}

        self.last_source_name = ''
        self.last_export_dir = ''

        # (x, y) pixel selection to seed once a matched cube finishes loading as
        # the new source; consumed in _on_cube_loaded. None when not pending.
        self._pending_source_selection = None

        ########################
        ########################
        ####### setup ui #######
        ########################
        ########################
        self.setWindowTitle('Hyperlyse v%s' % self.config.version)

        ### central widget and outer layout
        cw = QWidget(self)
        self.setCentralWidget(cw)
        layout_outer = QHBoxLayout(cw)

        #####################
        ### image display ###
        #####################

        layout_img = QVBoxLayout()
        layout_outer.addLayout(layout_img)

        layout_img_rotate = QHBoxLayout()
        layout_img.addLayout(layout_img_rotate)
        layout_img_rotate.addStretch()

        # Ctrl+Z shortcut to undo last selection
        self.shortcut_undo = QShortcut(QKeySequence('Ctrl+Z'), self)
        self.shortcut_undo.activated.connect(self.handle_undo_selection)

        # Cube nav bar: scrollable horizontal button row above the image
        self._cube_nav_scroll = QScrollArea(cw)
        self._cube_nav_scroll.setWidgetResizable(True)
        self._cube_nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._cube_nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cube_nav_scroll.setMaximumHeight(48)
        self._cube_nav_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._cube_nav_scroll.hide()

        self._cube_nav_inner = QWidget()
        self._cube_nav_layout = QHBoxLayout(self._cube_nav_inner)
        self._cube_nav_layout.setContentsMargins(2, 2, 2, 2)
        self._cube_nav_layout.setSpacing(3)
        self._cube_nav_layout.addStretch()
        self._cube_nav_scroll.setWidget(self._cube_nav_inner)

        layout_img.addWidget(self._cube_nav_scroll)

        self.lbl_img = QLabel(cw)
        self.lbl_img.setMouseTracking(True)
        self.lbl_img.mousePressEvent = self.handle_click_on_image
        self.lbl_img.mouseMoveEvent = self.handle_move_on_image
        self.lbl_img.mouseReleaseEvent = self.handle_release_on_image
        self.lbl_img.setAcceptDrops(True)
        self.lbl_img.dragEnterEvent = self.handle_drag_enter
        self.lbl_img.dropEvent = self.handle_drop
        self.rubberband_selector = QRubberBand(QRubberBand.Shape.Rectangle, self.lbl_img)
        self.rubberband_origin = QPoint(0, 0)
        self.rubberband_selector.setGeometry(QRect(0, 0, 0, 0))
        self.scroll_img = QScrollArea(cw)
        self.scroll_img.setWidget(self.lbl_img)
        self.scroll_img.setWidgetResizable(True)
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_img.mousePressEvent = self.handle_click_on_image_scroll
        self.scroll_img.mouseMoveEvent = self.handle_move_on_image_scroll
        self.scroll_img.wheelEvent = self.handle_wheel_on_image_scroll
        layout_img.addWidget(self.scroll_img)

        # viewing controls
        layout_img_ctrl = QGridLayout()
        layout_img_ctrl.setSpacing(12)
        layout_img_ctrl.setContentsMargins(0, 2, 0, 2)
        layout_img.addLayout(layout_img_ctrl)
        lbl_zoom_static = QLabel('Zoom')
        layout_img_ctrl.addWidget(lbl_zoom_static, 0, 0)
        layout_img_ctrl.setAlignment(lbl_zoom_static, Qt.AlignmentFlag.AlignVCenter)
        self.sl_zoom = QSlider(cw)
        self.sl_zoom.setOrientation(Qt.Orientation.Horizontal)
        self.sl_zoom.setMinimum(25)
        self.sl_zoom.setMaximum(800)
        self.sl_zoom.setValue(100)
        self.sl_zoom.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.sl_zoom.valueChanged.connect(self.update_image_label)
        layout_img_ctrl.addWidget(self.sl_zoom, 0, 1)
        layout_img_ctrl.setAlignment(self.sl_zoom, Qt.AlignmentFlag.AlignVCenter)
        self.lbl_zoom = QLabel('100%')
        layout_img_ctrl.addWidget(self.lbl_zoom, 0, 2)
        layout_img_ctrl.setAlignment(self.lbl_zoom, Qt.AlignmentFlag.AlignVCenter)

        lbl_brightness_static = QLabel('Brightness')
        layout_img_ctrl.addWidget(lbl_brightness_static, 1, 0)
        layout_img_ctrl.setAlignment(lbl_brightness_static, Qt.AlignmentFlag.AlignVCenter)
        self.sl_brightness = QSlider(cw)
        self.sl_brightness.setOrientation(Qt.Orientation.Horizontal)
        self.sl_brightness.setMinimum(0)
        self.sl_brightness.setMaximum(300)
        self.sl_brightness.setValue(100)
        self.sl_brightness.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.sl_brightness.valueChanged.connect(self.update_image_label)
        layout_img_ctrl.addWidget(self.sl_brightness, 1, 1)
        layout_img_ctrl.setAlignment(self.sl_brightness, Qt.AlignmentFlag.AlignVCenter)
        self.lbl_brightness = QLabel('100%')
        layout_img_ctrl.addWidget(self.lbl_brightness, 1, 2)
        layout_img_ctrl.setAlignment(self.lbl_brightness, Qt.AlignmentFlag.AlignVCenter)

        # Action buttons group (stacked)
        layout_buttons = QVBoxLayout()
        layout_buttons.setSpacing(4)
        layout_buttons.setContentsMargins(0, 0, 0, 0)

        # Rotation and clear buttons (use SVG icons)
        icons_dir = self.resource_path('icons')

        self.btn_rotate_left = QPushButton()
        self.btn_rotate_left.setMinimumWidth(110)
        self.btn_rotate_left.setMinimumHeight(30)
        icon_left = QIcon(os.path.join(icons_dir, 'rotate-ccw.svg'))
        self.btn_rotate_left.setIcon(icon_left)
        self.btn_rotate_left.setIconSize(QSize(14, 14))
        self.btn_rotate_left.setText('Rotate left')
        self.btn_rotate_left.setToolTip('Rotate Left 90\u00B0')
        self.btn_rotate_left.pressed.connect(self.handle_rotate_left_image)
        self.btn_rotate_left.setContentsMargins(0, 0, 0, 0)
        layout_buttons.addWidget(self.btn_rotate_left)
        layout_buttons.setAlignment(self.btn_rotate_left, Qt.AlignmentFlag.AlignHCenter)

        self.btn_rotate_right = QPushButton()
        self.btn_rotate_right.setMinimumWidth(110)
        self.btn_rotate_right.setMinimumHeight(30)
        icon_right = QIcon(os.path.join(icons_dir, 'rotate-cw.svg'))
        self.btn_rotate_right.setIcon(icon_right)
        self.btn_rotate_right.setIconSize(QSize(14, 14))
        self.btn_rotate_right.setText('Rotate right')
        self.btn_rotate_right.setToolTip('Rotate Right 90\u00B0')
        self.btn_rotate_right.pressed.connect(self.handle_rotate_right_image)
        self.btn_rotate_right.setContentsMargins(0, 0, 0, 0)
        layout_buttons.addWidget(self.btn_rotate_right)
        layout_buttons.setAlignment(self.btn_rotate_right, Qt.AlignmentFlag.AlignHCenter)

        self.btn_clear_selections = QPushButton('Clear selection')
        self.btn_clear_selections.setMinimumWidth(110)
        self.btn_clear_selections.setMinimumHeight(30)
        self.btn_clear_selections.setToolTip('Clear All Selections')
        self.btn_clear_selections.pressed.connect(self.handle_clear_selections)
        self.btn_clear_selections.setContentsMargins(0, 0, 0, 0)
        layout_buttons.addWidget(self.btn_clear_selections)
        layout_buttons.setAlignment(self.btn_clear_selections, Qt.AlignmentFlag.AlignHCenter)

        # push remaining space so save sits at the bottom edge
        

        # save image (aligned to bottom of the stack)
        self.btn_save_img = QPushButton('Save Image')
        self.btn_save_img.setMinimumWidth(110)
        self.btn_save_img.setMinimumHeight(30)
        self.btn_save_img.setContentsMargins(0, 0, 0, 0)
        self.btn_save_img.pressed.connect(self.handle_action_export_image)
        layout_buttons.addWidget(self.btn_save_img)

        layout_img_ctrl.addLayout(layout_buttons, 0, 3, 3, 2)

        # content controls
        lbl_img_display = QLabel(cw)
        lbl_img_display.setText('Mode')
        lbl_img_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_img_ctrl.addWidget(lbl_img_display, 2, 0)

        self.tabs_img_ctrl = QTabWidget(cw)
        self.tabs_img_ctrl.currentChanged.connect(self.update_image_label)
        self.tabs_img_ctrl.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        layout_img_ctrl.addWidget(self.tabs_img_ctrl, 2, 1, 1, 2)

        # RGB -> index 0
        tab_rgb = QWidget()
        tab_rgb.setLayout(QHBoxLayout())
        self.tabs_img_ctrl.addTab(tab_rgb, 'RGB')

        lbl_rgb = QLabel(tab_rgb)
        lbl_rgb.setText('Shows an RGB representation of the Hyperspectral cube.')
        tab_rgb.layout().addWidget(lbl_rgb)

        # layers -> index 1
        tab_layers = QWidget()
        tab_layers.setLayout(QHBoxLayout())
        self.tabs_img_ctrl.addTab(tab_layers, 'layers')

        lbl_layers = QLabel(tab_layers)
        lbl_layers.setText('Select layer:')
        tab_layers.layout().addWidget(lbl_layers)

        self.sl_lambda = QSlider(tab_layers)
        self.sl_lambda.setOrientation(Qt.Orientation.Horizontal)
        self.sl_lambda.valueChanged.connect(self.update_image_label)
        self.sl_lambda.setMinimumWidth(300)
        self.sl_lambda.setValue(0)
        tab_layers.layout().addWidget(self.sl_lambda)

        self.lbl_lambda = QLabel(tab_layers)
        self.lbl_lambda.setText("0")
        tab_layers.layout().addWidget(self.lbl_lambda)

        # similarity -> index 2
        tab_similarity = QWidget()
        tab_similarity.setLayout(QHBoxLayout())
        self.tabs_img_ctrl.addTab(tab_similarity, 'similarity')

        self.rb_sim_cube = QRadioButton('Current cube selection', tab_similarity)
        tab_similarity.layout().addWidget(self.rb_sim_cube)
        self.rb_sim_db = QRadioButton('Current database selection', tab_similarity)
        tab_similarity.layout().addWidget(self.rb_sim_db)
        self.rb_sim_cube.setChecked(True)
        self.rb_sim_cube.toggled.connect(self.set_recompute_errmap_flag)
        self.rb_sim_cube.toggled.connect(self.update_image_label)

        lbl_sim_t = QLabel(tab_similarity)
        lbl_sim_t.setText('t=')
        tab_similarity.layout().addWidget(lbl_sim_t)

        self.sl_sim_t = QSlider(tab_similarity)
        self.sl_sim_t.setOrientation(Qt.Orientation.Horizontal)
        self.sl_sim_t.setMinimum(0)
        self.sl_sim_t.setMaximum(99)
        self.sl_sim_t.setValue(0)
        self.sl_sim_t.valueChanged.connect(self.update_image_label)
        tab_similarity.layout().addWidget(self.sl_sim_t)

        # PCA -> index 3
        tab_pca = QWidget()
        tab_pca.setLayout(QHBoxLayout())
        self.tabs_img_ctrl.addTab(tab_pca, 'pca')

        lbl_components = QLabel(tab_pca)
        lbl_components.setText('Select component:')
        tab_pca.layout().addWidget(lbl_components)

        self.sl_component = QSlider(tab_pca)
        self.sl_component.setOrientation(Qt.Orientation.Horizontal)
        self.sl_component.valueChanged.connect(self.update_image_label)
        self.sl_component.setMinimumWidth(300)
        self.sl_component.setMinimum(0)
        self.sl_component.setMaximum(9)
        self.sl_component.setValue(0)
        tab_pca.layout().addWidget(self.sl_component)

        self.lbl_component = QLabel(tab_pca)
        self.lbl_component.setText("0")
        tab_pca.layout().addWidget(self.lbl_component)

        # add separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout_outer.addWidget(line)

        ########################
        ### specrtra display ###
        ########################
        layout_spectra = QVBoxLayout()
        layout_outer.addLayout(layout_spectra)

        #### plot canvas & range controls
        layout_plot = QHBoxLayout()
        layout_spectra.addLayout(layout_plot)

        # y-range control
        layout_yrangecontrol = QVBoxLayout()
        layout_plot.addLayout(layout_yrangecontrol)

        self.lbl_ymax = QLabel('ymax')
        layout_yrangecontrol.addWidget(self.lbl_ymax)
        self.sb_ymax = QDoubleSpinBox(cw)
        self.sb_ymax.setMinimum(0.5)
        self.sb_ymax.setMaximum(1.5)
        self.sb_ymax.setValue(1)
        self.sb_ymax.setSingleStep(0.1)
        self.sb_ymax.valueChanged.connect(self.update_spectrum_plot)
        layout_yrangecontrol.addWidget(self.sb_ymax)

        layout_yrangecontrol.addStretch()

        self.lbl_ymin = QLabel('ymin')
        layout_yrangecontrol.addWidget(self.lbl_ymin)
        self.sb_ymin = QDoubleSpinBox(cw)
        self.sb_ymin.setMinimum(0)
        self.sb_ymin.setMaximum(1)
        self.sb_ymin.setValue(0)
        self.sb_ymin.setSingleStep(0.1)
        self.sb_ymin.valueChanged.connect(self.update_spectrum_plot)
        layout_yrangecontrol.addWidget(self.sb_ymin)

        # plot
        self.plot = hyper.PlotCanvas(cw)
        layout_plot.addWidget(self.plot)


        ###### spectrum and database controls

        layout_spectra_ctrl = QGridLayout()
        layout_spectra.addLayout(layout_spectra_ctrl)

        # comparison / x-range control
        layout_spectra_ctrl.addWidget(QLabel('Comparison'), 0, 0)
        layout_compare_ctrl = QHBoxLayout()
        layout_spectra_ctrl.addLayout(layout_compare_ctrl, 0, 1)

        self.rs_xrange = hyper.QRangeSlider(cw)
        self.rs_xrange.setRange(0, 1000)
        self.rs_xrange.setMin(0)
        self.rs_xrange.setMax(1000)
        self.rs_xrange.startValueChanged.connect(self.update_spectrum_plot)
        self.rs_xrange.endValueChanged.connect(self.update_spectrum_plot)
        self.rs_xrange.startValueChanged.connect(self.set_recompute_errmap_flag)
        self.rs_xrange.endValueChanged.connect(self.set_recompute_errmap_flag)
        self.rs_xrange.startValueChanged.connect(self.set_recompute_pca_flag)
        self.rs_xrange.endValueChanged.connect(self.set_recompute_pca_flag)
        # Re-render the image AFTER the recompute flags are set, so the
        # similarity/PCA view refreshes immediately on a range change.
        self.rs_xrange.startValueChanged.connect(self.update_image_label)
        self.rs_xrange.endValueChanged.connect(self.update_image_label)

        layout_compare_ctrl.addWidget(self.rs_xrange)

        self.cb_squared = QCheckBox(self)
        self.cb_squared.setText('squared errors')
        self.cb_squared.setChecked(True)
        self.cb_squared.stateChanged.connect(self.update_spectrum_plot)
        self.cb_squared.stateChanged.connect(self.set_recompute_errmap_flag)
        self.cb_squared.stateChanged.connect(self.update_image_label)
        layout_compare_ctrl.addWidget(self.cb_squared)

        self.cb_gradient = QCheckBox(self)
        self.cb_gradient.setText('compare gradients')
        self.cb_gradient.setChecked(True)
        self.cb_gradient.stateChanged.connect(self.update_spectrum_plot)
        self.cb_gradient.stateChanged.connect(self.set_recompute_errmap_flag)
        self.cb_gradient.stateChanged.connect(self.update_image_label)
        layout_compare_ctrl.addWidget(self.cb_gradient)

        # comparison spectra source
        lbl_source = QLabel('Source')
        lbl_source.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_spectra_ctrl.addWidget(lbl_source, 1, 0)

        self.tabs_spectra_source = QTabWidget(cw)
        self.tabs_spectra_source.currentChanged.connect(self.update_spectrum_plot)
        self.tabs_spectra_source.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum))
        layout_spectra_ctrl.addWidget(self.tabs_spectra_source, 1, 1)

        # select spectrum -> index 0
        self.tab_select = QWidget()
        self.tab_select.setLayout(QHBoxLayout())
        self.tabs_spectra_source.addTab(self.tab_select, 'Select')
        self.tab_select.layout().addWidget(QLabel('Compare with spectrum'))
        self.cmb_comparison_ref = QComboBox(self.tab_select)
        self.cmb_comparison_ref.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.fill_db_spectra_combobox()
        self.cmb_comparison_ref.currentIndexChanged.connect(self.update_spectrum_plot)
        self.cmb_comparison_ref.currentIndexChanged.connect(self.set_recompute_errmap_flag)
        self.cmb_comparison_ref.currentIndexChanged.connect(self.update_image_label)
        self.tab_select.layout().addWidget(self.cmb_comparison_ref)
        self.tab_select.layout().addStretch()

        # db search -> index 1
        tab_search = QWidget()
        tab_search.setLayout(QHBoxLayout())
        self.tabs_spectra_source.addTab(tab_search, 'Search')
        tab_search.layout().addWidget(QLabel('show'))
        self.sb_nspectra = QSpinBox(self)
        self.sb_nspectra.setMinimum(0)
        self.sb_nspectra.setMaximum(10)
        self.sb_nspectra.setValue(3)
        self.sb_nspectra.valueChanged.connect(self.update_spectrum_plot)
        tab_search.layout().addWidget(self.sb_nspectra)
        tab_search.layout().addWidget(QLabel('most similar spectra in database'))
        tab_search.layout().addStretch()



        ### export controls
        # layout_export_ctrl = QHBoxLayout(cw)
        # layout_spectra.addLayout(layout_export_ctrl)
        #
        btn_export = QPushButton(cw)
        btn_export.setText('Save\nSpectra')
        btn_export.clicked.connect(self.handle_action_export_spectrum)
        btn_export.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum))
        layout_spectra_ctrl.addWidget(btn_export, 1, 2, 2, 1)

        ###################
        ### create menu ###
        ###################
        menubar = self.menuBar()

        # file menu
        menu_file = menubar.addMenu('&File')

        action_load_data = menu_file.addAction('&Load Hyperspectral Image...')
        action_load_data.triggered.connect(self.handle_action_load_data)

        action_save_img = menu_file.addAction('&Save current image...')
        action_save_img.triggered.connect(self.handle_action_export_image)

        action_export_spectrum = menu_file.addAction('&Save selected spectra...')
        action_export_spectrum.triggered.connect(self.handle_action_export_spectrum)

        # settings menu
        menu_settings = menubar.addMenu('S&ettings')

        action_preferences = menu_settings.addAction('&Preferences...')
        action_preferences.triggered.connect(self.handle_action_open_preferences)

        menu_settings.addSeparator()

        # number of hits submenu (duplicated in Preferences dialog)
        menu_num_hits = menu_settings.addMenu('Number of hits')
        self.num_hits_actions = []
        for n in [1, 2, 3, 5, 10]:
            action = menu_num_hits.addAction(str(n))
            action.setCheckable(True)
            action.setChecked(n == self.config.num_hits)
            action.triggered.connect(lambda checked, num=n: self.handle_set_num_hits(num))
            self.num_hits_actions.append(action)

        # search toggles (duplicated in Preferences dialog)
        self.action_search_in_db = menu_settings.addAction('Search in reference database')
        self.action_search_in_db.setCheckable(True)
        self.action_search_in_db.setChecked(self.config.search_in_db)
        self.action_search_in_db.toggled.connect(self.handle_toggle_search_in_db)

        self.action_search_in_cubes = menu_settings.addAction('Search in analyzed cubes')
        self.action_search_in_cubes.setCheckable(True)
        self.action_search_in_cubes.setChecked(self.config.search_in_cubes)
        self.action_search_in_cubes.toggled.connect(self.handle_toggle_search_in_cubes)

        menu_settings.addSeparator()

        self.action_analyze_cubes = menu_settings.addAction('&Analyze cubes now')
        self.action_analyze_cubes.triggered.connect(self.handle_action_analyze_cubes)
        self.action_analyze_cubes.setEnabled(bool(self.config.cube_folder_path))

        action_reset_cache = menu_settings.addAction('Reset cube cache...')
        action_reset_cache.triggered.connect(self.handle_action_reset_cube_cache)

        # info menu
        menu_info = menubar.addMenu('&?')
        action_info = menu_info.addAction('&Show info')
        action_info.triggered.connect(self.show_info)

        # very important
        quote = np.random.randint(0,len(hyper_quotes))
        self.statusBar().showMessage(hyper_quotes[quote])

        # additional windows (define here for better readability only)
        self.match_point_win = None

        # finito!
        ss = QGuiApplication.screens()[0].availableSize()
        self.setGeometry(int(ss.width() * 0.1), int(ss.height() * 0.1),
                         int(ss.width() * 0.8), int(ss.height() * 0.8))

        if self.rawfile is None:
            img_startup = QPixmap()
            img_startup.load(self.resource_path('startup.png'))
            wi = int(self.width() * self.config.initial_image_width_ratio)
            img_startup = img_startup.scaled(wi, wi, transformMode=Qt.TransformationMode.SmoothTransformation)
            self.lbl_img.setPixmap(img_startup)
            self.lbl_img.resize(wi, wi)
        else:
            self.load_data(self.rawfile)
        self.show()

    def show_info(self):
        mb = QMessageBox(QMessageBox.Icon.Information,
                         "About this software",
                         self.config.infotext(),
                         QMessageBox.StandardButton.Close)
        mb.exec()

    def resource_path(self, relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', relative_path))

    ##############
    # UI updates
    ##############
    def reset_ui(self):
        self.selections = []
        self.selection_counter = 0
        self.hovered_selection_idx = None
        self.point_selection = None
        self.rect_selection = None
        self.rotation_quadrants = 0
        self.spectrum_y = None
        self.error_map = None
        self.error_map_recompute_flag = True
        self.pca_recompute_flag = True
        self.cube_search_results = []
        self._hit_spectra_cache.clear()
        self._hit_error_map_cache.clear()
        self._hit_pca_cache.clear()
        self._update_cube_result_tabs()
        self.tabs_img_ctrl.setCurrentIndex(0)
        self.sl_lambda.setValue(0)
        self.lbl_lambda.setText(self.get_lambda_slider_text(0))
        if self.cube is not None:
            self.sl_lambda.setMaximum(self.cube.nbands - 1)
            self.rs_xrange.setMin(self.cube.bands[0])
            self.rs_xrange.setMax(self.cube.bands[-1])
            self.rs_xrange.setRange(self.cube.bands[0], self.cube.bands[-1])

        self.sl_component.setValue(0)
        self.plot.reset()

    def set_recompute_errmap_flag(self):
        self.error_map_recompute_flag = True
        # Matched-cube error maps depend on the same settings (range, gradient,
        # squared errors, reference spectrum) — drop them so they recompute too.
        self._hit_error_map_cache.clear()

    def set_recompute_pca_flag(self):
        self.pca_recompute_flag = True
        self._hit_pca_cache.clear()

    def update_image_label(self):
        # Check if we're viewing a cube hit tab (not the Source tab)
        if self._is_viewing_hit_tab():
            self._display_cube_hit_image()
            return

        img = None
        # I. get base image, depending on selected tab
        # 0 - RGB image
        if self.tabs_img_ctrl.currentIndex() == 0:
            if self.rgb is not None:
                img = self.rgb
        # 1 - single layer
        elif self.tabs_img_ctrl.currentIndex() == 1:
            layer = self.sl_lambda.value()
            if self.cube is not None:
                if 0 <= layer < self.cube.nbands:
                    img = self.cube.data[:, :, layer]
                    self.lbl_lambda.setText(self.get_lambda_slider_text(layer))
        # 2 - similarity
        elif self.tabs_img_ctrl.currentIndex() == 2:
            ref_x, ref_y = self._current_similarity_reference()
            if ref_y is not None and self.cube is not None:
                if self.error_map_recompute_flag:
                    self.error_map_recompute_flag = False
                    self.error_map = self.db.compare_spectra(np.array(self.cube.bands),
                                                             self.cube.data,
                                                             np.array(ref_x),
                                                             ref_y,
                                                             custom_range=(self.rs_xrange.start(), self.rs_xrange.end()),
                                                             use_gradient=self.cb_gradient.isChecked(),
                                                             squared_errs=self.cb_squared.isChecked())
                err_map_t = self.error_map.copy()
                t = (100 - self.sl_sim_t.value()) / 100 * err_map_t.max()
                err_map_t[err_map_t > t] = t
                img = self.visualize_error_map(err_map_t)

        # 3 - PCA
        elif self.tabs_img_ctrl.currentIndex() == 3:
            component = self.sl_component.value()
            if self.cube is not None:
                if self.pca is None or self.pca_recompute_flag:
                    self.pca_recompute_flag = False
                    band_min = self.cube.lambda2layer(self.rs_xrange.start())
                    band_max = self.cube.lambda2layer(self.rs_xrange.end())
                    self.pca = hyper.principal_component_analysis(self.cube.data[:,:,band_min:band_max],
                                                                  p_keep=0.01,
                                                                  n_components=10)
                if 0 <= component < self.pca.shape[2]:
                    img = self.pca[:, :, component]
                    img = (img - img.min()) / (img.max() - img.min())
                    self.lbl_component.setText(f'PC {component}')

        # II. if we have an image, draw the selected pixel and render it.
        if img is not None:
            # adjust brightness
            img = img * self.sl_brightness.value() / 100
            self.lbl_brightness.setText(f'{self.sl_brightness.value()}%')
            # clip
            img = np.clip(img, 0, 1)
            # float to normalized 8 bit
            img = np.uint8(img * 255)

            # draw cross
            img = self.draw_marker(img)

            # rotate for display
            img = self.rotate_for_display(img)

            width = img.shape[1]
            height = img.shape[0]
            if len(img.shape) == 3:
                qImg = QImage(img.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
            else:
                qImg = QImage(img.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)

            if qImg is not None:
                qPixmap = QPixmap.fromImage(qImg)
                # handle scaling
                self.lbl_zoom.setText(f'{self.sl_zoom.value()}%')
                scale = self.sl_zoom.value() / 100
                qPixmap = qPixmap.scaled(int(width * scale), int(height * scale),
                                         transformMode=Qt.TransformationMode.FastTransformation)
                # set image
                self.lbl_img.setPixmap(qPixmap)
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                self.lbl_img.resize(scaled_width, scaled_height)

    def _replot_cube_results(self):
        """Replot the already-computed cube search hits, reusing the colors they
        were assigned when the search ran. Used to redraw the plot (e.g. after a
        reference-DB toggle) without re-running the cube search."""
        for hit in self.cube_search_results:
            self.plot.plot(hit['spectrum_x'],
                           hit['spectrum_y'],
                           label=f"{hit['cube_name']} ({hit['x']},{hit['y']}) (mean err={hit['error']:10.3E})",
                           linewidth=1,
                           hold=True,
                           color=hit.get('_color_hex'))

    def update_spectrum_plot(self, run_cube_search=True):
        """Redraw the spectrum plot for the current selection.

        :param run_cube_search: When True (a new selection / range change),
            launch a fresh cross-cube search. When False (e.g. toggling the
            reference-DB source), the existing cube hits are replotted from cache
            instead of re-running the search.
        """

        self.plot.set_ranges(self.rs_xrange.start(),
                             self.rs_xrange.end(),
                             self.sb_ymin.value(),
                             self.sb_ymax.value())

        if not self.selections:
            self.plot.reset()
            return

        # Plot all selection spectra with their colors
        for i, sel in enumerate(self.selections):
            self.plot.plot(self.cube.bands,
                           sel.spectrum_y,
                           label=sel.label,
                           hold=(i > 0),
                           color=sel.color_mpl,
                           defer_draw=True)

        # database spectra (compared against last selection via self.spectrum_y)
        if self.spectrum_y is not None:
            # 0 - select
            if self.tabs_spectra_source.currentIndex() == 0:
                # Clear cube results when in Select mode
                self.cube_search_results = []
                self._update_cube_result_tabs()
                self.update_image_label()  # Refresh image display
                if self.db is not None:
                    if self.cmb_comparison_ref.currentData() >= 0:
                        reference = self.db.spectra[self.cmb_comparison_ref.currentData()]
                        error = hyper.Database.compare_spectra(self.cube.bands,
                                                               self.spectrum_y,
                                                               reference.x,
                                                               reference.y,
                                                               custom_range=(self.rs_xrange.start(), self.rs_xrange.end()),
                                                               use_gradient=self.cb_gradient.isChecked(),
                                                               squared_errs=self.cb_squared.isChecked())
                        self.plot.plot(reference.x,
                                       reference.y,
                                       label=f"{reference.display_string()} (mean err={error:10.3E})",
                                       linewidth=1,
                                       hold=True,
                                       defer_draw=True)
            # 1 - search
            elif self.tabs_spectra_source.currentIndex() == 1:
                if run_cube_search:
                    # Starting a fresh search: drop the previous cube results.
                    self.cube_search_results = []
                    self._update_cube_result_tabs()
                self.update_image_label()  # Refresh crosshairs for current results
                hit_index = 0  # global color index across all results

                # Search in JDX database
                if self.config.search_in_db and self.sb_nspectra.value() > 0:
                    results = self.db.search_spectrum(self.cube.bands,
                                                      self.spectrum_y,
                                                      custom_range=(self.rs_xrange.start(), self.rs_xrange.end()),
                                                      use_gradient=self.cb_gradient.isChecked(),
                                                      squared_errs=self.cb_squared.isChecked())
                    for result in results[:self.sb_nspectra.value()]:
                        color = self.hit_colors_hex[hit_index % len(self.hit_colors_hex)]
                        self.plot.plot(result['spectrum'].x,
                                       result['spectrum'].y,
                                       label=f"{result['spectrum'].display_string()} (mean err={result['error']:10.3E})",
                                       linewidth=1,
                                       hold=True,
                                       color=color)
                        hit_index += 1

                # Search in analyzed cubes (background worker with progress)
                if self.config.search_in_cubes and self.config.cube_folder_path:
                    if run_cube_search:
                        self._search_hit_index_offset = hit_index
                        self._start_cube_search()
                    else:
                        # Redraw only: reuse the existing cube hits instead of
                        # re-running the (expensive) search.
                        self._replot_cube_results()
                else:
                    # Cube search is off: stop any in-flight search so it can't
                    # re-plot stale hits, clear the current cube results, and
                    # refresh the image so its crosshairs go away too.
                    self._cancel_cube_search()
                    self.cube_search_results = []
                    self._update_cube_result_tabs()
                    self.update_image_label()

        # Single draw after all plot calls
        self.plot.draw()


    ##################
    # loading HS data
    ##################
    def load_data(self, filename):
        """Load cube data in a background thread with progress dialog."""
        self.loading_worker = CubeLoadingWorker(filename)
        self.loading_dialog = CubeLoadingDialog(self)

        # Connect worker signals
        self.loading_worker.progress.connect(self.loading_dialog.update_status)
        self.loading_worker.finished.connect(self._on_cube_loaded)
        self.loading_worker.error.connect(self._on_cube_load_error)
        self.loading_dialog.cancel_requested.connect(self.loading_worker.cancel)

        # Start loading in background thread
        self.loading_worker.start()

        # Show modal dialog
        self.loading_dialog.exec()

    def _on_cube_loaded(self, cube):
        """Handle successful cube loading."""
        try:
            self.cube = cube
            self.rgb = self.cube.to_rgb()
            self.pca = None
            self.reset_ui()
            self.update_image_label()
            self.rawfile = self.loading_worker.filename
            self.statusBar().showMessage(f'Loaded: {os.path.basename(self.rawfile)} | '
                                        f'{self.cube.ncols} x {self.cube.nrows} px | '
                                        f'{self.cube.nbands} bands.')
            self.sl_zoom.setValue(int(self.width() * self.config.initial_image_width_ratio / self.cube.ncols * 100))
            self.loading_dialog.accept()
            self._apply_pending_source_selection()
        except Exception as e:
            self._on_cube_load_error(str(e))

    def _on_cube_load_error(self, error_message):
        """Handle cube loading error."""
        print(f"Error loading file:\n{error_message}")
        self.loading_dialog.set_error(error_message)
        self._pending_source_selection = None

    def _apply_pending_source_selection(self):
        """After a matched cube loads as the new source, seed the point the user
        clicked as a selection and run a fresh search on it. No-op when nothing
        is pending."""
        pending = self._pending_source_selection
        self._pending_source_selection = None
        if pending is None or self.cube is None:
            return
        x, y = pending
        if not (0 <= x < self.cube.ncols and 0 <= y < self.cube.nrows):
            return

        spectrum_y = self.cube.data[y, x, :]
        label = f"P1 ({x},{y})"
        sel = self._make_selection('point', QPoint(x, y), None, spectrum_y, label)
        self.selections.append(sel)
        self._renumber_selections()
        self.set_recompute_errmap_flag()

        # Switch to Search mode without double-triggering the search via the
        # currentChanged signal, then run it once.
        self.tabs_spectra_source.blockSignals(True)
        self.tabs_spectra_source.setCurrentIndex(1)
        self.tabs_spectra_source.blockSignals(False)
        self.update_image_label()
        self.update_spectrum_plot()


    def handle_action_load_data(self):
        filename, _ = QFileDialog.getOpenFileName(None, "Select ENVI data file", "")
        if filename:
            self.load_data(filename)

    def handle_drag_enter(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def handle_drop(self, e):
        urls = e.mimeData().urls()
        if len(urls) > 1:
            print('Dropped multiple files. Loading one of them (good luck).')
        filename = urls[0].url(QUrl.UrlFormattingOption.RemoveScheme)
        # remove leading slashes
        while filename.startswith('/'):
            filename = filename[1:]
        self.load_data(filename)

    #############################
    # image & spectra operations
    #############################
    def _is_viewing_hit_tab(self):
        """Return True if user is viewing a cube hit tab (not Source)."""
        return (self._cube_nav_scroll.isVisible() and
                self._cube_current_index > 0 and
                bool(self._cube_tab_groups))

    def handle_click_on_image(self, event):
        if self._is_viewing_hit_tab():
            event.ignore()
            return
        if event.buttons() == Qt.MouseButton.LeftButton and self.cube is not None:
            self.rubberband_origin = event.pos()
            self.rubberband_selector.setGeometry(QRect(self.rubberband_origin, QSize()))
            self.rubberband_selector.show()
        else:
            event.ignore()
    def handle_move_on_image(self, event):
        if self._is_viewing_hit_tab():
            event.ignore()
            return
        if event.buttons() == Qt.MouseButton.LeftButton and self.cube is not None:
            x = np.clip(event.pos().x(), 0, self.lbl_img.width()-1)
            y = np.clip(event.pos().y(), 0, self.lbl_img.height()-1)
            self.rubberband_selector.setGeometry(QRect(self.rubberband_origin, QPoint(x, y)).normalized())
        elif event.buttons() == Qt.MouseButton.NoButton and self.cube is not None and self.selections:
            data_pt = self.display_to_data_point(event.pos())
            old_idx = self.hovered_selection_idx
            self.hovered_selection_idx = self._hit_test_selection(data_pt)
            if old_idx != self.hovered_selection_idx:
                if self.hovered_selection_idx is not None:
                    self.lbl_img.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.lbl_img.setCursor(Qt.CursorShape.ArrowCursor)
                self.update_image_label()
        else:
            event.ignore()
    def _make_selection(self, sel_type, point, rect, spectrum_y, label):
        """Create a Selection with the next color from the cycle."""
        color_rgb = SELECTION_COLORS[self.selection_counter % len(SELECTION_COLORS)]
        color_mpl = tuple(c / 255.0 for c in color_rgb)
        sel = Selection(
            sel_type=sel_type,
            point=point,
            rect=rect,
            spectrum_y=spectrum_y,
            color_rgb=color_rgb,
            color_mpl=color_mpl,
            label=label,
            index=self.selection_counter,
        )
        self.selection_counter += 1
        return sel

    def _renumber_selections(self):
        """Rebuild selection labels so numbering stays compact."""
        for i, sel in enumerate(self.selections):
            sel.index = i
            if sel.sel_type == 'point' and sel.point is not None:
                sel.label = f"P{i+1} ({sel.point.x()},{sel.point.y()})"
            elif sel.sel_type == 'rect' and sel.rect is not None:
                sel.label = f"R{i+1} ({sel.rect.x()},{sel.rect.y()},{sel.rect.width()}x{sel.rect.height()})"
        self._sync_legacy_state()

    def handle_release_on_image(self, event):
        if self._is_viewing_hit_tab():
            if event.button() == Qt.MouseButton.LeftButton:
                self._prompt_make_source(self._hit_click_to_data_point(event.pos()))
                event.accept()
            else:
                event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.cube is not None:
                shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

                # Click on hovered selection -> remove it
                if self.hovered_selection_idx is not None and not shift_held:
                    self.selections.pop(self.hovered_selection_idx)
                    self.hovered_selection_idx = None
                    self._renumber_selections()
                    self.set_recompute_errmap_flag()
                    self.update_image_label()
                    self.update_spectrum_plot()
                    self.rubberband_selector.hide()
                    return

                rect = self.display_to_data_rect(self.rubberband_selector.geometry())

                if not shift_held:
                    self.selections.clear()
                    self.selection_counter = 0

                if rect.width() > 1 and rect.height() > 1:
                    cube_slice = self.cube.data[rect.y():rect.y()+rect.height(),
                                                rect.x():rect.x()+rect.width(),
                                                :]
                    spectrum_y = np.mean(cube_slice, axis=(0, 1))
                    label = f"R{len(self.selections)+1} ({rect.x()},{rect.y()},{rect.width()}x{rect.height()})"
                    sel = self._make_selection('rect', None, rect, spectrum_y, label)
                else:
                    pos_img = self.display_to_data_point(event.pos())
                    if 0 <= pos_img.x() < self.cube.ncols and 0 <= pos_img.y() < self.cube.nrows:
                        print(f"Selected point: ({pos_img.x()}, {pos_img.y()})")
                        spectrum_y = self.cube.data[pos_img.y(), pos_img.x(), :]
                        label = f"P{len(self.selections)+1} ({pos_img.x()},{pos_img.y()})"
                        sel = self._make_selection('point', pos_img, None, spectrum_y, label)
                    else:
                        sel = None

                if sel is not None:
                    self.selections.append(sel)
                    self._renumber_selections()

                # update ui
                self.set_recompute_errmap_flag()
                self.update_image_label()
                self.update_spectrum_plot()
                self.rubberband_selector.hide()
        else:
            event.ignore()

    def _hit_click_to_data_point(self, pos):
        """Click position on the label -> (x, y) pixel in the matched cube being
        viewed. Hit images are shown without rotation, so only the zoom scale
        applies. Clamped to the matched cube's bounds."""
        point_on_pixmap = self.label_to_pixmap_point(pos)
        x = self.m2i(point_on_pixmap.x())
        y = self.m2i(point_on_pixmap.y())
        idx = self._cube_current_index
        hits = self._cube_tab_groups[idx - 1][1] if 0 < idx <= len(self._cube_tab_groups) else None
        if hits:
            x = int(np.clip(x, 0, hits[0]['ncols'] - 1))
            y = int(np.clip(y, 0, hits[0]['nrows'] - 1))
        return (x, y)

    def _prompt_make_source(self, point):
        """Offer to load the currently viewed matched cube as the new source.

        Selections and searches only operate on the active source cube, so a
        click inside a matched-cube tab can't do anything until that cube
        becomes the source. Rather than silently swapping (a heavy reload that
        clears the current selections and search), ask first. On accept, the
        clicked point is seeded as a selection once the cube loads and a fresh
        search is kicked off (see _on_cube_loaded).
        """
        idx = self._cube_current_index
        if idx <= 0 or idx > len(self._cube_tab_groups):
            return
        cube_file = self._cube_tab_groups[idx - 1][0]
        if not cube_file:
            return
        cube_name = os.path.basename(cube_file)
        reply = QMessageBox.question(
            self, 'Make source',
            f'You can only select and search within the active source cube.\n\n'
            f'Do you want to make "{cube_name}" the new source?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._pending_source_selection = point
            self.load_data(cube_file)

    def _sync_legacy_state(self):
        """Sync scalar point_selection/rect_selection/spectrum_y from selections list."""
        if self.selections:
            first = self.selections[0]
            self.spectrum_y = first.spectrum_y
            self.point_selection = first.point if first.sel_type == 'point' else None
            self.rect_selection = first.rect if first.sel_type == 'rect' else None
        else:
            self.spectrum_y = None
            self.point_selection = None
            self.rect_selection = None

    def _hit_test_selection(self, data_point):
        """Return index of selection under data_point, or None."""
        for i, sel in enumerate(self.selections):
            if sel.sel_type == 'point':
                dx = abs(data_point.x() - sel.point.x())
                dy = abs(data_point.y() - sel.point.y())
                if dx <= HOVER_TOLERANCE and dy <= HOVER_TOLERANCE:
                    return i
            elif sel.sel_type == 'rect':
                if sel.rect.contains(data_point):
                    return i
        return None

    def handle_rotate_left_image(self):
        self._rotate_image_with_center_preserved(-1)

    def handle_rotate_right_image(self):
        self._rotate_image_with_center_preserved(1)

    def _rotate_image_with_center_preserved(self, delta_quadrants):
        center_data_point = None
        if self.cube is not None and self.lbl_img.pixmap() is not None:
            h_scrollbar = self.scroll_img.horizontalScrollBar()
            v_scrollbar = self.scroll_img.verticalScrollBar()
            viewport = self.scroll_img.viewport()
            center_display_x = int(round(h_scrollbar.value() + viewport.width() / 2))
            center_display_y = int(round(v_scrollbar.value() + viewport.height() / 2))
            center_data_point = self.display_to_data_point(QPoint(center_display_x, center_display_y))

        self.rotation_quadrants = (self.rotation_quadrants + delta_quadrants) % 4
        self.update_image_label()

        if center_data_point is not None:
            h_scrollbar = self.scroll_img.horizontalScrollBar()
            v_scrollbar = self.scroll_img.verticalScrollBar()
            viewport = self.scroll_img.viewport()
            scale = self.sl_zoom.value() / 100
            center_display_after = self.data_to_display_point(center_data_point)
            center_label_x = center_display_after.x() * scale
            center_label_y = center_display_after.y() * scale
            h_scrollbar.setValue(max(h_scrollbar.minimum(),
                                     min(h_scrollbar.maximum(),
                                         int(round(center_label_x - viewport.width() / 2)))))
            v_scrollbar.setValue(max(v_scrollbar.minimum(),
                                     min(v_scrollbar.maximum(),
                                         int(round(center_label_y - viewport.height() / 2)))))

    def handle_clear_selections(self):
        self.selections.clear()
        self.selection_counter = 0
        self.hovered_selection_idx = None
        self._sync_legacy_state()
        self.set_recompute_errmap_flag()
        self.update_image_label()
        self.update_spectrum_plot()

    def handle_undo_selection(self):
        if self.selections:
            self.selections.pop()
            self.hovered_selection_idx = None
            self.selection_counter = len(self.selections)
            self._sync_legacy_state()
            self.set_recompute_errmap_flag()
            self.update_image_label()
            self.update_spectrum_plot()

    def handle_click_on_image_scroll(self, event):
        self.drag_start_x = event.pos().x()
        self.drag_start_y = event.pos().y()
        if event.buttons() == Qt.MouseButton.RightButton:
            self.drag_start_hs_v = self.scroll_img.horizontalScrollBar().value()
            self.drag_start_vs_v = self.scroll_img.verticalScrollBar().value()
        else:
            event.ignore()
    def handle_move_on_image_scroll(self, event):
        if event.buttons() == Qt.MouseButton.RightButton:
            # drag image
            hs_max = self.scroll_img.horizontalScrollBar().maximum()
            vs_max = self.scroll_img.verticalScrollBar().maximum()
            hs_min = self.scroll_img.horizontalScrollBar().minimum()
            vs_min = self.scroll_img.verticalScrollBar().minimum()

            dh = int(self.drag_start_x - event.position().x())
            dv = int(self.drag_start_y - event.position().y())

            self.scroll_img.horizontalScrollBar().setValue(min(hs_max, max(hs_min, self.drag_start_hs_v + dh)))
            self.scroll_img.verticalScrollBar().setValue(min(vs_max, max(vs_min, self.drag_start_vs_v + dv)))
        else:
            event.ignore()
    def handle_wheel_on_image_scroll(self, event):
        # Calculate new zoom level
        s = event.angleDelta().y() * self.config.scroll_speed
        if s < 0:
            s = -1 / s
        new_zoom_percent = int(self.sl_zoom.value() * s)

        # Clamp to valid zoom range
        new_zoom_percent = max(self.sl_zoom.minimum(), min(self.sl_zoom.maximum(), new_zoom_percent))

        # If zoom didn't change, don't process further
        if new_zoom_percent == self.sl_zoom.value():
            return

        # Get cursor position in viewport coordinates
        cursor_viewport_pos = event.position()
        cursor_vp_x = cursor_viewport_pos.x()
        cursor_vp_y = cursor_viewport_pos.y()

        # Get current scroll position and zoom
        h_scrollbar = self.scroll_img.horizontalScrollBar()
        v_scrollbar = self.scroll_img.verticalScrollBar()
        old_zoom = self.sl_zoom.value() / 100
        h_pos = h_scrollbar.value()
        v_pos = v_scrollbar.value()

        # Calculate cursor position in image coordinates (accounting for current zoom/scroll)
        cursor_img_x = (h_pos + cursor_vp_x) / old_zoom
        cursor_img_y = (v_pos + cursor_vp_y) / old_zoom

        # Apply zoom (triggers image resize via update_image_label)
        new_zoom = new_zoom_percent / 100
        self.sl_zoom.setValue(new_zoom_percent)

        # Calculate where cursor should be in new zoomed viewport coordinates
        new_cursor_viewport_x = cursor_img_x * new_zoom - cursor_vp_x
        new_cursor_viewport_y = cursor_img_y * new_zoom - cursor_vp_y

        # Apply scroll with bounds checking and rounding
        h_scrollbar.setValue(max(0, min(h_scrollbar.maximum(), round(new_cursor_viewport_x))))
        v_scrollbar.setValue(max(0, min(v_scrollbar.maximum(), round(new_cursor_viewport_y))))

    def handle_action_export_image(self):
        if self.cube is None or self.rawfile is None:
            return
        expdir = os.path.dirname(self.rawfile)
        basename = self.dataset_name()
        if self.tabs_img_ctrl.currentIndex() == 0:
            suffix = 'rgb'
        elif self.tabs_img_ctrl.currentIndex() == 1:
            suffix = self.lbl_lambda.text()
        elif self.tabs_img_ctrl.currentIndex() == 2:
            if self.rb_sim_cube.isChecked():
                suffix = f'sim{self.selection_str()}'
            else:
                suffix = f'sim({self.cmb_comparison_ref.currentText()})'
        elif self.tabs_img_ctrl.currentIndex() == 3:
            suffix = f'pc{self.sl_component.value()}'
        else:
            print("Your argument is invalid!")
            return
        expfile = os.path.join(expdir, f"{basename}_{suffix}.png")

        fileName, _ = QFileDialog.getSaveFileName(None, "Export image", expfile, "All Files (*)")
        if fileName:
            self.lbl_img.pixmap().save(fileName)

    def handle_action_export_spectrum(self):
        if not self.selections or self.cube is None:
            return

        export_dir_default = self.last_export_dir if self.last_export_dir else (self.db.root if self.db is not None else '.')
        export_dir = None
        export_ext = None
        image = np.uint8(self.rgb * (255 / self.rgb.max()))
        image = self.draw_marker(image)

        for selection in list(self.selections):
            selection_key = self._selection_short_label(selection)
            source_name_default = self.last_source_name if self.last_source_name else self.dataset_name()
            header_text = f'Enter metadata for {selection_key}'
            info_dialog = hyper.SaveSpectrumDialog(
                self,
                source_name_default,
                header_text=header_text,
                header_color=selection.color_rgb,
            )
            dialog_code = info_dialog.exec()
            info_dialog_result = info_dialog.get_data()

            if info_dialog_result['action'] == 'cancel' or dialog_code != QDialog.DialogCode.Accepted and info_dialog_result['action'] != 'skip':
                break
            if info_dialog_result['action'] == 'skip':
                continue

            self.last_source_name = info_dialog_result['source']
            metadata = hyper.Metadata(id=info_dialog_result['id'],
                                      description=info_dialog_result['description'],
                                      source_object=info_dialog_result['source'],
                                      source_file=self.dataset_name(),
                                      source_coordinates=self.selection_str(selection),
                                      device_info=f"{self.cube.device} / Hyperlyse {self.config.version}",
                                      intensity=info_dialog_result['intensity'])
            spectrum = hyper.Spectrum(self.cube.bands, selection.spectrum_y, metadata)

            if export_dir is None:
                filename_default = self._selection_export_filename(selection, spectrum.metadata.id, '.jdx')
                file_spectrum, _ = QFileDialog.getSaveFileName(
                    None,
                    'Save spectrum',
                    os.path.join(export_dir_default, filename_default),
                    'JCAMP-DX (*.jdx *.dx *jcm);;Plain x,y pairs (*.dpt *.csv *.txt )',
                )
                if not file_spectrum:
                    break
                export_dir = os.path.dirname(file_spectrum)
                export_ext = os.path.splitext(file_spectrum)[1] or '.jdx'
                self.last_export_dir = export_dir
            else:
                export_base = self._selection_export_stem(selection, spectrum.metadata.id)
                file_spectrum = os.path.join(export_dir, f'{export_base}{export_ext}')

            if not os.path.splitext(file_spectrum)[1]:
                file_spectrum = f'{file_spectrum}.jdx'

            hyper.Database.export_spectrum(
                file_spectrum,
                spectrum,
                image=image,
            )

    def _selection_short_label(self, selection):
        if selection is None or not selection.label:
            return ''
        return selection.label.split(' ', 1)[0]

    def _selection_export_stem(self, selection, spectrum_id):
        return f'{spectrum_id}_{self.selection_str(selection)}_{self.dataset_name()}'

    def _selection_export_filename(self, selection, spectrum_id, extension):
        return f'{self._selection_export_stem(selection, spectrum_id)}{extension}'


    def handle_action_open_preferences(self):
        dialog = hyper.SettingsDialog(self, self.config, exclude_cube_file=getattr(self, 'rawfile', None))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()

            # Apply database path change
            old_db_path = self.config.default_db_path
            new_db_path = data['default_db_path']
            if new_db_path and new_db_path != old_db_path:
                self.config.default_db_path = new_db_path
                self.db.refresh_from_disk(new_db_path)
                self.fill_db_spectra_combobox()

            # Apply cube folder change
            old_cube_path = self.config.cube_folder_path
            new_cube_path = data['cube_folder_path']
            if new_cube_path != old_cube_path:
                self.config.cube_folder_path = new_cube_path
                self.action_analyze_cubes.setEnabled(bool(new_cube_path))
                if new_cube_path:
                    self.statusBar().showMessage(f'Cube folder set: {new_cube_path}')

            # Apply sample rate
            old_sample_rate = self.config.sample_rate
            new_sample_rate = data['sample_rate']
            sample_rate_changed = new_sample_rate != old_sample_rate
            self.config.sample_rate = new_sample_rate

            # Apply num_hits and sync menu
            self.config.num_hits = data['num_hits']
            for action in self.num_hits_actions:
                action.blockSignals(True)
                action.setChecked(int(action.text()) == data['num_hits'])
                action.blockSignals(False)

            # Apply search toggles and sync menu
            self.config.search_in_db = data['search_in_db']
            self.action_search_in_db.blockSignals(True)
            self.action_search_in_db.setChecked(data['search_in_db'])
            self.action_search_in_db.blockSignals(False)

            self.config.search_in_cubes = data['search_in_cubes']
            self.action_search_in_cubes.blockSignals(True)
            self.action_search_in_cubes.setChecked(data['search_in_cubes'])
            self.action_search_in_cubes.blockSignals(False)

            self.config.use_pca = data.get('use_pca', False)

            old_search_cube_include = self.config.search_cube_include
            self.config.search_cube_include = data.get('search_cube_include', None)

            self.config.save()

            # Re-run the cube search if the cube filter changed and a search is
            # already active (selection exists and we are in Search tab).
            cube_filter_changed = (
                set(old_search_cube_include or []) != set(self.config.search_cube_include or [])
                if (old_search_cube_include is None) == (self.config.search_cube_include is None)
                else True
            )
            if (cube_filter_changed and self.config.search_in_cubes
                    and self.spectrum_y is not None
                    and self.tabs_spectra_source.currentIndex() == 1):
                self.update_spectrum_plot(run_cube_search=True)

            # The sample rate only affects cubes when they are (re-)analyzed;
            # existing caches keep the rate they were built at. Make that
            # explicit instead of letting the change silently do nothing.
            if sample_rate_changed:
                self._prompt_sample_rate_changed(old_sample_rate, new_sample_rate)

    def _prompt_sample_rate_changed(self, old_sample_rate, new_sample_rate):
        """After the sample rate changes, tell the user it won't take effect
        until cubes are re-analyzed and offer: keep it (Ok), re-analyze now,
        or revert to the previous value (Cancel)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle('Sample rate changed')
        box.setText(
            'The new sample rate does not affect cube searches until the cubes '
            'are re-analyzed.')
        box.setInformativeText(
            f'The cached cubes were analyzed at sample rate {old_sample_rate} '
            f'and searches keep using that until you re-analyze at '
            f'{new_sample_rate}.\n\n'
            'Analyze Now: re-analyze all cubes at the new rate.\n'
            'OK: keep the new rate for the next analysis.\n'
            'Cancel: revert to the previous rate.')
        analyze_btn = box.addButton('Analyze Now', QMessageBox.ButtonRole.AcceptRole)
        ok_btn = box.addButton(QMessageBox.StandardButton.Ok)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(ok_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is cancel_btn:
            self.config.sample_rate = old_sample_rate
            self.config.save()
        elif clicked is analyze_btn:
            self.handle_action_analyze_cubes()

    def handle_set_num_hits(self, num):
        self.config.num_hits = num
        self.config.save()
        for action in self.num_hits_actions:
            action.setChecked(int(action.text()) == num)

    def handle_toggle_search_in_db(self, checked):
        self.config.search_in_db = checked
        self.config.save()
        # Toggling the reference DB only affects DB curves; redraw without
        # re-running the cube search (it would pop the progress dialog and
        # recompute identical results). No-ops when nothing is selected.
        self.update_spectrum_plot(run_cube_search=False)

    def handle_toggle_search_in_cubes(self, checked):
        if checked and not self._has_analyzed_cubes():
            self._show_no_cubes_dialog()
            return
        self.config.search_in_cubes = checked
        self.config.save()
        # Checking it must run the cube search for the current selection;
        # unchecking it just removes the cube hits (no search needed).
        self.update_spectrum_plot(run_cube_search=checked)

    def _has_analyzed_cubes(self):
        if not self.config.cube_folder_path:
            return False
        cached = cube_analyzer.get_cached_cube_dirs(
            self.config.cube_folder_path,
            self.config.include_subfolders,
            self.config.sample_rate)
        return len(cached) > 0

    def _show_no_cubes_dialog(self):
        dlg = hyper.NoCubesAnalyzedDialog(self, self.config.cube_folder_path)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.wants_analysis():
            new_path = dlg.get_cube_folder_path()
            if new_path != self.config.cube_folder_path:
                self.config.cube_folder_path = new_path
                self.action_analyze_cubes.setEnabled(bool(new_path))
                self.config.save()
            self.handle_action_analyze_cubes()
        else:
            # Revert the checkbox without triggering the signal again
            self.action_search_in_cubes.blockSignals(True)
            self.action_search_in_cubes.setChecked(False)
            self.action_search_in_cubes.blockSignals(False)

    def handle_action_analyze_cubes(self):
        if not self.config.cube_folder_path:
            QMessageBox.information(self, 'Analyze cubes', 'No cube folder is set.')
            return

        # Show the progress dialog immediately. Discovery (which can take a few
        # seconds) runs in the worker thread; the range starts indeterminate
        # (0, 0) and is set once the worker reports the discovered count.
        self._analysis_progress = QProgressDialog(
            'Discovering cubes...', 'Cancel', 0, 0, self)
        self._analysis_progress.setWindowTitle('Analyzing Cubes')
        self._analysis_progress.setMinimumDuration(0)
        self._analysis_progress.setAutoClose(False)
        self._analysis_progress.setAutoReset(False)
        self._analysis_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._analysis_progress.setValue(0)

        # Create and start worker
        self._analysis_worker = CubeAnalysisWorker(
            self.config.cube_folder_path,
            self.config.sample_rate,
            self.config.include_subfolders)
        self._analysis_worker.discovered.connect(self._on_analysis_discovered)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_progress.canceled.connect(self._analysis_worker.terminate)
        self._analysis_worker.start()

    def _on_analysis_discovered(self, total):
        dlg = getattr(self, '_analysis_progress', None)
        if dlg is None:
            return
        if total == 0:
            dlg.close()
            self._analysis_progress = None
            QMessageBox.information(self, 'Analyze cubes',
                                   'No cube files found in the selected folder.')
            return
        dlg.setMaximum(total)
        dlg.setLabelText('Preparing analysis...')

    def _on_analysis_progress(self, current, total, name, avg_time, skipped):
        dlg = getattr(self, '_analysis_progress', None)
        if dlg is not None:
            dlg.setValue(current)
            if skipped:
                label = f'Cube {current}/{total}: {name} (cached, skipped)'
            else:
                label = f'Cube {current}/{total}: {name}'
                if avg_time > 0 and current >= 2:
                    remaining = avg_time * (total - current)
                    if remaining >= 60:
                        label += f' (~{remaining/60:.1f} min remaining)'
                    else:
                        label += f' (~{remaining:.0f}s remaining)'
            dlg.setLabelText(label)
        self.statusBar().showMessage(f'Analyzing cube {current}/{total}: {name}')

    def _on_analysis_finished(self, analyzed, skipped):
        if hasattr(self, '_analysis_progress') and self._analysis_progress is not None:
            self._analysis_progress.close()
            self._analysis_progress = None
        self.statusBar().showMessage(
            f'Cube analysis complete. {analyzed} analyzed, {skipped} skipped (cached).')
        self._analysis_worker = None

    def _on_analysis_error(self, message):
        if hasattr(self, '_analysis_progress') and self._analysis_progress is not None:
            self._analysis_progress.close()
            self._analysis_progress = None
        QMessageBox.critical(self, 'Analysis Error', f'An error occurred:\n{message}')
        self._analysis_worker = None

    ##############################
    # Cross-cube search (async)
    ##############################
    def _cancel_cube_search(self):
        """Stop any running cube-search worker and close its progress dialog, so
        a superseded search can't re-plot stale hits when it finishes."""
        if self._search_worker is not None:
            self._search_worker.terminate()
            self._search_worker.wait()
            self._search_worker = None
        dlg = getattr(self, '_search_progress', None)
        if dlg is not None:
            dlg.close()
            self._search_progress = None

    def _start_cube_search(self):
        """Launch background cross-cube search with a progress dialog."""
        if not self._has_analyzed_cubes():
            self._show_no_cubes_dialog()
            return

        # Cancel any running search
        if self._search_worker is not None:
            self._search_worker.terminate()
            self._search_worker.wait()
            self._search_worker = None

        # Close any leftover progress dialog before creating a new one,
        # otherwise the old one is orphaned on screen.
        old_dlg = getattr(self, '_search_progress', None)
        if old_dlg is not None:
            old_dlg.close()
            self._search_progress = None

        self._search_progress = QProgressDialog(
            'Searching cached cubes...', 'Cancel', 0, 0, self)
        self._search_progress.setWindowTitle('Searching Cubes')
        self._search_progress.setMinimumDuration(0)
        self._search_progress.setAutoClose(False)
        self._search_progress.setAutoReset(False)
        self._search_progress.setWindowModality(Qt.WindowModality.WindowModal)

        include = self.config.search_cube_include
        include_set = (
            {os.path.normcase(os.path.abspath(p)) for p in include}
            if include is not None else None
        )

        self._search_worker = CubeSearchWorker(
            self.config.cube_folder_path,
            self.cube.bands,
            self.spectrum_y,
            sample_rate=self.config.sample_rate,
            include_subfolders=self.config.include_subfolders,
            custom_range=(self.rs_xrange.start(), self.rs_xrange.end()),
            use_gradient=self.cb_gradient.isChecked(),
            squared_errs=self.cb_squared.isChecked(),
            num_hits=self.config.num_hits,
            use_pca=self.config.use_pca,
            exclude_cube_file=getattr(self, 'rawfile', None),
            include_cube_files=include_set)
        self._search_worker.progress.connect(self._on_search_progress)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)
        self._search_progress.canceled.connect(self._on_search_canceled)

        # Paint the dialog now, before the worker's parallel search threads
        # saturate the GIL. Otherwise the GUI thread can't repaint and the
        # dialog only pops up once the search is nearly finished.
        self._search_progress.show()
        QGuiApplication.processEvents()

        self._search_worker.start()

    def _on_search_progress(self, current, total, cube_name, avg_time):
        dlg = getattr(self, '_search_progress', None)
        if dlg is not None:
            if total > 0:
                dlg.setMaximum(total)
                dlg.setValue(current)
            label = f'Cube {current + 1}/{total}: {cube_name}'
            if avg_time > 0 and current >= 2:
                remaining = avg_time * (total - current)
                if remaining >= 60:
                    label += f' (~{remaining/60:.1f} min remaining)'
                else:
                    label += f' (~{remaining:.0f}s remaining)'
            dlg.setLabelText(label)
        self.statusBar().showMessage(f'Searching cube {current + 1}/{total}: {cube_name}')

    def _on_search_finished(self, results):
        dlg = getattr(self, '_search_progress', None)
        if dlg is not None:
            dlg.close()
            self._search_progress = None

        self.cube_search_results = results
        hit_index = getattr(self, '_search_hit_index_offset', 0)

        for i, hit in enumerate(self.cube_search_results):
            color = self.hit_colors_hex[(hit_index + i) % len(self.hit_colors_hex)]
            hit['_color_hex'] = color
            hit['_color_rgb'] = self.hit_colors_rgb[(hit_index + i) % len(self.hit_colors_rgb)]
            self.plot.plot(hit['spectrum_x'],
                           hit['spectrum_y'],
                           label=f"{hit['cube_name']} ({hit['x']},{hit['y']}) (mean err={hit['error']:10.3E})",
                           linewidth=1,
                           hold=True,
                           color=color)

        self._update_cube_result_tabs()
        self._search_worker = None
        n = len(self.cube_search_results)
        self.statusBar().showMessage(f'Cube search complete. {n} cube{"s" if n != 1 else ""} found.')

    def _on_search_error(self, message):
        dlg = getattr(self, '_search_progress', None)
        if dlg is not None:
            dlg.close()
            self._search_progress = None
        self.cube_search_results = []
        self._update_cube_result_tabs()
        self._search_worker = None
        self.statusBar().showMessage(f'Cube search error: {message}')

    def _on_search_canceled(self):
        if self._search_worker is not None:
            self._search_worker.terminate()
            self._search_worker.wait()
            self._search_worker = None
        dlg = getattr(self, '_search_progress', None)
        if dlg is not None:
            dlg.close()
            self._search_progress = None
        self.statusBar().showMessage('Cube search cancelled.')

    def handle_action_reset_cube_cache(self):
        if not self.config.cube_folder_path:
            QMessageBox.information(self, 'Reset cube cache', 'No cube folder is set.')
            return
        reply = QMessageBox.question(self, 'Reset cube cache',
                                     'This will delete all cached cube analysis data.\nAre you sure?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # Drop any search results first: hits may hold numpy memmap views
            # into the cached .npy files, and on Windows an open mmap handle
            # blocks deletion of the file it maps.
            self.cube_search_results = []
            self._hit_spectra_cache.clear()
            self._hit_error_map_cache.clear()
            self._hit_pca_cache.clear()
            self._update_cube_result_tabs()
            gc.collect()
            try:
                cube_analyzer.reset_cache(self.config.cube_folder_path)
                self.statusBar().showMessage('Cube cache cleared.')
            except OSError as e:
                QMessageBox.warning(self, 'Reset cube cache',
                                    f'Could not delete all cached files. '
                                    f'They may be in use.\n\n{e}')
                self.statusBar().showMessage('Cube cache partially cleared.')


    ###########
    # helpers
    ##########

    # --- Cube result navigation ---
    def _update_cube_result_tabs(self):
        """Rebuild cube nav bar from self.cube_search_results."""
        if not self.cube_search_results:
            self._cube_tab_groups = []
            self._cube_current_index = 0
            self._rebuild_cube_nav()
            return

        cube_groups = {}
        for hit in self.cube_search_results:
            key = hit['cube_file']
            if key not in cube_groups:
                cube_groups[key] = []
            cube_groups[key].append(hit)

        self._cube_tab_groups = sorted(cube_groups.items(),
                                       key=lambda kv: kv[1][0]['error'])
        self._cube_current_index = 0
        self._rebuild_cube_nav()

    def _rebuild_cube_nav(self):
        """Rebuild the scrollable button row above the image."""
        while self._cube_nav_layout.count():
            item = self._cube_nav_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._cube_tab_groups:
            self._cube_nav_scroll.hide()
            return

        self._add_nav_button('Source', 0)
        for i, (cube_file, hits) in enumerate(self._cube_tab_groups):
            cube_name = hits[0]['cube_name']
            display_name = cube_name if len(cube_name) <= 15 else cube_name[:12] + '...'
            self._add_nav_button(display_name, i + 1, tooltip=cube_name)
        self._cube_nav_layout.addStretch()

        self._update_nav_selection()
        self._cube_nav_scroll.show()

    def _add_nav_button(self, label, index, tooltip=None):
        btn = QPushButton(label)
        btn.setCheckable(True)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.clicked.connect(lambda _checked, i=index: self._on_cube_nav_select(i))
        self._cube_nav_layout.addWidget(btn)

    def _on_cube_nav_select(self, index):
        self._cube_current_index = index
        self._update_nav_selection()
        self.update_image_label()

    def _update_nav_selection(self):
        for i in range(self._cube_nav_layout.count()):
            item = self._cube_nav_layout.itemAt(i)
            btn = item.widget() if item else None
            if isinstance(btn, QPushButton):
                btn.setChecked(i == self._cube_current_index)

    def _current_similarity_reference(self):
        """Reference (x, y) for the similarity mode given the current UI state.

        Either the active cube selection spectrum (compared against the source
        cube's bands) or the chosen database spectrum. Returns (None, None) when
        nothing usable is selected. Shared by the source and matched-cube views
        so both compare against the same reference.
        """
        if self.rb_sim_cube.isChecked():
            if self.spectrum_y is not None and self.cube is not None:
                return np.array(self.cube.bands), self.spectrum_y
        else:
            idx = self.cmb_comparison_ref.currentData()
            if idx is not None and idx >= 0:
                ref = self.db.spectra[idx]
                return np.array(ref.x), ref.y
        return None, None

    @staticmethod
    def _lambda2layer(bands, lmd):
        """Nearest band index in `bands` to wavelength `lmd`."""
        diffs = [abs(lmd - l) for l in bands]
        return diffs.index(min(diffs))

    def _get_hit_spectra(self, cache_dir):
        """Load (and cache) the sampled spectra for a matched cube."""
        spectra = self._hit_spectra_cache.get(cache_dir)
        if spectra is None:
            spectra = cube_analyzer.load_spectra(cache_dir)
            if spectra is not None:
                self._hit_spectra_cache[cache_dir] = spectra
        return spectra

    def _get_hit_error_map(self, cache_dir, spectra, bands, ref_x, ref_y):
        """Similarity error map for a matched cube vs the reference spectrum."""
        error_map = self._hit_error_map_cache.get(cache_dir)
        if error_map is None:
            error_map = self.db.compare_spectra(
                bands, spectra, np.array(ref_x), ref_y,
                custom_range=(self.rs_xrange.start(), self.rs_xrange.end()),
                use_gradient=self.cb_gradient.isChecked(),
                squared_errs=self.cb_squared.isChecked())
            if error_map is None:
                return None
            self._hit_error_map_cache[cache_dir] = error_map
        return error_map

    def _get_hit_pca(self, cache_dir, spectra, bands):
        """PCA stack for a matched cube over the current comparison range."""
        pca = self._hit_pca_cache.get(cache_dir)
        if pca is None:
            band_min = self._lambda2layer(bands, self.rs_xrange.start())
            band_max = self._lambda2layer(bands, self.rs_xrange.end())
            if band_max <= band_min:
                return None
            sub = spectra[:, :, band_min:band_max]
            n_components = min(10, sub.shape[2], sub.shape[0] * sub.shape[1])
            if n_components < 1:
                return None
            pca = hyper.principal_component_analysis(sub, p_keep=0.01,
                                                     n_components=n_components)
            self._hit_pca_cache[cache_dir] = pca
        return pca

    @staticmethod
    def _upscale_to_full(img, nrows, ncols, sample_rate):
        """Nearest-neighbour upscale a sampled-resolution image to the cube's
        full pixel grid, so markers and zoom line up across all modes. Works for
        2D (grayscale) and 3D (RGB) arrays."""
        if img.shape[0] == nrows and img.shape[1] == ncols:
            return img
        sr = max(int(sample_rate), 1)
        row_idx = np.minimum(np.arange(nrows) // sr, img.shape[0] - 1)
        col_idx = np.minimum(np.arange(ncols) // sr, img.shape[1] - 1)
        return img[row_idx][:, col_idx]

    def _hit_base_image(self, hits):
        """Compute the base float image [0,1] for a matched cube according to the
        currently selected display mode. Returns HxW or HxWx3, upscaled to the
        cube's full pixel resolution, or None when unavailable."""
        cache_dir = hits[0]['cache_dir']
        nrows = hits[0]['nrows']
        ncols = hits[0]['ncols']
        mode = self.tabs_img_ctrl.currentIndex()

        # RGB: the cached preview is already at full resolution.
        if mode == 0:
            rgb = cube_analyzer.load_rgb_preview(cache_dir)
            if rgb is None:
                return None
            return rgb.astype(np.float64) / 255.0

        # All other modes operate on the cached (sampled) spectral data.
        meta = cube_analyzer.load_metadata(cache_dir)
        spectra = self._get_hit_spectra(cache_dir)
        if meta is None or spectra is None:
            return None
        bands = np.array(meta['bands'])
        sample_rate = meta.get('sample_rate', 1)

        img = None
        # 1 - single layer
        if mode == 1:
            layer = self.sl_lambda.value()
            if 0 <= layer < spectra.shape[2]:
                img = spectra[:, :, layer]
                self.lbl_lambda.setText('%.1fnm' % bands[layer])
        # 2 - similarity
        elif mode == 2:
            ref_x, ref_y = self._current_similarity_reference()
            if ref_y is not None:
                error_map = self._get_hit_error_map(cache_dir, spectra, bands, ref_x, ref_y)
                if error_map is not None:
                    err_map_t = error_map.copy()
                    t = (100 - self.sl_sim_t.value()) / 100 * err_map_t.max()
                    err_map_t[err_map_t > t] = t
                    img = self.visualize_error_map(err_map_t)
        # 3 - PCA
        elif mode == 3:
            component = self.sl_component.value()
            pca = self._get_hit_pca(cache_dir, spectra, bands)
            if pca is not None and 0 <= component < pca.shape[2]:
                img = pca[:, :, component]
                denom = img.max() - img.min()
                img = (img - img.min()) / denom if denom > 0 else np.zeros_like(img)
                self.lbl_component.setText(f'PC {component}')

        if img is None:
            return None
        return self._upscale_to_full(np.asarray(img, dtype=np.float64),
                                     nrows, ncols, sample_rate)

    def _display_cube_hit_image(self):
        """Render the cube hit image for the currently selected cube nav button."""
        idx = self._cube_current_index
        if idx <= 0 or idx > len(self._cube_tab_groups):
            return
        hits = self._cube_tab_groups[idx - 1][1]
        if not hits:
            return

        # Build the base image for the currently selected mode (RGB / layers /
        # similarity / PCA), so matched cubes respond to the mode tabs and their
        # sliders exactly like the source cube does.
        img = self._hit_base_image(hits)
        if img is None:
            return

        # Adjust brightness
        img = img * self.sl_brightness.value() / 100
        self.lbl_brightness.setText(f'{self.sl_brightness.value()}%')
        img = np.clip(img, 0, 1)
        img = np.uint8(img * 255)

        # Draw crosshair markers for each hit in this cube
        if len(img.shape) == 2:
            img = np.dstack([img, img, img])

        spotlight = self.spotlight_active
        pulse = self._spotlight_pulse() if spotlight else 0.0
        if spotlight:
            img = self._dim_desaturate(img)
        img_marker = img.copy()

        scale = self.sl_zoom.value() / 100
        if scale < 1:
            padding = int(np.ceil(1 / scale - 1))
            cross_size = int(np.ceil(self.config.cross_size / scale))
        else:
            padding = 0
            cross_size = self.config.cross_size

        nrows, ncols = img.shape[0], img.shape[1]

        for hit in hits:
            color = hit.get('_color_rgb', [255, 0, 0])
            px, py = hit['x'], hit['y']

            # Clamp to image bounds
            px = min(max(px, 0), ncols - 1)
            py = min(max(py, 0), nrows - 1)

            if spotlight:
                self._draw_emphasized_cross(img_marker, px, py, color,
                                            nrows - 1, ncols - 1,
                                            padding, cross_size, pulse)
                continue

            # Horizontal line
            img_marker[max(py - padding, 0):min(py + padding + 1, nrows),
                       max(px - cross_size, 0):min(px + cross_size + 1, ncols)] = color
            # Vertical line
            img_marker[max(py - cross_size, 0):min(py + cross_size + 1, nrows),
                       max(px - padding, 0):min(px + padding + 1, ncols)] = color

        alpha = 1.0 if spotlight else self.config.marker_alpha
        img = img * (1 - alpha) + img_marker * alpha
        img = img.astype(np.uint8)

        # Note: no rotation applied to hit cube images (rotation is for the source cube only)
        width = img.shape[1]
        height = img.shape[0]
        qImg = QImage(img.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)

        if qImg is not None:
            qPixmap = QPixmap.fromImage(qImg)
            self.lbl_zoom.setText(f'{self.sl_zoom.value()}%')
            zoom_scale = self.sl_zoom.value() / 100
            qPixmap = qPixmap.scaled(int(width * zoom_scale), int(height * zoom_scale),
                                     transformMode=Qt.TransformationMode.FastTransformation)
            self.lbl_img.setPixmap(qPixmap)
            self.lbl_img.resize(int(width * zoom_scale), int(height * zoom_scale))
    def rotate_for_display(self, img):
        if self.rotation_quadrants == 0:
            return img
        if self.rotation_quadrants == 1:
            return np.rot90(img, -1)
        if self.rotation_quadrants == 2:
            return np.rot90(img, 2)
        return np.rot90(img, 1)

    def display_to_data_point(self, point):
        if self.cube is None:
            return self.m2i(point)

        point_on_pixmap = self.label_to_pixmap_point(point)
        x_disp = self.m2i(point_on_pixmap.x())
        y_disp = self.m2i(point_on_pixmap.y())

        width = self.cube.ncols
        height = self.cube.nrows

        if self.rotation_quadrants in (0, 2):
            disp_width = width
            disp_height = height
        else:
            disp_width = height
            disp_height = width

        x_disp = int(np.clip(x_disp, 0, max(disp_width - 1, 0)))
        y_disp = int(np.clip(y_disp, 0, max(disp_height - 1, 0)))

        if self.rotation_quadrants == 0:
            x_data = x_disp
            y_data = y_disp
        elif self.rotation_quadrants == 1:
            x_data = y_disp
            y_data = height - 1 - x_disp
        elif self.rotation_quadrants == 2:
            x_data = width - 1 - x_disp
            y_data = height - 1 - y_disp
        else:
            x_data = width - 1 - y_disp
            y_data = x_disp

        return QPoint(x_data, y_data)

    def display_to_data_rect(self, rect):
        p1 = self.display_to_data_point(rect.topLeft())
        p2 = self.display_to_data_point(rect.bottomRight())
        return QRect(p1, p2).normalized()

    def label_to_pixmap_point(self, point):
        pixmap = self.lbl_img.pixmap()
        if pixmap is None:
            return point

        x_offset = max((self.lbl_img.width() - pixmap.width()) // 2, 0)
        y_offset = max((self.lbl_img.height() - pixmap.height()) // 2, 0)
        return QPoint(point.x() - x_offset, point.y() - y_offset)

    def data_to_display_point(self, point):
        if self.cube is None:
            return point

        x_data = point.x()
        y_data = point.y()
        width = self.cube.ncols
        height = self.cube.nrows

        if self.rotation_quadrants == 0:
            x_disp = x_data
            y_disp = y_data
        elif self.rotation_quadrants == 1:
            x_disp = height - 1 - y_data
            y_disp = x_data
        elif self.rotation_quadrants == 2:
            x_disp = width - 1 - x_data
            y_disp = height - 1 - y_data
        else:
            x_disp = y_data
            y_disp = width - 1 - x_data

        return QPoint(x_disp, y_disp)

    def m2i(self, object):
        # convert mouse coordinates on scaled image label to image coordinates
        if isinstance(object, numbers.Number):
            return int(object / self.sl_zoom.value() * 100)
        if isinstance(object, QPoint):
            return QPoint(self.m2i(object.x()), self.m2i(object.y()))
        elif isinstance(object, QSize):
            return QSize(self.m2i(object.width()), self.m2i(object.height()))
        elif isinstance(object, QRect):
            return QRect(self.m2i(object.topLeft()), self.m2i(object.size()))
        else:
            raise ValueError()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F and not event.isAutoRepeat():
            self._set_spotlight(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_F and not event.isAutoRepeat():
            self._set_spotlight(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _set_spotlight(self, active):
        if active == self.spotlight_active:
            return
        self.spotlight_active = active
        if active:
            self.spotlight_phase = 0.0
            self.spotlight_timer.start()
        else:
            self.spotlight_timer.stop()
        self.update_image_label()

    def _on_spotlight_tick(self):
        self.spotlight_phase += 0.20
        self.update_image_label()

    def _spotlight_pulse(self):
        """Current pulse value in [0, 1] driven by the spotlight phase."""
        return float((np.sin(self.spotlight_phase) + 1.0) / 2.0)

    def _dim_desaturate(self, img, sat=0.15, dim=0.45):
        """Push a uint8 H*W*3 image toward a dark, near-grayscale backdrop so
        coloured markers pop. sat keeps a little colour; dim darkens overall."""
        f = img.astype(np.float64)
        lum = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        gray = np.dstack([lum, lum, lum])
        out = (sat * f + (1.0 - sat) * gray) * dim
        return np.clip(out, 0, 255).astype(np.uint8)

    def _draw_emphasized_cross(self, buf, px, py, color, row_hi, col_hi, padding, cross_size, pulse):
        """Draw a high-visibility marker: white-haloed colour cross plus a
        breathing white ring around the point. Coordinates are clamped."""
        px = int(min(max(px, 0), col_hi))
        py = int(min(max(py, 0), row_hi))
        white = [255, 255, 255]
        cs = cross_size
        hp = padding + 1  # halo is one pixel thicker than the colour arm

        # white halo cross (drawn first, slightly thicker)
        buf[max(py - hp, 0):min(py + hp + 1, row_hi), max(px - cs, 0):min(px + cs + 1, col_hi)] = white
        buf[max(py - cs, 0):min(py + cs + 1, row_hi), max(px - hp, 0):min(px + hp + 1, col_hi)] = white
        # colour cross on top
        buf[max(py - padding, 0):min(py + padding + 1, row_hi), max(px - cs, 0):min(px + cs + 1, col_hi)] = color
        buf[max(py - cs, 0):min(py + cs + 1, row_hi), max(px - padding, 0):min(px + padding + 1, col_hi)] = color
        # central white dot
        buf[max(py - padding, 0):min(py + padding + 1, row_hi), max(px - padding, 0):min(px + padding + 1, col_hi)] = white

        # breathing ring (hollow square) around the point
        r = int(round(cs + 2 + pulse * (cs + 2)))
        th = padding + 1
        x0, x1 = max(px - r, 0), min(px + r + 1, col_hi)
        y0, y1 = max(py - r, 0), min(py + r + 1, row_hi)
        buf[max(py - r, 0):min(py - r + th, row_hi), x0:x1] = white          # top
        buf[max(py + r - th + 1, 0):min(py + r + 1, row_hi), x0:x1] = white  # bottom
        buf[y0:y1, max(px - r, 0):min(px - r + th, col_hi)] = white          # left
        buf[y0:y1, max(px + r - th + 1, 0):min(px + r + 1, col_hi)] = white  # right

    def draw_marker(self, img):
        """
        Draws crosses (for point selections) or rectangles (for area selections) onto an image.
        :param img: numpy array, r*c or r*c*3
        :return: image with markers drawn
        """
        if not self.selections:
            return img

        if len(img.shape) == 2:
            img = np.dstack([img, img, img])

        spotlight = self.spotlight_active
        pulse = self._spotlight_pulse() if spotlight else 0.0
        if spotlight:
            img = self._dim_desaturate(img)
        img_marker = img.copy()

        scale = self.sl_zoom.value() / 100
        if scale < 1:
            padding = int(np.ceil(1 / scale - 1))
            cross_size = int(np.ceil(self.config.cross_size / scale))
        else:
            padding = 0
            cross_size = self.config.cross_size

        hover_color = [255, 255, 255]

        for i, sel in enumerate(self.selections):
            color = hover_color if i == self.hovered_selection_idx else sel.color_rgb

            if spotlight and sel.sel_type == 'point':
                p = sel.point
                self._draw_emphasized_cross(img_marker, p.x(), p.y(), color,
                                            self.cube.nrows - 1, self.cube.ncols - 1,
                                            padding, cross_size, pulse)
            elif sel.sel_type == 'rect':
                r = sel.rect
                # top line
                img_marker[max(r.top() - padding, 0):min(r.top() + padding + 1, self.cube.nrows - 1),
                           max(r.left(), 0):min(r.right(), self.cube.ncols - 1)] = color
                # bottom line
                img_marker[max(r.bottom() - padding, 0):min(r.bottom() + padding + 1, self.cube.nrows - 1),
                           max(r.left(), 0):min(r.right(), self.cube.ncols - 1)] = color
                # left line
                img_marker[max(r.top(), 0):min(r.bottom()+1, self.cube.nrows - 1),
                           max(r.left() - padding, 0):min(r.left() + padding + 1, self.cube.ncols - 1)] = color
                # right line
                img_marker[max(r.top(), 0):min(r.bottom()+1, self.cube.nrows - 1),
                           max(r.right() - padding, 0):min(r.right() + padding + 1, self.cube.ncols - 1)] = color

            elif sel.sel_type == 'point':
                p = sel.point
                # horizontal line
                img_marker[max(p.y() - padding, 0):min(p.y() + padding + 1, self.cube.nrows - 1),
                          max(p.x() - cross_size, 0):min(p.x() + cross_size + 1, self.cube.ncols - 1)] = color
                # vertical line
                img_marker[max(p.y() - cross_size, 0):min(p.y() + cross_size + 1, self.cube.nrows - 1),
                          max(p.x() - padding, 0):min(p.x() + padding + 1, self.cube.ncols - 1)] = color
                # central dot
                img_marker[max(p.y() - padding, 0):min(p.y() + padding + 1, self.cube.nrows - 1),
                          max(p.x() - padding, 0):min(p.x() + padding + 1, self.cube.ncols - 1)] = [255, 255, 255]

        alpha = 1.0 if spotlight else self.config.marker_alpha
        img = img * (1 - alpha) + img_marker * alpha
        return img.astype(np.uint8)

    def selection_coords(self, selection=None):
        if selection is None:
            if not self.selections:
                return []
            selection = self.selections[-1]
        if selection is None:
            return []
        if selection.sel_type == 'rect':
            r = selection.rect
            return [r.left(), r.top(), r.width(), r.height()]
        elif selection.sel_type == 'point':
            p = selection.point
            return [p.x(), p.y()]
        else:
            return []

    def selection_str(self, selection=None):
        return f'({",".join([str(c) for c in self.selection_coords(selection)])})'

    def get_lambda_slider_text(self, layer_idx):
        return '%.1fnm' % self.cube.bands[layer_idx]

    def fill_db_spectra_combobox(self):
        # Block signals while repopulating: clear()/addItem() would otherwise
        # fire currentIndexChanged repeatedly, re-triggering update_spectrum_plot
        # (and a cross-cube search) just from reloading the DB list.
        self.cmb_comparison_ref.blockSignals(True)
        self.cmb_comparison_ref.clear()
        self.cmb_comparison_ref.addItem('(none)', -1)
        if self.db is not None:
            for i, s in enumerate(self.db.spectra):
                self.cmb_comparison_ref.addItem(s.display_string(with_description=True), i)
        self.cmb_comparison_ref.adjustSize()
        self.cmb_comparison_ref.blockSignals(False)

    def visualize_error_map(self, error_map):
        # invert and map to [0, 1]:
        similarity_map = 1 - (error_map / error_map.max())
        # apply color map
        cm = plt.get_cmap('viridis')
        return cm(similarity_map)[:, :, :3]

    def dataset_name(self):
        if self.rawfile is not None:
            return os.path.splitext(os.path.basename(self.rawfile))[0]