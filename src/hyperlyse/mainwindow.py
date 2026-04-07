import os
import numpy as np
import numbers
from dataclasses import dataclass
from typing import Optional
from PyQt6.QtGui import QPixmap, QImage, QGuiApplication, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QUrl, QRect, QPoint, QSize
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QRubberBand, QDoubleSpinBox, QRadioButton
from PyQt6.QtWidgets import QWidget, QLabel, QCheckBox, QSlider, QPushButton, QComboBox, QSpinBox, QFrame, QLineEdit
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QTabWidget, QScrollArea, QSizePolicy, QDialog
from matplotlib import pyplot as plt
import hyperlyse as hyper

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

        self.db = hyper.Database(config.default_db_path)


        self.last_source_name = ''
        self.last_export_dir = ''

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

        self.btn_rotate_left = QPushButton('Rotate Left\n90°')
        self.btn_rotate_left.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))
        self.btn_rotate_left.pressed.connect(self.handle_rotate_left_image)
        layout_img_rotate.addWidget(self.btn_rotate_left)

        self.btn_rotate_right = QPushButton('Rotate Right\n90°')
        self.btn_rotate_right.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))
        self.btn_rotate_right.pressed.connect(self.handle_rotate_right_image)
        layout_img_rotate.addWidget(self.btn_rotate_right)

        self.btn_clear_selections = QPushButton('Clear All\nSelections')
        self.btn_clear_selections.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))
        self.btn_clear_selections.pressed.connect(self.handle_clear_selections)
        layout_img_rotate.addWidget(self.btn_clear_selections)

        layout_img_rotate.addStretch()

        # Ctrl+Z shortcut to undo last selection
        self.shortcut_undo = QShortcut(QKeySequence('Ctrl+Z'), self)
        self.shortcut_undo.activated.connect(self.handle_undo_selection)

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
        self.scroll_img.mousePressEvent = self.handle_click_on_image_scroll
        self.scroll_img.mouseMoveEvent = self.handle_move_on_image_scroll
        self.scroll_img.wheelEvent = self.handle_wheel_on_image_scroll
        layout_img.addWidget(self.scroll_img)

        # viewing controls
        layout_img_ctrl = QGridLayout()
        layout_img.addLayout(layout_img_ctrl)
        lbl_zoom_static = QLabel('Zoom')
        layout_img_ctrl.addWidget(lbl_zoom_static, 0, 0)
        self.sl_zoom = QSlider(cw)
        self.sl_zoom.setOrientation(Qt.Orientation.Horizontal)
        self.sl_zoom.setMinimum(25)
        self.sl_zoom.setMaximum(800)
        self.sl_zoom.setValue(100)
        self.sl_zoom.valueChanged.connect(self.update_image_label)
        layout_img_ctrl.addWidget(self.sl_zoom, 0, 1)
        self.lbl_zoom = QLabel('100%')
        layout_img_ctrl.addWidget(self.lbl_zoom, 0, 2)

        lbl_brightness_static = QLabel('Brightness')
        layout_img_ctrl.addWidget(lbl_brightness_static, 1, 0)
        self.sl_brightness = QSlider(cw)
        self.sl_brightness.setOrientation(Qt.Orientation.Horizontal)
        self.sl_brightness.setMinimum(0)
        self.sl_brightness.setMaximum(300)
        self.sl_brightness.setValue(100)
        self.sl_brightness.valueChanged.connect(self.update_image_label)
        layout_img_ctrl.addWidget(self.sl_brightness, 1, 1)
        self.lbl_brightness = QLabel('100%')
        layout_img_ctrl.addWidget(self.lbl_brightness, 1, 2)

        # content controls
        lbl_img_display = QLabel(cw)
        lbl_img_display.setText('Mode')
        lbl_img_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_img_ctrl.addWidget(lbl_img_display, 2, 0)

        self.tabs_img_ctrl = QTabWidget(cw)
        self.tabs_img_ctrl.currentChanged.connect(self.update_image_label)
        self.tabs_img_ctrl.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum))
        layout_img_ctrl.addWidget(self.tabs_img_ctrl, 2, 1)

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

        # save image
        self.btn_save_img = QPushButton('Save\nImage')
        self.btn_save_img.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))
        layout_img_ctrl.addWidget(self.btn_save_img, 2, 2)
        self.btn_save_img.pressed.connect(self.handle_action_export_image)


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

        layout_spectra_ctrl.addWidget(QLabel('Database'), 0, 0)
        self.le_db_path = QLineEdit(self.config.default_db_path)
        self.le_db_path.setEnabled(False)
        layout_spectra_ctrl.addWidget(self.le_db_path, 0, 1)
        self.btn_change_db = QPushButton('change..')
        self.btn_change_db.pressed.connect(self.handle_action_set_db_dir)
        layout_spectra_ctrl.addWidget(self.btn_change_db, 0, 2)



        # comparison / x-range control
        layout_spectra_ctrl.addWidget(QLabel('Comparison'), 1, 0)
        layout_compare_ctrl = QHBoxLayout()
        layout_spectra_ctrl.addLayout(layout_compare_ctrl, 1, 1)

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

        layout_compare_ctrl.addWidget(self.rs_xrange)

        self.cb_squared = QCheckBox(self)
        self.cb_squared.setText('squared errors')
        self.cb_squared.setChecked(True)
        self.cb_squared.stateChanged.connect(self.update_spectrum_plot)
        self.cb_squared.stateChanged.connect(self.set_recompute_errmap_flag)
        layout_compare_ctrl.addWidget(self.cb_squared)

        self.cb_gradient = QCheckBox(self)
        self.cb_gradient.setText('compare gradients')
        self.cb_gradient.setChecked(True)
        self.cb_gradient.stateChanged.connect(self.update_spectrum_plot)
        self.cb_gradient.stateChanged.connect(self.set_recompute_errmap_flag)
        layout_compare_ctrl.addWidget(self.cb_gradient)

        # comparison spectra source
        lbl_source = QLabel('Source')
        lbl_source.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_spectra_ctrl.addWidget(lbl_source, 2, 0)

        self.tabs_spectra_source = QTabWidget(cw)
        self.tabs_spectra_source.currentChanged.connect(self.update_spectrum_plot)
        self.tabs_spectra_source.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum))
        layout_spectra_ctrl.addWidget(self.tabs_spectra_source, 2, 1)

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
        btn_export.setText('Save\nSpectrum')
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

        action_load_data = menu_file.addAction('&Set database...')
        action_load_data.triggered.connect(self.handle_action_set_db_dir)

        action_save_img = menu_file.addAction('&Save current image...')
        action_save_img.triggered.connect(self.handle_action_export_image)

        action_export_spectrum = menu_file.addAction('&Save selected spectrum...')
        action_export_spectrum.triggered.connect(self.handle_action_export_spectrum)

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
            img_startup.load('startup.png')
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

    def set_recompute_pca_flag(self):
        self.pca_recompute_flag = True

    def update_image_label(self):
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
            ref_x = None
            ref_y = None
            if self.rb_sim_cube.isChecked():
                if self.spectrum_y is not None:
                    ref_x = self.cube.bands
                    ref_y = self.spectrum_y
            elif self.cmb_comparison_ref.currentData() >= 0:
                ref_x = self.db.spectra[self.cmb_comparison_ref.currentData()].x
                ref_y = self.db.spectra[self.cmb_comparison_ref.currentData()].y
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
                self.lbl_img.resize(int(width * scale), int(height * scale))

    def update_spectrum_plot(self):

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
                if self.sb_nspectra.value() > 0:
                    results = self.db.search_spectrum(self.cube.bands,
                                                      self.spectrum_y,
                                                      custom_range=(self.rs_xrange.start(), self.rs_xrange.end()),
                                                      use_gradient=self.cb_gradient.isChecked(),
                                                      squared_errs=self.cb_squared.isChecked())
                    for result in results[:self.sb_nspectra.value()]:
                        self.plot.plot(result['spectrum'].x,
                                       result['spectrum'].y,
                                       label=f"{result['spectrum'].display_string()} (mean err={result['error']:10.3E})",
                                       linewidth=1,
                                       hold=True,
                                       defer_draw=True)

        # Single draw after all plot calls
        self.plot.draw()


    ##################
    # loading HS data
    ##################
    def load_data(self, filename):
        try:
            self.cube = hyper.Cube(filename)
            self.rgb = self.cube.to_rgb()
            self.pca = None
            self.reset_ui()
            self.update_image_label()
            self.rawfile = filename
            self.statusBar().showMessage(f'Loaded: {os.path.basename(filename)} | '
                                         f'{self.cube.ncols} x {self.cube.nrows} px | '
                                         f'{self.cube.nbands} bands.')
            self.sl_zoom.setValue(int(self.width() * self.config.initial_image_width_ratio / self.cube.ncols * 100))

        except Exception as e:
            print("Error loading file: ")
            print(e)

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
    def handle_click_on_image(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.cube is not None:
            self.rubberband_origin = event.pos()
            self.rubberband_selector.setGeometry(QRect(self.rubberband_origin, QSize()))
            self.rubberband_selector.show()
        else:
            event.ignore()
    def handle_move_on_image(self, event):
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

    def handle_release_on_image(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.cube is not None:
                shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

                # Click on hovered selection -> remove it
                if self.hovered_selection_idx is not None and not shift_held:
                    self.selections.pop(self.hovered_selection_idx)
                    self.hovered_selection_idx = None
                    self.selection_counter = len(self.selections)
                    # Sync legacy state from remaining selections
                    self._sync_legacy_state()
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
                    label = f"R{self.selection_counter+1} ({rect.x()},{rect.y()},{rect.width()}x{rect.height()})"
                    sel = self._make_selection('rect', None, rect, spectrum_y, label)
                else:
                    pos_img = self.display_to_data_point(event.pos())
                    if 0 <= pos_img.x() < self.cube.ncols and 0 <= pos_img.y() < self.cube.nrows:
                        print(f"Selected point: ({pos_img.x()}, {pos_img.y()})")
                        spectrum_y = self.cube.data[pos_img.y(), pos_img.x(), :]
                        label = f"P{self.selection_counter+1} ({pos_img.x()},{pos_img.y()})"
                        sel = self._make_selection('point', pos_img, None, spectrum_y, label)
                    else:
                        sel = None

                if sel is not None:
                    self.selections.append(sel)
                    # Sync legacy scalar state for backward compat
                    self._sync_legacy_state()

                # update ui
                self.set_recompute_errmap_flag()
                self.update_image_label()
                self.update_spectrum_plot()
                self.rubberband_selector.hide()
        else:
            event.ignore()

    def _sync_legacy_state(self):
        """Sync scalar point_selection/rect_selection/spectrum_y from selections list."""
        if self.selections:
            last = self.selections[-1]
            self.spectrum_y = last.spectrum_y
            self.point_selection = last.point if last.sel_type == 'point' else None
            self.rect_selection = last.rect if last.sel_type == 'rect' else None
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
        self.rotation_quadrants = (self.rotation_quadrants - 1) % 4
        self.update_image_label()

    def handle_rotate_right_image(self):
        self.rotation_quadrants = (self.rotation_quadrants + 1) % 4
        self.update_image_label()

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
        if self.scroll_img.horizontalScrollBar().maximum() > 0:
            hs_r = self.scroll_img.horizontalScrollBar().value() / self.scroll_img.horizontalScrollBar().maximum()
        else:
            hs_r = 0.5
        if self.scroll_img.verticalScrollBar().maximum() > 0:
            vs_r = self.scroll_img.verticalScrollBar().value() / self.scroll_img.verticalScrollBar().maximum()
        else:
            vs_r = 0.5
        s = event.angleDelta().y() * self.config.scroll_speed
        if s < 0:
            s = -1 / s
        self.sl_zoom.setValue(int(self.sl_zoom.value() * s))
        self.scroll_img.horizontalScrollBar().setValue(int(self.scroll_img.verticalScrollBar().maximum() * hs_r))
        self.scroll_img.verticalScrollBar().setValue(int(self.scroll_img.verticalScrollBar().maximum() * vs_r))

    def handle_action_export_image(self):
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
        if self.spectrum_y is not None:
            source_name_default = self.last_source_name if self.last_source_name else self.dataset_name()
            info_dialog = hyper.SaveSpectrumDialog(self, source_name_default)
            if info_dialog.exec() == QDialog.DialogCode.Accepted:
                info_dialog_result = info_dialog.get_data()
                self.last_source_name = info_dialog_result['source']
                metadata = hyper.Metadata(id=info_dialog_result['id'],
                                          description=info_dialog_result['description'],
                                          source_object=info_dialog_result['source'],
                                          source_file=self.dataset_name(),
                                          source_coordinates=self.selection_str(),
                                          device_info=f"{self.cube.device} / Hyperlyse {self.config.version}",
                                          intensity=info_dialog_result['intensity'])
                spectrum = hyper.Spectrum(self.cube.bands, self.spectrum_y, metadata)
                if self.last_export_dir:
                    dir_default = self.last_export_dir
                elif self.db is not None:
                    dir_default = self.db.root
                else:
                    dir_default = '.'
                filename_default = f"{self.dataset_name()}_{spectrum.metadata.id}_{self.selection_str()}.jdx"
                file_spectrum, _ = QFileDialog.getSaveFileName(None, "Save spectrum", os.path.join(dir_default, filename_default),
                                                               "JCAMP-DX (*.jdx *.dx *jcm);;Plain x,y pairs (*.dpt *.csv *.txt )")
                if file_spectrum:
                    img = np.uint8(self.rgb * (255 / self.rgb.max()))
                    img = self.draw_marker(img)
                    hyper.Database.export_spectrum(file_spectrum,
                                                   spectrum,
                                                   image=img)


    def handle_action_set_db_dir(self):
        db_dir = QFileDialog.getExistingDirectory(self, "Select directory containing reference spectra in jcamp-dx format",
                                                  self.db.root)
        if db_dir:
            self.db.refresh_from_disk(db_dir)
            self.fill_db_spectra_combobox()
            self.le_db_path.setText(self.db.root)


    ###########
    # helpers
    ##########
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

        x_disp = self.m2i(point.x())
        y_disp = self.m2i(point.y())

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

            if sel.sel_type == 'rect':
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

        img = img * (1 - self.config.marker_alpha) + img_marker * self.config.marker_alpha
        return img.astype(np.uint8)

    def selection_coords(self):
        if not self.selections:
            return []
        sel = self.selections[-1]
        if sel.sel_type == 'rect':
            r = sel.rect
            return [r.left(), r.top(), r.width(), r.height()]
        elif sel.sel_type == 'point':
            p = sel.point
            return [p.x(), p.y()]
        else:
            return []

    def selection_str(self):
        return f'({",".join([str(c) for c in self.selection_coords()])})'

    def get_lambda_slider_text(self, layer_idx):
        return '%.1fnm' % self.cube.bands[layer_idx]

    def fill_db_spectra_combobox(self):
        self.cmb_comparison_ref.clear()
        self.cmb_comparison_ref.addItem('(none)', -1)
        if self.db is not None:
            for i, s in enumerate(self.db.spectra):
                self.cmb_comparison_ref.addItem(s.display_string(with_description=True), i)
        self.cmb_comparison_ref.adjustSize()

    def visualize_error_map(self, error_map):
        # invert and map to [0, 1]:
        similarity_map = 1 - (error_map / error_map.max())
        # apply color map
        cm = plt.get_cmap('viridis')
        return cm(similarity_map)[:, :, :3]

    def dataset_name(self):
        if self.rawfile is not None:
            return os.path.splitext(os.path.basename(self.rawfile))[0]