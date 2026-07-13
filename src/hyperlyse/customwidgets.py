from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import os
from PyQt6.QtWidgets import (QSizePolicy, QDialog, QFormLayout, QLabel, QLineEdit, QComboBox,
                              QPushButton, QVBoxLayout, QHBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QSpinBox,
                              QFileDialog, QDialogButtonBox, QScrollArea, QWidget, QMessageBox)
from PyQt6.QtCore import QSize, Qt

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0.0
        self.ymax = 1.0

    def set_ranges(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def plot(self,
             x,
             y,
             label='',
             linewidth=2,
             hold=False,
             color=None,
             defer_draw=False):
        if not hold:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            range_color = 'green' if self.xmin < self.xmax else 'red'
            self.ax.axvspan(self.xmin, self.xmax, alpha=0.1, color=range_color)
        kwargs = dict(label=label, linewidth=linewidth)
        if color is not None:
            kwargs['color'] = color
        self.ax.plot(x, y, **kwargs)
        self.ax.set_ylim(self.ymin, self.ymax)
        if label:
            self.ax.legend()
        if not defer_draw:
            self.draw()

    def reset(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.draw()

    def save(self, fileName):
        self.figure.savefig(fileName, transparent=True)


class SaveSpectrumDialog(QDialog):
    def __init__(self, parent, default_object='', header_text='', header_color=None):
        super(QDialog, self).__init__(parent)

        self.setWindowTitle('Enter metadata for this spectrum')
        self._action = 'cancel'

        outer_layout = QVBoxLayout(self)
        header_label = QLabel(header_text or 'Enter metadata for this spectrum', self)
        header_label.setWordWrap(True)
        header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header_styles = ['font-weight: 600;']
        if header_color is not None:
            header_styles.append(f'color: rgb({header_color[0]}, {header_color[1]}, {header_color[2]});')
        header_label.setStyleSheet(' '.join(header_styles))
        outer_layout.addWidget(header_label)

        layout = QFormLayout()
        outer_layout.addLayout(layout)

        self.resize(QSize(300, 100))

        self.le_id = QLineEdit(self)
        self.le_description = QLineEdit(self)
        self.le_source = QLineEdit(self)
        self.le_source.setText(default_object)
        self.cb_intensity = QComboBox(self)
        for intensity in ['(undefined intensity)', 'light', 'medium', 'dark']:
            self.cb_intensity.addItem(intensity)

        layout.addRow(QLabel('Sample ID'), self.le_id)
        layout.addRow(QLabel('Description'), self.le_description)
        layout.addRow(QLabel('Intensity'), self.cb_intensity)
        layout.addRow(QLabel('Source object'), self.le_source)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self.btn_save = QPushButton('Save', self)
        self.btn_skip = QPushButton('Skip', self)
        self.btn_cancel = QPushButton('Cancel', self)

        self.btn_save.clicked.connect(self._accept_save)
        self.btn_skip.clicked.connect(self._accept_skip)
        self.btn_cancel.clicked.connect(self._reject_cancel)

        button_row.addWidget(self.btn_save)
        button_row.addWidget(self.btn_skip)
        button_row.addWidget(self.btn_cancel)
        outer_layout.addLayout(button_row)

    def _accept_save(self):
        self._action = 'save'
        self.accept()

    def _accept_skip(self):
        self._action = 'skip'
        self.done(QDialog.DialogCode.Accepted)

    def _reject_cancel(self):
        self._action = 'cancel'
        self.reject()

    def _browse_db(self):
        path = QFileDialog.getExistingDirectory(
            self, 'Select directory containing reference spectra in jcamp-dx format',
            self.le_db_path.text())
        if path:
            self.le_db_path.setText(path)

    def _browse_cube(self):
        start = self.le_cube_path.text() or '.'
        path = QFileDialog.getExistingDirectory(
            self, 'Select folder containing hyperspectral cubes', start)
        if path:
            self.le_cube_path.setText(path)

    def get_data(self):
        return {
            'id': self.le_id.text(),
            'description': self.le_description.text(),
            'source': self.le_source.text(),
            'intensity': self.cb_intensity.currentText(),
            'action': self._action,
        }


class CubeSelectionDialog(QDialog):
    """Dialog for selecting which analyzed cubes are included in search."""

    def __init__(self, parent, analyzed_cubes, current_include):
        """
        :param analyzed_cubes: list of (display_name, filepath) from list_analyzed_cubes()
        :param current_include: None (all included) or list of filepath strings
        """
        super().__init__(parent)
        self.setWindowTitle('Select cubes for search')
        self._result = current_include
        self._checkboxes = []

        if current_include is not None:
            norm_include = {os.path.normcase(os.path.abspath(p)) for p in current_include}
        else:
            norm_include = None

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Check all / Uncheck all row
        btn_row = QHBoxLayout()
        btn_check_all = QPushButton('Check all')
        btn_uncheck_all = QPushButton('Uncheck all')
        btn_check_all.clicked.connect(self._check_all)
        btn_uncheck_all.clicked.connect(self._uncheck_all)
        btn_row.addWidget(btn_check_all)
        btn_row.addWidget(btn_uncheck_all)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Scrollable list of checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(2)
        content_layout.setContentsMargins(4, 4, 4, 4)

        for name, filepath in analyzed_cubes:
            cb = QCheckBox(name)
            cb.setToolTip(filepath)
            if norm_include is None:
                cb.setChecked(True)
            else:
                cb.setChecked(os.path.normcase(os.path.abspath(filepath)) in norm_include)
            content_layout.addWidget(cb)
            self._checkboxes.append((cb, filepath))

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # OK / Cancel
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        n = len(analyzed_cubes)
        self.resize(QSize(420, min(100 + n * 26 + 60, 520)))

    def _check_all(self):
        for cb, _ in self._checkboxes:
            cb.setChecked(True)

    def _uncheck_all(self):
        for cb, _ in self._checkboxes:
            cb.setChecked(False)

    def _on_accept(self):
        if all(cb.isChecked() for cb, _ in self._checkboxes):
            self._result = None
        else:
            self._result = [fp for cb, fp in self._checkboxes if cb.isChecked()]
        self.accept()

    def get_include(self):
        """None if all cubes included, else list of selected cube filepaths."""
        return self._result


class SettingsDialog(QDialog):
    def __init__(self, parent, config, exclude_cube_file=None):
        super(QDialog, self).__init__(parent)
        self._config = config
        self._search_cube_include = config.search_cube_include
        self._exclude_cube_file = exclude_cube_file

        self.setWindowTitle('Preferences')

        layout = QFormLayout(self)
        self.resize(QSize(500, 280))

        # Database path
        db_row = QHBoxLayout()
        self.le_db_path = QLineEdit(config.default_db_path)
        self.le_db_path.setReadOnly(True)
        self.btn_browse_db = QPushButton('...')
        self.btn_browse_db.setFixedWidth(40)
        self.btn_browse_db.clicked.connect(self._browse_db)
        db_row.addWidget(self.le_db_path)
        db_row.addWidget(self.btn_browse_db)
        layout.addRow(QLabel('Database path'), db_row)

        # Cube folder path
        cube_row = QHBoxLayout()
        self.le_cube_path = QLineEdit(config.cube_folder_path)
        self.le_cube_path.setReadOnly(True)
        self.btn_browse_cube = QPushButton('...')
        self.btn_browse_cube.setFixedWidth(40)
        self.btn_browse_cube.clicked.connect(self._browse_cube)
        cube_row.addWidget(self.le_cube_path)
        cube_row.addWidget(self.btn_browse_cube)
        layout.addRow(QLabel('Cube folder'), cube_row)

        # Sample rate
        self.sp_sample_rate = QSpinBox()
        self.sp_sample_rate.setMinimum(1)
        self.sp_sample_rate.setMaximum(64)
        self.sp_sample_rate.setValue(config.sample_rate)
        layout.addRow(QLabel('Sample rate'), self.sp_sample_rate)

        # Number of hits
        self.cb_num_hits = QComboBox()
        for n in [1, 2, 3, 5, 10]:
            self.cb_num_hits.addItem(str(n), n)
        for i in range(self.cb_num_hits.count()):
            if self.cb_num_hits.itemData(i) == config.num_hits:
                self.cb_num_hits.setCurrentIndex(i)
                break
        layout.addRow(QLabel('Number of hits'), self.cb_num_hits)

        # Search checkboxes
        self.chk_search_db = QCheckBox('Search in reference database')
        self.chk_search_db.setChecked(config.search_in_db)
        layout.addRow('', self.chk_search_db)

        self.chk_search_cubes = QCheckBox('Search in analyzed cubes')
        self.chk_search_cubes.setChecked(config.search_in_cubes)
        layout.addRow('', self.chk_search_cubes)

        self.chk_use_pca = QCheckBox('Use fast search (PCA prefilter)')
        self.chk_use_pca.setChecked(config.use_pca)
        self.chk_use_pca.setToolTip(
            'When enabled, a PCA prefilter narrows each cube to a small set of\n'
            'candidate pixels, which are then re-ranked with the exact metric.\n'
            'Dramatically faster than the brute-force scan (tens of ms vs ~1s per\n'
            'cube) and returns the same top hits.')
        layout.addRow('', self.chk_use_pca)

        self.btn_cube_filter = QPushButton('Manage analyzed cube filter...')
        self.btn_cube_filter.clicked.connect(self._open_cube_filter)
        layout.addRow('', self.btn_cube_filter)

        # Button box
        self.bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        layout.addWidget(self.bb)

    def _browse_db(self):
        path = QFileDialog.getExistingDirectory(
            self, 'Select directory containing reference spectra in jcamp-dx format',
            self.le_db_path.text())
        if path:
            self.le_db_path.setText(path)

    def _browse_cube(self):
        start = self.le_cube_path.text() or '.'
        path = QFileDialog.getExistingDirectory(
            self, 'Select folder containing hyperspectral cubes', start)
        if path:
            self.le_cube_path.setText(path)

    def _open_cube_filter(self):
        folder = self.le_cube_path.text()
        if not folder:
            QMessageBox.information(self, 'No cube folder', 'Set a cube folder first.')
            return
        from hyperlyse import cube_analyzer
        analyzed = cube_analyzer.list_analyzed_cubes(folder, self._config.include_subfolders)
        if self._exclude_cube_file:
            excl_key = cube_analyzer._scene_key_normalized(self._exclude_cube_file)
            analyzed = [(name, fp) for name, fp in analyzed
                        if cube_analyzer._scene_key_normalized(fp) != excl_key]
        if not analyzed:
            QMessageBox.information(self, 'No cubes analyzed',
                                    'No analyzed cubes found in the cube folder.\n'
                                    'Run "Analyze cubes" first.')
            return
        dlg = CubeSelectionDialog(self, analyzed, self._search_cube_include)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._search_cube_include = dlg.get_include()

    def get_data(self):
        return {
            'default_db_path': self.le_db_path.text(),
            'cube_folder_path': self.le_cube_path.text(),
            'sample_rate': self.sp_sample_rate.value(),
            'num_hits': self.cb_num_hits.currentData(),
            'search_in_db': self.chk_search_db.isChecked(),
            'search_in_cubes': self.chk_search_cubes.isChecked(),
            'use_pca': self.chk_use_pca.isChecked(),
            'search_cube_include': self._search_cube_include,
        }


class NoCubesAnalyzedDialog(QDialog):
    """Dialog shown when cube search is enabled but no cubes have been analyzed yet."""

    def __init__(self, parent, cube_folder_path=''):
        super().__init__(parent)
        self.setWindowTitle('No cubes analyzed')
        self._start_analysis = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        msg = QLabel(
            'No cubes have been analyzed yet.\n'
            'Set the cube folder path below and start the analysis to enable cube search.')
        msg.setWordWrap(True)
        layout.addWidget(msg)

        form = QFormLayout()
        path_row = QHBoxLayout()
        self.le_cube_path = QLineEdit(cube_folder_path)
        self.btn_browse = QPushButton('...')
        self.btn_browse.setFixedWidth(40)
        self.btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self.le_cube_path)
        path_row.addWidget(self.btn_browse)
        form.addRow(QLabel('Cube folder'), path_row)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_analyze = QPushButton('Start Analysis')
        self.btn_analyze.setDefault(True)
        self.btn_cancel = QPushButton('Cancel')
        self.btn_analyze.clicked.connect(self._on_start)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_analyze)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.resize(QSize(500, 160))

    def _browse(self):
        start = self.le_cube_path.text() or '.'
        path = QFileDialog.getExistingDirectory(
            self, 'Select folder containing hyperspectral cubes', start)
        if path:
            self.le_cube_path.setText(path)

    def _on_start(self):
        self._start_analysis = True
        self.accept()

    def get_cube_folder_path(self):
        return self.le_cube_path.text()

    def wants_analysis(self):
        return self._start_analysis