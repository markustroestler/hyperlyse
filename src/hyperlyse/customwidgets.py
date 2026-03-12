from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QSizePolicy, QDialog, QFormLayout, QLabel, QLineEdit, QComboBox,
                              QDialogButtonBox, QHBoxLayout, QPushButton, QCheckBox, QSpinBox,
                              QFileDialog)
from PyQt6.QtCore import QSize

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
             color=None):
        if not hold:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            span_color = 'green' if self.xmin < self.xmax else 'red'
            self.ax.axvspan(self.xmin, self.xmax, alpha=0.1, color=span_color)
        kwargs = dict(label=label, linewidth=linewidth)
        if color is not None:
            kwargs['color'] = color
        self.ax.plot(x, y, **kwargs)
        self.ax.set_ylim(self.ymin, self.ymax)
        if label:
            self.ax.legend()
        self.draw()

    def reset(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.draw()

    def save(self, fileName):
        self.figure.savefig(fileName, transparent=True)


class SaveSpectrumDialog(QDialog):
    def __init__(self, parent, default_object=''):
        super(QDialog, self).__init__(parent)

        self.setWindowTitle('Enter metadata for this spectrum')

        layout = QFormLayout(self)

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

        self.bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        layout.addWidget(self.bb)

    def get_data(self):
        return {
            'id': self.le_id.text(),
            'description': self.le_description.text(),
            'source': self.le_source.text(),
            'intensity': self.cb_intensity.currentText()
        }


class SettingsDialog(QDialog):
    def __init__(self, parent, config):
        super(QDialog, self).__init__(parent)

        self.setWindowTitle('Preferences')

        layout = QFormLayout(self)
        self.resize(QSize(500, 250))

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

        self.chk_use_pca = QCheckBox('Use fast approximate search (PCA + BallTree)')
        self.chk_use_pca.setChecked(config.use_pca)
        self.chk_use_pca.setToolTip(
            'When enabled, uses PCA dimensionality reduction for faster k-NN search.\n'
            'Builds PCA from extracted features on-the-fly for each cube.\n'
            'Results should be similar to brute-force search.')
        layout.addRow('', self.chk_use_pca)

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

    def get_data(self):
        return {
            'default_db_path': self.le_db_path.text(),
            'cube_folder_path': self.le_cube_path.text(),
            'sample_rate': self.sp_sample_rate.value(),
            'num_hits': self.cb_num_hits.currentData(),
            'search_in_db': self.chk_search_db.isChecked(),
            'search_in_cubes': self.chk_search_cubes.isChecked(),
            'use_pca': self.chk_use_pca.isChecked(),
        }