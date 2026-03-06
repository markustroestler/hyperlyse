from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QSizePolicy, QDialog, QFormLayout, QLabel, QLineEdit, QComboBox, QDialogButtonBox
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