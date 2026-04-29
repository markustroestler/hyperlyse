from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QSizePolicy, QDialog, QFormLayout, QLabel, QLineEdit, QComboBox, QPushButton, QVBoxLayout, QHBoxLayout
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

    def get_data(self):
        return {
            'id': self.le_id.text(),
            'description': self.le_description.text(),
            'source': self.le_source.text(),
            'intensity': self.cb_intensity.currentText(),
            'action': self._action,
        }