import sys
from PyQt5.QtWidgets import QApplication
from probe_visualizer.app import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Probe Data Visualizer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
