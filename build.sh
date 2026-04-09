#!/bin/bash
# Build standalone executable with PyInstaller
set -e

echo "Installing PyInstaller..."
pip3 install pyinstaller

echo "Building ProbeVisualizer..."
pyinstaller --onefile \
    --name ProbeVisualizer \
    --hidden-import=pyqtgraph.opengl \
    --hidden-import=OpenGL.platform.glx \
    --hidden-import=OpenGL.GL \
    --hidden-import=numpy \
    --hidden-import=matplotlib.backends.backend_qt5agg \
    --collect-submodules=pyqtgraph \
    --collect-all=OpenGL \
    main.py

echo ""
echo "Done! Executable: dist/ProbeVisualizer"
