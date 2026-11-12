# Reproducibility and environment limitations

## Software

- Python: source was labeled **v62.0**. The original Python logic is preserved (only the opening command example was updated to reflect the new filename) in `python/Python_FaceAssymmetry.py`.
- Python test dependencies: NumPy and OpenCV (headless build for CI).
- Camera/UI dependencies described by the source: OAK DepthAI v3 with a v2 compatibility branch; MediaPipe FaceMesh 0.10.21; Kivy 2.3.1.
- MATLAB: the file header states R2020b+ and base MATLAB only.
- An OAK-D Pro or other supported calibrated OAK stereo setup, suitable USB/power, and appropriate permissions are required for actual capture.

## Checks performed at packaging time

1. `python -m py_compile` — **passed**.
2. `python python/Python_FaceAssymmetry.py --self-test` — **passed** using installed NumPy/OpenCV, synthetic geometry, CSV save/load and export, and a **mock** OAK device.
3. Python source compared with the submitted file — **passed** after excluding the single entry-file command example in its docstring.
4. MATLAB measurement logic and CSV importer are preserved; compared with the previous public package, only the function identity, usage comments, and Open-scan callback reference were changed to match the renamed file.

## Not checked in the packaging environment

- MATLAB interpretation, graphical layout, data import, or numeric output with actual CSV scans (MATLAB was unavailable).
- The real OAK camera, onboard IR/projector behavior, USB synchronization, latency, exposure and accuracy.
- Touchscreen/Kivy display: Kivy, MediaPipe and DepthAI were not installed in the test container.
- Independent anatomical placement or any clinical diagnosis.

The provided GitHub Actions workflow repeats the Python syntax and synthetic/mock tests but **does not substitute** for validation on the actual hardware and MATLAB installation.
