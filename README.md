# FaceASI — 3D Facial Asymmetry Measurement and Visualization

FaceASI is an experimental workflow for capturing facial RGB and stereo-depth data with a Luxonis OAK camera, extracting MediaPipe face landmarks, calculating three-dimensional facial asymmetry measurements, exporting scan data, and reviewing the measurements in MATLAB.

The **Python program** provides a touch-friendly capture/review interface, device acquisition, measurements, smoothing, and CSV export. The **MATLAB program** imports the exported scan CSV and provides 3D cloud visualization, landmark-to-landmark distances, frame-based measurements, ASI plots, and related quality-control views.

**Research only:** The project does not establish diagnostic validity, clinical thresholds, millimeter measurement accuracy, or suitability for clinical decision-making.

## Repository contents

```text
FaceASI-3D-Facial-Asymmetry/
├── python/
│   └── Python_FaceAssymmetry.py   
├── matlab/
│   └── MATLAB_FaceAssymmetry.m   # MATLAB scan viewer;
├── docs/
│   ├── METHODS.md               # ASI definitions and Python/MATLAB differences
│   ├── DATA_FORMAT.md           # Python-to-MATLAB long-format CSV contract
│   ├── PRIVACY.md               # Data/consent and release restrictions
│   ├── CODE_REVIEW.md           # Verified checks and outstanding hardware tests
├── examples/
│   └── README.md                # Use synthetic demo
├── scripts/
│   └── verify_sources.py        # Verify source integrity and public-release boundary
├── requirements-test.txt
├── requirements-device.txt
├── LICENSE                     # CC BY-SA 4.0 license and official legal-code link
└── .gitignore
```

## Installation

### Python: tests and demo (no camera required)

Use **Python 3.10 or later** (tested locally with Python 3.11). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-test.txt
python python/Python_FaceAssymmetry.py --self-test
```

The tests exercise synthetic ASI calculations, geometry, import/export, temporal filtering, and mock camera behavior; they **do not** establish compatibility with the physical device.

The optional graphical synthetic demo requires Kivy as well:

```bash
pip install "Kivy==2.3.1"
python python/Python_FaceAssymmetry.py --demo --windowed
```

### Python: real OAK camera

Use a working Raspberry Pi / Linux environment with a compatible OAK-D Pro or supported OAK stereo device, USB/power, drivers, DepthAI, MediaPipe, OpenCV, NumPy, and Kivy. The project's device environment was described as **Kivy 2.3.1**, **MediaPipe 0.10.21**, and **DepthAI v3** (with a v2 adapter in the source). Availability of these packages can vary by OS, Python release, and CPU architecture.

An indicative dependency file is provided at `requirements-device.txt`; validate the versions against the device's existing working environment before installing anything.

```bash
python python/Python_FaceAssymmetry.py --camera-check
python python/Python_FaceAssymmetry.py --windowed
```

For OAK-D Pro, the default IR intensity settings are `--ir-dot 0.35 --ir-flood 0.10`. If the connected hardware does not support these emitters, use the correct device settings rather than treating them as universally compatible.

Common command-line options:

```bash
python python/Python_FaceAssymmetry.py --demo --windowed
python python/Python_FaceAssymmetry.py --frames 90 --pose Neutral
python python/Python_FaceAssymmetry.py --frames 90 --pose Smiling
python python/Python_FaceAssymmetry.py --frames 90 --pose Blinking
python python/Python_FaceAssymmetry.py --export-capture /path/to/capture --export-to /path/to/output-folder
```

The export destination must already exist for `--export-capture`. Scan output defaults to a local results folder; the configured automatic CSV export directory in this program is `/home/face/FaceASIScope/results` and may require changes/permissions on other machines.

### MATLAB: scan review

Use **MATLAB R2020b or later** with base MATLAB. No MATLAB toolbox is listed as a requirement by this source.

```matlab
addpath('matlab');
Results = MATLAB_FaceAssymmetry('ScanCSV', '/path/to/Scan_001_Neutral.csv');
```

Or call `MATLAB_FaceAssymmetry` with no arguments to browse for a scan CSV. The CSV must be the **single long-format FaceASI export**, not `cloud.csv`, `landmarks.csv`, or `burst_asi_frames.csv` from an internal capture folder.

## Workflow

1. Capture a research scan with the Python program using a suitable camera and an approved study protocol.
2. Find the exported `Scan_###_Neutral.csv`, `Scan_###_Smile.csv`, or `Scan_###_Blink.csv` long-format scan file.
3. Import that CSV into MATLAB by using the code above or MATLAB's file dialog.
4. Review measured 3D landmarks, anatomical pair measurements, regional ASI, moving-window summaries, and selected A-to-B distances.
5. Export MATLAB result tables as needed; do not interpret the two programs' scores as numerically identical without harmonizing reference/validity definitions. See `docs/METHODS.md`.

## Validation and reproducibility

- Python syntax was checked, and the final source's built-in `--self-test` **passed** in a non-device environment.
- Included tests use synthetic data and mocked OAK calls. They do not verify real camera accuracy, real-time touchscreen layout, or clinical performance.
- MATLAB was not available in the packaging environment; MATLAB execution, GUI display, and Python-to-MATLAB matching need to be checked by the team before publication.
- The MATLAB release retains the original CSV processing logic; the old embedded demonstration is not distributed.

## Licensing

The repository's original project source code and documentation are provided under the **[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)** license, subject to the scope and exclusions in [`LICENSE`](LICENSE). Reuse requires appropriate credit, a license link, indication of changes, and ShareAlike for adaptations. This license permits commercial reuse; it does not mean a reuse is clinically validated.
