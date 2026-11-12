# Python → MATLAB scan CSV interface

`export_matlab_csv()` in `python/Python_FaceAssymmetry.py` exports a **single long-format CSV** using schema `faceasi.scan.v1`. MATLAB's `readScanCSV()` imports that file and uses the `record_type` field to distinguish rows.

| `record_type` | Purpose |
|---|---|
| `frame` | Frame metadata, raw and filtered ASI values, quality information, acquisition metadata |
| `landmark` | Recorded labels (up to 21 automatic IDs plus optional manual trichion), 3D XYZ and pixel coordinates |
| `dense_landmark` | MediaPipe mesh indices with 2D pixels and optional stereo 3D XYZ |
| `cloud` | Recorded camera point-cloud XYZ and RGB values |
| `scan_summary` | Whole-scan ASI means under the source-defined valid-pair subset rule |

The CSV includes identifiers such as `scan_id`, `subject_id`, `frame_index`, landmark ID, and the X/Y/Z coordinates in **millimeters**. `NaN` / blank coordinates are treated as missing; these entries should not be fabricated or silently replaced. `frame_index` is the join key for cloud, landmark, and frame records.

Example filename patterns created by the Python exporter: `Scan_001_Neutral.csv`, `Scan_001_Smile.csv`, `Scan_001_Blink.csv`. Filename naming and the `mode` field use different human-readable labels for some conditions; use `record_type` and frame/scan identifiers rather than deriving mode from a filename.

**Do not use** `landmarks.csv`, `cloud.csv`, or `burst_asi_frames.csv` from the internal packet folder as a substitute for the one-file export. The MATLAB importer requires `schema_version`, `record_type`, `frame_index`, `landmark_id`, and `x_mm/y_mm/z_mm`.

No real subject CSV or scan data is included in this public repository. The Python synthetic demo and self-tests generate non-human test observations programmatically.
