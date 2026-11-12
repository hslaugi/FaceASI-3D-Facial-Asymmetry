# Example data policy

This repository intentionally ships **no real facial scans** or photo/point-cloud datasets.

Generate synthetic regression observations using:

```bash
python python/Python_FaceAssymmetry.py --self-test
```

Or run the interactive `--demo` mode after installing the Kivy UI dependency. The Python-generated demo is explicitly labeled artificial and **must not be described as a participant recording or validation dataset**.

For actual MATLAB import, use one authorized Python scan export generated with `export_matlab_csv()`; keep that file out of this public repository.
