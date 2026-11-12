# Code review — final submitted sources

## Python application (`Python_FaceAssymmetry.py`)

Source features directly confirmed in v62 include: OAK camera acquisition with v2/v3 DepthAI branches, RGB/depth frame guards, calibrated deprojection, 21 predefined MediaPipe facial labels, optional dense FaceMesh vertices, experimental three-region H/V/D ASI, legacy calculation modes, a Kivy UI, temporal smoothing, local CSV packet save/load, one-file scan exports, and internal regression tests.

**Checks:** Python syntax check **passed** and the script's full `--self-test` suite **passed** in this environment. This includes 3D geometry regression cases, partial/missing data handling, 90-frame CSV export/reload tests, and simulated IR-camera controls. The tests do not prove real-hardware functionality.

**Source preservation:** The public version preserves the supplied Python algorithms, thresholds, callbacks, hardware parameters and exports. Only the command example in the opening docstring was updated to match the requested renamed file.

**Publication observations:** The default automatic export path is `/home/face/FaceASIScope/results` and may require machine-specific permissions. The Python and MATLAB calculations are **related but not numerically guaranteed identical**, because of documented differences in reference axes and validity gates. The source contains heuristic expression labels, not a trained validated classifier.

## MATLAB application (`MATLAB_FaceAssymmetry.m`)

The final source has the updated GUI positioning from the prior cleaning pass and includes 3D point cloud/landmark displays, selected A–B distances, regional scores, optional midline contributions, moving means, and CSV-derived audit exports.

**Public release alteration:** The earlier public-release packaging removed the embedded face scan and demo-only loader, replacing `UseEmbedded` with a descriptive error. In this updated package, the MATLAB entry function and the internal Open-scan callback reference were also renamed to match the requested filename. The numerical ASI/geometry callbacks and the `readScanCSV` importer remain unchanged. The private full original is not part of the public repository.

**Not verified:** MATLAB runtime, GUI layout on the publication team's display, and comparison of outputs using a real same-scan CSV. This release must receive a MATLAB smoke test before the conference-paper code-availability link is finalized.
