# Conference repository packaging notes

This package is prepared for upload to the GitHub repository:
https://github.com/hslaugi/FaceASI-3D-Facial-Asymmetry

## Application files

- `python/Python_FaceAssymmetry.py` — final Python v62 implementation. The only application source modification relative to the submitted Python file is the command example in the opening docstring, changed to match the requested new filename. **No Python algorithm or behavior was changed.**
- `matlab/MATLAB_FaceAssymmetry.m` — final public MATLAB application. Its primary function name, references in usage comments, and its GUI's Open-scan callback call were renamed to match the `.m` file name; all measurement and processing logic was retained. As already documented for the previous public package, the embedded human facial recording and the demo-only loader are omitted; real exported CSV scan import remains available.
- Both filenames intentionally use the spelling **`FaceAssymmetry`**, as requested.

## Repository files

- `CITATION.cff` was removed as requested.
- References in the README, instructions, verification script, and GitHub Actions were updated to the actual file names and GitHub owner/repository.
- `LICENSE` and `ATTRIBUTION.md` specify Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- No real facial scans, identifying photographs, or raw measurements are included.

## Validation and publication boundaries

The Python self-tests are synthetic and use simulated camera behavior; they cannot verify real OAK hardware, MATLAB GUI interaction, or clinical accuracy. MATLAB execution and real-scan interoperability still need to be tested by the research team. Public release of inherited work requires the necessary rights-holder permissions.
