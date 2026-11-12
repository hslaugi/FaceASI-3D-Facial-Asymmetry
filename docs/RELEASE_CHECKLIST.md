# Before making this GitHub repository public

- [x] Set the repository name and owner: `hslaugi/FaceASI-3D-Facial-Asymmetry`.
- [ ] Confirm publication rights for code inherited from previous project teams and contributions by faculty, students and sponsor(s).
- [x] State the requested **CC BY-SA 4.0 International** license in `LICENSE`, `README.md`, and `ATTRIBUTION.md` (preparation only; this does not establish the team's authority to license others' work).
- [ ] Confirm with all relevant copyright holders that **CC BY-SA 4.0** may be applied to every included original source contribution; check sponsor and institutional rules.
- [ ] Confirm that repository attribution in `ATTRIBUTION.md` and the conference author list correctly acknowledge current and previous contributors.
- [ ] Confirm that any third-party code or assets included in the repository may be legally redistributed under their own terms and are clearly distinguished from CC-licensed material.
- [ ] Confirm that this **public** package contains no embedded face scan or identifiable study/subject files.
- [ ] Run `python python/Python_FaceAssymmetry.py --self-test` on the chosen Python environment.
- [ ] Run `python python/Python_FaceAssymmetry.py --camera-check` on the **real** OAK hardware and confirm synchronized depth and intended IR settings.
- [ ] Run the Kivy UI and collect/review an approved test scan on the actual device.
- [ ] Open the public MATLAB file in MATLAB R2020b+ and import a **permitted** Python long-format scan CSV.
- [ ] Verify the 3D cloud, selection dropdowns, header/button spacing, distance measurements, plots and CSV exports on the actual display.
- [ ] Decide which ASI implementation created the reported conference results. Do not state that Python and MATLAB values match without a direct study.
- [ ] Confirm all quantitative claims and hardware accuracy/clinical-limit statements in the paper match completed validation.
- [ ] Create and upload the repository to the configured GitHub URL and confirm it is public.
- [ ] Freeze a final GitHub release/commit/tag and copy that permanent release URL or DOI into the paper if required.

## One-time GitHub publishing commands

From the extracted `faceasi-conference-github` directory:

```bash
git init
git add .
git status
git commit -m "Initial conference code release: Python v62 and MATLAB ASI viewer"
git branch -M main
git remote add origin https://github.com/hslaugi/FaceASI-3D-Facial-Asymmetry.git
git push -u origin main
```

Create the empty GitHub repository first with **Add README = Off, .gitignore = None, License = None**, since all three are supplied. You can also upload the contents using GitHub’s **Add file → Upload files** page and commit them to `main`, without running Git commands. Confirm licensing, authorship and privacy permissions **before** making the repository public.
