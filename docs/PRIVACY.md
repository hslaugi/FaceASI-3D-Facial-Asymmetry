# Privacy, consent and human data

The **original finalized MATLAB file includes a 30-frame human facial dataset embedded directly inside the `.m` source**. Its content includes numerical anatomical landmarks and a compressed cloud/RGB payload. A source file that contains encoded or compressed face data is still a human-data file. A pseudonym such as “Anonymous” does not make a biometric scan safe to publish.

To avoid accidentally committing that recording, the MATLAB file in **this public GitHub package** retains the full ASI/measurement/GUI/CSV-import logic while omitting the embedded scan and its demonstration loader. Calling it with `UseEmbedded=true` now produces a clear error; normal `ScanCSV` imports continue to be supported.

Before publishing scans, screenshots, identifiable face images, facial point clouds, or participant metadata, obtain appropriate written permissions and follow the research team's institutional, sponsor, and consent/IRB requirements where applicable. Strip subject identifiers from screenshots and example filenames. Even anonymized 3D facial geometry may remain identifiable.

The `.gitignore` excludes common CSV/MAT/NPY/image/video outputs and capture directories, but `.gitignore` is not a privacy guarantee. Check staged files with `git status` and `git diff --cached --name-only` before every push, including history and GitHub Release attachments. Once sensitive data has been pushed, deleting the file in a later commit does not erase it from Git history.

The original unmodified MATLAB file should be kept only in restricted internal storage unless all publication rights and privacy/consent conditions have been cleared.
