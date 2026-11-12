# ASI measurement definitions — source-derived notes

These descriptions are based on the **included source code**, not on a claim that the method is clinically validated. Measurements are experimental.

## Python v62 (device-side)

`compute_three_region_asi()` builds a local head coordinate system from **Subnasale** (origin), the **Nasion–Subnasale** vertical direction, and three valid eye/brow bilateral reference pairs (medial canthi, lateral canthi, and superciliare proxies). The selected upper pairs are brow proxies, lateral canthi, and medial canthi; lower pairs are alare and cheilion. Zygoma is not selected by default.

For each valid left/right pair, with local coordinates `U`, `V`, `D`, define:

- `width = |U_L| + |U_R|`
- `dH = ||U_L| - |U_R||`
- `dV = |V_L - V_R|`
- `dD = |D_L - D_R|`

For a region, normalize the pooled component sums by pooled pair widths:

```text
H = 100 * sum(dH) / sum(width)
V = 100 * sum(dV) / sum(width)
D = 100 * sum(dD) / sum(width)
ASI = sqrt(H^2 + V^2 + D^2)
```

The score is an **index reported using percentage-style labels in the interface and CSV**, not a validated clinical percent of facial impairment. No clinical cutoff is established. One experimental result can exceed 100. The overall result pools eligible pairs; its coverage must be read with the result.

## MATLAB (analysis-side)

The MATLAB viewer recomputes its own face coordinate basis using a different reference procedure: **Nasion–Subnasale** defines its `V` direction, but the `U` direction averages up to five usable left/right pairs (brow proxies, lateral and medial canthi, alare, and cheilion). The MATLAB validity gates also differ, including minimum widths, face-local depth checks, and regional requirements. By default, the MATLAB score uses 3 upper pairs and 2 lower pairs (zygoma is initially off). MATLAB additionally provides an **optional** separately normalized midline-offset term `M`, so when midline features are selected:

```text
MATLAB TOTAL = sqrt(H^2 + V^2 + D^2 + M^2)
```

The MATLAB **overall** score is set unavailable unless there is at least one valid selected pair from each upper and lower group; the Python implementation can pool available pairs even where one region is missing. The MATLAB public file contains the same computation and CSV import code as the finalized full source; only the embedded example recording was stripped.

## Important comparison limitation

**Do not describe the Python and MATLAB outputs as identical, or directly equate their numerical ASIs, without a documented harmonization/validation study.** Although both use pooled `H`, `V`, and `D` terms, their face basis construction and validity/coverage rules are different. The MATLAB heading distinguishes its recalculated indices from imported device results.

For a conference methods section, describe which implementation produced each result, document which landmark pairs contributed, and state whether filtering or optional midline contributions were enabled. A numerical validation requires known-distance physical objects, repeated scans, and independent anatomical landmark verification; neither the self-tests nor the synthetic demo provide that evidence.
