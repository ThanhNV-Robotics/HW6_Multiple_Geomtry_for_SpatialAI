# HW6 — Multiple View Geometry for Spatial AI ✅

This repository contains the Python and MATLAB solutions for Homework 6 of the "Multiple View Geometry for Spatial AI" course. Each Python file corresponds to a separate question in the homework and is runnable independently.

---

## Project structure 🔧

- `calcErr.py` — photometric error computation (Python)
- `deriveErrAnalytic.py` — analytic derivative of the photometric error (Python)
- `deriveErrNumeric.py` — numeric derivative (Python)
- `doAlignment.py` — image alignment (Python)
- `downscale.py` — utility for downscaling RGB/depth images (Python)
- `rgb/` — example RGB images used by the scripts
- `depth/` — example depth images used by the scripts
- `matlab/` — reference MATLAB implementations of the same problems
- `results/` — example outputs (e.g., downscaled results)
- `HW6_Multiple_View_Geometry.pdf` — Solution summary

---


## How to run ▶️

Each Python file can be run directly from the repository root. Example:

```bash
python calcErr.py
python deriveErrAnalytic.py
python deriveErrNumeric.py
python doAlignment.py
python downscale.py
```

Notes:
- The scripts expect the test images to exist in the `rgb/` and `depth/` folders (provided here). Paths are relative to the repository root.
- Some scripts expect grayscale images (2D arrays). If your image loader returns RGB images, convert to grayscale before passing to the function (e.g., average the RGB channels or use luminance conversion).

---