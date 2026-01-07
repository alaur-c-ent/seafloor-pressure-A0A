**Version:** 0.1  
**Date:** 20 December 2025  

# A0A – Seafloor Pressure Processing Tools

## Overview

This repository contains research codes developed for the processing of seafloor pressure data acquired with A0A (Ambient–Zero–Ambient) pressure gauges.

Each instrument integrates **three pressure sensors**:
- **two external Paroscientific Digiquartz pressure recorders**, measuring
  seafloor pressure,
- **one internal barometric pressure sensor**, measuring the pressure inside a
  sealed atmospheric cylinder.

This repository implements processing strategies developed and applied to recent A0A deployments in active submarine settings, including:
- long-term monitoring in the Mayotte region within the framework of the
  **REVOSIMA** observatory,
- deployments along the South-East Indian Ridge east of New Amsterdam Island.

The repository provides research-grade tools to process raw A0A pressure records into drift- and tide-corrected time series suitable for geophysical analysis. This includes:
- extracting instrumental drift using in situ self-calibration (zero pressure) sequences,
- modelling drift curves using a least square regression.
- validating drift correction through differential pressure (∆P) computed between co-located sensors within the same instrument. 

Exemples are provided for each step of the applied corretion process from raw (uncorrected) pressure to drift and tide corrected records. 

Oceanic variations is only partially addressed at this stage as only tides signals are removed using **UTide**, while other oceanographic contributions are not yet corrected.

The codes are primarily intended for long-term deployments (typically ~12 months). 


---
## Scientific context

Seafloor pressure gauges provide a direct proxy for vertical ground motion, with approximately 1 dbar corresponding to 1 m of water column height. However, the detectability of slow or small-amplitude deformation is limited by:
- instrumental drift of quartz pressure sensors,
- instrumental artifacts,
- ocean dynamic signals.
The A0A method mitigates part of these limitations by performing periodic in situ zero-pressure measurements, enabling the estimation of each sensors drift during deployment. Zero-pressure measurements (internal vave rotation into the instrument housing) are then used to correct raw seafloor pressure signals from drift.

---
## Scope of the repository

This repository includes tools for:

- reading raw A0A pressure, temperature and barometric data,
- identifying and extracting calibration (zero-pressure) sequences from the event-log,
- modelling instrumental drift (least square regression method),
- correcting seafloor pressure records from instrumental drift,
- computing pressure differences between co-located sensors (ΔP),
- correcting from tides signal,
- preparing cleaned and corrected time series for further analysis.

The repository **does not** aim at providing yet:
- a turnkey operational processing chain,
- real-time processing tools,
- finalised oceanographic corrections (advanced tides or circulation models).

---
## Processing workflow

The processing strategy implemented in this repository follows a modular,
stepwise approach:

- STEP 1 – Parsing and quality control of raw A0A data
  - reading raw pressure, temperature and barometric records,
  - identification of calibration (zero-pressure) sequences,
  - flagging and extraction of valid data segments.
  - `examples/plt_logs_n_flag.py`

- STEP 2 – In situ calibration and drift estimation
  - extraction of calibration sequences (zero-pressure measurements),
  - computation of per sequence calibration values,
  - control by differential pressure (ΔP) signal.
  - `examples/compute_calibrations.py`

- STEP 3 – Correction of pressure records
  - STEP 3.1: modelling of instrumental drift using exponential + linear regression,
  - `examples/fit_drift_curves.py`
  - STEP 3.2: correction of long-term drift on pressure time series,
  - `examples/drift_correction.py`
  - STEP 3.3: removal of tidal signals using harmonic analysis (UTide).
  - `examples/remove_tides.py`

Another example of the data process from raw to drift corrected is available : `examples/example.py`

The exponential + linear drift models implemented here are directly applied
to long-term deployments and subsequently used to correct full-resolution
pressure time series prior to tidal analysis.

The output consists of cleaned, drift and tide corrected pressure records
ready to be published and/or pursue geophysical/oceanogrpahic analysis.

---
## Data policy

No raw scientific data are distributed in this repository.

When examples are provided, they rely on:
- synthetic datasets, or
- reduced / anonymized excerpts for demonstration purposes only.

Users are expected to apply the codes to their own datasets, respecting the data policies of the corresponding projects and institutions.

---
## Methodological notes

Instrumental drift is modelled following approaches commonly used for quartz pressure sensors, combining:

* an exponential term accounting for post-deployment relaxation,
* a linear term representing long-term secular drift,

These formulations are consistent with previous long-term A0A seafloor pressure
study and manufacturer-led validation experiments.

Users are strongly encouraged to:
* visually inspect calibration sequences,
* assess residuals after correction,
* interpret corrected signals in combination with independent observations
  (e.g. GNSS, seismicity, oceanographic data).

---
## References

Key references underlying the A0A pressure gauge and drift-correction methodology include:

* Bürgmann, R., & Chadwell, D. (2014).
  *Seafloor Geodesy.*
  Annual Review of Earth and Planetary Sciences.

* Paros, J. M., & Kobayashi, T. (2015a).
  *Mathematical models of quartz sensor stability.*

* Paros, J. M., & Kobayashi, T. (2015b).
  *Root causes of quartz sensor drift.*

* Wilcock, W. S. D., et al. (2021).
  *A Thirty-Month Seafloor Test of the A-0-A Method for Calibrating Pressure Gauges.*
  Frontiers in Earth Science.

---
## Code status and disclaimer

This repository contains **research-grade code**.

* The code is under active development.
* It is provided **without warranty**.
* It has not been optimized for performance or robustness.

Users are responsible for validating results obtained with these tools.

This work was carried out during a postdoctoral (A.T.E.R) position at La Rochelle Université, in collaboration with researchers from LIENS, the University of Tasmania, and RBR Ltd. (Canada).

Initial developments in seafloor pressure data processing were initiated by Yann-Treden Tranchant during his PhD at La Rochelle Université and are further extended in this repository.

---
## License

This project is released under the GNU General Public License v3.0 (GPL-3.0).

Any redistributed or modified versions of this code must remain open source
under the same license.

---
## Acknowledgements

The author thanks Pierre Sakic (IPGP) for his knowledges about the development of seafloor geodetic processing tools and for insightful discussions that indirectly supported the work presented in this repository.

---
## Citation

If you use this repository for scientific work, please cite it as:

> Laurent, A., *A0A – Seafloor Pressure Processing Tools*, GitHub repository, 2025.

A more formal citation (DOI) may be added in the future.

Associated scientific article is referenced as :

> - **Laurent, A.**, Tranchant, Y. T., Duvernay, A., Dausse, D., Leconte, J.-M., Hanyuan L., Testut, L., Ballu, V. in review. _Vertical deformation at the seafloor using pressure gauges : impact and mitigation of instrumental drifts and jumps_. Submitted on November, 24, 2025, in _Proceedings of the 2025 IAG scientific assembly._  
---
## Contact

For questions, comments or suggestions about the scripts, please contact:

- Angèle Laurent (anlaurent@ipgp.fr) 
  (post-doc researcher at OVPF-IPGP, part of the REVOSIMA consortium)

For questions about the instrument, please contact :
- Valérie Ballu (valerie.ballu@univ-lr.fr)
- Jean Michel Leconte (oem@rbr-global.com)

