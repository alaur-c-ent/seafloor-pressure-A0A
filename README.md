**Version:** 0.1  
**Date:** 20 December 2025  

# A0A – Seafloor Pressure Processing Tools

## Overview

This repository contains research codes developed for the processing of seafloor pressure data acquired with A0A (Ambient–Zero–Ambient) pressure gauges.

Each instrument integrates **three pressure sensors**:
- **two external Paroscientific Digiquartz pressure recorders**, measuring
  seafloor pressure,
- **one internal barometric pressure sensor**, measuring the pressure inside a
  sealed atmospheric chamber.

Periodical switches of the external sensors between seawater pressure and the internal atmospheric reference, producing so-called **zero-pressure sequences**. These sequences enable in situ estimation of instrumental drift.

Additional internal and external temperature sensors monitor thermal conditions
of both the seawater and the instrument housing, supporting drift analysis and quality control.

The main objective is to detect vertical seafloor deformation in active submarine environments by processing raw absolute pressure records. This includes:
- extracting and correcting instrumental drift using in situ self-calibration (zero pressure) sequences,
- modelling drift curves using a least square regression.
- validating drift correction through differential pressure (∆P) computed between co-located sensors within the same instrument. 
Oceanic variations  is only partially addressed at this stage.  In the current version, tidal signals are removed using **UTide**, while other oceanographic contributions (currents, etc) are not yet corrected.

The codes are primarily intended for long-term deployments (typically ~12 months). According to RBR Ltd. manufacturer specifications, pressure sensors are capable of resolving millimeter-scale vertical seafloor motions

---
## Scientific context

Seafloor pressure gauges provide a direct proxy for vertical ground motion, with approximately 1 dbar corresponding to 1 m of water column height. However, the detectability of slow or small-amplitude deformation is limited by:
- instrumental drift of quartz pressure sensors,
- instrumental artifacts (e.g. pressure jumps),
- ocean dynamic signals (tides, currents, regional variability, etc).
The A0A method addresses part of these limitations by performing periodic in situ zero-pressure measurements, enabling the estimation of each sensors drift during deployment. Zero-pressure measurements are used to correct raw seafloor pressure signals from drift.

This repository implements processing strategies developed and applied to recent A0A deployments in active submarine settings, including:
- long-term monitoring in the Mayotte region within the framework of the
  **REVOSIMA** observatory,
- deployments along the South-East Indian Ridge east of New Amsterdam Island.
---
## Scope of the repository

This repository includes tools for:

- reading raw A0A pressure, temperature and barometric data,
- identifying and extracting calibration (zero-pressure) sequences from the event-log,
- modelling instrumental drift (least square regression method),
- correcting seafloor pressure records from instrumental drift,
- computing pressure differences between co-located sensors (ΔP),
- preparing cleaned time series for further analysis.

The repository **does not** aim at providing:
- a turnkey operational processing chain,
- real-time processing tools,
- finalised oceanographic corrections (advanced tide or circulation models).

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

Key references underlying the methodology include:

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

The author thanks Pierre Sakic (IPGP) for his  contributions to the development of seafloor geodetic processing tools and for insightful discussions that indirectly supported the work presented in this repository.

---
## Citation

If you use this repository for scientific work, please cite it as:

> Laurent, A., *A0A – Seafloor Pressure Processing Tools*, GitHub repository, 2025.

A more formal citation (DOI) may be added in the future.

Associated scientific article is referenced as :

> - **Laurent, A.**, Tranchant, Y. T., Duvernay, A., Dausse, D., Leconte, J.-M., Hanyuan L., Testut, L., Ballu, V. in review. _Vertical deformation at the seafloor using pressure gauges : impact and mitigation of instrumental drifts and jumps_. Submitted on November, 24, 2025, in _Proceedings of the 2025 IAG scientific assembly._  
---
## Contact

For questions, comments or suggestions about the code, please contact:

- Angèle Laurent (anlaurent@ipgp.fr) 
  (sismo-volcanology, geophysical marine)

For questions about the instrument, please contact :
- Valérie Ballu (valerie.ballu@univ-lr.fr)
- Jean Michel Leconte (oem@rbr-global.com)

