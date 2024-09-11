MUSTER: Multi-Session Temporal Registration
============================================

MUSTER is a Python package designed to perform **longitudinal registration** of images over multiple time points. The primary objective of the package is find the deformations that best describes the changes in the images over the time points. The package is designed to work with medical images, such as MRI and CT scans, but can be used with any type of image data that are in 3D.


Key Features
------------

- **Registration**: Core class for performing registration of image time series.  
- **Multiple image similarity metrics** including normalized cross-correlation (NCC), L2 norm, and gradient-based metrics.
- **Fast and accurate**: MUSTER is designed to be fast and accurate, and can be used for large datasets.

Installation
============

Installing the package is as simple as:

```bash
pip install pymuster
```

