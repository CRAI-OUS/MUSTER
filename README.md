# MUSTER: Multi Session Temporal Registration
**MUSTER** is a registration tool for the analysis of longitudinal 3D medical images. The tool is built using PyTorch and can therefor take advantage of GPU acceleration. A timeseries of 8 images with a resolution of ``[160, 160, 160]`` typically takes only 2 min. 
## Installation
To install make sure you have python installed and run
```bash
pip install pymuster
```
## Inctructions
**MUSTER** can be used either as a command line tool or as a Python package

### Command Line Interface
After pip installing the package run

```bash
muster registration <in_dir> <out_dir>
```
Where ``<in_dir>`` is a path to a folder of a subject in BIDS format. That is, the folder must contain a subfolder for each session, and the sessions folders must be named such that when they are sorted alphabetically the correct ordering is achieved. ``<out_dir>`` is the output folder, where the same session folder will be created. In each folder the deformation from that session to all the other sessions will be stored. 

### Python Package
The python package is more flexible then the CLI, and enable expert settings. See the incode documentation for details
