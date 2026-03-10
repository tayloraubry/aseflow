# aseflow
ASE-based workflow utilities for automated NEB calculations, including interpolation, MACE NEB runs, and optional DFT single-point calculations.

## Installation
Clone the repo and install into the desired Python environment:

git clone <repo-url>  
cd aseflow  
pip install -e .

The environment must contain ASE, MACE, and PyTorch for MACE NEBs.

## Full workflow

Running the full workflow can be executed as follows:

aseneb --interpolate N --neb-config mace.yaml --run-neb mace --final-sp --dft-inputs dft_inputs/

This will:

1. Interpolate images  
2. Run MACE NEB with the settings defined in the mace.yaml  
3. Read final images from neb.traj  
4. Write CONTCAR into each image folder  
5. Copy DFT inputs  
6. Run single-point calculations  

---

## Individual steps

The workflow is modular so individual pieces can be run as follows:

### Interpolation only can be done via:

aseneb --interpolate N

### Running just the interpolation and MACE NEB

aseneb --interpolate 15 --neb-config mace.yaml --run-neb mace

### Runing NEB if images already exist

aseneb --neb-config ../mace.yaml --run-neb mace

### Run final DFT single-point calculations only

aseneb --final-sp --dft-inputs ../dft_inputs/

---
