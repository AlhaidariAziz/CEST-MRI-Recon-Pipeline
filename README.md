# CEST MRI Reconstruction Pipeline

A research pipeline for preparing Multi-contrast, multi-dimensional (CEST) MRI data
and reconstructing accelerated 3D acquisitions. The repository provides 
Python implementations for SENSE/L2 and locally low-rank (LLR) reconstruction,
together with GRAPPA and compressed-sensing workflows.

The pipeline supports Siemens raw-data conversion, Cartesian and CAIP
retrospective undersampling, Temporat/Contrast-dependent sampling viarations,
ESPIRiT coil-sensitivity estimation, and CPU/GPU execution with SigPy and CuPy.
Besides, reconstructions comparison and CEST Analysis Notebooks. 

## Reconstruction methods

| Method | Entry point | Status |
| --- | --- | --- |
| SENSE / L2 | `scripts/HybVolSENSE.py` | Primary workflow |
| Locally low rank (LLR) | `scripts/HybVolLLR.py` | Primary workflow |
| GRAPPA | `scripts/HybVolGRAPPA.py` | Experimental |
| Compressed sensing | `scripts/HybVolCS.py` | Experimental |

## Repository layout

```text
CEST-MRI-Recon-Pipeline/
├── env/       Conda environment definition
├── kdat/      optional folder for k-space data (Git-ignored)
├── maps/      Generated ESPIRiT sensitivity maps  (Git-ignored)
├── recons/    optional folder for reconstruction results (Git-ignored)
├── sbatch/    SLURM job scripts and templates
└── scripts/   Data preparation, reconstruction, and visualization tools
```

## Configure your paths

Each user must provide paths appropriate for their storage system.
The following shell variables make the commands in this README easier to reuse:

```bash
export PROJECT_DIR="/path/to/CEST_Recon_Pipeline"
export DATA_DIR="$PROJECT_DIR/kdat"
export OUTPUT_DIR="$PROJECT_DIR/recons"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
```

These variables are shell conveniences; the Python scripts do not read them
automatically. Pass input files explicitly with `--data`.

The repository folders `kdat` and `recons` are placeholders only.
Users working with large data on another filesystem can instead set
`DATA_DIR` and `OUTPUT_DIR` to their own storage paths. Reproducing the original
experiments requires access to the complementary data submitted with the
project.

## Environment setup

Create and activate the supplied Conda environment:

```bash
cd "$PROJECT_DIR"
conda env create -f env/LLRv3_30052026.yml
conda activate LLRv3
```

The main software dependencies include SigPy, CuPy/CUDA, h5py, NumPy,
twixtools, PyTorch, and psutil. GPU reconstruction requires a working
CUDA installation compatible with CuPy.

## Data conventions

The scripts operate on HDF5 files. Depending on the acquisition and processing
stage, the files may contain:

- `kdat_01`, `kdat_02`, ...: k-space data subsets
- `ref` or `refs`: reference/calibration data
- `W`: optional prewhitening matrix
- `mps`: coil-sensitivity maps


## Workflows

There are two reconstruction workflows, depending on whether sensitivity maps
are calculated directly from 3-D k-space or separately for each 2-D k-space
after conversion to hybrid space. Workflow 1 is the primary path implemented
and evaluated in this project.

### Workflow 1: sensitivity maps from 3-D k-space

```text
Siemens .dat
    |
    v
CEST_Data_Handling.py
    |
    v
3-D HDF5 k-space
    |
    +--> optional Retro_us.py
    |
    +--> mps.py --space ksp ---------> 3-D k-space sensitivity maps
    |
    v
Hyb_ifft.py
    |
    v
Hybrid-space k-data + 3-D k-space sensitivity maps
    |
    v
HybVolSENSE.py or HybVolLLR.py
```

### Workflow 2: sensitivity maps from hybrid space

```text
Siemens .dat
    |
    v
CEST_Data_Handling.py
    |
    v
3-D HDF5 k-space
    |
    +--> optional Retro_us.py
    |
    v
Hyb_ifft.py
    |
    v
Hybrid-space k-data
    |
    v
mps.py --space Hyb
    |
    v
One sensitivity-map set for each 2-D k-space
    |
    v
HybVolSENSE.py or HybVolLLR.py
```


```bash
cd "$PROJECT_DIR/scripts"
```

### 1. Convert Siemens data to HDF5

```bash
python CEST_Data_Handling.py \
    --mode map \
    --data "$DATA_DIR/input.dat"
```

`--mode` accepts `map` or `read`. Use `--acs True` for a separate ACS scan and
`--os` when readout oversampling should be retained.

### 2. Apply retrospective undersampling (optional)

Cartesian example:

```bash
python Retro_us.py --type Cart --R 3
```

CAIP example with total acceleration 12 (RPE1=4, RPE2=R2):

```bash
python Retro_us.py --type CAIP --R 12 --RPE2 3
```

Before running these commands, edit `DIR` and `infile` near the beginning of
`Retro_us.py` for the desired dataset. Here, `RPE1 = R / RPE2`; choose values
for which this division is meaningful.

### 3. Estimate coil-sensitivity maps

Sensitivity maps can be generated from an HDF5 file containing merged reference/ACS data:

```bash
python mps.py \
    --merged_refs \
    --space ksp \
    --data "$DATA_DIR/kdat_3D.h5" \
    --w 36
```

For a separate reference/ACS file in the same directory as the k-space data:

```bash
python mps.py \
    --space ksp \
    --data "$DATA_DIR/kdat_3D.h5" \
    --ref_data "ACS_3D.h5" \
    --w 36
```

To estimate maps after the hybrid-space transformation, use the hybrid-space
file and select `Hyb`:

```bash
python mps.py \
    --merged_refs \
    --space Hyb \
    --data "$DATA_DIR/kdat_2D.h5" \
    --w 36
```

`mps.py` options include:

- `--space ksp|Hyb`: calibration space
- `--w`: calibration-region width (default: `24`, recommended: `36`)
- `--kw`: ESPIRiT kernel width (default: `6`)
- `--merged_refs`: read the reference/ACS data from the same input HDF5 file
- `--ref_data`: name of a separate reference file in the same input directory

The generated map is saved under `$PROJECT_DIR/maps/`

### 4. Transform 3-D k-space into hybrid space

```bash
python Hyb_ifft.py \
    --refs \
    --data "$DATA_DIR/kdat_3D.h5"
```

Use `--refs` when reference data are stored in the same HDF5 file. The `--dim`
option selects the inverse-FFT dimension (default for readout dimension: `--dim -1` )


The resulting `kdat_2D.h5` file is saved beside the input file.

Steps 3 and 4 may be reversed when sensitivity maps are to be estimated in
hybrid space.

### 5. Run a SENSE reconstruction

```bash
python HybVolSENSE.py \
    --data "$DATA_DIR/kdat_2D.h5" \
    --mps "mps_example.h5" \
    --i 30 \
    --r 0.001
```

### 6. Run an LLR reconstruction

```bash
python HybVolLLR.py \
    --data "$DATA_DIR/kdat_2D.h5" \
    --mps "mps_example.h5" \
    --i 30 \
    --r 0.01 \
    --blk_shape 1 5 5 \
    --blk_strides 1 1 1 \
    --splits 2
```

The principal LLR options are:

- `--i`: maximum number of iterations (default: `20`)
- `--r`: LLR regularization weight (default: `1e-2`)
- `--blk_shape`: three-dimensional LLR block shape (default: `1 5 5`)
- `--blk_strides`: block strides (default: `1 1 1`)
- `--splits`: number of reconstruction subsets; accepted values are `1`, `2`,
  `4`, or `8`. LLR regularization is coupled over the contrast/temporal
  dimension within each subset.

## Running on SLURM

The files in `sbatch/` are templates for CPU and GPU jobs. Review every
`#SBATCH` option and replace all absolute paths before submission:

```bash
sbatch sbatch/CBJ.sh
sbatch sbatch/GBJ.sh
```

For interactive commands, request resources according to the cluster policy,
load the required modules, activate the `LLRv3` environment, and run the same
commands shown above.


## Results analysis and visualization

- `view_v2.ipynb`: reconstruction visualization
- `Z_v2.ipynb`: CEST Z-spectrum analysis

Split reconstruction outputs may need to be assembled before visualization.
The utilities in `h5.ipynb` demonstrate this process.

## Additional scripts

- `HybVolLLR_v2.py`: parallel implementation for CPU clusters
- `HybVolCS.py`: compressed-sensing reconstruction
- `HybVolGRAPPA.py`: GRAPPA reconstruction; currently uses configured filenames
- `recon_ifft.py`: direct inverse-FFT reconstruction; currently uses configured
  filenames
- `cmd_prompts.tex`: working command notes and experiment examples

Use each script's help output to see its current command-line interface:

```bash
python SCRIPT_NAME.py --help
```

## References

1. Tan Z, Liebig PA, Heidemann RM, Laun FB, Knoll F. Accelerated
   diffusion-weighted magnetic resonance imaging at 7 T: Joint reconstruction
   for shift-encoded navigator-based interleaved echo planar imaging
   (JETS-NAViEPI). *Imaging Neuroscience*. 2024;2:1–15.
   [doi:10.1162/imag_a_00085](https://doi.org/10.1162/imag_a_00085)

2. Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity
   encoding for fast MRI. *Magnetic Resonance in Medicine*.
   1999;42(5):952–962.
   [doi:10.1002/(SICI)1522-2594(199911)42:5<952::AID-MRM16>3.0.CO;2-S](https://doi.org/10.1002/%28SICI%291522-2594%28199911%2942%3A5%3C952%3A%3AAID-MRM16%3E3.0.CO%3B2-S)

3. Griswold MA, Jakob PM, Heidemann RM, et al. Generalized autocalibrating
   partially parallel acquisitions (GRAPPA). *Magnetic Resonance in Medicine*.
   2002;47(6):1202–1210.
   [doi:10.1002/mrm.10171](https://doi.org/10.1002/mrm.10171)

4. Uecker M, Lai P, Murphy MJ, et al. ESPIRiT—an eigenvalue approach to
   autocalibrating parallel MRI: Where SENSE meets GRAPPA. *Magnetic Resonance
   in Medicine*. 2014;71(3):990–1001.
   [doi:10.1002/mrm.24751](https://doi.org/10.1002/mrm.24751)

5. Trzasko J, Manduca A. Local versus global low-rank promotion in dynamic MRI
   series reconstruction. In: *Proceedings of the 19th Annual Meeting of
   ISMRM*; 2011. Abstract 4371.
   [ISMRM abstract](https://archive.ismrm.org/2011/4371.html)

6. Ong F, Lustig M. SigPy: A Python package for high performance iterative
   reconstruction. In: *Proceedings of the 27th Annual Meeting of ISMRM*;
   2019. Abstract 4819.
   [ISMRM abstract](https://cds.ismrm.org/protected/19MProceedings/PDFfiles/4819.html)
