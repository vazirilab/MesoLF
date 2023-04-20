# MesoLF — Mesoscale volumetric light-field imaging

The MesoLF hard- and software solution allows recording from thousands of neurons within volumes of ⌀4 × 0.2 mm, located at up to 350 µm depth in the mouse cortex, at 18 volumes per second and an effective voxel rate of ~40 megavoxels per second.

MesoLF was developed by the [Vaziri Lab](http://vaziria.com/) at [Rockefeller University](https://www.rockefeller.edu/), and is described and demonstrated in the following publication:

Nöbauer*, T., Zhang*, Y., Kim, H. & Vaziri, A.  
[*Mesoscale volumetric light-field (MesoLF) imaging of neuroactivity across cortical areas at 18 Hz.*](https://www.nature.com/articles/s41592-023-01789-z)  
Nature Methods 20(4) (2023). doi:10.1038/s41592-023-01789-z

Auxiliary data (trained models, PSF, demo data; see placement instructions in the Installation section below) is available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7306113.svg)](https://doi.org/10.5281/zenodo.7306113)

Please submit any issues or questions to this repository's [issue tracker](https://github.com/vazirilab/mesolf/issues).

## System requirements
* Software:
  * Matlab-compatible version of Windows or Linux (see https://www.mathworks.com/support/requirements/matlab-system-requirements.html)
  * Matlab R2020a or newer
  * Toolboxes: signal processing, image processing, statistics and machine learning, curve fitting, parallel computing
  * Matlab-compatible CUDA driver (>= 10.1 for Matlab R2020a)
  * Tested on Ubuntu Linux 20.04 with Matlab R2020a, CUDA driver v11, CUDA toolkit v10.1
* Hardware:
  * Matlab system requirements (see https://www.mathworks.com/support/requirements/matlab-system-requirements.html)
  * RAM should be about twice the size of the raw data in one patch. For data from 4-mm MesoLF optical system and 6 x 6 patches, ~128 GB RAM are recommended
  * Matlab-compatible Nvidia GPU (see https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html], >~ 10 GB GPU-RAM recommended
  * Tested on a workstation with two Intel Xeon Gold 6136 CPUs (12 cores each), 256 GB RAM, 4 TB NVMe flash disk, three Nvidia Titan V GPUs with 12 GB GPU-RAM each

## Installation
1. Add folder containing `mesolf.m` and all subfolders to Matlab path
2. Ensure Matlab's `mex` compiler is configured with a compatible compiler, as listed here: https://www.mathworks.com/support/requirements/supported-compilers.html
3. Change directory to subfolder `src/cuda` and run `build.m` to run the mexcuda compiler to build CUDA and C++/MEX binaries. This should result in a file `mex_lfm_convolution_vec.mexa64` (file extension may differ depending on platform)
4. [Download zip file](https://zenodo.org/record/7306113/files/mesolf_cnn_svm_models.zip?download=1) (1 GB) containing trained CNN and SVM models for calcium trace classification, and place the files contained in the zip file into folder `utility/`
5. [Download simulated point spread function file](https://zenodo.org/record/7306113/files/PSFmatrix_lfm2pram_M10_FN12p5_pm200_from-200_to200_zspacing4_Nnum15_lambda520_OSR3.mat?download=1) (.mat, 3.3 GB) and save it to an arbitrary directory, which we refer to below as `<psfdir>`

Typical installation time: 5 minutes. Download times depend on connection speed; should be <1 hour on a reasonably fast connection

## Usage
* See comments in `src/mesolf_set_params.m` for documentation of required and optional arguments
* A complete demo is provided as a Matlab live script, `mesolf_demo.mlx`, which downloads a ~33 GB demo dataset (two patches of ~600 × 600 μm field of view each, 7-minute MesoLF recording at 18 fps from mouse cortex labelled with SomaGCaMP7f, depth range 0–200 μm), sets parameters, runs the full MesoLF pipeline and plots the results for inspection. The demo requires ~60 GB of RAM and took 18 minutes to run on a workstation with two Intel Xeon Gold 6136 3.00GHz CPUs with 12 cores each, 260 GB RAM, and a 1 TB NVMe SSD hard disk. One nVidia TITAN V GPU with 12 GB RAM was used.
* In general, to run the MesoLF pipeline, pass at least the required arguments to the main function, `mesolf()`. Arguments can be passed in two alternative ways:
  1. As a single struct with fields named like the parameters defined in mesolf_set_params.m
  2. As a series of key-value pairs, with keys matching the parameter names defined in mesolf_set_params.m
  
  Example using the first option:
  ```
  % Prepare parameter struct
  myparams = struct;
  
  % Required arguments
  myparams.indir = '/disk1/dataset1_raw';
  myparams.outdir = '/disk1/dataset1_mesolf';
  myparams.psffile = '<psfdir>/PSFmatrix_lfm2pram_M10_FN12p5_from-100_to100_zspacing4_Nnum15_lambda520_OSR3.mat';
  myparams.frames_x_offset = 1234.567;
  myparams.frames_y_offset = 2345.678;
  myparams.frames_dx = 15.123;
  
  % Some optional arguments
  myparams.frameRate = 18;  % frame rate of raw data
  myparams.n_patches = 6;  % number of tiles per side to divide FOV into
  myparams.patch_mode = 'carpet';  % tile FOV into a "carpet" of n_patches by n_patches tiles
  myparams.filt_method = 'cnn';  % use pre-trained CNN for final trace classification
  myparams.gpu_ids = [1 2];  % use GPUs 1 and 2

  % Run MesoLF
  mesolf(myparams)
  ```
  * Parameters `frames_x_offset`, `frames_y_offset`, `frames_dx` (central microlens offets in x and y and microlens pitch, all in units of pixels) can be conveniently determined using the LFDisplay software published by the Levoy lab at Stanford University: http://graphics.stanford.edu/software/LFDisplay/
  * Parameter `indir` should point to a folder containing a series of .tif files containing one frame of raw data each. Files will be read in in alphabetic order.
  * Replace `<psfdir>` with the path to the directory containing the PSF file.
