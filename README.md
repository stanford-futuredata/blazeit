# BlazeIt

This is the official project page for the BlazeIt project. 

Please read the [paper](https://arxiv.org/abs/1805.01046) for full technical details.


# Requirements

This repository contains the code for the optimization step in the paper. 

You will need the following installed:
- python 3.x
- CUDA, CUDNN
- torch, torchvision, pandas, opencv (with FFMpeg bindings)

Your machine will need at least:
- 300+GB of memory
- 500+GB of space
- A GPU (this has only been tested with NVIDIA P100 and V100)


# Installation

You will need to install the following packages:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge opencv
conda install -c conda-forge pyclipper
```
(`conda` can be replaced with `pip`).

You will also need to `swag` (GitHub [here](https://github.com/stanford-futuredata/swag-python/)) and `blazeit`. 


# Reproducing experiments

*IMPORTANT*: your runtimes may vary depending on the GPU you are using.

1. Download the data below. The data is expected to be in `/lfs/1/ddkang/blazeit/data`.

2. Extract the videos into `npy` files (this is currently required). For exampe, run:
```
python gen_small_vid.py --base_name jackson-town-square --date 2017-12-14
```
in the `scripts` directory.

3. To reproduce the aggregation experiments, in the `aggregation` folder, run
```
mkdir csvs
time python run_counter.py --base_name jackson-town-square \
  --train_date 2017-12-14 --thresh_date 2017-12-16 --test_date 2017-12-17 \
  --objects car --no-load_video --out_csv csvs/jackson-town-square-2017-12-17.csv
```
to generate predicted counts per frame.

Then run
```
time python run_ebs_sampling.py \
  --obj_name car --err_tol 0.01 \
  --base_name jackson-town-square --test_date 2017-12-17 --train_date 2017-12-14
```
to run EBS sampling.


4. To reproduce the limit query experiments, in the `scrubbing` folder, run:
```
time python taipei-scrubbing.py --base_name jackson-town-square \
  --train_date 2017-12-14 --thresh_date 2017-12-16 --test_date 2017-12-17 \
  --objects car --no-load_video --counts 5 --labeler mock-detectron --limit 10
```

# Datasets

We currently have released the `night-street` (i.e., `jackson-town-square`) data. The data is available [here](https://drive.google.com/drive/folders/1riFVI6QZGf8X6lyFphyRighAYMDTAH4Z?usp=sharing).

Please email the first author directly for other datasets.
