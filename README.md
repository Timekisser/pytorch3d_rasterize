# Pytorch3d Rasterization to Pointcloud

This repository contains code to rasterize 3D mesh models into multi-view RGB-D and normal images using Pytorch3D, and generate point clouds with color and normal.

![teaser](asset/teaser.png)


## 1. Installation
This code has been tested on Ubuntu 22.04 with Nvidia 3090 GPUs.

1.1 Install Conda and create a Conda environment
```bash
conda create --name raster2pc python=3.11
conda activate raster2pc
```
1.2 Clone this repository.
```bash
git clone https://github.com/Timekisser/pytorch3d_rasterize.git
cd pytorch3d_rasterize
```
1.3 Install PyTorch=2.4.1 with conda according to the official documentation.
```
pip install torch torchvision torchaudio
```
1.4 Install other dependencies.
```
pip install -r requirements.txt
```
## 2. Prepare Dataset
Download your 3D mesh dataset in /data directory. Now we provide support for [ShapeNet](https://shapenet.org/) and [Objaverse](https://objaverse.allenai.org/) datasets. 
And then, you need to generate the filelist for your dataset. We provide an example filelist for ShapeNet in the `data/filelist/` folder. And we also provide a script to generate the filelist for Objaverse dataset in `data/utils/gen_objaverse_filelist.py`. You can run the script as:
``` 
python data/utils/gen_objaverse_filelist.py
```
If you want to use other datasets, you need to:
1. Define the dataset class in `data/datasets/` folder, as we shown in `dataset/shapenet.py` and `dataset/objaverse.py`.
2. Generate the filelist for your dataset.

## 3. Running
The main code to generate point clouds and multi-view images is in `main.py`. We provide an example bash script `run.sh` to run the code on ShapeNet dataset. You can simply run:
```
sh run.sh
```
to generate point clouds and multi-view images for ShapeNet dataset by default. 

Also, you can modify the parameters provided in `main.py` to fit your needs. 
Here are some important parameters:
- `--dataset`: Dataset name, we provide support for ShapeNet and Objaverse
- `--file_list`: Specifies the filelists to use; accepts multiple filenames .
- `--save_file_type`: ['pointcloud', 'image', 'data'], 'pointcloud' saves point clouds as .ply files, 'image' saves multi-view RGB-D and normal images, and data saves point clouds as .npz files.
- `--camera_mode`: Camera mode for rendering, can be 'Perspective' or 'Orthographic'

More parameters can be found in `main.py`.

## 4. Licence
This repository is released under the MIT License. Please see the LICENSE file for more details.