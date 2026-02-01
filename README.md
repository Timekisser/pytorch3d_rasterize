# Pytorch3d Rasterization to Pointcloud

This repository contains code to rasterize 3D mesh models into multi-view RGB-D images using Pytorch3D, and generate point clouds with color and normal.

## Requirements
- Python 3.11
- Pytorch 2.4.1
Other dependencies can be installed via:
```
pip install -r requirements.txt
```

## Usage

```
sh run.sh
```

Parameters in `run.sh` can be modified to fit your needs. Some of the important parameters are:
- `--dataset`: Dataset name, we provide support for ShapeNet and Objaverse
- `--save_file_type`: ['pointcloud', 'image', 'data'], 'pointcloud' saves point clouds as .ply files, 'image' saves multi-view RGB-D and normal images, and data saves point clouds as .npz files.
- `--camera_mode`: Camera mode for rendering, can be 'Perspective' or 'Orthographic'