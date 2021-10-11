# Neural Radiance Fields (Re-Implementation)

This repository implements a minimal training and inference package around Neural Radiance Fields (NeRF). See the original paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," by Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng at ECCV 2020.

<div style="background-color: white;"><img src='render.png'/></div>

## Installation

The package may be installed via pip:

```bash
pip install nerf-pytorch
```

## Usage

The NeRF class implements all core functionality for volume rendering of a neural radiance field. You can initialize your own NeRF model and render an image using your (untrained) model with the following snippet.

```python
from nerf.model import NeRF
import torch

# build the NeRF model with default parameters
model = NeRF().cuda()

# select a pose for the camera in homogeneous coordinates
pose = torch.eye(4).unsqueeze(0)

# settings for the pinhole camera
image_h = 100
image_w = 100
focal_length = 130.0

# settings for ray sampling and pose normalization
near = 2.0
far = 6.0
pose_limit = 6.0

# number of points sampled along the ray
num_samples_per_ray = 64
max_chunk_size = 1024

# whether to sample points randomly
random = True

# standard deviation of gaussian noise added to density head
density_noise_std = 1.0

# render the image using volume rendering
with torch.no_grad():

    image = model.render_image(
        pose.cuda(), 
        image_h, 
        image_w, 
        focal_length, 
        near, 
        far, 
        num_samples_per_ray, 
        max_chunk_size=max_chunk_size,
        random=random, 
        density_noise_std=density_noise_std, 
        pose_limit=pose_limit)
```

