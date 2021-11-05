from nerf.model import NeRF
from nerf.dataset import PixelRayDataset

import pickle as pkl
import numpy as np
import os
import io
import argparse
import json

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-dir",
                        type=str, default="experiment")
    parser.add_argument("--data",
                        type=str, default="examples/data_for_nerf.pkl")
    parser.add_argument("--epochs",
                        type=int, default=100)
    parser.add_argument("--camera-focal-length",
                        type=float, default=50.0)
    parser.add_argument("--camera-ccd-width",
                        type=float, default=36.0)
    parser.add_argument("--batch-size",
                        type=int, default=1024)
    parser.add_argument("--normalize-position",
                        type=float, default=20.0)
    parser.add_argument("--learning-rate",
                        type=float, default=0.0001)
    parser.add_argument("--near-plane",
                        type=float, default=0.0)
    parser.add_argument("--far-plane",
                        type=float, default=20.0)
    parser.add_argument("--num-samples-per-ray",
                        type=int, default=64)
    parser.add_argument("--density-noise-std",
                        type=float, default=1.0)
    parser.add_argument("--log-interval",
                        type=int, default=1000)

    args = parser.parse_args()

    os.makedirs(args.logging_dir, exist_ok=True)

    with open(os.path.join(
            args.logging_dir, "params.json"), "w") as f:

        json.dump(dict(
            logging_dir=args.logging_dir,
            data=args.data,
            epochs=args.epochs,
            camera_focal_length=args.camera_focal_length,
            camera_ccd_width=args.camera_ccd_width,
            batch_size=args.batch_size,
            normalize_position=args.normalize_position,
            learning_rate=args.learning_rate,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            num_samples_per_ray=args.num_samples_per_ray,
            density_noise_std=args.density_noise_std,
            log_interval=args.log_interval), f)

    with open(args.data, "rb") as f:
        images_poses_states = pkl.load(f)

    images = images_poses_states['images']
    poses = images_poses_states['poses']
    states_x = images_poses_states['states']

    H, W = images[0].shape[:2]
    focal_length = float(W) * (args.camera_focal_length /
                               args.camera_ccd_width)

    images = torch.FloatTensor(images).cuda()
    poses = torch.FloatTensor(poses).cuda()
    states_x = torch.FloatTensor(states_x).cuda() / np.pi

    poses = torch.cat([NeRF.direction_to_rotation_matrix(
        poses[:, 3:]), poses[:, :3].unsqueeze(2)], dim=2)

    test_image = images[-1:]
    test_pose = poses[-1:]
    test_state_x = states_x[-1:]

    images = images[:-1]
    poses = poses[:-1]
    states_x = states_x[:-1]

    dataset = PixelRayDataset(
        images, poses, focal_length, states_x=states_x)

    data_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    nerf = NeRF(normalize_position=args.normalize_position,
                density_inputs=3 + states_x.shape[-1]).cuda()

    nerf_optimizer = optim.Adam(
        nerf.parameters(), lr=args.learning_rate)

    psnrs = []
    iternums = []
    rendered_images = []
    ground_truth_images = []

    iteration = -1
    for epoch in range(args.epochs):
        for batch in data_loader:

            iteration += 1

            pixels = nerf.render_rays(
                batch['rays_o'],
                batch['rays_d'],
                args.near_plane,
                args.far_plane,
                args.num_samples_per_ray,
                states_x=batch['states_x'],
                randomly_sample=True,
                density_noise_std=args.density_noise_std)

            nerf_optimizer.zero_grad()

            loss = ((pixels - batch['pixels'].unsqueeze(1)) ** 2).mean()
            loss.backward()

            nerf_optimizer.step()

            if iteration % args.log_interval == 0:

                with torch.no_grad():

                    test_render = nerf.render_image(
                        test_pose[..., :3, 3],
                        test_pose[..., :3, :3],
                        H,
                        W,
                        focal_length,
                        args.near_plane,
                        args.far_plane,
                        args.num_samples_per_ray,
                        states_x=test_state_x)

                psnr = -10.0 * torch.log(((
                    test_render - test_image) ** 2).mean()) / 2.30258509299

                psnrs.append(psnr.cpu().numpy())
                iternums.append(iteration)
                rendered_images.append(test_render.cpu().numpy())
                ground_truth_images.append(test_image.cpu().numpy())

                torch.save(nerf.state_dict(),
                           os.path.join(args.logging_dir, "model.pth"))

                np.save(os.path.join(args.logging_dir, "psnrs.npy"),
                        np.asarray(psnrs))

                np.save(os.path.join(args.logging_dir, "iternums.npy"),
                        np.asarray(iternums))

                np.save(os.path.join(args.logging_dir, "rendered_images.npy"),
                        np.asarray(rendered_images))

                np.save(os.path.join(args.logging_dir,
                                     "ground_truth_images.npy"),
                        np.asarray(ground_truth_images))
