import torch.utils.data as data
import torch
from nerf.model import NeRF


class ImageRayDataset(data.Dataset):
    """Generate a dataset containing rays from a pinhole camera with the
    specified focal length with an imaging plane of size equal to the
    height and width of the provided images tensor

    Arguments:

    images: torch.Tensor
        a torch tensor of values on [0, 1] representing images to be
        generated by a neural radiance field: [batch, height, width, 3]
    poses: torch.Tensor
        a torch tensor containing poses of a camera represented by a
        transformation in homogeneous coordinates: [batch, 4, 4]
    focal_length: float
        a float representing the focal length of a pinhole camera and
        used to scale the physical size of the imaging plane

    """

    def __init__(self, images, poses, states, focal_length,
                 num_vertical_blocks=8, num_horizontal_blocks=8,
                 num_samples_per_block=2):
        """Generate a dataset containing rays from a pinhole camera with the
        specified focal length with an imaging plane of size equal to the
        height and width of the provided images tensor

        Arguments:

        images: torch.Tensor
            a torch tensor of values on [0, 1] representing images to be
            generated by a neural radiance field: [batch, height, width, 3]
        poses: torch.Tensor
            a torch tensor containing poses of a camera represented by a
            transformation in homogeneous coordinates: [batch, 4, 4]
        states: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects their geometry
        focal_length: float
            a float representing the focal length of a pinhole camera and
            used to scale the physical size of the imaging plane
        num_vertical_blocks: int
            the number of blocks to randomly sample from on a grid spanning
            the height of the image, for every sample in the dataset
        num_horizontal_blocks: int
            the number of blocks to randomly sample from on a grid spanning
            the width of the image, for every sample in the dataset
        num_samples_per_block: int
            the number of ray samples to draw from each block in the image
            typically used to estimate loss statistics per block

        """

        self.images = images
        self.poses = poses
        self.states = states
        self.num_vertical_blocks = num_vertical_blocks
        self.num_horizontal_blocks = num_horizontal_blocks
        self.num_samples_per_block = num_samples_per_block

        # generate all possible rays to cast through the scene
        kwargs = dict(dtype=images.dtype, device=images.device)
        self.rays = NeRF.generate_rays(
            images.shape[1], images.shape[2], focal_length, **kwargs)

        # calculate the size of each rectangular bin of rays per image
        self.horizontal_block_size = (self.images.shape[2] //
                                      self.num_horizontal_blocks)
        self.vertical_block_size = (self.images.shape[1] //
                                    self.num_vertical_blocks)

        # calculate how many samples are needed to cover each image
        self.total_block_area = (self.horizontal_block_size *
                                 self.vertical_block_size)

    def __len__(self):
        """Return the size of the a dataset containing pixels and rays,
        where each example in the dataset represents a single pixel
        and ray cast through the scene in world coordinates

        Returns:

        dataset_size: int
            the integer size of the dataset, which represents the total
            number of pixels in all images in the dataset

        """

        return (self.images.shape[0] *
                self.total_block_area) // self.num_samples_per_block

    def __getitem__(self, idx):
        """Retrieve a single example from the dataset by selecting the
        corresponding pose, pixel, and ray, transforming each ray to world
        coordinates, and returning pairs of rays and pixels

        Arguments:

        idx: int
            the example id in the dataset to load, which encodes the image
            id, and height width location of a pixel

        Returns:

        sample['pixels']: torch.Tensor
            a tensor that represents a pixel to be rendered for the given
            rays by alpha compositing predicted colors in space
        sample['rays']: torch.Tensor
            a tensor that represents the direction of the ray exiting the
            camera in the camera coordinate system

        sample['rays_o']: torch.Tensor
            a single ray location in world coordinates, represented as a
            3-vector and paired with a single ray direction
        sample['rays_d']: torch.Tensor
            a single ray direction in world coordinates, represented as a
            3-vector and paired with a single ray location

        sample['pose_o']: torch.Tensor
            a single camera location in world coordinates, represented as a
            3-vector and paired with a camera viewing direction
        sample['pose_d']: torch.Tensor
            a single camera viewing direction in world coordinates,
            represented as a 3-vector and paired with a camera location

        """

        # randomly sample the locations of rays in the image plane using
        # a stratified sampling procedure using rectangular bins
        block_ray_idx = torch.multinomial(
            torch.ones(self.num_samples_per_block, self.total_block_area),
            self.num_vertical_blocks * self.num_horizontal_blocks,
            replacement=True).to(self.images.device)

        # assign ids to determine which block is being sampled from
        # note that blocks are index according to row-major order
        block_idx = torch.arange(self.num_vertical_blocks *
                                 self.num_horizontal_blocks)\
            .unsqueeze(0).to(self.images.device)

        # check which rays are sampled in each bin along the image width
        block_wx = block_ray_idx % self.horizontal_block_size
        image_wi = block_wx + ((block_idx % self.num_horizontal_blocks) *
                               self.horizontal_block_size)

        # check which rays are sampled in each bin along the image height
        block_hx = block_ray_idx // self.horizontal_block_size
        image_hi = block_hx + ((block_idx // self.num_horizontal_blocks) *
                               self.vertical_block_size)

        # select the image id to sample pixels and rays from
        image_bi = torch.broadcast_to(
            torch.LongTensor([[idx // self.total_block_area]]),
            image_hi.shape).to(self.images.device)

        # select a pixel from the image and a corresponding ray
        pixel = self.images[image_bi, image_hi, image_wi]
        ray = self.rays[image_hi, image_wi]
        pose = self.poses[image_bi]
        state = self.states[image_bi]

        # then transform the given ray to world coordinates
        pose_o, pose_d = pose[..., :3, 3], pose[..., :3, :3]
        rays_o, rays_d = NeRF.rays_to_world_coordinates(ray, pose_o, pose_d)
        return dict(image_bi=image_bi, image_hi=image_hi, image_wi=image_wi,
                    pixels=pixel, states=state, rays=ray,
                    pose_o=pose_o, pose_d=pose_d,
                    rays_o=rays_o, rays_d=rays_d)


class PixelRayDataset(data.Dataset):
    """Generate a dataset containing rays from a pinhole camera with the
    specified focal length with an imaging plane of size equal to the
    height and width of the provided images tensor

    Arguments:

    images: torch.Tensor
        a torch tensor of values on [0, 1] representing images to be
        generated by a neural radiance field: [batch, height, width, 3]
    poses: torch.Tensor
        a torch tensor containing poses of a camera represented by a
        transformation in homogeneous coordinates: [batch, 4, 4]
    focal_length: float
        a float representing the focal length of a pinhole camera and
        used to scale the physical size of the imaging plane

    """

    def __init__(self, images, segmentation, poses,
                 focal_length, states_x=None, states_d=None):
        """Generate a dataset containing rays from a pinhole camera with the
        specified focal length with an imaging plane of size equal to the
        height and width of the provided images tensor

        Arguments:

        images: torch.Tensor
            a torch tensor of values on [0, 1] representing images to be
            generated by a neural radiance field: [batch, height, width, 3]
        poses: torch.Tensor
            a torch tensor containing poses of a camera represented by a
            transformation in homogeneous coordinates: [batch, 4, 4]
        focal_length: float
            a float representing the focal length of a pinhole camera and
            used to scale the physical size of the imaging plane
        states_x: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects density and color
        states_d: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects only the color

        """

        self.images = images
        self.segmentation = segmentation
        self.poses = poses
        self.states_x = states_x
        self.states_d = states_d

        # generate all possible rays to cast through the scene
        kwargs = dict(dtype=images.dtype, device=images.device)
        self.rays = NeRF.generate_rays(images.shape[1], images.shape[2],
                                       focal_length, **kwargs)

    def __len__(self):
        """Return the size of the a dataset containing pixels and rays,
        where each example in the dataset represents a single pixel
        and ray cast through the scene in world coordinates

        Returns:

        dataset_size: int
            the integer size of the dataset, which represents the total
            number of pixels in all images in the dataset

        """
        return (self.images.shape[0] *
                self.images.shape[1] * self.images.shape[2])

    def __getitem__(self, idx):
        """Retrieve a single example from the dataset by selecting the
        corresponding pose, pixel, and ray, transforming each ray to world
        coordinates, and returning pairs of rays and pixels

        Arguments:

        idx: int
            the example id in the dataset to load, which encodes the image
            id, and height width location of a pixel

        Returns:

        sample['pixels']: torch.Tensor
            a tensor that represents a pixel to be rendered for the given
            rays by alpha compositing predicted colors in space
        sample['rays']: torch.Tensor
            a tensor that represents the direction of the ray exiting the
            camera in the camera coordinate system

        sample['rays_o']: torch.Tensor
            a single ray location in world coordinates, represented as a
            3-vector and paired with a single ray direction
        sample['rays_d']: torch.Tensor
            a single ray direction in world coordinates, represented as a
            3-vector and paired with a single ray location

        sample['pose_o']: torch.Tensor
            a single camera location in world coordinates, represented as a
            3-vector and paired with a camera viewing direction
        sample['pose_d']: torch.Tensor
            a single camera viewing direction in world coordinates,
            represented as a 3-vector and paired with a camera location

        """

        # select the column of the image to select pixels from
        image_wi = idx % self.images.shape[2]
        idx = idx // self.images.shape[2]

        # select the row of the image to select pixels from
        image_hi = idx % self.images.shape[1]
        idx = idx // self.images.shape[1]

        # select the image id to select pixels from
        image_bi = idx % self.images.shape[0]

        # select a pixel from the image and a corresponding ray
        pixel = self.images[image_bi, image_hi, image_wi]
        label = self.segmentation[image_bi, image_hi, image_wi]
        ray = self.rays[image_hi, image_wi]
        pose = self.poses[image_bi]

        # select the state if provided in the original dataset
        state_x = (self.states_x[image_bi]
                   if self.states_x is not None
                   else torch.zeros(0, device=pixel.device))
        state_d = (self.states_d[image_bi]
                   if self.states_d is not None
                   else torch.zeros(0, device=pixel.device))

        # then transform the given ray to world coordinates
        pose_o, pose_d = pose[:3, 3], pose[:3, :3]
        rays_o, rays_d = NeRF.rays_to_world_coordinates(ray, pose_o, pose_d)
        return dict(image_wi=torch.LongTensor([image_wi]).to(pixel.device),
                    image_hi=torch.LongTensor([image_hi]).to(pixel.device),
                    image_bi=torch.LongTensor([image_bi]).to(pixel.device),
                    states_x=state_x, states_d=state_d,
                    pixels=pixel, label=label, rays=ray,
                    pose_o=pose_o, pose_d=rays_d,
                    rays_o=rays_o, rays_d=rays_d)
