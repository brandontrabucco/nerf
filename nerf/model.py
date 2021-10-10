import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class NeRF(nn.Module):
    """Create a neural network model that learns a Neural Radiance Field
    by mapping positions and camera orientations in 3D space into the
    RBG and density values of a Neural Radiance Field (NeRF)

    Arguments:

    output_init_scale: float
        the scale of the weights in the final layer of each of the
        density and color prediction heads of the model
    hidden_size: int
        an integer that represents the hidden size of each hidden layer
        in each fully connected layer in the model
    x_positional_encoding_size: int
        an integer that represents the number of elements in the
        positional encoding for the xyz positions in space
    d_positional_encoding_size: int
        an integer that represents the number of elements in the
        positional encoding for theta and phi of the camera
    density_inputs: int
        the number of inputs received by the network for conditioning
        density and should typically be set to three
    color_inputs: int
        the number of inputs received by the network for conditioning
        color and should typically be set to three
    color_outputs: int
        the number of outputs made by the network representing color
        and should typically be set to three

    """

    @staticmethod
    def positional_encoding(x, size):
        """Generate a positional embedding of size d_model for a tensor by
        treating the last dimension as a sequence of scalars and using
        their value as the position in a positional embedding

        Arguments:

        x: torch.Tensor
            a float tensor of arbitrary shape that will be embedded
            using a positional encoding along the last dimension

        Returns:

        embedding: torch.Tensor
            a frequency encoding with the same shape as x, but with
            an additional dimension of size d_model appended to the end

        """

        # generate frequency scales for a sequence of sine waves
        idx = torch.arange(0, size // 2).to(dtype=x.dtype, device=x.device)
        frequency_scale = torch.pow(2.0, idx) * np.pi

        # ensure x and frequency_scale have the same tensor shape
        x = x.unsqueeze(-1)
        while len(frequency_scale.shape) < len(x.shape):
            frequency_scale = frequency_scale.unsqueeze(0)

        # generate a sine and cosine embedding for each value in x
        return torch.cat([torch.sin(x * frequency_scale),
                          torch.cos(x * frequency_scale)], dim=-1)

    @staticmethod
    def generate_rays(image_h, image_w, focal_length,
                      dtype=torch.float32, device='cpu'):
        """Generate a ray for each pixel in an image with a particular height
        and width by setting up a pinhole camera and sending out a ray from
        the camera to the imaging plane at each pixel location

        Arguments:

        image_h: int
            an integer that described the height of the imaging plane in
            pixels and determines the number of rays sampled vertically
        image_w: int
            an integer that described the width of the imaging plane in
            pixels and determines the number of rays sampled horizontally
        focal_length: float
            the focal length of the pinhole camera, which corresponds to
            the physical distance from the camera to the imaging plane

        Returns:

        rays: torch.Tensor
            a tensor that represents the directions of sampled rays in the
            coordinate system of the camera, placed at the origin

        """

        # generate pixel locations for every ray in the imaging plane
        kwargs = dict(dtype=dtype, device=device)
        y, x = torch.meshgrid(torch.arange(
            image_h, **kwargs), torch.arange(image_w, **kwargs))

        # convert pixel coordinates to the camera coordinate system
        # y is negated to conform to OpenGL convention
        x = (x - 0.5 * float(image_w - 1)) / focal_length
        y = (y - 0.5 * float(image_h - 1)) / focal_length
        return torch.stack([x, -y, -torch.ones_like(x)], dim=-1)

    @staticmethod
    def direction_to_rotation_matrix(camera_d):
        """Given a batch of orientations represented as vectors in R3, build
        a rotation matrix in SO(3) that maps from rays in the camera
        coordinate system to the orientation specific by camera_d

        Arguments:

        camera_d: torch.Tensor
            a batch of camera directions in world coordinates, represented
            as 3-vectors paired with every ray in the tensor rays

        Returns:

        rotation_r: torch.Tensor
            a batch or ray locations in world coordinates, represented
            as 3-vectors and paired with a batch of ray directions

        """

        # build the direction of the primal axis of the default camera
        # with the a shape that can be broadcast to camera_d.shape
        kwargs = dict(dtype=camera_d.dtype, device=camera_d.device)
        broadcast_shape = [1 for _ in range(len(camera_d.shape[:-1]))]
        default_camera = torch.Tensor([0, 0, -1], **kwargs)
        default_camera = default_camera.reshape(*(broadcast_shape + [3]))

        # calculate the axis of rotation k and angle of rotation a
        # from the default camera to world coordinates
        rotation_k = torch.cross(default_camera, camera_d, dim=-1)
        cos_a = functional.cosine_similarity(default_camera, camera_d, dim=-1)

        # broadcast the rotation angles to the correct tensor shape
        cos_a = cos_a.reshape(*(list(cos_a.shape) + [1, 1]))
        sin_a = torch.sqrt(1.0 - cos_a ** 2)

        # three matrices to build a 'cross product matrix'
        a0 = torch.Tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], **kwargs)
        a1 = torch.Tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], **kwargs)
        a2 = torch.Tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], **kwargs)

        # a matrix that represents a cross product with k
        rotation_k = torch.stack([torch.matmul(rotation_k, a0),
                                  torch.matmul(rotation_k, a1),
                                  torch.matmul(rotation_k, a2)], dim=-1)

        # build an identity matrix of the same shape as rotation_k
        eye_k = torch.eye(3, **kwargs)
        eye_k = eye_k.reshape(*(broadcast_shape + [3, 3]))

        # build a rotation matrix using Rodrigues' rotation formula
        return (eye_k + rotation_k * sin_a +
                torch.matmul(rotation_k, rotation_k) * (1.0 - cos_a))

    @staticmethod
    def rays_to_world_coordinates(rays, camera_o, camera_r):
        """Given a batch of positions and orientations of the camera, convert
        rays to the world coordinate system from camera coordinates, which
        assumes the camera parameters are the same for each scene

        Arguments:

        rays: torch.Tensor
            a batch of rays that have been generated in the coordinate system
            of the camera and will be converted to world coordinates
        camera_o: torch.Tensor
            a batch of camera locations in world coordinates, represented
            as 3-vectors paired with every ray in the tensor rays
        camera_r: torch.Tensor
            a batch of camera rotations in world coordinates, represented
            as a rotation matrix in SO(3) found by Rodrigues' formula

        Returns:

        rays_o: torch.Tensor
            a batch of ray locations in world coordinates, represented
            as 3-vectors and paired with a batch of ray directions
        rays_d: torch.Tensor
            a batch of ray directions in world coordinates, represented
            as 3-vectors and paired with a batch of ray locations

        """

        # the ray origins are the camera origins, and the ray viewing
        # directions are rotated by the given rotation matrix
        return camera_o, (camera_r * rays.unsqueeze(-2)).sum(dim=-1)

    @staticmethod
    def sample_along_rays(rays_o, rays_d,
                          near, far, num_samples, random=True):
        """Generate a set of stratified samples along each ray by first
        splitting the ray into num_samples bins between near and far clipping
        planes, and then sampling uniformly from each bin

        Arguments:

        rays_o: torch.Tensor
            a batch of ray locations in world coordinates, represented
            as 3-vectors and paired with a batch of ray directions
        rays_d: torch.Tensor
            a batch of ray directions in world coordinates, represented
            as 3-vectors and paired with a batch of ray locations
        near: float
            the distance in world coordinates of the near plane from the
            camera origin along the ray defined by rays_d
        far: float
            the distance in world coordinates of the far plane from the
            camera origin along the ray defined by rays_d
        num_samples: int
            the number of samples along each ray to generate by sampling
            uniformly from num_samples bins between clipping planes

        Returns:

        points: torch.Tensor
            num_samples points generated along each ray defined by rays_o
            and rays_d between the near and far clipping planes

        """

        # sample uniformly between the near and far plane and copy the result
        # to the same device with the same data type as rays_o
        reshape_shape = [1 for _ in range(len(rays_o.shape[:-1]))]
        kwargs = dict(dtype=rays_o.dtype, device=rays_o.device)
        samples = torch.linspace(near, far, num_samples, **kwargs)

        # broadcast the sample bins to a compatible shape with rays_o
        samples = samples.reshape(*(reshape_shape + [num_samples]))
        samples = torch.broadcast_to(
            samples, list(rays_o.shape[:-1]) + [num_samples])

        if random:

            # calculate the midpoints between each sample along the ray
            midpoints = 0.5 * (samples[..., 1:] + samples[..., :-1])

            # calculate bounds for uniform distributions over bins
            lower = torch.cat([samples[..., :1],  midpoints], dim=-1)
            upper = torch.cat([midpoints, samples[..., -1:]], dim=-1)

            # sample points along each ray uniformly over the bins
            samples = torch.rand(*samples.shape, **kwargs)
            samples = lower + (upper - lower) * samples

        # generate points in 3D space along each ray using samples
        return (rays_o.unsqueeze(-2) +
                rays_d.unsqueeze(-2) * samples.unsqueeze(-1))

    @staticmethod
    def alpha_compositing_coefficients(points, density_outputs):
        """Generate coefficients for alpha compositing along a set of points
        sampled along each ray in order to produce an rgb image, where the
        volumetric density is evaluated densely at all points

        Arguments:

        points: torch.Tensor
            num_samples points generated along each ray defined by rays_o and
            rays_d between the near and far clipping planes
        density_outputs: torch.Tensor
            the density scalar output by the neural radiance field along each
            point sampled along each ray from the camera

        Returns:

        weights: torch.Tensor
            a set of alpha compositing coefficients generated by aggregating
            volumetric density coefficients across all points for each ray

        """

        # record the distances between adjacent points along each rays
        # where the final distance is infinity because there is one point
        dists = points[..., 1:, :] - points[..., :-1, :]
        dists = functional.pad(torch.linalg.norm(
            dists, dim=-1, keepdim=True), (0, 0, 0, 1), value=1e10)

        # generate coefficients for alpha compositing across each ray
        alpha = torch.exp(-functional.relu(density_outputs) * dists)
        return (1.0 - alpha) * functional.pad(torch.cumprod(
            alpha[..., :-1, :] + 1e-10, dim=-2), (0, 0, 1, 0), value=1.0)

    def __init__(self, density_inputs=3, color_inputs=3, color_outputs=3,
                 hidden_size=128, x_positional_encoding_size=20,
                 d_positional_encoding_size=12):
        """Create a neural network model that learns a Neural Radiance Field
        by mapping positions and camera orientations in 3D space into the
        RBG and density values of a Neural Radiance Field (NeRF)

        Arguments:

        output_init_scale: float
            the scale of the weights in the final layer of each of the
            density and color prediction heads of the model
        hidden_size: int
            an integer that represents the hidden size of each hidden layer
            in each fully connected layer in the model
        x_positional_encoding_size: int
            an integer that represents the number of elements in the
            positional encoding for the xyz positions in space
        d_positional_encoding_size: int
            an integer that represents the number of elements in the
            positional encoding for theta and phi of the camera
        density_inputs: int
            the number of inputs received by the network for conditioning
            density and should typically be set to three
        color_inputs: int
            the number of inputs received by the network for conditioning
            color and should typically be set to three
        color_outputs: int
            the number of outputs made by the network representing color
            and should typically be set to three

        """

        super(NeRF, self).__init__()

        # hyper-parameters that define the network construction
        self.density_inputs = density_inputs
        self.color_inputs = color_inputs
        self.density_outputs = 1
        self.color_outputs = color_outputs

        # record the size of inputs to the network and
        # corresponding positional encodings
        self.d_positional_encoding_size = d_positional_encoding_size
        self.x_positional_encoding_size = x_positional_encoding_size
        self.d_size = (self.density_inputs *
                       self.d_positional_encoding_size)
        self.x_size = (self.color_inputs *
                       self.x_positional_encoding_size)

        # a final fully connected layer that outputs opacity
        self.density = nn.Linear(hidden_size, self.density_outputs)
        self.color = nn.Linear(hidden_size, self.color_outputs)
        self.hidden_size = hidden_size

        # a neural network block that processes position
        self.block_0 = nn.Sequential(
            nn.Linear(self.x_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU())

        # a neural network block with a skip connection
        self.block_1 = nn.Sequential(
            nn.Linear(self.x_size + hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU())

        # a neural network block that processes direction
        self.block_2 = nn.Sequential(
            nn.Linear(self.d_size + hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU())

    def forward(self, x, d):
        """Perform a forward pass on a Neural Radiance Field given a position
        and viewing direction, and predict both the density of the position
        and the color of the position and viewing direction

        Arguments:

        x: torch.Tensor
            an input vector that represents the positions in 3D space the
            NeRF model is evaluated at, using a positional encoding
        d: torch.Tensor
            an input vector that represents the viewing direction the NeRF
            model is evaluated at, using a positional encoding

        Returns:

        density: torch.Tensor
            an positive floating point tensor representing the density of the
            shape at the current location in space independent to view
        shape: torch.Tensor
            an positive floating point tensor on [0, 1] representing the color
            of at the current location in space dependent on view direction

        """

        # embed the position in a high dimensional positional encoding
        x_shape = list(x.shape)
        x_shape[-1] *= self.x_positional_encoding_size
        x_embed = self.positional_encoding(
            x, self.x_positional_encoding_size).reshape(*x_shape)

        # embed the direction in a high dimensional positional encoding
        d_shape = list(d.shape)
        d_shape[-1] *= self.d_positional_encoding_size
        d_embed = self.positional_encoding(
            d, self.d_positional_encoding_size).reshape(*d_shape)

        # perform a forward pass on several fully connected layers
        block_0 = self.block_0(x_embed)
        block_1 = self.block_1(torch.cat([block_0, x_embed], dim=-1))
        block_2 = self.block_2(torch.cat([block_1, d_embed], dim=-1))

        # final activation functions to output density and color
        return self.density(block_1), self.color(block_2)

    def render_rays(self, rays_o, rays_d, near, far, num_samples,
                    random=False, density_noise_std=0.0, pose_limit=6.0):
        """Render pixels in the imaging plane for a batch of rays given the
        specified parameters of the near and far clipping planes by alpha
        compositing colors predicted by a neural radiance field

        Arguments:

        rays_o: torch.Tensor
            a batch of ray locations in world coordinates, represented
            as 3-vectors and paired with a batch of ray directions
        rays_d: torch.Tensor
            a batch of ray directions in world coordinates, represented
            as 3-vectors and paired with a batch of ray locations
        near: float
            the distance in world coordinates of the near plane from the
            camera origin along the ray defined by rays_d
        far: float
            the distance in world coordinates of the far plane from the
            camera origin along the ray defined by rays_d
        num_samples: int
            the number of samples along each ray to generate by sampling
            uniformly from num_samples bins between clipping planes

        Returns:

        pixels: torch.Tensor
            a tensor that represents pixels rendered for the given rays
            by alpha compositing predicted colors in space

        """

        # generate samples along each ray
        kwargs = dict(dtype=rays_d.dtype, device=rays_d.device)
        points = self.sample_along_rays(rays_o, rays_d, near,
                                        far, num_samples, random=random)

        # normalize the viewing direction of the camera and add
        # an additional dimension of ray samples
        rays_d = rays_d / torch.linalg.norm(rays_d,
                                            dim=-1, keepdim=True)
        rays_d = torch.broadcast_to(rays_d.unsqueeze(-2),
                                    [points.shape[0], num_samples, 3])

        # predict a density and color for every point along each ray
        # potentially add random noise to each density prediction
        density, color = self.forward(points / pose_limit, rays_d)
        density += torch.randn(density.shape,
                               **kwargs) * density_noise_std

        # generate alpha compositing coefficients along each ray and
        # and composite the color values onto the imaging plane
        weights = self.alpha_compositing_coefficients(points, density)
        return (torch.sigmoid(color) * weights).sum(dim=-2)

    def render_image(self, pose, image_h, image_w, focal_length,
                     near, far, num_samples, max_chunk_size=1024,
                     random=False, density_noise_std=0.0, pose_limit=6.0):
        """Render an image with the specified height and width using a camera
        with the specified pose and focal length using a neural radiance
        field evaluated densely at points along rays through the scene

        Arguments:

        pose: torch.Tensor
            a batch of camera poses represented as a 4x4 matrix in homogeneous
            coordinates containing the camera position and rotation
        image_h: int
            an integer that described the height of the imaging plane in
            pixels and determines the number of rays sampled vertically
        image_w: int
            an integer that described the width of the imaging plane in
            pixels and determines the number of rays sampled horizontally
        focal_length: float
            the focal length of the pinhole camera, which corresponds to
            the physical distance from the camera to the imaging plane
        near: float
            the distance in world coordinates of the near plane from the
            camera origin along the ray defined by rays_d
        far: float
            the distance in world coordinates of the far plane from the
            camera origin along the ray defined by rays_d
        num_samples: int
            the number of samples along each ray to generate by sampling
            uniformly from num_samples bins between clipping planes

        Returns:

        image: torch.Tensor
            an image tensor with the specified height and width rendered
            using alpha compositing with a neural radiance field

        """

        # generate all possible rays to cast through the scene
        kwargs = dict(dtype=pose.dtype, device=pose.device)
        rays = self.generate_rays(image_h, image_w, focal_length, **kwargs)

        # ensure the rays and pose have compatible shape, by adding a
        # batch size dimension to the rays tensor and spatial size to pose
        rays = torch.broadcast_to(rays.unsqueeze(0),
                                  [pose.shape[0], image_h, image_w, 3])
        pose = torch.broadcast_to(pose.unsqueeze(1).unsqueeze(1),
                                  [pose.shape[0], image_h, image_w, 4, 4])

        # transform the rays to the world coordinate system using pose
        rays_o, rays_d = self.rays_to_world_coordinates(
            rays, pose[..., :3, 3], pose[..., :3, :3])

        # flatten the batch size with the spatial size for batching
        rays_o = torch.flatten(rays_o, start_dim=0, end_dim=2)
        rays_d = torch.flatten(rays_d, start_dim=0, end_dim=2)

        # shard the rendering process in case the image is too large
        # or we are generating many samples along each ray
        image = []
        for rays_o_i, rays_d_i in zip(torch.split(rays_o, max_chunk_size),
                                      torch.split(rays_d, max_chunk_size)):

            # an inner function that wraps the volume rendering process for
            # each pixel, and then append pixels to an image buffer
            image.append(self.render_rays(
                rays_o_i, rays_d_i, near, far, num_samples, random=random,
                pose_limit=pose_limit, density_noise_std=density_noise_std))

        # merge the sharded predictions into a single image and
        # inflate the tensor to have the correct shape
        return torch.cat(image, dim=0).reshape(pose.shape[0],
                                               image_h, image_w, 3)
