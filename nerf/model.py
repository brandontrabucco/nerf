import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class NeRF(nn.Module):
    """Create a neural network model that learns a Neural Radiance Field
    by mapping positions and camera orientations in 3D space into the
    RBG and density values of a Neural Radiance Field (NeRF)

    Arguments:

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
    learn_cdf_for_volume_sampling: bool
        whether to use a learned inverse cdf transform for sampling
        points to evaluate radiance along each ray

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
            a d_model times larger dimension at the end

        """

        # generate frequency scales for a sequence of sine waves
        idx = torch.arange(0, size // 2).to(dtype=x.dtype, device=x.device)
        frequency_scale = torch.pow(2.0, idx) * np.pi

        # ensure x and frequency_scale have the same tensor shape
        x = x.unsqueeze(-1)
        while len(frequency_scale.shape) < len(x.shape):
            frequency_scale = frequency_scale.unsqueeze(0)

        # generate a sine and cosine embedding for each value in x
        embedding = torch.cat([torch.sin(x * frequency_scale),
                               torch.cos(x * frequency_scale)], dim=-1)

        # flatten the features and positional embedding axes
        return torch.flatten(embedding, start_dim=-2, end_dim=-1)

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
        default_camera = torch.Tensor([0, 0, -1]).to(**kwargs)
        default_camera = default_camera.reshape(*(broadcast_shape + [3]))
        default_camera = torch.broadcast_to(default_camera, camera_d.shape)

        # normalize the camera viewing direction to unit norm and calculate
        # the axis of rotation by taking a cross product
        unit_d = camera_d / torch.linalg.norm(camera_d, dim=-1, keepdim=True)
        rotation_k = torch.cross(default_camera, unit_d, dim=-1)

        # calculate the cosine and sine of the angle between the vectors
        # using two identities from linear algebra
        cos_a = (default_camera * unit_d).sum(dim=-1, keepdim=True)
        sin_a = torch.linalg.norm(rotation_k, dim=-1, keepdim=True)

        # normalize the axis of rotation to unit norm and handle when the
        # viewing direction is parallel to the primal camera axis
        fallback_k = torch.Tensor([0, -1, 0]).to(**kwargs)
        fallback_k = fallback_k.reshape(*(broadcast_shape + [3]))
        fallback_k = torch.broadcast_to(fallback_k, rotation_k.shape)
        rotation_k = torch.where(torch.eq(sin_a, 0),
                                 fallback_k,
                                 rotation_k / sin_a).detach()

        # three matrices to build a 'cross product matrix'
        a0 = torch.Tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]]).to(**kwargs)
        a1 = torch.Tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]).to(**kwargs)
        a2 = torch.Tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]]).to(**kwargs)

        # a matrix that represents a cross product with k
        rotation_k = torch.stack([torch.matmul(rotation_k, a0),
                                  torch.matmul(rotation_k, a1),
                                  torch.matmul(rotation_k, a2)], dim=-1)

        # build an identity matrix of the same shape as rotation_k
        # and broadcast the rotation angles to the correct tensor shape
        eye_k = torch.eye(3, **kwargs)
        eye_k = eye_k.reshape(*(broadcast_shape + [3, 3]))
        cos_a, sin_a = cos_a.unsqueeze(-1), sin_a.unsqueeze(-1)

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
    def sample_along_rays(rays_o, rays_d, near, far, num_samples,
                          states_x=None, states_d=None, randomly_sample=True):
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
        states_x: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects density and color
        states_d: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects only the color
        randomly_sample: bool
            whether to randomly sample the evaluation positions along each
            ray or to use a deterministic set of points

        Returns:

        samples: torch.Tensor
            num_samples samples generated along each ray defined by rays_o
            and rays_d between the near and far clipping planes

        """

        # sample uniformly between the near and far plane and copy the result
        # to the same device with the same data type as rays_o
        reshape_shape = [1 for _ in range(len(rays_o.shape[:-1]))]
        kwargs = dict(dtype=rays_o.dtype, device=rays_o.device)
        samples = torch.linspace(0.0, 1.0, num_samples, **kwargs)

        # broadcast the sample bins to a compatible shape with rays_o
        samples = samples.reshape(*(reshape_shape + [num_samples]))
        samples = torch.broadcast_to(
            samples, list(rays_o.shape[:-1]) + [num_samples])

        if randomly_sample:

            # calculate the midpoints between each sample along the ray
            midpoints = 0.5 * (samples[..., 1:] + samples[..., :-1])

            # calculate bounds for uniform distributions over bins
            lower = torch.cat([samples[..., :1],  midpoints], dim=-1)
            upper = torch.cat([midpoints, samples[..., -1:]], dim=-1)

            # sample points along each ray uniformly over the bins
            samples = torch.rand(*samples.shape, **kwargs)
            samples = lower + (upper - lower) * samples

        # scale samples between the near and far clipping planes
        return near + samples * (far - near)

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

    @staticmethod
    def inverse_transform_sampling(samples, weights,
                                   num_samples, randomly_sample=False):
        """Sample points to evaluate the neural radiance field from the
        empirical pdf formed by the weights for alpha compositing assuming
        that many points should be sampled in close proximity to high weights

        Arguments:

        samples: torch.Tensor
            a tensor containing samples along each ray that have been
            evaluated by a neural radiance field
        weights: torch.Tensor
            a tensor representing the probability that a ray terminates at the
            corresponding location given by elements in samples
        num_samples: int
            the number of samples to generate from the empirical pdf induced
            by the alpha compositing weights provided to the function
        randomly_sample: bool
            whether to randomly sample the evaluation positions along each
            ray or to use a deterministic set of points

        Returns:

        fine_samples: torch.Tensor
            num_samples samples generated along each ray defined by rays_o
            and rays_d between the near and far clipping planes

        """

        # use the provided alpha compositing weights to build an empirical
        # distribution over locations along the ray
        pdf = weights / torch.sum(weights, -1, keepdim=True).clamp(min=1e-10)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if not randomly_sample:

            # take samples spaced evenly along the length of the ray
            fine_samples = torch.linspace(0., 1., steps=num_samples)
            fine_samples = fine_samples.expand(list(cdf.shape[:-1]) +
                                               [num_samples])

        else:

            # take samples by uniform random sampling
            fine_samples = torch.rand(list(cdf.shape[:-1]) + [num_samples])

        # invert the empirical cdf by determining which bin of the empirical
        # cdf the new sampled points fall into
        fine_samples = fine_samples.contiguous().to(samples.device)
        sample_indices = torch.searchsorted(cdf, fine_samples, right=True)

        # for each sample determine the index of the upper and lower bounds
        # of the cdf bin to interpolate between them
        below = torch.max(torch.zeros_like(sample_indices - 1),
                          sample_indices - 1)
        above = torch.min((cdf.shape[-1] - 1) *
                          torch.ones_like(sample_indices), sample_indices)

        # prepare to broadcast cdf and samples to the right shape
        bounds_indices = torch.stack([below, above], dim=-1)
        matched_shape = [bounds_indices.shape[0],
                         bounds_indices.shape[1], cdf.shape[-1]]

        # acquire the bounds for each bin of the empirical cdf and also which
        # points along each ray those bounds correspond to
        cdf_bounds = torch.gather(
            cdf.unsqueeze(1).expand(matched_shape), 2, bounds_indices)
        samples_bounds = torch.gather(
            samples.unsqueeze(1).expand(matched_shape), 2, bounds_indices)

        # linearly interpolate between the upper and lower bounds
        # depending on how far through each bin the fine-grain samples are
        denom = (cdf_bounds[..., 1] - cdf_bounds[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        interpolation = (fine_samples - cdf_bounds[..., 0]) / denom
        return samples_bounds[..., 0] + interpolation * (
                samples_bounds[..., 1] - samples_bounds[..., 0])

    def __init__(self, density_inputs=3, color_inputs=3, color_outputs=3,
                 hidden_size=256, x_positional_encoding_size=20, num_stages=2,
                 d_positional_encoding_size=12, normalize_position=1.0):
        """Create a neural network model that learns a Neural Radiance Field
        by mapping positions and camera orientations in 3D space into the
        RBG and density values of a Neural Radiance Field (NeRF)

        Arguments:

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
        learn_cdf_for_volume_sampling: bool
            whether to use a learned inverse cdf transform for sampling
            points to evaluate radiance along each ray

        """

        super(NeRF, self).__init__()

        # hyper-parameters that define the network construction
        self.density_inputs = density_inputs
        self.color_inputs = color_inputs
        self.density_outputs = 1
        self.color_outputs = color_outputs
        self.normalize_position = normalize_position
        self.num_stages = num_stages

        # record the size of inputs to the network and
        # corresponding positional encodings
        self.hidden_size = hidden_size
        self.d_positional_encoding_size = d_positional_encoding_size
        self.x_positional_encoding_size = x_positional_encoding_size
        self.d_size = (self.color_inputs *
                       self.d_positional_encoding_size)
        self.x_size = (self.density_inputs *
                       self.x_positional_encoding_size)

        # a final fully connected layer that outputs opacity
        self.density = nn.ModuleList([
            nn.Linear(hidden_size, self.density_outputs)
            for i in range(self.num_stages)])
        self.color = nn.ModuleList([
            nn.Linear(hidden_size, self.color_outputs)
            for i in range(self.num_stages)])

        # a neural network block that processes position
        self.block_0 = nn.ModuleList([nn.Sequential(
            nn.Linear(self.x_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size))
            for i in range(self.num_stages)])

        # a neural network block with a skip connection
        self.block_1 = nn.ModuleList([nn.Sequential(
            nn.Linear(self.x_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size))
            for i in range(self.num_stages)])

        # a neural network block that processes direction
        self.block_2 = nn.ModuleList([nn.Sequential(
            nn.Linear(self.d_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size))
            for i in range(self.num_stages)])

    def forward(self, rays_x, rays_d, stage=0, states_x=None, states_d=None):
        """Perform a forward pass on a Neural Radiance Field given a position
        and viewing direction, and predict both the density of the position
        and the color of the position and viewing direction

        Arguments:

        rays_x: torch.Tensor
            an input vector that represents the positions in 3D space the
            NeRF model is evaluated at, using a positional encoding
        rays_d: torch.Tensor
            an input vector that represents the viewing direction the NeRF
            model is evaluated at, using a positional encoding
        states_x: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects density and color
        states_d: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects only the color

        Returns:

        density: torch.Tensor
            an positive floating point tensor representing the density of the
            shape at the current location in space independent to view
        shape: torch.Tensor
            an positive floating point tensor on [0, 1] representing the color
            of at the current location in space dependent on view direction

        """

        # embed the position in a high dimensional positional encoding
        x_embed = self.positional_encoding(
            rays_x / self.normalize_position, self.x_positional_encoding_size)

        # embed the direction in a high dimensional positional encoding
        d_embed = self.positional_encoding(rays_d / torch.linalg.norm(
            rays_d, dim=-1, keepdim=True), self.d_positional_encoding_size)

        # concatenate an additional state vector to the density inputs
        if states_x is not None:
            x_embed = torch.cat([self.positional_encoding(
                states_x, self.x_positional_encoding_size), x_embed], dim=-1)

        # concatenate an additional state vector to the density inputs
        if states_d is not None:
            d_embed = torch.cat([self.positional_encoding(
                states_d, self.d_positional_encoding_size), d_embed], dim=-1)

        # perform a forward pass on several fully connected layers
        block_0 = self.block_0[stage](x_embed)
        block_1 = self.block_1[stage](torch.cat([block_0, x_embed], dim=-1))
        block_2 = self.block_2[stage](torch.cat([block_1, d_embed], dim=-1))

        # final activation functions to output density and color
        return self.density[stage](block_1), self.color[stage](block_2)

    def render_rays(self, rays_o, rays_d, near, far, num_samples,
                    states_x=None, states_d=None,
                    randomly_sample=False, density_noise_std=0.0):
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
        states_x: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects density and color
        states_d: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects only the color
        randomly_sample: bool
            whether to randomly sample the evaluation positions along each
            ray or to use a deterministic set of points
        density_noise_std: float
            the standard deviation of gaussian noise added to the density
            predictions made by the model before alpha compositing

        Returns:

        pixels: torch.Tensor
            a tensor that represents pixels rendered for the given rays
            by alpha compositing predicted colors in space

        """

        # generate samples along each ray
        kwargs = dict(dtype=rays_d.dtype, device=rays_d.device)
        samples = previous_samples = self.sample_along_rays(
            rays_o, rays_d, near, far, num_samples, states_x=states_x,
            states_d=states_d, randomly_sample=randomly_sample)

        # normalize the viewing direction of the camera and add
        # an additional dimension of ray samples
        unit_d = torch.broadcast_to(rays_d.unsqueeze(-2),
                                    [rays_d.shape[0], num_samples * 2, 3])
        if states_x is not None:
            states_x = torch.broadcast_to(states_x.unsqueeze(
                -2), [rays_d.shape[0], num_samples * 2, states_x.shape[-1]])
        if states_d is not None:
            states_d = torch.broadcast_to(states_d.unsqueeze(
                -2), [rays_d.shape[0], num_samples * 2, states_d.shape[-1]])

        image_stages = []

        # iterate through every stage from course to fine
        for stage_idx in range(self.num_stages):

            if stage_idx > 0: # only later stages invert the cdf

                # sample points in space from an empirical distribution
                # specified by alpha compositing weights
                new_samples = self.inverse_transform_sampling(
                    0.5 * (samples[..., 1:] + samples[..., :-1]),
                    weights[..., 1:-1, 0],
                    num_samples, randomly_sample=randomly_sample)

                # combine the previously generates ray samples with the
                # newly generated samples
                samples, previous_samples = torch.sort(
                    torch.cat([previous_samples,
                               new_samples], dim=-1), dim=-1)[0], new_samples

            # generate points in 3D space along each ray using samples
            points = (rays_o.unsqueeze(-2) +
                      rays_d.unsqueeze(-2) * samples.unsqueeze(-1))

            # calculate the number of sampes taken along each ray
            max_n = num_samples if stage_idx == 0 else num_samples * 2

            # predict a density and color for every point along each ray
            # potentially add random noise to each density prediction
            density, color = self.forward(
                points, unit_d[:, :max_n], stage=stage_idx,
                states_x=None if states_x is None else states_x[:, :max_n],
                states_d=None if states_d is None else states_d[:, :max_n])
            density += torch.randn(density.shape,
                                   **kwargs) * density_noise_std

            # generate alpha compositing coefficients along each ray and
            # and composite the color values onto the imaging plane
            weights = self.alpha_compositing_coefficients(points, density)
            image_stages.append((weights *
                                 torch.sigmoid(color)).sum(dim=-2))

        # stack the stages of image generation into a tensor where the
        # second to last axis index over each stage of the model
        return torch.stack(image_stages, dim=-2)

    def render_image(self, camera_o, camera_r, image_h, image_w, focal_length,
                     near, far, num_samples, states_x=None, states_d=None,
                     max_chunk_size=1024,
                     randomly_sample=False, density_noise_std=0.0):
        """Render an image with the specified height and width using a camera
        with the specified pose and focal length using a neural radiance
        field evaluated densely at points along rays through the scene

        Arguments:

        camera_o: torch.Tensor
            a batch of camera locations in world coordinates, represented
            as 3-vectors paired with every ray in the tensor rays
        camera_r: torch.Tensor
            a batch of camera rotations in world coordinates, represented
            as a rotation matrix in SO(3) found by Rodrigues' formula
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
        states_x: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects density and color
        states_d: torch.Tensor
            a batch of input vectors that represent the visual state of
            objects in the scene, which affects only the color
        max_chunk_size: int
            the maximum number of elements in a single batch chunk to process
            in parallel, in order to save memory for large images
        randomly_sample: bool
            whether to randomly sample the evaluation positions along each
            ray or to use a deterministic set of points
        density_noise_std: float
            the standard deviation of gaussian noise added to the density
            predictions made by the model before alpha compositing

        Returns:

        image: torch.Tensor
            an image tensor with the specified height and width rendered
            using alpha compositing with a neural radiance field

        """

        # generate all possible rays to cast through the scene
        kwargs = dict(dtype=camera_o.dtype, device=camera_o.device)
        batch_size = camera_o.shape[0]
        rays = self.generate_rays(image_h, image_w, focal_length, **kwargs)

        # ensure the rays and pose have compatible shape, by adding a
        # batch size dimension to the rays tensor and spatial size to pose
        rays = torch.broadcast_to(
            rays.unsqueeze(0), [batch_size, image_h, image_w, 3])

        # broadcast the camera pose to have a spatial size equal to the
        # number of pixels being rendered in the image
        camera_o = torch.broadcast_to(camera_o.unsqueeze(
            1).unsqueeze(1), [batch_size, image_h, image_w, 3])
        camera_r = torch.broadcast_to(camera_r.unsqueeze(
            1).unsqueeze(1), [batch_size, image_h, image_w, 3, 3])

        # broadcast the states of the radiance field to the correct shape
        # accounting for the number of raysd being cast
        if states_x is not None:
            states_x = torch.broadcast_to(states_x.unsqueeze(
                1).unsqueeze(1), [batch_size, image_h,
                                  image_w, states_x.shape[-1]])
        if states_d is not None:
            states_d = torch.broadcast_to(states_d.unsqueeze(
                1).unsqueeze(1), [batch_size, image_h,
                                  image_w, states_d.shape[-1]])

        # transform the rays to the world coordinate system using pose
        rays_o, rays_d = self\
            .rays_to_world_coordinates(rays, camera_o, camera_r)

        # shard the rendering process in case the image is too large
        # or we are generating many samples along each ray
        rays_o = torch.flatten(rays_o, start_dim=0, end_dim=2)
        rays_d = torch.flatten(rays_d, start_dim=0, end_dim=2)
        batched_inputs = [torch.split(rays_o, max_chunk_size),
                          torch.split(rays_d, max_chunk_size)]

        # if visual states are provided that also create shards for
        # these tensors and add them to the batch
        if states_x is not None:
            batched_inputs.append(torch.split(torch.flatten(
                states_x, start_dim=0, end_dim=2), max_chunk_size))
        if states_d is not None:
            batched_inputs.append(torch.split(torch.flatten(
                states_d, start_dim=0, end_dim=2), max_chunk_size))

        # an inner function that wraps the volume rendering process for
        # each pixel, and then append pixels to an image buffer
        image = [self.render_rays(
            rays_o_i, rays_d_i, near, far, num_samples,
            randomly_sample=randomly_sample,
            density_noise_std=density_noise_std,
            states_x=None if states_x is None else states_i[0],
            states_d=None if states_d is None else states_i[-1])[:, -1]
            for rays_o_i, rays_d_i, *states_i in zip(*batched_inputs)]

        # merge the sharded predictions into a single image and
        # inflate the tensor to have the correct shape
        return torch.cat(image, dim=0).reshape(batch_size,
                                               image_h, image_w, 3)
