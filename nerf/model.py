import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class ResidualBlock(nn.Module):

    def __init__(self, hidden_size, feedforward_size):

        super(ResidualBlock, self).__init__()

        self.linear_one = nn.Linear(hidden_size, feedforward_size)
        self.activation = nn.GELU()
        self.linear_two = nn.Linear(feedforward_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):

        h = self.linear_two(self.activation(self.linear_one(x)))
        return self.layer_norm(x + h)


def expected_sin(x, x_var):
  """Estimates mean and variance of sin(z), z ~ N(x, var)."""
  # When the variance is wide, shrink sin towards zero.
  y = torch.exp(-0.5 * x_var) * torch.sin(x)
  y_var = (0.5 * (1 - torch.exp(-2 * x_var) *
                  torch.cos(2 * x)) - y**2).clamp(min=0.0)
  return y, y_var


def lift_gaussian(d, t_mean, t_var, r_var, diag):
  """Lift a Gaussian defined along a ray to 3D coordinates."""
  mean = d[..., None, :] * t_mean[..., None]

  d_mag_sq = torch.sum(d**2, dim=-1, keepdim=True).clamp(min=1e-10)

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[..., :, None] * d[..., None, :]
    eye = jnp.eye(d.shape[-1])
    null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
    t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
    xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
    cov = t_cov + xy_cov
    return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
  """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: jnp.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

  Returns:
    a Gaussian (mean and covariance).
  """
  if stable:
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                      (3 * mu**2 + hw**2)**2)
    r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                              (hw**4) / (3 * mu**2 + hw**2))
  else:
    t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
    r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
    t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_var = t_mosq - t_mean**2
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
  """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: jnp.float32 3-vector, the axis of the cylinder
    t0: float, the starting distance of the cylinder.
    t1: float, the ending distance of the cylinder.
    radius: float, the radius of the cylinder
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
  t_mean = (t0 + t1) / 2
  r_var = radius**2 / 4
  t_var = (t1 - t0)**2 / 12
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
  """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    t_vals: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
  t0 = t_vals[..., :-1]
  t1 = t_vals[..., 1:]
  if ray_shape == 'cone':
    gaussian_fn = conical_frustum_to_gaussian
  elif ray_shape == 'cylinder':
    gaussian_fn = cylinder_to_gaussian
  else:
    assert False
  means, covs = gaussian_fn(directions, t0, t1, radii, diag)
  means = means + origins[..., None, :]
  return means, covs


def integrated_pos_enc(x_coord, min_deg, max_deg):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

    Args:
        x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
            be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diag: bool, if true, expects input covariances to be diagonal (full
            otherwise).

    Returns:
        encoded: jnp.ndarray, encoded variables.
    """

    x, x_cov_diag = x_coord
    scales = torch.as_tensor([2**i for i in range(
        min_deg, max_deg)]).to(dtype=x.dtype, device=x.device)
    shape = list(x.shape[:-1]) + [-1]
    y = (x[..., None, :] * scales[:, None]).reshape(*shape)
    y_var = (x_cov_diag[..., None, :] * scales[:, None]**2).reshape(*shape)

    return expected_sin(
        torch.cat([y, y + 0.5 * np.pi], dim=-1),
        torch.cat([y_var] * 2, dim=-1))[0]


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
    num_stages: int
        the number of stages to use when generating an image, where
        later stages sample along rays using an empirical cdf

    """

    def positional_encoding(self, x, diag_covariance, size):
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

        starting_frequency = -torch.log2(
            torch.maximum(self.rays_max.abs().amax(),
                          self.rays_min.abs().amax()))

        # generate frequency scales for a sequence of sine waves
        frequency_scale = torch.pow(2.0, torch.linspace(
            starting_frequency,
            starting_frequency + size / 2 - 1.0,
            size // 2, dtype=x.dtype, device=x.device)) * np.pi / 2

        # ensure x and frequency_scale have the same tensor shape
        x = x.unsqueeze(-1)
        while len(frequency_scale.shape) < len(x.shape):
            frequency_scale = frequency_scale.unsqueeze(0)

        amplitude = torch.exp(-0.5 * (frequency_scale ** 2) *
                              diag_covariance.unsqueeze(-1))

        # generate a sine and cosine embedding for each value in x
        embedding = torch.cat([
            torch.sin(x * frequency_scale),
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
    def spherical_to_cartesian(yaw, elevation):
        """Helper function to convert from a spherical coordinate system
        parameterized by a yaw and elevation to the xyz cartesian coordinate
        with a unit radius, where the z-axis points upwards.

        Arguments:

        yaw: torch.Tensor
            a tensor representing the top-down yaw in radians of the coordinate,
            starting from the positive x-axis and turning counter-clockwise.
        elevation: torch.Tensor
            a tensor representing the elevation in radians of the coordinate,
            about the x-axis, with positive corresponding to upwards tilt.

        Returns:

        point: torch.Tensor
            a tensor corresponding to a point specified by the given yaw and
            elevation, in spherical coordinates.

        """

        # zero elevation and zero yaw points along the positive x-axis
        return torch.stack([torch.cos(yaw) * torch.cos(elevation),
                            torch.sin(yaw) * torch.cos(elevation),
                            torch.sin(elevation)], dim=-1)

    @staticmethod
    def get_rotation_matrix(eye_vector, up_vector):
        """Given a batch of camera orientations, specified with a viewing
        direction and up vector, generate a rotation matrix that transforms
        rays from the camera frame to the world frame of reference.

        Arguments:

        eye_vector: torch.Tensor
            a batch of viewing directions that are represented as three
            vectors in the world coordinate system with shape: [batch, 3]
        up_vector: torch.Tensor
            a batch of up directions in the imaging plane represented as
            three vectors in the world coordinate system with shape: [batch, 3]

        Returns:

        rotation_matrix: torch.Tensor
            a batch of rotation matrices that represent the orientation of
            cameras in the scene with shape: [batch, 3, 3]

        """

        # create a rotation matrix that transforms rays from the camera
        # coordinate system to the world coordinate system
        return torch.stack([torch.cross(eye_vector, up_vector),
                            up_vector, -eye_vector], dim=-1)

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

    def sample_along_rays(self, rays_o, rays_d, num_samples,
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
        samples = torch.pow(2.0, torch.linspace(-9.43633744014, 0.0,
                                                num_samples, **kwargs))

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

        return samples * torch.linalg.norm(self.rays_max - self.rays_min)

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

    def __init__(self, color_outputs=3, segmentation_outputs=50,
                 hidden_size=256, encoding_size=32, focal_length=112.0,
                 min_x=-20.0, max_x=20.0,
                 min_y=-20.0, max_y=20.0,
                 min_z=-20.0, max_z=20.0):
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
        segmentation_outputs: int
            the number of semantic segmentation categories output
            by the semantic segmentation head of the neural radiance field.
        num_stages: int
            the number of stages to use when generating an image, where
            later stages sample along rays using an empirical cdf

        """

        super(NeRF, self).__init__()

        self.focal_length = focal_length

        self.color_outputs = color_outputs
        self.segmentation_outputs = segmentation_outputs

        self.hidden_size = hidden_size
        self.encoding_size = encoding_size

        self.register_buffer("rays_min", torch.as_tensor(
            [[[min_x, min_y, min_z]]], dtype=torch.float32))

        self.register_buffer("rays_max", torch.as_tensor(
            [[[max_x, max_y, max_z]]], dtype=torch.float32))

        self.prediction_heads = nn.Sequential(
            nn.Linear(3 * encoding_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1 + color_outputs +
                      segmentation_outputs))

    def integrated_pe(self, rays_o, rays_d, samples):

        r_dot = 1 / (np.sqrt(3) * self.focal_length)

        means, covs = cast_rays(samples, rays_o, rays_d, r_dot, "cone")

        return means, covs, integrated_pos_enc(
            (means, covs), -4, self.encoding_size // 2 - 4)

    def forward(self, rays_o, rays_d, samples, states_x=None, states_d=None):
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
        stage int
            the stage of the nerf model to use during thie forward pass
            where from zero to n the models run course to fine
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

        mean, diag_covariance, h = self.integrated_pe(rays_o, rays_d, samples)

        head_predictions = self.prediction_heads(h)

        density, color, segmentation = head_predictions.split([
            1, self.color_outputs, self.segmentation_outputs], dim=2)

        return mean, density, color, segmentation

    def render_rays(self, rays_o, rays_d, num_samples,
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
        samples = previous_samples = self.sample_along_rays(
            rays_o, rays_d, num_samples, randomly_sample=randomly_sample)

        image_stages, segmentation_stages = [], []

        # predict a density and color for every point along each ray
        # potentially add random noise to each density prediction
        points, density, color, segmentation = self.forward(
            rays_o, rays_d, samples)

        density = density + torch.randn(
            density.shape, dtype=density.dtype,
            device=density.device) * density_noise_std

        # generate alpha compositing coefficients along each ray and
        # and composite the color values onto the imaging plane
        weights = self.alpha_compositing_coefficients(points, density)

        image_stages.append((weights * torch.sigmoid(color)).sum(dim=-2))
        segmentation_stages.append((torch.log(
            weights + 1e-10) + torch.log_softmax(
            segmentation, dim=-1)).logsumexp(dim=-2))

        # stack the stages of image generation into a tensor where the
        # second to last axis index over each stage of the model
        return torch.stack(image_stages, dim=-2), \
            torch.stack(segmentation_stages, dim=-2)

    def render_image(self, camera_o, camera_r, image_h, image_w, focal_length,
                     num_samples, states_x=None, states_d=None, max_chunk_size=1024,
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

        # transform the rays to the world coordinate system using pose
        rays_o, rays_d = self\
            .rays_to_world_coordinates(rays, camera_o, camera_r)

        # shard the rendering process in case the image is too large
        # or we are generating many samples along each ray
        rays_o = torch.flatten(rays_o, start_dim=0, end_dim=2)
        rays_d = torch.flatten(rays_d, start_dim=0, end_dim=2)
        batched_inputs = [torch.split(rays_o, max_chunk_size),
                          torch.split(rays_d, max_chunk_size)]

        # an inner function that wraps the volume rendering process for
        # each pixel, and then append pixels to an image buffer
        image = [[x[:, -1] for x in self.render_rays(
            rays_o_i, rays_d_i, num_samples,
            randomly_sample=randomly_sample,
            density_noise_std=density_noise_std)]
            for rays_o_i, rays_d_i in zip(*batched_inputs)]

        image, segmentation = zip(*image)

        # merge the sharded predictions into a single image and
        # inflate the tensor to have the correct shape
        return torch.cat(image, dim=0).reshape(batch_size, image_h,
                                               image_w, self.color_outputs), \
            torch.cat(segmentation, dim=0).reshape(
                batch_size, image_h, image_w, self.segmentation_outputs)

