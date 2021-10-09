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
    density_outputs: int
        the number of outputs made by the network representing density
        and should typically be set to one
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

        idx = torch.arange(0, size // 2).to(dtype=x.dtype, device=x.device)
        frequency_scale = torch.exp(2.0, idx) * np.pi
        x = x.unsqueeze(-1)
        while len(frequency_scale.shape) < len(x.shape):
            frequency_scale = frequency_scale.unsqueeze(0)
        return torch.cat([torch.sin(x * frequency_scale),
                          torch.cos(x * frequency_scale)], dim=-1)

    def __init__(self, output_init_scale=3e-3, hidden_size=256,
                 density_inputs=3, color_inputs=3,
                 density_outputs=1, color_outputs=3,
                 x_positional_encoding_size=20,
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
        density_outputs: int
            the number of outputs made by the network representing density
            and should typically be set to one
        color_outputs: int
            the number of outputs made by the network representing color
            and should typically be set to three

        """

        super(NeRF, self).__init__()

        self.output_init_scale = output_init_scale
        self.hidden_size = hidden_size
        self.density_inputs = density_inputs
        self.color_inputs = color_inputs
        self.density_outputs = density_outputs
        self.color_outputs = color_outputs

        self.d_positional_encoding_size = d_positional_encoding_size
        self.x_positional_encoding_size = x_positional_encoding_size
        self.d_size = (self.density_inputs *
                       self.d_positional_encoding_size)
        self.x_size = (self.color_inputs *
                       self.x_positional_encoding_size)

        self.density = nn.Linear(hidden_size, self.density_outputs)
        self.density.bias.data.fill_(0)
        self.density.weight.data.uniform_(
            -output_init_scale, output_init_scale)

        self.color = nn.Linear(hidden_size, self.color_outputs)
        self.color.bias.data.fill_(0)
        self.color.weight.data.uniform_(
            -output_init_scale, output_init_scale)

        self.block_0 = nn.Sequential(
            nn.Linear(self.x_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size))

        self.block_1 = nn.Sequential(
            nn.Linear(self.x_size + hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size))

        self.block_2 = nn.Sequential(
            nn.Linear(self.d_size + hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size))

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

        x_embed = self.positional_encoding(x, self.x_positional_encoding_size)
        d_embed = self.positional_encoding(d, self.d_positional_encoding_size)
        block_0 = self.block_0(x_embed)
        block_1 = self.block_1(torch.cat([block_0, x_embed], dim=-1))
        block_2 = self.block_2(torch.cat([block_1, d_embed], dim=-1))
        return (functional.relu(self.density(block_1)),
                functional.sigmoid(self.color(block_2)))

    def get_rays_np(self, H, W, focal, c2w):
        """Get ray origins, directions from a pinhole camera."""
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d
