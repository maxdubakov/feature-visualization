import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(0)


class TransformationRobustness(nn.Module):
    def __init__(self, transformations):
        super(TransformationRobustness, self).__init__()
        self.transformations = transformations

    def forward(self, x):
        for transformation in self.transformations:
            x = transformation(x)
        return x


# I really like the idea with inner once I got why we do it...
def pad_image(pad_size=16, pad_mode='reflect'):
    def inner(x):
        padding = (pad_size, pad_size, pad_size, pad_size)
        return F.pad(x, padding, mode=pad_mode)

    return inner


def jitter(jitter_size=None):
    def inner(x):
        batch_size, channels, height, width = x.size()
        crop_height = height - jitter_size
        crop_width = width - jitter_size

        top = torch.randint(0, jitter_size + 1, (batch_size,))
        left = torch.randint(0, jitter_size + 1, (batch_size,))

        crops = []
        for i in range(batch_size):
            crop = x[i:i + 1, :, top[i]:top[i] + crop_height, left[i]:left[i] + crop_width]
            crops.append(crop)

        return torch.cat(crops, dim=0)

    return inner


def random_scale(scale_factors=None):
    if scale_factors is None:
        scale_factors = [1, 0.975, 1.025, 0.95, 1.05]

    def inner(x):

        scale = random.choice(scale_factors)

        batch_size, channels, height, width = x.size()
        new_height = int(height * scale)
        new_width = int(width * scale)

        # Use F.interpolate for scaling
        x_scaled = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

        return x_scaled
    return inner


def random_rotate(angles=None):
    if angles is None:
        angles = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    def inner(x):

        angle = random.choice(angles)

        # Convert angle to radians
        angle_rad = torch.tensor(angle * (3.14159 / 180))

        batch_size, channels, height, width = x.size()

        # Create rotation matrix
        rot_mat = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
            [torch.sin(angle_rad), torch.cos(angle_rad), 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Create grid
        grid = F.affine_grid(rot_mat, x.size(), align_corners=False)

        # Apply rotation
        x_rotated = F.grid_sample(x, grid, align_corners=False, mode='bilinear')

        return x_rotated

    return inner


def crop_padding(pad_size=16):
    """
    Crop the padding added to the image.

    Args:
    x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

    Returns:
    torch.Tensor: Cropped tensor with padding removed
    """
    def inner(x):
        if pad_size == 0:
            return x

        _, _, height, width = x.size()
        return x[:, :, pad_size:height - pad_size, pad_size:width - pad_size]

    return inner
