import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(0)


class TransformationRobustness(nn.Module):
    def __init__(self, pad_size=16, pad_mode='reflect', jitter_size=16, scale_factors=None, angles=None):
        super(TransformationRobustness, self).__init__()
        if angles is None:
            angles = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        if scale_factors is None:
            scale_factors = [1, 0.975, 1.025, 0.95, 1.05]
        self.pad_size = pad_size
        self.pad_mode = pad_mode
        self.jitter_size = jitter_size
        self.scale_factors = scale_factors
        self.angles = angles

    def forward(self, x):
        # Apply padding
        x = self._pad_image(x)

        x = self._jitter(x)

        x = self._random_scale(x)

        x = self._random_rotate(x)

        x = self._jitter(x, self.jitter_size // 2)

        x = self._crop_padding(x)

        return x

    def _pad_image(self, x):
        padding = (self.pad_size, self.pad_size, self.pad_size, self.pad_size)
        return F.pad(x, padding, mode=self.pad_mode)

    def _jitter(self, x, jitter_size=None):
        if jitter_size is None:
            jitter_size = self.jitter_size
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

    def _random_scale(self, x):
        scale = random.choice(self.scale_factors)

        batch_size, channels, height, width = x.size()
        new_height = int(height * scale)
        new_width = int(width * scale)

        # Use F.interpolate for scaling
        x_scaled = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

        return x_scaled

    def _random_rotate(self, x):
        angle = random.choice(self.angles)

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

    def _crop_padding(self, x):
        """
        Crop the padding added to the image.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
        torch.Tensor: Cropped tensor with padding removed
        """
        if self.pad_size == 0:
            return x

        _, _, height, width = x.size()
        return x[:, :, self.pad_size:height - self.pad_size, self.pad_size:width - self.pad_size]