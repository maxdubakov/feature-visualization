import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision.models import GoogLeNet_Weights

from model import googlenet, global_step_tracker

global_step = 0


def pixel_image(shape, sd=0.01):
    return np.random.normal(size=shape, scale=sd).astype(np.float32)


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


class ParameterizedImage(nn.Module):
    def __init__(self,
                 w,
                 h=None,
                 fft=True,
                 decorrelate=True,
                 decay_power=1,
                 batch=1,
                 image_value_range=(-117/255, (255-117)/255),
                 sd=0.01,
                 channels=3,
                 device='cpu'):
        super().__init__()
        h = h or w
        shape = [batch, channels, h, w]
        self.w = w
        self.h = h
        self.batch = batch
        self.channels = channels
        self.fft = fft
        self.decay_power = decay_power
        self.decorrelate = decorrelate

        if fft:
            self.freqs = rfft2d_freqs(h, w)
            shape = (batch, channels, 2) + self.freqs.shape
        init_val = pixel_image(shape, sd=sd)
        self.register_buffer('low', torch.tensor(image_value_range[0], device=device))
        self.register_buffer('high', torch.tensor(image_value_range[1], device=device))

        # fft
        self.freqs = rfft2d_freqs(h, w)

        self.param = nn.Parameter(torch.tensor(init_val))

        # Color decorrelation matrix
        if decorrelate:
            color_correlation_svd_sqrt = torch.tensor([
                [0.26, 0.09, 0.02],
                [0.27, 0.00, -0.05],
                [0.27, -0.09, 0.03]
            ], dtype=torch.float32)
            max_norm_svd_sqrt = torch.norm(color_correlation_svd_sqrt, dim=0).max()
            self.color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

            self.color_mean = torch.tensor([0.48, 0.46, 0.41])


    def forward(self):
        x = self.param

        if self.fft:
            spectrum_t = torch.complex(x[:, :, 0], x[:, :, 1])
            scale = 1.0 / np.maximum(self.freqs, 1.0 / max(self.w, self.h)) ** self.decay_power
            scale *= np.sqrt(self.w * self.h)
            scale = torch.from_numpy(scale).float()
            scaled_spectrum_t = scale * spectrum_t
            x = torch.fft.irfft2(scaled_spectrum_t, s=(self.h, self.w))
            x = x[:self.batch, :self.channels, :self.h, :self.w]
            x = x / 4.0  # TODO: is that a magic constant?

        if self.decorrelate:
            x_flat = x.reshape(-1, 3)
            x_flat = torch.matmul(x_flat, self.color_correlation_normalized.t().to(x.device))
            x = x_flat.reshape(x.shape)

        x = torch.sigmoid(x)
        scaled = self.low + x * (self.high - self.low)
        return x, scaled


# function where we perform forward pass, extract average activation (as a single number) and return negation of it
# negation because we have to _maximize_ the objective and _minimize_ the loss
# so if loss will go from 15 to 7 (lowering), objective will go from -15 to -7 (growing)
def visualize(model, image, activation, channel_nr, regularizers, transformations):
    image_params, image_params_scaled = image()

    if transformations is None:
        transformations = nn.Identity() # no-op

    transformed_image = transformations(image_params_scaled)

    model(transformed_image)  # Forward pass
    loss = -activation['target'][0, channel_nr].mean()

    for reg in regularizers:
        loss += reg['weight'] * reg['func'](image_params_scaled)

    loss.backward()
    return image_params, loss.item()


# Render function
def render_vis(
        channel: str,
        optimizer,
        image: ParameterizedImage,
        regularizers=[],
        transformations=None,
        visualize=visualize,
        thresholds=None,
        use_fixed_seed=True,
        device='cpu'):
    global global_step
    global_step = 0

    if use_fixed_seed:
        np.random.seed(0)
        torch.manual_seed(0)

    layer, branch, channel_nr = tuple(channel.split(':'))
    channel_nr = int(channel_nr)

    if thresholds is None:
        thresholds = [1, 32, 128, 256, 2048, 2559]

    # init the model
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval()
    model.to(device)

    target_layer = getattr(model, layer)

    activation = {}

    def get_activation():
        def hook(module, input, output):
            activation['target'] = output

        return hook

    if int(branch[-1]) > 1:
        getattr(target_layer, branch)[1].bn.register_forward_hook(get_activation())
    else:
        getattr(target_layer, branch).bn.register_forward_hook(get_activation())

    images = []
    num_iterations = thresholds[-1] + 1
    pbar = tqdm(range(num_iterations), total=num_iterations)
    for i in pbar:
        optimizer.zero_grad()
        image_params, loss = visualize(model, image, activation, channel_nr, regularizers, transformations)
        optimizer.step()
        global_step_tracker.increment()

        pbar.set_postfix({'loss': f'{loss:.4f}'})
        global_step += 1

        if i in thresholds:
            image.eval()
            img_np = (image_params
                      .squeeze()  # remove batch dimension
                      .detach()  # detach from graph (will not be included in autograd and never require the gradient)
                      .cpu()  # from mps/cuda -> cpu
                      .numpy()  # to numpy array instead of pytorch.Tensor
                      )
            images.append(np.copy(img_np))
            image.train()

    return images, thresholds
