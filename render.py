import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights


def pixel_image(shape, sd=0.01):
    return np.random.normal(size=shape, scale=sd).astype(np.float32)


class ParameterizedImage(nn.Module):
    def __init__(self, w, h=None, batch=1, sd=0.01, channels=3):
        super().__init__()
        h = h or w
        shape = [batch, channels, h, w]
        init_val = pixel_image(shape, sd=sd)
        self.param = nn.Parameter(torch.tensor(init_val))

    def forward(self):
        return torch.sigmoid(self.param)


# function where we perform forward pass, extract average activation (as a single number) and return negation of it
# negation because we have to _maximize_ the objective and _minimize_ the loss
# so if loss will go from 15 to 7 (lowering), objective will go from -15 to -7 (growing)
def visualize_naive(model, image, activation, channel_nr, regularizers, transformations, **params):
    image_params = image()
    model(image_params)  # Forward pass
    loss = -activation['target'][0, channel_nr].mean()

    for reg in regularizers:
        loss += reg['weight'] + reg['func'](image_params)

    loss.backward()
    return loss.item()


# Render function
def render_vis(
        channel,
        optimizer,
        image,
        regularizers=[],
        transformations=None,
        visualize=visualize_naive,
        num_iterations=2560,
        device='cpu',
        **_visualize_params):
    layer, branch, channel_nr = tuple(channel.split(':'))
    channel_nr = int(channel_nr)

    # init the model
    model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval()
    model.to(device)

    target_layer = getattr(model, layer)

    activation = {}

    def get_activation():
        def hook(module, input, output):
            activation['target'] = output

        return hook

    if int(branch[-1]) > 1:
        getattr(target_layer, branch)[1].conv.register_forward_hook(get_activation())
    else:
        getattr(target_layer, branch).conv.register_forward_hook(get_activation())

    thresholds = [1, 32, 128, 256, 2048, 2559]
    images = []
    pbar = tqdm(range(num_iterations), total=num_iterations)
    for i in pbar:
        optimizer.zero_grad()
        loss = visualize(model, image, activation, channel_nr, regularizers, transformations, **_visualize_params)
        optimizer.step()

        pbar.set_postfix({'loss': f'{loss:.4f}'})

        if i in thresholds:
            img_np = (image()
                      .squeeze()  # remove batch dimension
                      .detach()  # detach from graph (will not be included in autograd and never require the gradient)
                      .cpu()  # from mps/cuda -> cpu
                      .numpy()  # to numpy array instead of pytorch.Tensor
                      )
            images.append(np.copy(img_np))

    return images, thresholds
