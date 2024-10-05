import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision.models import GoogLeNet_Weights

from model import googlenet

global_step = 0


def pixel_image(shape, sd=0.01):
    return np.random.normal(size=shape, scale=sd).astype(np.float32)


def scale_input(image, image_value_range=(-117, 255-117)):
    lo, hi = image_value_range
    image = lo + image * (hi - lo)

    return image


class ParameterizedImage(nn.Module):
    def __init__(self, w, h=None, batch=1, sd=0.01, channels=3):
        super().__init__()
        h = h or w
        shape = [batch, channels, h, w]
        init_val = pixel_image(shape, sd=sd)
        normalized_init_val = scale_input(torch.sigmoid(torch.tensor(init_val)))
        self.param = nn.Parameter(normalized_init_val)

    def forward(self):
        return self.param


# function where we perform forward pass, extract average activation (as a single number) and return negation of it
# negation because we have to _maximize_ the objective and _minimize_ the loss
# so if loss will go from 15 to 7 (lowering), objective will go from -15 to -7 (growing)
def visualize(model, image, activation, channel_nr, regularizers, transformations):
    image_params = image()

    if transformations is None:
        transformations = nn.Identity() # no-op

    transformed_image = transformations(image_params)

    model(transformed_image)  # Forward pass
    loss = -activation['target'][0, channel_nr].mean()

    for reg in regularizers:
        loss += reg['weight'] * reg['func'](image_params)

    loss.backward()
    return loss.item()


# Render function
def render_vis(
        channel,
        optimizer,
        image,
        regularizers=[],
        transformations=None,
        visualize=visualize,
        num_iterations=2560,
        device='cpu'):
    global global_step
    global_step = 0

    layer, branch, channel_nr = tuple(channel.split(':'))
    channel_nr = int(channel_nr)

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

    thresholds = [1, 32, 128, 256, 2048, 2559]
    images = []
    pbar = tqdm(range(num_iterations), total=num_iterations)
    for i in pbar:
        optimizer.zero_grad()
        loss = visualize(model, image, activation, channel_nr, regularizers, transformations)
        optimizer.step()

        pbar.set_postfix({'loss': f'{loss:.4f}'})
        global_step += 1

        if i in thresholds:
            img_np = (image()
                      .squeeze()  # remove batch dimension
                      .detach()  # detach from graph (will not be included in autograd and never require the gradient)
                      .cpu()  # from mps/cuda -> cpu
                      .numpy()  # to numpy array instead of pytorch.Tensor
                      )
            images.append(np.copy(img_np))

    return images, thresholds
