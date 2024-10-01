import io
import torch
import base64
import IPython
import numpy as np
from PIL import Image
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from IPython.display import HTML
from torchvision.transforms.functional import resize


def show_result(_imgs, _thresholds, factor=2):
    fig, axs = plt.subplots(1, len(_imgs), figsize=(15, 3))
    image_size = _imgs[0].shape[1]
    for i, img in enumerate(_imgs):
        # Resize image to double its size
        img_resized = resize(torch.from_numpy(img), [int(image_size * factor), int(image_size * factor)])
        # in InceptionV1 the dimensions of the resulting image are (channels, height, width)
        # matplotlib expects (height, width, channels)
        img_resized = img_resized.permute(1, 2, 0).numpy()

        # display the image on a specific axis
        axs[i].imshow(img_resized)
        # disable x/y number grid
        axs[i].axis('off')
        axs[i].set_title(f'Iteration {_thresholds[i]}')

    plt.tight_layout()  # makes arrangement of the images a bit nicer
    plt.show()


def __serialize_array(arr, fmt='png'):
    img = Image.fromarray(np.uint8(arr * 255))
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def __image_url(array, fmt='png'):
    image_data = __serialize_array(array, fmt=fmt)
    base64_byte_string = base64.b64encode(image_data).decode('ascii')
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


def show_result_html(_imgs, _thresholds):
    s = '<div style="display: flex; flex-direction: row;">'
    for i, img in enumerate(_imgs):
        img_resized = nd.zoom(img, [1, 2, 2], order=0)
        img_resized = torch.from_numpy(img_resized).permute(1, 2, 0).numpy()
        label = f'Iteration {_thresholds[i]}'
        img_html = f'<img src="{__image_url(img_resized)}" style="image-rendering: pixelated; image-rendering: crisp-edges;">'
        s += f"""<div style="margin-right:10px; margin-top: 4px;">
                    {label} <br/>
                    {img_html}
                 </div>"""
    s += "</div>"
    IPython.display.display(IPython.display.HTML(s))
