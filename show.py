import io

import PIL
import torch
import base64
import IPython
import numpy as np
from PIL import Image
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from IPython.display import HTML
from torchvision.transforms.functional import resize


def _normalize_array(array, domain=(0, 1)):
  """Given an arbitrary rank-3 NumPy array, produce one representing an image.

  This ensures the resulting array has a dtype of uint8 and a domain of 0-255.

  Args:
    array: NumPy array representing the image
    domain: expected range of values in array,
      defaults to (0, 1), if explicitly set to None will use the array's
      own range of values and normalize them.

  Returns:
    normalized PIL.Image
  """
  # first copy the input so we're never mutating the user's data
  array = np.array(array)
  # squeeze helps both with batch=1 and B/W and PIL's mode inference
  array = np.squeeze(array)
  assert len(array.shape) <= 3
  assert np.issubdtype(array.dtype, np.number)
  assert not np.isnan(array).any()

  low, high = np.min(array), np.max(array)
  if domain is None:
    print(f"No domain specified, normalizing from measured ({low:.2f}, {high:.2f})")
    domain = (low, high)

  # clip values if domain was specified and array contains values outside of it
  if low < domain[0] or high > domain[1]:
    print(f"Clipping domain from ({low:.2f}, {high:.2f}) to ({domain[0]:.2f}, {domain[1]:.2f}).")
    array = array.clip(*domain)

  min_value, max_value = np.iinfo(np.uint8).min, np.iinfo(np.uint8).max  # 0, 255
  # convert signed to unsigned if needed
  if np.issubdtype(array.dtype, np.inexact):
    offset = domain[0]
    if offset != 0:
      array -= offset
      print(f"Converting inexact array by subtracting {offset:.2f}")
    if domain[0] != domain[1]:
      scalar = max_value / (domain[1] - domain[0])
      if scalar != 1:
        array *= scalar
        print(f"Converting inexact array by scaling by {scalar:.2f}.")

  return array.clip(min_value, max_value).astype(np.uint8)


def _serialize_normalized_array(array, fmt='png', quality=70):
  """Given a normalized array, returns byte representation of image encoding.

  Args:
    array: NumPy array of dtype uint8 and range 0 to 255
    fmt: string describing desired file format, defaults to 'png'
    quality: specifies compression quality from 0 to 100 for lossy formats

  Returns:
    image data as BytesIO buffer
  """
  dtype = array.dtype
  assert np.issubdtype(dtype, np.unsignedinteger)
  assert np.max(array) <= np.iinfo(dtype).max
  assert array.shape[-1] > 1  # array dims must have been squeezed

  image = PIL.Image.fromarray(array)
  image_bytes = io.BytesIO()
  image.save(image_bytes, fmt, quality=quality)
  image_data = image_bytes.getvalue()
  return image_data


def serialize_array(array, domain=(0, 1), fmt='png', quality=70):
  """Given an arbitrary rank-3 NumPy array,
  returns the byte representation of the encoded image.

  Args:
    array: NumPy array of dtype uint8 and range 0 to 255
    domain: expected range of values in array, see `_normalize_array()`
    fmt: string describing desired file format, defaults to 'png'
    quality: specifies compression quality from 0 to 100 for lossy formats

  Returns:
    image data as BytesIO buffer
  """
  normalized = _normalize_array(array, domain=domain)
  return _serialize_normalized_array(normalized, fmt=fmt, quality=quality)


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


def __image_url(array, fmt='png', quality=90, domain=None):
    image_data = serialize_array(array, fmt=fmt, quality=quality, domain=domain)
    base64_byte_string = base64.b64encode(image_data).decode('ascii')
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


def show_result_html(_imgs, _thresholds, fmt='png', quality=70, domain=(0, 1)):
    s = '<div style="display: flex; flex-direction: row;">'
    for i, img in enumerate(_imgs):
        img_resized = nd.zoom(img, [1, 2, 2], order=0)
        img_resized = torch.from_numpy(img_resized).permute(1, 2, 0).numpy()
        label = f'Iteration {_thresholds[i]}'
        img_html = f'<img src="{__image_url(img_resized, fmt=fmt, quality=quality, domain=domain)}" style="image-rendering: pixelated; image-rendering: crisp-edges;">'
        s += f"""<div style="margin-right:10px; margin-top: 4px;">
                    {label} <br/>
                    {img_html}
                 </div>"""
    s += "</div>"
    IPython.display.display(IPython.display.HTML(s))
