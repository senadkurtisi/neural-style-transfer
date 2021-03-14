from PIL import Image

import numpy as np
import torch
import torchvision.transforms.transforms as transforms

import os

from config import cfg


def preprocess_img(img_path):
    """ Loads the desired image and prepares it
        for VGG19 model.

    Parameters:
        img_path: path to the image
    Returns:
        processed: loaded image after preprocessing
    """
    prep = transforms.Compose([transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])

    img = Image.open(img_path)
    processed = prep(img)

    if cfg.cuda:
        processed = processed.cuda()

    return processed.unsqueeze(0)


def get_init_img(mode='noise', source_img=None):
    """ Constructs the initial image for the NST algorithm.

    Parameters:
        mode: how to initialize the image? {'noise', 'other'}
        source_img: image used for initialization of @mode is set to 'other'
    Returns:
        opt_image: initialized image
    """
    assert mode in ['noise', 'other'], f"{mode} is and illegal initialization mode!"

    if mode == 'style' or mode == 'other':
        assert (source_img is not None), f"Can't initialize from {mode}!"

    if mode == 'noise':
        if cfg.cuda:
            opt_image = np.random.normal(loc=0, scale=90.,
                                         size=(1, 3, cfg.IMG_SIZE,
                                               cfg.IMG_SIZE)).astype(np.float32)
            opt_image = torch.from_numpy(opt_image).float().cuda()
        else:
            pass
    else:
        opt_image = (source_img.detach()).clone()

    # Make sure that gradients are being calculated for this image
    # During forward pass
    opt_image.requires_grad = True

    return opt_image


def gram_matrix(x):
    """ Calculates the Gram matrix for the
        feature maps contained in x.

    Parameters:
          x: feature maps
    Returns:
          G: gram matrix
    """
    b, c, h, w = x.size()
    F = x.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h * w)
    return G


def postprocess(img):
    """ Prepares the image for display and saving. """
    postp = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                               transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                               ])

    img = postp(img)
    # In order to have more visually appealing images
    # We need to clip the pixel values
    img[img > 1] = 1
    img[img < 0] = 0
    img = transforms.ToPILImage()(img)

    return img


def get_file_name(path):
    """ Extracts only the filename from the given
        path. Extension is removed as well.
    """
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

