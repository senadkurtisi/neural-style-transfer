import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import LBFGS

from utils import *

from config import cfg


def optimize(model, opt_img, target_content, style_features, save=False):
    """ Performs the Neural Style Transfer algorithm by
        using the L-BFGS optimizer.

    Parameters:
         model: pretrained VGG19 model used for forward-pass
         opt_img: image which will be optimized
         target_content: target feature maps of content image
         style_features: style feature maps used for target Gram matrix
    """
    mse_func = nn.MSELoss()
    iter_ = [0]
    optimizer = LBFGS([opt_img])
    target_style = [gram_matrix(st) for st in style_features]

    while iter_[0] <= cfg.MAX_ITER:
        def closure():
            optimizer.zero_grad()
            opt_style_features, opt_content_features = model(opt_img, feat_type='both')

            style_loss = [cfg.style_w * cfg.style_layer_norm[i]
                          * mse_func(gram_matrix(opt_style_features[i]), target_style[i])
                          for i in range(len(opt_style_features))]

            content_loss = [cfg.content_w * mse_func(opt_content_features[i], target_content[i])
                            for i in range(len(opt_content_features))]

            loss = sum(style_loss + content_loss)
            loss.backward()

            iter_[0] += 1
            if iter_[0] % cfg.disp_iter == (cfg.disp_iter - 1):
                print('Iteration: %d, loss: %f' % (iter_[0] + 1, loss.item()))

            return loss

        optimizer.step(closure)

    opt_img = postprocess(opt_img.detach().cpu().squeeze())

    plt.figure(figsize=(15, 8))
    plt.imshow(opt_img)
    plt.axis('off')

    if save:
        img_name = f"{get_file_name(cfg.content_path)}_{get_file_name(cfg.style_path)}.png"
        plt.savefig(img_name, bbox_inches='tight')

    plt.show()


def reconstruct_from_content(model, target_content, layers, save=False):
    """ Reconstructs the original content image by using
        feature maps from specified layer/s.

    Parameters:
         model: pretrained VGG19 model used for forward-pass
         target_content: target feature maps of content image
         layers: list of layers used for reconstruction
    """
    opt_img = get_init_img(mode='noise')
    optimizer = LBFGS([opt_img])
    mse_func = nn.MSELoss(reduction='mean')

    iter_ = [0]
    max_iter = 500

    while iter_[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            content_features = model(opt_img, 'content', layers)
            loss = mse_func(content_features[0], target_content[0]) * (cfg.style_w ** 3)
            loss.backward()

            iter_[0] += 1
            if iter_[0] % cfg.disp_iter == (cfg.disp_iter - 1):
                print('Iteration: %d, loss: %f' % (iter_[0] + 1, loss.item()))

            return loss

        optimizer.step(closure)

    opt_img = postprocess(opt_img.detach().cpu().squeeze())

    plt.figure(figsize=(15, 8))
    plt.imshow(opt_img)
    plt.axis('off')

    if save:
        plt.savefig('from_content.png', bboxs_inches='tight')

    plt.show()


def reconstruct_from_style(model, style_features, layers, save=False):
    """ Reconstructs the original style image by using
        Gram matrices from specified layer's feature maps.

    Parameters:
         model: pretrained VGG19 model used for forward-pass
         style_features: target feature maps of style image
         layers: list of layers used for reconstruction
    """
    opt_img = get_init_img(mode='noise')
    optimizer = LBFGS([opt_img])
    mse_func = nn.MSELoss(reduction='sum')

    iter_ = [0]
    max_iter = 500

    target_style = [gram_matrix(style) for style in style_features]

    while iter_[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            style_features_opt = model(opt_img, 'style', layers_s=layers)
            style_loss = [mse_func(gram_matrix(style_features_opt[i])/3, target_style[i]/3)
                          for i in range(len(target_style))]
            loss = sum(style_loss)/len(style_loss)
            loss.backward()

            iter_[0] += 1
            if iter_[0] % cfg.disp_iter == (cfg.disp_iter - 1):
                print('Iteration: %d, loss: %f' % (iter_[0] + 1, loss.item()))

            return loss

        optimizer.step(closure)

    opt_img = postprocess(opt_img.detach().cpu().squeeze())

    plt.figure(figsize=(15, 8))
    plt.imshow(opt_img)
    plt.axis('off')

    if save:
        plt.savefig('from_style.png', bbox_inches='tight')

    plt.show()


