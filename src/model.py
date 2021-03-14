import torch
import torchvision
import torch.nn as nn

from config import cfg, DEVICE


class NSTModel(nn.Module):
    def __init__(self, local_weights=True, params_path='vgg19.pth'):
        super().__init__()

        self.vgg19 = torchvision.models.vgg19(pretrained=(not local_weights))
        # We don't need the fully-connected 'Classifier' part of the network
        self.vgg19 = self.vgg19.features

        if local_weights:
            self.load_state_dict(torch.load(params_path, map_location=DEVICE))

        # Freeze the layers. Make sure that we don't calculate gradients for
        # the VGG19 layers, and therefore don't update those weights.
        for child in self.vgg19.children():
            for param in child.parameters():
                param.requires_grad = False

        if cfg.cuda:
            self.vgg19 = self.vgg19.cuda()

        self.vgg19.eval()
        self.vgg_features = list(self.vgg19)

    def forward(self, img, feat_type='both', layers_c=None, layers_s=None):
        """ Parameters @layers_c or @layers_s are used when we want to
            specify the exact layers from the function will extract the
            feature maps. If this parameters aren't specified default layers
            in the cfg ArgumentParser will be used.
        """
        if layers_c is not None:
            content_layers = layers_c
        else:
            content_layers = cfg.content_layers

        if layers_s is not None:
            style_layers = layers_s
        else:
            style_layers = cfg.style_layers

        outputs = []
        if feat_type == 'both':
            outputs = [[], []]

        x = img

        for i, layer in enumerate(self.vgg_features):
            x = layer(x)

            if feat_type == 'both':
                if i in style_layers:
                    outputs[0].append(x)
                elif i in content_layers:
                    outputs[1].append(x)

                if i == max(content_layers+style_layers):
                    break
            elif feat_type == 'content':
                if i in content_layers:
                    outputs.append(x)
                if i == max(content_layers):
                    break

            elif feat_type == 'style':
                if i in style_layers:
                    outputs.append(x)
                if i == max(style_layers):
                    break

        return outputs
