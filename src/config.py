import torch
from argparse import ArgumentParser


using_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if using_cuda else 'cpu')

parser = ArgumentParser()
parser.add_argument("--IMG_SIZE", type=str, default=400)
parser.add_argument("--content_path", type=str, default='images/content/tubingen.jpg')
parser.add_argument("--style_path", type=str, default='images/style/starry_night.jpg')
parser.add_argument("--MAX_ITER", type=int, default=500)
parser.add_argument("--disp_iter", type=int, default=50)
parser.add_argument("--style_w", type=float, default=1e3)
parser.add_argument("--content_w", type=float, default=2e1)
parser.add_argument("--cuda", type=bool, default=using_cuda)
parser.add_argument("--style_layers", type=list, default=[1, 6, 11, 20, 29])
parser.add_argument("--content_layers", type=list, default=[22])
# Norms based on the number of filters in the according 'style layers'
parser.add_argument("--style_layer_norm", type=list,
                    default=[1/(n**2) for n in [64, 128, 256, 512, 512]])

cfg = parser.parse_args()
