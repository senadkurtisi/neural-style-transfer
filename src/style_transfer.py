from style_utils import *
from model import NSTModel


if __name__ == "__main__":
    # Load style and content image
    style = preprocess_img(cfg.style_path)
    content = preprocess_img(cfg.content_path)

    model = NSTModel()
    if cfg.cuda:
        model = model.cuda()

    # Precalculate the style and content image features of through the VGG19
    style_features = model(style.detach(), 'style')
    content_features = model(content.detach(), 'content')

    # Get the initial image
    opt_image = get_init_img('other', style)
    # Start the Style Transfer process
    optimize(model, opt_image, content_features, style_features)
