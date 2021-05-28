import argparse
from pathlib import Path

import cv2

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, device, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def read_image_opencv(content_path, style_path):
    content = cv2.imread(content_path)
    style = cv2.imread(style_path)
    content = content[:, :, ::-1]
    style = style[:, :, ::-1]

    return content, style


def initialize_model(args, device):
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    return vgg, decoder


def run_forward(content, style, vgg, decoder, device, preserve_color, alpha, interpolation_weights):
    assert content.shape[2] == 3 and style.shape[2] == 3
    content = Image.fromarray(content)
    style = Image.fromarray(style)
    content = content_tf(content)
    style = style_tf(style)

    if preserve_color:
        style = coral(style, content)

    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, device, alpha, interpolation_weights)

    if torch.cuda.is_available():
        output = output.cpu()

    output = make_grid(output)
    output = output.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    return output


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content", type=str, default=None)
    parser.add_argument("--style", type=str, default=None)

    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='models/decoder.pth')

    parser.add_argument('--content_size', type=int, default=512,
                        help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=512,
                        help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
    parser.add_argument(
        '--style_interpolation_weights', type=str, default=None,
        help='The weight for blending the style of multiple style images')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    if args.style_interpolation_weights is not None:
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
    else:
        interpolation_weights = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vgg, decoder = initialize_model(args, device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    #content = content_tf(Image.open(args.content))
    #style = style_tf(Image.open(args.style))

    content, style = read_image_opencv(args.content, args.style)

    output = run_forward(content=content, style=style, vgg=vgg, decoder=decoder, device=device,
                         preserve_color=args.preserve_color,
                         alpha=args.alpha, interpolation_weights=interpolation_weights)
    #img = Image.fromarray(output)
    #img.show()

    cv2.imshow("Anh", output[:, :, ::-1])
    cv2.waitKey(0)
