import torch
import torch.nn
import argparse
import os
import time
import numpy as np
from options import HiDDenConfiguration
from collections import defaultdict
from average_meter import AverageMeter

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms,datasets
from noise_argparser import NoiseArgParser

def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='/home/audience/Desktop/2021824hidden3/runs/hidden 2021.08.28--23-30-50/options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', default='/home/audience/Desktop/2021824hidden3/runs/hidden 2021.08.28--23-30-50/checkpoints/hidden--epoch-546.pyt', type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=1, type=int, help='The batch size.')
    parser.add_argument('--noise', nargs='*', action=NoiseArgParser)



    args = parser.parse_args()

    noise_config = args.noise if args.noise is not None else []
    test_images = "/home/audience/dataset/mscoco/val4"
    need_pic = True
    train_options, hidden_config, _ = utils.load_options(args.options_file)
    noiser = Noiser(noise_config,device)

    checkpoint = torch.load(args.checkpoint_file)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    transform = transforms.Compose([
        transforms.CenterCrop((hidden_config.H, hidden_config.W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_images = datasets.ImageFolder(test_images, transform)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=1,
                                                    shuffle=False, num_workers=4, drop_last=True)

    validation_losses = defaultdict(AverageMeter)

    for i,(image,_) in enumerate(test_loader):
        image = image.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0],
                                                         hidden_config.message_length))).to(device)
        losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image, message])

        for name, loss in losses.items():
            validation_losses[name].update(loss)
        if need_pic:
            if hidden_config.enable_fp16:
                image = image.float()
                encoded_images = encoded_images.float()
            message = utils.reshape_watermark(message)
            decoded_messages = torch.round(decoded_messages)
            decoded_messages = utils.reshape_watermark(decoded_messages)
            watermark = torch.cat([message, decoded_messages], dim=0)
            utils.save_images(image.cpu(),
                              encoded_images.cpu(),
                              noised_images.cpu(),
                              i,
                              os.path.join("/home/audience/Desktop/2021824hidden3/runs/hidden 2021.08.28--23-30-50/test/dropout", 'images'),
                              watermark,
                              resize_to=256)

        if i%20 == 0:
            print(i)


    # for t in range(args.times):

    utils.log_progress(validation_losses)
    utils.write_losses(os.path.join("/home/audience/Desktop/2021824hidden3/runs/hidden 2021.08.28--23-30-50/test/dropout", 'validation.csv'), validation_losses, 0,
                       time.time() - 0)




if __name__ == '__main__':
    main()
