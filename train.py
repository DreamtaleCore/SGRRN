""" train.py

train entry for SGRRN

@autor: DreamTale
"""


import argparse
import os
import shutil

import cv2
import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import tqdm
from torchvision import transforms

from trainer import Trainer
from utils import get_local_time
from utils import get_reflection_data_loader, prepare_sub_folder, get_config, \
    write_2images, write_loss, write_html

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/semantic_rr.yaml', help='Path to the configs file.')
parser.add_argument('--output_path', type=str, default='checkpoints', help="outputs path")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument('--gpu_id', type=int, default=0, help="gpu id")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
display_size = config['display_size']

cudnn.benchmark = True
torch.cuda.set_device(opts.gpu_id)

# Setup model and data loader
trainer = Trainer(config)

trainer.cuda()
train_loader = get_reflection_data_loader(config, train_mode='train')
val_loader = get_reflection_data_loader(config, train_mode='val')

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'configs.yaml'))  # copy configs file to output folder

# Start training
start_epoch = trainer.resume(checkpoint_directory, param=config) if opts.resume else 0

print('Start from epoch {}, total epochs {}. {} images for each epoch.'.format(start_epoch, config['n_epoch'],
                                                                               len(train_loader)))

to_pil = transforms.ToPILImage()

iterations = start_epoch * len(train_loader)

for epoch in range(start_epoch, config['n_epoch']):
    for it, (img_in, targets, flag) in enumerate(train_loader):
        if torch.any(flag == False):
            continue
        if targets['reflection'] is not None:
            images_i, images_b, images_r = img_in.cuda().detach(), \
                                           targets['background'].cuda().detach(), \
                                           targets['reflection'].cuda().detach()
        else:
            images_i, images_b, images_r = img_in.cuda().detach(), \
                                           targets['background'].cuda().detach(), None
        seman_b = targets['bg-semantic'].cuda().detach()

        trainer.gen_update(images_i, images_b, images_r, seman_b, config)
        trainer.dis_update(images_i, images_b, images_r, config)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print('<{}> [Epoch: {}/{}] [Iter: {}/{}] | {}'.format(get_local_time(), epoch, config['n_epoch'],
                                                                  it, len(train_loader),
                                                                  trainer.print_loss()))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                outputs = trainer.sample(images_i, images_b, images_r, None)
            display_size = outputs[1][0].shape[0] * len(outputs[1])
            write_2images(outputs[1], display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        iterations += 1

        trainer.update_learning_rate()

        # if it > 5:
        #     break

    if epoch % config['snapshot_save_epoch'] == 0:
        psnr_list = []
        t_bar = tqdm.tqdm(val_loader)
        t_bar.set_description('\nEvaluating')

        for it, (img_in, targets, flag) in enumerate(t_bar):
            if torch.any(flag == False):
                continue
            if targets['reflection'] is not None:
                images_i, images_b, images_r = img_in.cuda().detach(), \
                                               targets['background'].cuda().detach(), \
                                               targets['reflection'].cuda().detach()
                semantic_b, semantic_r = targets['bg-semantic'].cuda().detach(), targets['rf-semantic'].cuda().detach()
            else:
                images_i, images_b, images_r = img_in.cuda().detach(), \
                                               targets['background'].cuda().detach(), None
                semantic_b, semantic_r = targets['bg-semantic'].cuda().detach(), None

            psnr_list.append(trainer.evaluate(images_i, images_b, images_r))

            # if it > 5:
            #     break

        psnr = np.mean(psnr_list)
        if psnr > trainer.best_result:
            trainer.best_result = psnr
            print()
            print('=' * 40)
            print('<{}> Save the model, the best background PNSR is {}'.format(get_local_time(), psnr))
            print('=' * 40)
            trainer.best_result = psnr
            trainer.save(checkpoint_directory, epoch)
