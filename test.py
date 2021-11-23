from tqdm import tqdm

from utils import get_config, label2colormap
from trainer import Trainer
import argparse
from torch.autograd import Variable
import torch
import os
import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/semantic_rr.yaml', help="net configuration")
parser.add_argument('--checkpoint', type=str, default='checkpoints/outputs/semantic_rr-wo_adv/checkpoints/gen_148.pt',
                    help="pwd of checkpoints")
parser.add_argument('--input_dir', type=str, default='/home/ros/ws/dataset/RRdataset/test/input',
                    help="input image path")
parser.add_argument('--gt_bg_dir', type=str, default='/home/ros/ws/dataset/RRdataset/test/background',
                    help="ground truth background image path")
parser.add_argument('--gt_rf_dir', type=str, default='/home/ros/ws/dataset/RRdataset/test/reflection',
                    help="ground truth reflection image path")
parser.add_argument('--gt_sm_dir', type=str, default='/home/ros/ws/dataset/RRdataset/test/semantic',
                    help="ground truth reflection image path")
parser.add_argument('--output_dir', type=str, default='resutls/semantic_rr-127',
                    help="output image path")
opts = parser.parse_args()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


if not os.path.exists(opts.output_dir):
    os.makedirs(opts.output_dir)

# Load experiment setting
config = get_config(opts.config)

trainer = Trainer(config)

state_dict = torch.load(opts.checkpoint, map_location='cuda:0')
trainer.generator.load_state_dict(state_dict['generator'])

trainer.cuda()
trainer.eval()

if 'new_size' in config:
    new_size = config['new_size']
else:
    new_size = config['new_size_i']

with torch.no_grad():

    image_paths = os.listdir(opts.input_dir)
    image_paths = [x for x in image_paths if is_image_file(x)]
    t_bar = tqdm(image_paths)
    t_bar.set_description('Processing')
    for image_name in t_bar:
        image_pwd = os.path.join(opts.input_dir, image_name)
        image_base = image_name.split('.')[0]

        img_in = cv2.imread(image_pwd)
        tf_img_in = Variable(torch.from_numpy(np.transpose((img_in / 255. - 0.5) * 2, (2, 0, 1)))).float().cuda().unsqueeze(0)

        # Start testing
        tf_pred_bg, tf_pred_rf, tf_pred_sm, _ = trainer.generator(tf_img_in)
        
        pred_bg = np.uint8(np.clip((tf_pred_bg / 2. + 0.5).cpu().squeeze().data.numpy().transpose((1, 2, 0)), 0, 1) * 255)
        pred_rf = np.uint8(np.clip((tf_pred_rf / 2. + 0.5).cpu().squeeze().data.numpy().transpose((1, 2, 0)), 0, 1) * 255)
        if tf_pred_sm is not None:
            pred_sm = torch.argmax(tf_pred_sm[0], dim=0).detach().long().cpu().numpy()
            pred_sm_color = label2colormap(pred_sm)[:, :, ::-1]

        cv2.imwrite(os.path.join(opts.output_dir, '{}-input.jpg'.format(image_base)), img_in)
        cv2.imwrite(os.path.join(opts.output_dir, '{}-predict-background.jpg'.format(image_base)), pred_bg)
        cv2.imwrite(os.path.join(opts.output_dir, '{}-predict-reflection.jpg'.format(image_base)), pred_rf)
        if tf_pred_sm is not None:
            cv2.imwrite(os.path.join(opts.output_dir, '{}-predict-semantic.jpg'.format(image_base)), pred_sm)
            cv2.imwrite(os.path.join(opts.output_dir, '{}-predict-semantic-color.jpg'.format(image_base)), pred_sm_color)

        if os.path.exists(os.path.join(opts.gt_bg_dir, image_name)):
            img_gt = cv2.imread(os.path.join(opts.gt_bg_dir, image_name))
            cv2.imwrite(os.path.join(opts.output_dir, "{}-label-background.jpg".format(image_base)), img_gt)
        
        if os.path.exists(os.path.join(opts.gt_rf_dir, image_name)):
            img_gt = cv2.imread(os.path.join(opts.gt_bg_dir, image_name))
            cv2.imwrite(os.path.join(opts.output_dir, "{}-label-reflection.jpg".format(image_base)), img_gt)
        
        if os.path.exists(os.path.join(opts.gt_sm_dir, image_name)):
            img_gt = cv2.imread(os.path.join(opts.gt_sm_dir, image_name), 0)
            img_gt_color = label2colormap(img_gt)[:, :, ::-1]
            cv2.imwrite(os.path.join(opts.output_dir, "{}-label-semantic.png".format(image_base)), img_gt)
            cv2.imwrite(os.path.join(opts.output_dir, "{}-label-semantic-color.jpg".format(image_base)), img_gt_color)

print('Done!')
