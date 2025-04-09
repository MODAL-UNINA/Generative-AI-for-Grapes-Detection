# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
import torch
import argparse

np.set_printoptions(suppress=True)
plt.rcParams['figure.figsize'] = [15, 8]

# Import Crop GAN related libs
root_dir = os.getcwd()
gan_dir = os.path.join(root_dir, "CycleGAN")
sys.path.append(gan_dir)

from options.train_options import TrainOptions
from options.test_options import TestOptions

from data import create_dataset
from models import create_model
import util.util as utils

# %%
# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='Dataset_CycleGAN/created_dataset', help='path to images (should have subfolders trainA, trainB, valA, valB)')
parser.add_argument('--num_threads', default=10, type=int, help='# threads for loading data')
parser.add_argument('--name', type=str, default='Grapes', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
parser.add_argument('--checkpoints_dir', type=str, default='CycleGAN/model_saves', help='models are saved here')
parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. CycleGAN is cycle_gan')
parser.add_argument('--load_size', type=int, default=640, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=416, help='then crop to this size')
parser.add_argument('--batch_size', type=int, default=3, help='input batch size')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--save_epoch_freq', type=int, default=100, help='frequency of saving checkpoints')

args, _ = parser.parse_known_args()
arguments = " ".join([f"--{k} {v}" for k, v in vars(args).items()])
sys.argv = arguments.split()
opt = TrainOptions().parse_notebook(arguments.split())

# %%
# Load the model
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()

load_suffix = "latest"
model.load_networks(load_suffix)
# %%
# Generate images
folder_generated = f"Generated_images/{args.name}/"
folder = 'Synthetic_images'
os.makedirs(folder_generated, exist_ok=True)

for img in os.listdir(folder):
    img_path = os.path.join(folder, img)
    real_a_img = Image.open(img_path).convert("RGB")

    # Preprocess image
    real_a_img_tensor, real_a_img_np = utils.preprocess_images(real_a_img, resize=[args.load_size, args.load_size])

    with torch.no_grad():
        fake_b_img = model.netG_A(real_a_img_tensor)


    fake_b_img_np = (
        fake_b_img.detach().cpu().squeeze(0).permute([1, 2, 0]).numpy() * 0.5 + 0.5
    ) * 255
    fake_b_img_np = fake_b_img_np.astype("uint8")

    fake_b_img_pil = Image.fromarray(fake_b_img_np)

    # Saving the generated image
    save_path = os.path.join(folder_generated, f"generated_{img}")
    fake_b_img_pil.save(save_path)

    print(f"Saved: {save_path}")

# %%
