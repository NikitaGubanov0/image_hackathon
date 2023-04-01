import argparse
import os
import warnings
import pickle
from os import listdir

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LinearRegression
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007

from utils.XYZ_to_SRGB import XYZ_TO_SRGB

class DenoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Conv2d(64, 64, (3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ) for _ in range(5)],
            nn.Conv2d(64, 3, (3, 3), padding=1)
        )

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class ImagePipeline:
    def __init__(self, path, output_path, model, demo, pattern, lin, srgb):
        self.path = path
        self.sample_paths = [f for f in listdir(path) if f.endswith('sample.npy')]
        self.gt_paths = [path.replace('sample', 'gt') for path in self.sample_paths]
        self.model = model
        self.demo = demo
        self.pattern = pattern
        self.lin = lin
        self.srgb = srgb
        self.output_path = output_path

    def generate(self):
        for i, (sample_path, gt_path) in enumerate(zip(self.sample_paths, self.gt_paths)):
            sample = np.load(os.path.join(self.path, sample_path), allow_pickle=True)
            gt = np.load(os.path.join(self.path, gt_path), allow_pickle=True)
            gt_xyz = np.clip(gt.item().get('xyz'), 0., 1.)
            sample_xyz = sample.item().get('image') / 255.
            sample_xyz = self.demo(sample_xyz, pattern=self.pattern)
            sample_xyz = np.clip(sample_xyz, 0., 1.)

            self.model.eval()
            device = torch.device('cuda')
            pred = self.model(torch.permute(torch.from_numpy(sample_xyz.astype('float32')), (2, 0, 1)).to(device).unsqueeze(0))
            pred = (pred - pred.min()) / pred.max()
            x = torch.permute(pred.detach().cpu().squeeze(0), (1, 2, 0)).numpy()

            self.lin.fit(x.reshape(-1, 3), gt_xyz.reshape(-1, 3))
            pred = self.lin.predict(x.reshape(-1, 3)).reshape(512, 512, 3)
            pred = (pred - pred.min()) / pred.max()
            pred_srgb = self.srgb.XYZ_to_sRGB(pred)
            gt_srgb = self.srgb.XYZ_to_sRGB(gt_xyz)

            pred_img = Image.fromarray((pred_srgb * 255).astype(np.uint8))
            pred_img.save(os.path.join(self.output_path, 'predicted', f"{i}.png"))

            gt_img = Image.fromarray((gt_srgb * 255).astype(np.uint8))
            gt_img.save(os.path.join(self.output_path, 'target', f"{i}.png"))

            print(f'saved {i}')
        return len(self.sample_paths)
      
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the quality of the predicted image.')
    parser.add_argument(
        'input_path',
        type=str,
        help='The path to the input images.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='The path to the output images.'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.join(args.output_path, 'predicted'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'target'), exist_ok=True)

    device = torch.device('cuda')
    model = DenoNet().to(device)
    lin = pickle.load(open('lin.sav', 'rb'))
    model.load_state_dict(torch.load('denoise.pt'))
    demo = demosaicing_CFA_Bayer_Menon2007
    pattern = 'GBRG'
    srgb = XYZ_TO_SRGB()

    image_pipeline = ImagePipeline(args.input_path, args.output_path, model, demo, pattern, lin, srgb)
    image_pipeline.generate()

    print(f'Generated {len(image_pipeline.sample_paths)} images.')

if __name__ == '__main__':
    main()

