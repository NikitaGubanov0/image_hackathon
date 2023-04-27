import argparse
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from os import listdir
from utils.XYZ_to_SRGB import XYZ_TO_SRGB
import pickle
import warnings

class DenoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv7 = nn.Conv2d(64, 3, (3, 3), padding=1)

    def forward(self, x):
        var = self.conv1(x)
        var = self.conv2(var)
        var = self.conv3(var)
        var = self.conv4(var)
        var = self.conv5(var)
        var = self.conv6(var)
        var = self.conv7(var)
        return x + var

class ImagePipeline:
    def __init__(self, path, output_path, model, demo, pattern, lin, srgb):
        self.path = path
        self.sample_paths = [f for f in listdir(path) if f.endswith('sample.npy')]
        self.gt_paths = [el.replace('sample', 'gt') for el in self.sample_paths]
        self.sample_data = []
        self.gt_data = []
        self.model = model
        self.demo = demo
        self.pattern=pattern
        self.lin = lin
        self.srgb = srgb
        self.output_path = output_path

    def generate(self):
        for i in range(len(self.sample_paths)):
            sm = np.load(self.path + '/' + self.sample_paths[i], allow_pickle=True)
            gt = np.load(self.path + '/' + self.gt_paths[i], allow_pickle=True)
            gt_xyz  = gt.item().get('xyz')
            gt_xyz = np.clip(gt_xyz, 0., 1.)
            sm_xyz  = sm.item().get('image')
            sm_xyz = sm_xyz / 255.
            sm_xyz = self.demo(sm_xyz, pattern=self.pattern)
            sm_xyz = np.clip(sm_xyz, 0., 1.)
            test = sm_xyz
            self.model.eval()
            device = torch.device('cuda')
            pred = self.model(torch.permute(torch.from_numpy(test.astype('float32')), (2, 0, 1)).to(device).unsqueeze(0))
            pred = pred - pred.min()
            pred = pred / pred.max()
            x = torch.permute(pred.detach().cpu().squeeze(0), (1, 2, 0)).numpy()
            y = gt_xyz
            self.lin.fit(x.reshape(-1, 3), y.reshape(-1, 3))
            pred = self.lin.predict(x.reshape(-1, 3)).reshape(512, 512, 3)
            pred = pred - pred.min()
            pred = pred / pred.max()
            pred = self.srgb.XYZ_to_sRGB(pred)
            gt_xyz = self.srgb.XYZ_to_sRGB(gt_xyz)
            im = Image.fromarray((pred * 255).astype(np.uint8))
            im.save(self.output_path + '/predicted/' + f"{i}.png")
            gt = Image.fromarray((gt_xyz * 255).astype(np.uint8))
            gt.save(self.output_path + '/ground_truth/' + f"{i}.png")
            print(f'Successfully saved {i}!')
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
    os.makedirs(args.output_path + '/' + 'predicted', exist_ok=True)
    os.makedirs(args.output_path + '/' + 'ground_truth', exist_ok=True)
    device = torch.device('cuda')
    model = DenoNet().to(device)
    lin = pickle.load(open('lin.sav', 'rb'))
    model.load_state_dict(torch.load('denoise.pt'))
    demo = demosaicing_CFA_Bayer_Menon2007
    pattern='GBRG'
    srgb = XYZ_TO_SRGB()
    image_pipeline = ImagePipeline(args.input_path, args.output_path, model, demo, pattern, lin, srgb)
    n = image_pipeline.generate()
    print(f'Generated {n} images.')
    
if __name__ == '__main__':
    main()
