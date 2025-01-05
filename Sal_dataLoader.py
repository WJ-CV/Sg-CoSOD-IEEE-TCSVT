import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from torch.utils import data

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, depth_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = []
        self.depths = []
        self.gts = []
        images_files= os.listdir(gt_root)
        for file in images_files:
            self.images.append(image_root + file[:-4] + '.jpg')
            self.depths.append(depth_root + file[:-4] + '.jpg')
            self.gts.append(gt_root + file)

        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        image, depth, gt = randomFlip(image, depth, gt)
        image, depth, gt = randomCrop(image, depth, gt)
        image, depth, gt = randomRotation(image, depth, gt)
        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        depth = self.depths_transform(depth)
        gt = self.gt_transform(gt)
        return image, depth, gt

    def __len__(self):
        return self.size

    def filter_files(self):
        """ Check whether a set of images match in size. """
        assert len(self.images) == len(self.depths) == len(self.gts)
        images = []
        depths = []
        gts = []
        for img_path, depth_path, gt_path in zip(self.images, self.depths, self.gts):
            # Notes: On DUT dataset, the size of training depth images are [256, 256],
            # it is not matched with RGB images and GT [600, 400].
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
            else:
                print(img_path)
                raise Exception("Image sizes do not match, please check.")
        self.images = images
        self.depths = depths
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            # Removing alpha channel.
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

def randomFlip(img, depth, gt):
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_flag == 2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
    return img, depth, gt

def randomRotation(image, depth, gt):
    mode = Image.BICUBIC
    if random.random() > 0.3:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return image, depth, gt

def randomCrop(image, depth, gt):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), depth.crop(random_region), gt.crop(random_region)

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

def get_loader(image_root, depth_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    dataset = SalObjDataset(image_root, depth_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

