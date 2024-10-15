import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pickle
import numpy as np
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, ColorJitter
class AnimalDataset(Dataset):
    def __init__(self, root="data/animals", train=True, transform=None):
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        if train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        self.image_paths = []
        self.labels = []

        for category in self.categories:
            category_path = os.path.join(data_path, category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(self.categories.index(category))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        if self.transform:
            image = self.transform(image)

        label = self.labels[item]
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224,244)),
    ])
    dataset = AnimalDataset(root="./data/animals", train=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=6,
        drop_last=True,
    )
    for images, labels in dataloader:
        print(images.shape)
        print(labels)