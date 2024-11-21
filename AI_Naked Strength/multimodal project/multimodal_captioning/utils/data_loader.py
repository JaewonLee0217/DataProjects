import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Flickr30kDataset(Dataset):
    def __init__(self, img_dir, caption_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.captions = []

        with open(caption_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, caption = parts
                    self.images.append(img_name)
                    self.captions.append(caption)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        caption = self.captions[idx]

        if self.transform:
            image = self.transform(image)

        return image, caption


def get_loader(img_dir, caption_file, batch_size, shuffle, num_workers):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = Flickr30kDataset(img_dir, caption_file, transform=transform)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader