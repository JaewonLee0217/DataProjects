import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class Flickr30kDataset(Dataset):
    def __init__(self, img_dir, caption_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.captions = []

        print(f"Loading data from {caption_file}")
        with open(caption_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # 헤더 건너뛰기
            for row in csv_reader:
                if len(row) == 2:
                    img_name, caption = row
                    self.images.append(img_name)
                    self.captions.append(caption)

        print(f"Loaded {len(self.images)} images and captions")

        if len(self.images) == 0:
            raise ValueError("No data loaded. Check the caption file path and content.")

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

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check the data loading process.")

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader
