import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class Flickr30kDataset(Dataset):
    # csv_file에서 annotations, img 주소, 형태를 정의
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,idx):
        img_name = os.path.join(self.img_dir,self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB') #image열어서 RGB로 변형
        caption = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, caption

def get_loader(csv_file, img_dir, batch_size, shuffle, num_workers):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    dataset = Flickr30kDataset(csv_file, img_dir, transform=transform)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return data_loader