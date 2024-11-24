import os
from PIL import Image
from torch.utils.data import Dataset

class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = 0 if "cat" in self.images[idx] else 1  # Simplified binary classification (cat vs dog)

        if self.transform:
            image = self.transform(image)

        return image, label
