import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class PokemonDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv("data\\pokemon.csv")
        self.root_dir = "data\\images"
        self.transform = transforms.Compose([
                    transforms.Resize((196, 196)),  # Resize the image to (224, 224)
                    transforms.ToTensor(),  # Convert the image to a tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
                ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['name'] + '.png'
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)
        # Preprocess the image if needed
        processed_image = self.transform(image)
        label = self.data.iloc[idx]['type1']
        return processed_image, label

