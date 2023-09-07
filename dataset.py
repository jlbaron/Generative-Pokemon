import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt

class PokemonDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv("data\\pokemon.csv")
        self.root_dir = "data\\images"
        self.transform = transforms.Compose([
                    transforms.CenterCrop((96, 96)), # going with center crop as subject has a lot of space around it
                    transforms.Resize((64, 64)), # size it down to 64x64
                    transforms.ToTensor(),  # Convert the image to a tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
                ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['Name'] + '.png'
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        # Preprocess the image
        processed_image = self.transform(image)
        return processed_image

def view_image_samples(idx=0):
    df = pd.read_csv("data\\pokemon.csv")
    image_name = df.iloc[idx]['Name'] + '.png'
    image_path = os.path.join("data\\images", image_name)
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
                    transforms.CenterCrop((96, 96)), # going with center crop instead as subject has a lot of space around it
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),  # Convert the image to a tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
                ])
    transformed_image = transform(image)
    plt.imshow(transformed_image.permute(1, 2, 0))
    plt.savefig('visualizations\\transformed_samples\\'+image_name)

# for i in range(50):
#     view_image_samples(i)

def convert_jpg_to_png(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(directory, filename))
            png_filename = os.path.splitext(filename)[0] + ".png"
            img.save(os.path.join(directory, png_filename))
            # img.show()

convert_jpg_to_png('data\\images')