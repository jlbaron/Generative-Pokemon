import yaml
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pandas as pd


def read_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
    plt.figure(figsize=(2, 2))
    plt.axis("off")
    plt.title(image_name)
    plt.imshow(transformed_image.permute(1, 2, 0))
    plt.savefig('visualizations\\transformed_samples\\'+image_name)


# for i in range(50):
#     view_image_samples(i)

# had to use an outside tool to remove background from jpg images
# stored results in a folder and used this to rename (Thanks Zyro)
def move_converted_files():
    names = [file for file in os.listdir("visualizations\\convert_me")]
    for i in range(len(names)):
        old_file_name = f"visualizations\\converted\\zyro-image ({i}).png"
        name = names[i]
        new_file_name = f"visualizations\\converted_images\\{name[:-4]}.png"
        img = Image.open(old_file_name)
        img.save(new_file_name)
# move_converted_files()