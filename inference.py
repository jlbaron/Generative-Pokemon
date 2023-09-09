'''
start with random noise and produce an image
save to generated_samples/
'''
import argparse
import torch
import random
from models.generator import Generator
import matplotlib.pyplot as plt
from utils import read_config
import numpy as np

parser = argparse.ArgumentParser(description='Pokemon GAN')
parser.add_argument('--config', default='.\\configs\\config.yaml', help='Path to the configuration file. Default: .\\configs\\config_CNN.yaml')
parser.add_argument('--random', type=bool, default=False, help='True/False for random seed on generation')
parser.add_argument('--gen-file', default='checkpoints\\generator_trained.pt', help='Path to trained generator.')
parser.add_argument('--image-name', default='sample', help='Name for outputted images (omit file extension).')
parser.add_argument('--image-count', type=int, default=1, help='Number of samples to produce.')

def load_model(gen_path, config):
    generator = Generator(nz=config['nz'], ngf=config['ngf'], nc=config['nc'])  # Replace with your actual model class
    generator.load_state_dict(torch.load(gen_path))
    generator.eval()
    return generator

def inference(generator, filename, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator.to(device)
    
    noise = torch.randn(1, config['nz'], 1, 1, device=device)
    fake = generator(noise).detach().cpu()
    plt.figure(figsize=(2, 2))
    plt.axis("off")
    plt.title(filename)
    plt.imshow(np.transpose(fake[0], (1, 2, 0)))
    plt.savefig('generated_samples\\'+filename)
    plt.show


# Add arguments for config file, generator, file name, files_count, random_seed
'''
Usage: python inference.py [OPTIONS]

Options:
  --config CONFIG_PATH  Path to the configuration file.
                        Default: .\\configs\\config.yaml

  --random IS_RANDOM  True/False for random seed on generation, False for reproducible results.
                        Default: False

  --gen-file GENERATOR_FILE  Path to trained generator
                        Default: "checkpoints\\generator_trained.pt"
  --image-name IMAGE_NAME  Name for outputted images (omit file extension)
                        Default: "sample"
  --image-count IMAGE_COUNT  Number of samples to produce
                        Default: 1
'''
if __name__ == "__main__":
    # get command line args
    global args
    args = parser.parse_args()

    # Set random seed for reproducibility
    manualSeed = 999
    if args.random:
        manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    config = read_config(args.config)

    gen_file = args.gen_file
    image_name = args.image_name
    image_count = args.image_count

    generator = load_model(gen_path=gen_file, config=config)
    if image_count > 1:
        for i in range(image_count):
            modified_image_name = image_name + f"({i}).png"
            inference(generator=generator, filename=modified_image_name, config=config)
    else:
        inference(generator=generator, filename=image_name+".png", config=config)