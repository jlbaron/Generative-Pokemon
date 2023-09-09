import os
import yaml
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from dataset import PokemonDataset
from models.discriminator import Discriminator
from models.generator import Generator

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

def plot_curves(train_losses, val_losses):
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("visualizations\\training_plots\\loss.png")
    plt.show()

def plot_imgs(img_list, real_batch, device):
    # create GIF
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    writer = animation.PillowWriter(fps=30)
    ani.save('visualizations\\animation.gif', writer=writer)
    plt.show()

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig("visualizations\\Real_vs_Fake.png")
    plt.show()

def read_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    config = read_config("configs\\config.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    full_dataset = PokemonDataset()
    dataloader = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=True)

    # Define your model
    generator = Generator(nz=config['nz'], ngf=config['ngf'], nc=config['nc'])
    generator.apply(weights_init)
    discriminator = Discriminator(nc=config['nc'], ndf=config['ndf'])
    discriminator.apply(weights_init)

    # Binary loss with options [real (1), fake (0)]
    criterion = nn.BCELoss()
    real_label, fake_label = 1., 0.

    # fixed noise for visualizations
    fixed_noise = torch.randn(64, config['nz'], 1, 1, device=device)

    # optimizer for each net
    optimizerG = torch.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
    optimizerD = torch.optim.Adam(generator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

    # Training loop
    img_list = []
    gen_losses = []
    dis_losses = []
    for epoch in range(config['epochs']):
        for idx, data in enumerate(dataloader):

            # generator generates on noise
            # discriminator discriminates between real and generated


            ## update discriminator: maximize log(D(x)) + log(1-D(G(z)))
            ## Train with all-real batch
            discriminator.zero_grad()

            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update G
            optimizerG.step()
            # Update D
            optimizerD.step()
        
            gen_losses.append(errG.item())
            dis_losses.append(errD.item())
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, config['epochs'], idx, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if (epoch % 10 == 0) or (epoch == config['epochs']-1):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                    img_list.append(make_grid(fake, padding=2, normalize=True))

    # save trained models
    gen_path = os.path.join('checkpoints', f"generator_trained.pt")
    torch.save(generator.state_dict(), gen_path)
    print(f"Generator saved to '{gen_path}'.")
    dis_path = os.path.join('checkpoints', f"discriminator_trained.pt")
    torch.save(generator.state_dict(), dis_path)
    print(f"Discriminator saved to '{dis_path}'.")

    # plot curves and create animation
    plot_curves(gen_losses, dis_losses)
    real_batch = next(iter(dataloader))
    print(real_batch.shape)
    plot_imgs(img_list, real_batch, device)

if __name__ == '__main__':
    main()