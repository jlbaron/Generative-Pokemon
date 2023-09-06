from dataset import PokemonDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import yaml
import os
import matplotlib.pyplot as plt
from models.discriminator import Discriminator
from models.generator import Generator

def plot_curves(train_losses, val_losses):
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("visualizations\\training_plots\\loss.png")
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
    # train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2])
    dataloader = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Define your model
    generator = Generator()
    generator.apply(weights_init)
    discriminator = Discriminator()
    discriminator.apply(weights_init)

    # Define your criterion and optimizer
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, config['latent_vector'], 1, 1, device=device)
    real_label, fake_label = 1., 0.
    optimizerG = nn.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(beta1, 0.999))
    optimizerD = nn.optim.Adam(generator.parameters(), lr=config['lr'], betas=(beta1, 0.999))

    # Training loop
    gen_losses = []
    dis_losses = []
    for epoch in range(config['epochs']):
        total_loss = 0.
        for idx, data in enumerate(full_dataset):
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
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
            noise = torch.randn(b_size, nz, 1, 1, device=device)
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
            # Update D
            optimizerD.step()

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
        
            gen_losses.append(errG.item())
            dis_losses.append(errD.item())
        print(f"--------EPOCH {epoch+1}, GEN LOSS: {train_loss}, DIS LOSS: {val_loss}---------")
        print("---------------------------------------------------")
    gen_path = os.path.join('checkpoints', f"generator_trained.pt")
    torch.save(generator.state_dict(), gen_path)
    print(f"Generator saved to '{gen_path}'.")
    dis_path = os.path.join('checkpoints', f"discriminator_trained.pt")
    torch.save(generator.state_dict(), dis_path)
    print(f"Discriminator saved to '{dis_path}'.")
    plot_curves(gen_losses, dis_losses)

if __name__ == '__main__':
    main()