from torch import nn, optim 
from torch.utils.data import DataLoader
from models.discriminator import Discriminator
from configs import config
from models.generator import Generator
from configs.utils import *
from configs.dataset import MapDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
def train(disc,gen,loader,opt_disc,opt_gen,l1,bce,g_scaler,d_scaler):
    loop = tqdm(loader,leave = True)
    generator_loss = []
    epoch = []
    discriminator_loss =[]
    i=0
    for idx,(x,y) in enumerate(loop):
        x,y = x.to(config.DEVICE),y.to(config.DEVICE)
    #   Train discriminator
        with torch.cuda.amp.autocast():
            y_fake =gen(x)
            D_real = disc(x,y)
            D_fake = disc(x,y_fake.detach())
            D_real_loss = bce(D_real,torch.ones_like(D_real))
            D_fake_loss = bce(D_fake,torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2 # try removing the 2
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        print(type(D_loss.item()))
        discriminator_loss.append(D_loss.item())

    # training the generator 
        with torch.cuda.amp.autocast():
            D_fake = disc(x,y_fake)
            G_fake_loss = bce(D_fake,torch.ones_like(D_fake))
            L1 =l1(y_fake,y)* config.L1_LAMBDA
            G_loss = G_fake_loss + L1
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        print(type(G_loss.item()))
        generator_loss.append(G_loss.item())
        i+=1
        epoch.append(i)
        plotLoss(epoch,generator_loss,discriminator_loss)

        
def main():
    disc =Discriminator(in_channels=3).to(config.DEVICE) 
    gen =Generator(in_channels=3).to(config.DEVICE) 
    opt_disc = optim.Adam(disc.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.99))
    opt_gen = optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.99))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC,disc,opt_disc,config.LEARNING_RATE)
    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)

    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)
    for epoch in range(config.NUM_EPOCHS):
        train(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE,g_scaler,d_scaler)
        if epoch%5==0:
            save_checkpoint(gen,opt_gen,filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc,opt_disc,filename=config.CHECKPOINT_DISC)
        save_some_examples(gen,val_loader,epoch,folder="evaluation")

def plotLoss(epoch, generator_loss, discriminator_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Convert tensor to numpy array
    print(epoch,generator_loss,discriminator_loss)
    # Plot data on the first subplot
    ax1.plot(epoch, generator_loss)
    ax1.set_title('Generator ')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('L1Loss + CGAN Loss')

    # Plot data on the second subplot
    ax2.plot(epoch, discriminator_loss)
    ax2.set_title('Discriminator')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Loss')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.savefig("generator_loss_vs_discriminator_loss.png")
if __name__ == "__main__":
    main()
