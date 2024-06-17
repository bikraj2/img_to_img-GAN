from torch import nn, optim 
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.discriminator import Discriminator
from models.generator import Generator
import configs.config as config
from configs.utils import *
from configs.dataset import TestDataset 
def main():
    disc =Discriminator(in_channels=3).to(config.DEVICE) 
    gen =Generator(in_channels=3).to(config.DEVICE) 
    opt_disc = optim.Adam(disc.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.99))
    opt_gen = optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.99))
    load_checkpoint("data/" + config.TRAIN_DATA+"/gen.pth.tar",gen,opt_gen,config.LEARNING_RATE)
    test_dataset = TestDataset(root_dir=config.TEST_DIR)

    test_loader = DataLoader(test_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)
    index = 0
    folder = config.DESTINATION
    for x, y in tqdm(test_dataset, desc="Generating images"):
        with torch.no_grad():
            x = x.unsqueeze(0).to(config.DEVICE)  # Add batch dimension and move to device
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # Rescale to [0, 1]
            
            # Save images
            save_image(y_fake, f"{folder}/y_gen_{index}.png")
            save_image(x * 0.5 + 0.5, f"{folder}/input_{index}.png")
        
        index += 1    


if __name__ == "__main__":
    main()
