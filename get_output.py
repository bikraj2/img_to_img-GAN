from models.generator import Generator
from configs.dataset import TestDataset
from configs.utils import save_image
from tqdm import tqdm 
import torch
import uuid
import configs.config as config
def get_output(model:Generator,dataDir):
    test_dataset = TestDataset(root_dir=dataDir)

    index = 0
    folder = "mobile_processed_images"
    for x, y in tqdm(test_dataset, desc="Generating images"):
        with torch.no_grad():
            x = x.unsqueeze(0).to(config.DEVICE)  # Add batch dimension and move to device
            y_fake = model(x)
            y_fake = y_fake * 0.5 + 0.5  # Rescale to [0, 1]
            # Save images
            save_image(y_fake, f"{folder}/y_gen{uuid.uuid4()}_.png")
            # save_image(x * 0.5 + 0.5, f"{folder}/input_{index}.png")
    return  f"{folder}/y_gen{uuid.uuid4()}_.png"
        


