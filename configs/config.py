import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/map/train"
VAL_DIR = "data/map/val"
TEST_DIR ="testImages/map"
TRAIN_DATA="map"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = True 
SAVE_MODEL = True
CHECKPOINT_DISC = "data/map/disc.pth.tar"
CHECKPOINT_GEN = "data/map/gen.pth.tar"
DESTINATION = "results/map"

both_transform = A.Compose(
    
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
    is_check_shapes=False
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
