from model import config
from model.dataset import dataset
from model.model import UNet

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from torchvision import transforms as T
from imutils import paths

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
# loading the image and masks in sorted order

imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

split = train_test_split(
    imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42
)

(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

print("[INFO] saving testing image paths >>>> ")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close

# Transformation
transforms = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
        T.ToTensor(),
    ]
)

# create train test datasets
trainDS = dataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
testDS = dataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)

print(f"[INFO] found {len(trainDS)} images in train dataset")
print(f"[INFO] found {len(testDS)} images in test dataset")


class UNetDataLoader:
    def __init__(
        self, trainDS, testDS, batch_size, num_workers, pin_memory, transforms
    ):
        self.trainDS = trainDS
        self.testDS = testDS

        self.trainLoader = DataLoader(
            self.trainDS,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers if torch.cuda.is_available() else 0,
        )

        self.testLoader = DataLoader(
            self.testDS,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers if torch.cuda.is_available() else 0,
        )

    def get_loaders(self):
        return self.trainLoader, self.testLoader


if __name__ == "__main__":
    # Instantiate DataLoader class
    data_loader = UNetDataLoader(
        trainDS=trainDS,
        testDS=testDS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        transforms=transforms,
    )

    trainLoader, testLoader = data_loader.get_loaders()


# ?Model Loading
unet = UNet().to(config.DEVICE)
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

h = {"train_loss": [], "test_loss": []}

print("[INFO] training the network")
tic = time.time()

for e in tqdm(range(config.NUM_EPOCHS)):
    unet.train()

    totalTrainLoss = 0
    totalTestLoss = 0

    for i, (x, y) in enumerate(trainLoader):
        (X, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        pred = unet(x)
        loss = lossFunc(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        totalTrainLoss += loss

    with torch.no_grad():
        unet.eval()

        for x, y in testLoader:
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            pred = unet(x)
            loss = lossFunc(pred, y)

            totalTestLoss += loss

    avgTrainLoss = totalTrainLoss / trainSteps
    avgTestLoss = totalTestLoss / testSteps

    h["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    h["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    print(f"[INFO] EPOCH:{e + 1}/{config.NUM_EPOCHS}")
    print(f"[INFO] Train loss: {avgTrainLoss:.4f}, Test loss: {avgTestLoss:.4f}")

    if e % 10 == 0:
        print("[INFO] model saved")
        torch.save(unet, os.path.join(config.BASE_OUTPUT, f"unet_tgs_salt{e}.pth"))

toc = time.time()
print(f"[INFO] Time taken: {toc - tic:.2f}")


# plotting
plt.style.use("ggplot")
plt.figure()
plt.plot(h["train_loss"], label="Train loss")
plt.plot(h["test_loss"], label="Test loss")
plt.title("Training loss on Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

torch.save(unet, config.MODEL_PATH)
