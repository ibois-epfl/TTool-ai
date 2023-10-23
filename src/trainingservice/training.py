"""
Provides the API for transferlerning on an EfficientNet.
"""

import pathlib

import torch
import torch.nn as nn
import torchvision


class TransferEfficientNet(nn.Module):
    def __init__(
        self,
        num_classes=14,
        **kwargs,
    ):
        """
        EfficientNet model with default weights from torchvision
        and a custom classification head to match the number
        of classes we have.
        """
        super().__init__()
        backbone = torchvision.models.efficientnet_v2_s(weights="DEFAULT")
        layers = list(backbone.children())
        num_filters = layers[-1][1].in_features
        self.feature_extractor = nn.Sequential(*layers[:-2]).eval()

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Dropout(p=0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(num_filters, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classification_head(x)
        return x


class ToolDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        transform=None,
        target_transform=None,
        subsampling=1,
    ):
        """
        Data set class that loads png images from a folder.
        The images have to respect the following naming
        convention: tool_name__ect.png, where the name of the
        tool comes first and is seperated by a double underscore
        from the rest of the file name.

        Parameters
        ----------
        img_dir : pathlib.Path
            Path to folder with png images.
        transform : function
            Function applied to the image, e.g.
            to normalize and/or augment the images.
            Input and output of the function should be
            and image.
        target_transform : function
            Function used to convert the target values.
            Usually this is a function that converts the tool
            name string into a one hot vector encoding.
        subsampling : int
            Integer used to subsample the data by only using every
            nth image.
        """
        self.img_dir = img_dir

        # Get list of all image paths
        self.img_paths = sorted(list(img_dir.glob("*.png")))

        # Subsample
        self.img_paths = self.img_paths[::subsampling]

        self.transform = transform
        self.target_transform = target_transform

        # Define the function used to extract the ground truth tool name
        # from the file name
        self.get_tool = lambda x: x.stem.split("__")[0]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = torchvision.io.read_image(str(img_path))
        # Convert image to float
        image = image.float() / 255

        # Extract ground truth tool name
        label = self.get_tool(img_path)

        if self.transform:
            # Apply normalization and augmentation if provided
            image = self.transform(image)
        if self.target_transform:
            # Convert tool name string into target for network (usually one hot encoding)
            label = self.target_transform(label)
        return image, label

    def get_labels(self):
        labels = set()
        for img_path in self.img_paths:
            label = self.get_tool(img_path)
            labels.add(label)
        return labels


def train(training_data, validation_data, max_epochs, batch_size):
    torch.set_float32_matmul_precision("high")

    train_dataset = ToolDataset(pathlib.Path("."))

    labels = train_dataset.get_labels()

    normalization_transform = (
        torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms()
    )
    network = TransferEfficientNet(num_classes=10)

    augmentation_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomRotation(degrees=30),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ColorJitter(hue=0.3),
        ]
    )
    transform = torchvision.transforms.Compose(
        [augmentation_transforms, normalization_transform]
    )

    def label_transform(label):
        """
        Converts the name of a tool into a one hot
        vector encoding.
        """
        return labels.index(label)
