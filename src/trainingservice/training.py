"""
Provides the API for transferlerning on an EfficientNet.
"""

import itertools
import pathlib

import tensorboardX
import torch
import torch.nn as nn
import torchvision
import tqdm


class TransferEfficientNet(nn.Module):
    def __init__(
        self,
        num_classes,
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
        self.get_tool = lambda x: x.parents[2].stem

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
            # Convert tool name string into target for network
            # (usually one hot encoding)
            label = self.target_transform(label)
        return image, label

    def get_labels(self):
        labels = set()
        for img_path in self.img_paths:
            label = self.get_tool(img_path)
            labels.add(label)
        return labels


def train(data_dirs, max_epochs, batch_size, log_dir):
    log_dir = pathlib.Path(log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir()

    data_dirs = list(map(pathlib.Path, data_dirs))

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on {device}")

    train_datasets = []
    val_datasets = []
    for data_dir in data_dirs:
        train_datasets.append(ToolDataset(data_dir / "train"))
        val_datasets.append(ToolDataset(data_dir / "val"))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    normalization_transform = (
        torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms()
    )

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

    label_list = map(lambda x: x.get_labels(), train_datasets)
    label_list = itertools.chain.from_iterable(label_list)
    label_list = list(set(label_list))

    label_mapping = {label: index for index, label in enumerate(label_list)}
    label_mapping_str = "\n".join(f"{label}: {index}" for label, index in label_mapping.items())
    label_map_file = log_dir / "label_map.txt"

    with open(label_map_file, "w") as f:
        f.write(label_mapping_str)


    def label_transform(label):
        """
        Converts the name of a tool into a one hot
        vector encoding.
        """
        index = label_list.index(label)
        return torch.tensor(index)

    for dataset in train_datasets:
        dataset.transform = transform
        dataset.target_transform = label_transform
    for dataset in val_datasets:
        dataset.transform = normalization_transform
        dataset.target_transform = label_transform

    model = TransferEfficientNet(num_classes=len(label_list))
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 30], gamma=0.1
    )

    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    def save_model(name, epoch, step):
        for path in checkpoint_dir.glob(f"{name}*"):
            path.unlink()
        file_name = f"{name}_epoch_{epoch}_step_{step}"
        torch.save(model.state_dict(), checkpoint_dir / f"{file_name}.pth")
        traced_model = torch.jit.trace(
            model, torch.rand(1, 3, 720, 1280, device=device)
        )
        traced_model.save(checkpoint_dir / f"{file_name}.pt")

    writer = tensorboardX.SummaryWriter(log_dir)

    max_val_acc = 0
    min_val_loss = float("inf")

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        # Training phase
        model.train()
        train_correct = 0
        for batch_idx, batch in tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            batch_loss = loss.item()
            writer.add_scalar(
                "Loss/Train", batch_loss, epoch * len(train_loader) + batch_idx
            )

            _, predictions = torch.max(outputs, 1)
            batch_correct = (predictions == labels).sum().item()
            train_correct += batch_correct

        train_accuracy = train_correct / len(train_loader.dataset)
        writer.add_scalar(
            "Accuracy/Train", train_accuracy, (epoch + 1) * len(train_loader)
        )

        # Validation phase
        model.eval()
        val_correct = 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predictions = torch.max(outputs, 1)

                val_correct += (predictions == labels).sum().item()

        average_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)
        writer.add_scalar("Loss/Validation", average_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        if average_val_loss < min_val_loss or val_accuracy > max_val_acc:
            step = (epoch + 1) * len(train_loader)
            save_model("min_val_loss", epoch, step)

        if val_accuracy > max_val_acc:
            step = (epoch + 1) * len(train_loader)
            save_model("max_val_accuracy", epoch, step)

        scheduler.step()
    writer.close()

    weights_file = list(checkpoint_dir.glob("min_val_loss*.pth"))[0]
    trace_file = list(checkpoint_dir.glob("min_val_loss*.pt"))[0]
    return weights_file, trace_file, label_map_file
