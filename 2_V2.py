# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchmetrics

# from tqdm import tqdm
# from torchvision.transforms import v2
# from torch.utils.data import DataLoader, Dataset
# from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights, resnet50, EfficientNet_B7_Weights
# from PIL import Image

# sns.set(style="whitegrid", palette="Set2")

# PATH = "/kaggle/input/tammathon-task-2"

# train_df = pd.read_csv(f"{PATH}/train.csv")
# valid_df = pd.read_csv(f"{PATH}/val.csv")
# test_df = pd.read_csv(f"{PATH}/test.csv")

# train_df.iloc[0]['path'].split('/')[0]
# class ICAODataset(Dataset):
#     def __init__(self, root_dir, dataframe, transforms=None):
#         self.root_dir = root_dir
#         self.dataframe = dataframe.reset_index(drop=True)
#         self.transforms = transforms
#         self.subset = dataframe.iloc[0]['path'].split('/')[0]

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]
#         img_path = os.path.join(self.root_dir, self.subset, row['path'])
#         img = Image.open(img_path).convert("RGB")  # ensure 3 channels
        
#         label = torch.tensor(row['label'], dtype=torch.float32)  # binary case
#         if self.transforms:
#             img = self.transforms(img)
#         return img, torch.tensor([label])


# train_transforms = v2.Compose([
#     v2.Resize(600),
#     v2.RandomHorizontalFlip(p=0.3),
#     v2.ColorJitter(0.4, 0.4, 0.4, 0.4),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=[0.485, 0.456, 0.406], 
#                  std=[0.229, 0.224, 0.225])
# ])

# valid_transforms = v2.Compose([
#     v2.Resize(600),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=[0.485, 0.456, 0.406], 
#                  std=[0.229, 0.224, 0.225])
# ])

# train_data = ICAODataset(PATH, train_df, transforms=train_transforms)
# valid_data = ICAODataset(PATH, valid_df, transforms=valid_transforms)

# train_loader = DataLoader(train_data, num_workers=0, shuffle=True, 
#                           pin_memory=True)
# valid_loader = DataLoader(valid_data, num_workers=0, shuffle=False, 
#                           pin_memory=True)

# class EffNetB7Base(nn.Module):
#     def __init__(self, n_classes=1):
#         super().__init__()
#         model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
#         in_features = model.classifier[1].in_features
        
#         self.backbone = nn.Sequential(*list(model.children())[:-1])
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(in_features, 1)

#     def forward(self, x):
#         out = self.backbone(x)
#         out = self.pool(out)
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#         return out  # raw logits


# class ResNet50Base(nn.Module):
#     def __init__(self, n_classes=1):
#         super().__init__()
#         model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         in_features = model.fc.in_features

#         # Remove the original classifier
#         self.backbone = nn.Sequential(*list(model.children())[:-1])
#         self.classifier = nn.Linear(in_features, n_classes)

#     def forward(self, x):
#         out = self.backbone(x)
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#         return out

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         probs = torch.sigmoid(inputs)
#         ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)  # pt = e^(-BCE)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
#         return focal_loss.mean()

# imgs, labels = next(iter(train_loader))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = {
#     "lr": 1e-4,
#     "weight_decay": 1e-4
# }

# effnet_model = EffNetB7Base(1).to(device)
# class_weights = torch.tensor([0.67, 0.33]).to(device)

# focal_loss = FocalLoss(alpha=0.75)
# bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# optimizer = torch.optim.AdamW(effnet_model.parameters(), 
#                               lr=config['lr'],
#                               weight_decay=config['weight_decay'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2) # So LR doesn't plateau smh
# metric_fn = torchmetrics.F1Score(task='binary')

# def train_model(
#     model,
#     train_loader, 
#     loss_fn,
#     optimizer,
#     metric_fn,
#     device,
#     scheduler=None,
# ):
#     model.train()
#     running_metric = 0.0
#     running_loss = 0.0

#     with tqdm(train_loader, unit="batch", desc="Training") as progress_bar:
#         for batch, (x, y) in enumerate(progress_bar):
#             x, y = x.to(device), y.to(device)

#             optimizer.zero_grad()
            
#             yhat = model(x) # Model returns logits,
#             yhat = torch.sigmoid(yhat) # Apply sigmoid for pred probs !!! 

#             loss = loss_fn(yhat, y)
#             loss.backward()
#             optimizer.step()

#             metric_fn.update(yhat, y)
#             running_loss += loss.item()
#             running_metric += metric_fn.compute().item()

#             progress_bar.set_postfix(loss=loss.item(), 
#                                      metric=running_metric / (batch + 1))

#     epoch_loss = running_loss / len(train_loader)
#     epoch_metric = running_metric / len(train_loader)

#     metric_fn.reset()

#     print(f"\nTraining Loss: {epoch_loss:.4f}, F1 Score: {epoch_metric:.4f}")
#     return epoch_loss, epoch_metric


# def validate_model(
#     model,
#     valid_loader,
#     loss_fn,
#     metric_fn,
#     device
# ):
#     model.eval()
#     running_loss = 0.0
#     running_metric = 0.0

#     with torch.inference_mode():
#         with tqdm(valid_loader, unit="batch", desc="Validation") as progress_bar:
#             for x, y in progress_bar:
#                 x, y = x.to(device), y.to(device)

#                 yhat = model(x)
#                 yhat = torch.sigmoid(yhat)

#                 loss = loss_fn(yhat, y)

#                 metric_fn.update(yhat, y)

#                 running_loss += loss.item()
#                 running_metric += metric_fn.compute().item()

#                 progress_bar.set_postfix(loss=loss.item(), 
#                                          metric=running_metric / (len(progress_bar) + 1))

#     epoch_loss = running_loss / len(valid_loader)
#     epoch_metric = running_metric / len(valid_loader)

#     metric_fn.reset()

#     print(f"\nValidation Loss: {epoch_loss:.4f}, F1 Score: {epoch_metric:.4f}")
#     return epoch_loss, epoch_metric


# def training_pipeline(
#     epochs,
#     model,
#     train_loader,
#     valid_loader,
#     loss_fn,
#     metric_fn,
#     optimizer,
#     device,
#     scheduler=None,
# ):
#     train_losses, valid_losses = [], []
#     train_metrics, valid_metrics = [], []
    
#     for epoch in range(epochs):
#         print(f"\nEpoch {epoch + 1}/{epochs}")

#         train_loss, train_metric = train_model(
#             model, train_loader, loss_fn, optimizer, metric_fn, device, scheduler
#         )
#         valid_loss, valid_metric = validate_model(
#             model, valid_loader, loss_fn, metric_fn, device
#         )

#         train_losses.append(train_loss)
#         valid_losses.append(valid_loss)
#         train_metrics.append(train_metric)
#         valid_metrics.append(valid_metric)

#         if scheduler:
#             scheduler.step(valid_loss)

#         print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
#         print(f"Validation Loss: {valid_loss:.4f}, Validation Metric: {valid_metric:.4f}")

#     return train_losses, valid_losses, train_metrics, valid_metrics

# train_losses, valid_losses, train_metrics, valid_metrics = training_pipeline(
#     10, effnet_model, train_loader, valid_loader, focal_loss, metric_fn, optimizer, device, scheduler
# )


# results_df = pd.DataFrame({
#     "epoch": list(range(1, len(train_losses) + 1)),
#     "train_loss": train_losses,
#     "valid_loss": valid_losses,
#     "train_f1": train_metrics,
#     "valid_f1": valid_metrics
# })

# results_df.to_csv("training_results.csv", index=False)



import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from tqdm import tqdm
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights, resnet50, EfficientNet_B7_Weights
from PIL import Image

sns.set(style="whitegrid", palette="Set2")

                    # PATH = "/kaggle/input/tammathon-task-2"  # path to data in the kaggle
                    # PATH = "/root/tammathon/data" # path to data in the server

                    # train_df = pd.read_csv(f"{PATH}/train.csv")
                    # valid_df = pd.read_csv(f"{PATH}/val.csv")
                    # test_df = pd.read_csv(f"{PATH}/test.csv")

                    # # train_df.iloc[0]['path'].split('/')[0]
                    # train_df.iloc[0]['filename'].split('/')[0]

                    # class ICAODataset(Dataset):
                    #     def __init__(self, root_dir, dataframe, transforms=None):
                    #         self.root_dir = root_dir
                    #         self.dataframe = dataframe.reset_index(drop=True)
                    #         self.transforms = transforms
                    #         self.subset = dataframe.iloc[0]['path'].split('/')[0]

                    #     def __len__(self):
                    #         return len(self.dataframe)

                    #     def __getitem__(self, idx):
                    #         row = self.dataframe.iloc[idx]
                    #         img_path = os.path.join(self.root_dir, self.subset, row['path'])
                    #         img = Image.open(img_path).convert("RGB")  # ensure 3 channels
                            
                    #         label = torch.tensor(row['label'], dtype=torch.float32)  # binary case
                    #         if self.transforms:
                    #             img = self.transforms(img)
                    #         return img, torch.tensor([label])

PATH = "/root/tammathon/data" # path to data in the server

train_df = pd.read_csv(f"{PATH}/train.csv")
valid_df = pd.read_csv(f"{PATH}/val.csv")
test_df = pd.read_csv(f"{PATH}/test.csv")


train_df['label'] = train_df['label'].astype(float)
valid_df['label'] = valid_df['label'].astype(float)

# train_df.iloc[0]['path'].split('/')[0]
train_df.iloc[0]['filename'].split('/')[0]

class ICAODataset(Dataset):
    def __init__(self, root_dir, dataframe, transforms=None):
        self.root_dir = root_dir
        self.dataframe = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.subset = dataframe.iloc[0]['path'].split('/')[0]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, self.subset, row['path'])
        img = Image.open(img_path).convert("RGB")

        label = float(row['label'])
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        if self.transforms:
            img = self.transforms(img)
        return img, label
    

                    # train_transforms = v2.Compose([
                    #     v2.Resize(600),
                    #     v2.RandomHorizontalFlip(p=0.3),
                    #     v2.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    #     v2.ToImage(),
                    #     v2.ToDtype(torch.float32, scale=True),
                    #     v2.Normalize(mean=[0.485, 0.456, 0.406], 
                    #                  std=[0.229, 0.224, 0.225])
                    # ])

                    # valid_transforms = v2.Compose([
                    #     v2.Resize(600),
                    #     v2.ToImage(),
                    #     v2.ToDtype(torch.float32, scale=True),
                    #     v2.Normalize(mean=[0.485, 0.456, 0.406], 
                    #                  std=[0.229, 0.224, 0.225])
                    # ])

                    # train_data = ICAODataset(PATH, train_df.rename(columns={"filename": "path"}), transforms=train_transforms)
                    # valid_data = ICAODataset(PATH, valid_df.rename(columns={"filename": "path"}), transforms=valid_transforms)

                    # train_loader = DataLoader(train_data, num_workers=0, shuffle=True, 
                    #                           pin_memory=True)
                    # valid_loader = DataLoader(valid_data, num_workers=0, shuffle=False, 
                    #                           pin_memory=True)

train_transforms = v2.Compose([
    v2.Resize(600),
    v2.RandomHorizontalFlip(p=0.3),
    v2.ColorJitter(0.4, 0.4, 0.4, 0.4),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225])
])

valid_transforms = v2.Compose([
    v2.Resize(600),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225])
])

train_data = ICAODataset(PATH, train_df.rename(columns={"filename": "path"}), transforms=train_transforms)
valid_data = ICAODataset(PATH, valid_df.rename(columns={"filename": "path"}), transforms=valid_transforms)

train_loader = DataLoader(train_data, num_workers=0, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_data, num_workers=0, shuffle=False, pin_memory=True)


class EffNetB7Base(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        out = self.backbone(x)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out  # raw logits


class ResNet50Base(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features

        # Remove the original classifier
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = e^(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

imgs, labels = next(iter(train_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "lr": 1e-4,
    "weight_decay": 1e-4
}

effnet_model = EffNetB7Base(1).to(device)
class_weights = torch.tensor([0.67, 0.33]).to(device)

focal_loss = FocalLoss(alpha=0.75)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)

optimizer = torch.optim.AdamW(effnet_model.parameters(), 
                              lr=config['lr'],
                              weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2) # So LR doesn't plateau smh
metric_fn = torchmetrics.F1Score(task='binary')

def train_model(
    model,
    train_loader, 
    loss_fn,
    optimizer,
    metric_fn,
    device,
    scheduler=None,
):
    model.train()
    running_metric = 0.0
    running_loss = 0.0

    with tqdm(train_loader, unit="batch", desc="Training") as progress_bar:
        for batch, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            
            yhat = model(x) # Model returns logits,
            yhat = torch.sigmoid(yhat) # Apply sigmoid for pred probs !!! 

            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()

            metric_fn.update(yhat, y)
            running_loss += loss.item()
            running_metric += metric_fn.compute().item()

            progress_bar.set_postfix(loss=loss.item(), 
                                     metric=running_metric / (batch + 1))

    epoch_loss = running_loss / len(train_loader)
    epoch_metric = running_metric / len(train_loader)

    metric_fn.reset()

    print(f"\nTraining Loss: {epoch_loss:.4f}, F1 Score: {epoch_metric:.4f}")
    return epoch_loss, epoch_metric

def validate_model(
    model,
    valid_loader,
    loss_fn,
    metric_fn,
    device
):
    model.eval()
    running_loss = 0.0
    running_metric = 0.0

    with torch.inference_mode():
        with tqdm(valid_loader, unit="batch", desc="Validation") as progress_bar:
            for x, y in progress_bar:
                x, y = x.to(device), y.to(device)

                yhat = model(x)
                yhat = torch.sigmoid(yhat)

                loss = loss_fn(yhat, y)

                metric_fn.update(yhat, y)

                running_loss += loss.item()
                running_metric += metric_fn.compute().item()

                progress_bar.set_postfix(loss=loss.item(), 
                                         metric=running_metric / (len(progress_bar) + 1))

    epoch_loss = running_loss / len(valid_loader)
    epoch_metric = running_metric / len(valid_loader)

    metric_fn.reset()

    print(f"\nValidation Loss: {epoch_loss:.4f}, F1 Score: {epoch_metric:.4f}")
    return epoch_loss, epoch_metric


def training_pipeline(
    epochs,
    model,
    train_loader,
    valid_loader,
    loss_fn,
    metric_fn,
    optimizer,
    device,
    scheduler=None,
):
    train_losses, valid_losses = [], []
    train_metrics, valid_metrics = [], []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_metric = train_model(
            model, train_loader, loss_fn, optimizer, metric_fn, device, scheduler
        )
        valid_loss, valid_metric = validate_model(
            model, valid_loader, loss_fn, metric_fn, device
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)

        if scheduler:
            scheduler.step(valid_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
        print(f"Validation Loss: {valid_loss:.4f}, Validation Metric: {valid_metric:.4f}")

    return train_losses, valid_losses, train_metrics, valid_metrics


train_losses, valid_losses, train_metrics, valid_metrics = training_pipeline(
    10, effnet_model, train_loader, valid_loader, focal_loss, metric_fn, optimizer, device, scheduler
)




# results_df = pd.DataFrame({
#     "epoch": list(range(1, len(train_losses) + 1)),
#     "train_loss": train_losses,
#     "valid_loss": valid_losses,
#     "train_f1": train_metrics,
#     "valid_f1": valid_metrics
# })

# results_df.to_csv("training_results.csv", index=False)


