import os
import math
import faiss
import timm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
import torch.nn.functional as F
import timm

from collections import defaultdict
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from pytorch_metric_learning import miners, losses, samplers, distances

PATH = "/root/tammathon/data/"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_csv = pd.read_csv(f"{PATH}/train.csv")
valid_csv = pd.read_csv(f"{PATH}/val.csv")
test_csv = pd.read_csv(f"{PATH}/test.csv")

len(train_csv), len(valid_csv), len(test_csv)

train_csv["id"] = [
    train_csv.iloc[i]["filename"].split("/")[1]
    for i in range(len(train_csv["filename"].values))
]
valid_csv["id"] = [
    valid_csv.iloc[i]["filename"].split("/")[1]
    for i in range(len(valid_csv["filename"].values))
]

combined_csv = pd.concat([train_csv, valid_csv])
combined_csv = combined_csv.reset_index(drop=True)
combined_csv.head()

id_to_images = defaultdict(list)
for i in tqdm(range(len(combined_csv)), desc="Creating ID to Image Mapping"):
    row = combined_csv.iloc[i]
    filename, label, id = row
    id_to_images[label].append(filename)

train_data, val_data = [], []
for label, paths in tqdm(
    id_to_images.items(), desc="Redistributing Classes between Datasets"
):
    if len(paths) < 2:
        train_data.extend((p, label) for p in paths)
        continue
    train_imgs, val_imgs = train_test_split(paths, test_size=0.2, random_state=42)
    train_data.extend((p, label) for p in train_imgs)
    val_data.extend((p, label) for p in val_imgs)

test_csv["id"] = [
    test_csv.iloc[i]["filename"].split("/")[1]
    for i in range(len(test_csv["filename"].values))
]
test_csv["label"] = [i for i in range(len(test_csv))]
test_csv["id"] = [
    test_csv.iloc[i]["id"].split("-")[1].split(".")[0] for i in range(len(test_csv))
]


class CatDataset(Dataset):
    def __init__(self, root_dir, dataframe, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # was causing error so I changed it to the next line
        # row = self.dataframe.iloc[idxlabel]
        row = self.dataframe.iloc[idx]

        subset = row["filename"].split("/")[0]
        img_path = os.path.join(self.root_dir, subset, row["filename"])
        label = row["label"]
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, label


class ResNetFE(nn.Module):
    def __init__(self, embedding_size, weights=None, freeze_until="layer1"):
        super().__init__()
        base_model = resnet50(weights=weights)

        # Freeze layers
        freeze = True
        for name, param in base_model.named_parameters():
            if freeze:
                param.requires_grad = False
            if freeze_until in name:
                freeze = False

        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(base_model.fc.in_features, embedding_size)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        return F.normalize(emb, p=2, dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(
        self, embedding_size, num_classes, scale=64.0, margin=0.5, easy_margin=False
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - cosine**2 + 1e-6)

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


class RetrievalAtK(torchmetrics.Metric):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, knn_labels, query_labels):
        top_k = knn_labels[:, : self.k]
        matches = top_k == query_labels.unsqueeze(1)
        correct_per_sample = matches.any(dim=1).sum()
        self.correct += correct_per_sample
        self.total += query_labels.numel()

    def compute(self):
        return self.correct.float() / self.total


train_df = pd.DataFrame(train_data, columns=["filename", "label"])
valid_df = pd.DataFrame(val_data, columns=["filename", "label"])

# weights_path = "/kaggle/input/resnet50-arcmargin-baseline/pytorch/resnet50_arcface/1/best_model.pth"
weights_path = (
    "./putout/best_model.pt"
)

model = ResNetFE(embedding_size=512).to(device)
arc_margin = ArcMarginProduct(512, len(train_df["label"].unique()))

checkpoint = torch.load(weights_path, weights_only=True)
print(checkpoint.keys())

model.load_state_dict(checkpoint["model"])
arc_margin.load_state_dict(checkpoint["arc_margin"])

train_transforms = v2.Compose(
    [
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = v2.Compose(
    [
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ResNetArcFace(nn.Module):
    def __init__(self, embedding_size, n_classes, weights_path=None):
        super().__init__()
        self.fe = ResNetFE(embedding_size)
        self.arc_margin = ArcMarginProduct(embedding_size, n_classes)
        if weights_path is not None:
            self._load_weights()

    def _load_weights(self):
        try:
            checkpoint = torch.load(weights_path, weights_only=True)
            self.fe.load_state_dict(checkpoint["model"])
            self.arc_margin.load_state_dict(checkpoint["arc_margin"])
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error: {e}")

    def forward(self, x, labels=None):
        embed = self.fe(x)
        if labels is not None:
            return self.arc_margin(embed, labels)
        return embed


class TimmArcFace(nn.Module):
    def __init__(self, embedding_size, n_classes, timm_path, arc_weights=None):
        super().__init__()
        self.fe = timm.create_model(timm_path, pretrained=True, num_classes=0)
        self.arc_margin = ArcMarginProduct(embedding_size, n_classes)
        if arc_weights is not None:
            self._load_arc_weights()

    def _load_arc_weights(self):
        try:
            checkpoint = torch.load(weights_path, weights_only=True)
            self.arc_margin.load_state_dict(checkpoint["arc_margin"])
            print("ArcMargin Weights loaded successfully.")
        except Exception as e:
            print(f"Error: {e}")

    def forward(self, x, labels=None):
        embed = self.fe(x)
        if labels is not None:
            return self.arc_margin(embed, labels)
        return embed


batch_size = 128
embedding_size = 512

train_set = CatDataset(PATH, train_df, train_transforms)
test_set = CatDataset(PATH, test_csv, test_transforms)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

base_model = ResNetArcFace(
    embedding_size, len(train_df["label"].unique()), weights_path
).to(device)

def embed(dataloader, model, device):
    model.eval()
    outputs = {"embeddings": [], "labels": []}

    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc="Embedding", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = model(inputs)
            outputs["embeddings"].append(embeddings.cpu())
            outputs["labels"].append(labels.cpu())

    outputs["embeddings"] = torch.cat(outputs["embeddings"])
    outputs["labels"] = torch.cat(outputs["labels"])
    return outputs


def build_faiss_index(embeddings):
    embeddings = embeddings.numpy().astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product index
    index.add(embeddings)
    return index


train_out = embed(train_loader, model, device)
test_out = embed(test_loader, model, device)

# Building FAISS index using train embeddings
#index = build_faiss_index(train_out["embeddings"])

print("indexing")
# Performing the search on test embeddings
#_, indices = index.search(test_out["embeddings"].cpu().numpy().astype("float32"), 3)

index = build_faiss_index(train_out["embeddings"])
_, indices = index.search(test_out["embeddings"].numpy().astype("float32"), 3)
test_csv.drop(["id", "label"], axis=1, inplace=True)


for i in range(3):
    test_csv[f"label_{i + 1}"] = indices[:, i]

test_csv.head()

test_csv.to_csv("submission.csv", index=False)
