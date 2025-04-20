import os
import faiss
import torch
import torchmetrics
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_metric_learning import miners, losses, samplers, distances

class CatDataset(Dataset):
    def __init__(self, root_dir, dataframe, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.dataframe = dataframe
        self.subset = dataframe['filename'][0].split('/')[0]
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir,self.subset, row['filename'])
        label = int(row["label"])
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, label

train_transforms = v2.Compose([
    v2.RandomApply([
        v2.ColorJitter(0.5, 0.3, 0.4, 0.2),
        v2.RandomRotation(30),
        v2.RandomHorizontalFlip(p=0.3)
    ], p=0.2),
    v2.RandomResizedCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

valid_transforms = v2.Compose([
    v2.Resize((256, 256)), 
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

class ResNetFE(nn.Module):
    def __init__(self, embedding_size, weights=None, freeze_until='layer1'):
        super().__init__()
        base_model = resnet50(weights=weights)

        # Freeze all layers up to and including `freeze_until`
        freeze = True
        for name, param in base_model.named_parameters():
            if freeze:
                param.requires_grad = False
            if freeze_until in name:
                freeze = False  # Unfreeze after the target layer

        self.n_features = base_model.fc.in_features
        self.pool = base_model.avgpool
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.final_embedding = nn.Linear(self.n_features, embedding_size)

    def forward(self, x):
        # Extract features
        ft = self.model(x)
        pooled_output = self.pool(ft)
        
        # Flatten the pooled output to (batch_size, n_features)
        embedding = torch.flatten(pooled_output, 1)  # Flatten all dimensions except batch size
        output = self.final_embedding(embedding)
        
        return F.normalize(output, p=2, dim=1)
    
class RetrievalAtK(torchmetrics.Metric):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, knn_labels, query_labels):
        top_k =  knn_labels[:, :self.k]
        matches = (top_k == query_labels.unsqueeze(1))
        correct_per_sample = matches.any(dim=1).sum()
        self.correct += correct_per_sample
        self.total += query_labels.numel()
    
    def compute(self):
        return self.correct.float() / self.total

def add_id_column(dataframe):
    dataframe['id'] = [filename.split('/')[1] for filename in dataframe['filename']]

def prepare_dataloaders(path, batch_size):
    # train_csv = pd.read_csv(f"{path}\\train.csv")
    # valid_csv = pd.read_csv(f"{path}\\val.csv")
    
	# Better cross-platform solution
    train_csv = pd.read_csv(os.path.join(path, "train.csv"))
    valid_csv = pd.read_csv(f"{path}\\val.csv")


    add_id_column(train_csv)
    add_id_column(valid_csv)
    
    train_transforms = v2.Compose([
        v2.RandomApply([
            v2.ColorJitter(0.5, 0.3, 0.4, 0.2),
            v2.RandomRotation(30),
            v2.RandomHorizontalFlip(p=0.3)
        ], p=0.2),
        v2.RandomResizedCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = v2.Compose([
        v2.Resize((256, 256)), 
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    train_data = CatDataset(path, train_csv, transforms=train_transforms)
    valid_data = CatDataset(path, valid_csv, transforms=valid_transforms)
    
    sampler = samplers.MPerClassSampler(train_data.dataframe['id'].values, m=6, 
                                        batch_size=batch_size, length_before_new_iter=len(train_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=min(2, os.cpu_count()-1))
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=min(2, os.cpu_count()-1))

    return train_loader, valid_loader

def embed(data_loader, model, device):
    model.eval()
    outputs = {
        'embeddings': [],
        'labels': []
    }

    with torch.inference_mode():
        for inputs, labels in tqdm(data_loader, desc="Embedding", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = model(inputs)
            outputs['embeddings'].append(embeddings.cpu())
            outputs['labels'].append(labels.cpu())

    outputs['embeddings'] = torch.cat(outputs['embeddings'], dim=0)
    outputs['labels'] = torch.cat(outputs['labels'], dim=0)
    return outputs

def get_knn_labels(ref_embeddings, query_embeddings, ref_labels, k=5):
    ref_embeddings = ref_embeddings.numpy()
    query_embeddings = query_embeddings.numpy()
    ref_labels = ref_labels.numpy()

    index = faiss.IndexFlatIP(ref_embeddings.shape[1])
    index.add(ref_embeddings)
    
    _, indices = index.search(query_embeddings, k)
    knn_labels = torch.tensor(ref_labels[indices], dtype=torch.long)
    return knn_labels

def run_training_loop(model, loss_fn, miner, train_loader, val_loader, optimizer, device, epochs=10, k=5):
    metric = RetrievalAtK(k).to(device)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", leave=False)
        for batch_idx, (x, y) in train_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            embeddings = model(x)

            miner_pairs = miner(embeddings, y)
            loss_value = loss_fn(embeddings, y, miner_pairs)
            loss_value.backward()
            optimizer.step()

            epoch_loss += loss_value.item()

            train_bar.set_postfix({
                "Loss": f"{loss_value.item():.4f}",
                "Triplets": miner.num_triplets
            })

        print(f"Epoch {epoch + 1} Finished | Avg Loss = {epoch_loss / len(train_loader):.4f}")

        print("Running validation...")

        train_embeddings = embed(train_loader, model, device)
        val_embeddings = embed(val_loader, model, device)

        knn_labels = get_knn_labels(
            ref_embeddings=train_embeddings['embeddings'],
            query_embeddings=val_embeddings['embeddings'],
            ref_labels=train_embeddings['labels'],
            k=k
        )

        y_true = val_embeddings['labels']

        metric.reset()
        metric.update(knn_labels, y_true)
        retrieval_score = metric.compute().item()
        print(f"Retrieval@{k}: {retrieval_score * 100:.4f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # data_path = os.path.join(os.getcwd(), "Task-1-Pawsitive", "data")
    data_path = "/root/data1" # Change this to your data path

    embedding_size = 512
    batch_size = 48
    learning_rate = 1e-4
    epochs = 10
    k = 3

    print("Preparing dataloaders...")
    train_loader, val_loader = prepare_dataloaders(data_path, batch_size)

    print("Initializing model...")
    model = ResNetFE(embedding_size=embedding_size, weights=ResNet50_Weights.IMAGENET1K_V2, freeze_until='layer1').to(device)

    distance = distances.CosineSimilarity()
    loss_fn = losses.TripletMarginLoss(margin=0.2, distance=distance, triplets_per_anchor="all")
    miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run_training_loop(model, loss_fn, miner, train_loader, val_loader, optimizer, device, epochs=epochs, k=k)

if __name__ == '__main__':
    main()
