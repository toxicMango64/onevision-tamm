import os
import math
import faiss
import torch
import torchmetrics
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_metric_learning import miners, losses, samplers, distances

class CatDataset(Dataset):
    def __init__(self, root_dir, dataframe, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        subset = row['filename'].split('/')[0]
        img_path = os.path.join(self.root_dir, subset, row['filename'])
        label = row['label']
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, label

def redistribute_classes(train_csv, valid_csv, test_size=0.2, random_state=42):
    combined_csv = pd.concat([train_csv, valid_csv])
    combined_csv = combined_csv.reset_index(drop=True)
    
    id_to_images = defaultdict(list)
    for i in tqdm(range(len(combined_csv)), desc="Creating ID to Image Mapping"):
        row = combined_csv.iloc[i]
        filename, label = row['filename'], row['label']
        id_to_images[label].append(filename)

    train_data, valid_data = [], []
    for label, paths in tqdm(id_to_images.items(), desc="Redistributing Classes between Datasets"):
        if len(paths) < 2:
            train_data.extend((p, label) for p in paths)
            continue
        train_imgs, valid_imgs = train_test_split(paths, 
                                                test_size=test_size,
                                                random_state=random_state)
        train_data.extend((p, label) for p in train_imgs)
        valid_data.extend((p, label) for p in valid_imgs)

    train_df = pd.DataFrame(train_data, columns=['filename', 'label'])
    valid_df = pd.DataFrame(valid_data, columns=['filename', 'label'])
    
    return train_df, valid_df

def get_train_transforms():
    return v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomGrayscale(p=0.15),
        v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        v2.RandomAffine(degrees=25, translate=(0.05, 0.05), scale=(0.85, 1.15), shear=10),
        v2.RandomPerspective(distortion_scale=0.3, p=0.2),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_valid_transforms():
    return v2.Compose([
        v2.Resize((256, 256)), 
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class ResNetFE(nn.Module):
    def __init__(self, embedding_size, weights=None, freeze_until='layer1'):
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
    def __init__(self, embedding_size, num_classes, scale=64.0, margin=0.5, easy_margin=False):
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
        top_k = knn_labels[:, :self.k]
        matches = (top_k == query_labels.unsqueeze(1))
        correct_per_sample = matches.any(dim=1).sum()
        self.correct += correct_per_sample
        self.total += query_labels.numel()
    
    def compute(self):
        return self.correct.float() / self.total

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

def prepare_dataloaders(path, train_csv, valid_csv, batch_size, use_arc_margin=False):
    train_data = CatDataset(path, train_csv, transforms=get_train_transforms())
    valid_data = CatDataset(path, valid_csv, transforms=get_valid_transforms())

    if use_arc_margin:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=min(4, os.cpu_count()-1), pin_memory=True, persistent_workers=True, prefetch_factor=2)
    else:
        sampler = samplers.MPerClassSampler(train_data.dataframe['label'].values, m=8, 
                                      batch_size=batch_size, length_before_new_iter=len(train_data))
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, 
                             num_workers=min(4, os.cpu_count()-1), pin_memory=True, persistent_workers=True, prefetch_factor=2)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, 
                             num_workers=min(4, os.cpu_count()-1), pin_memory=True, persistent_workers=True)

    return train_loader, valid_loader, len(train_csv['label'].unique())

def run_training_loop(model, loss_fn, train_loader, val_loader, optimizer, 
                     device, scheduler, epochs=15, k=3, save_path="best_model.pt", 
                     arc_margin=None, validate_every=2):
    
    metric = RetrievalAtK(k).to(device)
    best_score = 0.0
    history = {'loss': [], 'val_acc': []}
    
    # Automatic Mixed Precision setup
    use_amp = torch.cuda.get_device_capability()[0] >= 7
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        if arc_margin:
            arc_margin.train()
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                embeddings = model(x)
                logits = arc_margin(embeddings, y) if arc_margin else None
                if arc_margin:
                    loss = loss_fn(logits, y)
                else:
                    loss = loss_fn(embeddings, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # Step the scheduler after each epoch
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Validation
        if (epoch + 1) % validate_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.inference_mode():
                train_embeddings = embed(train_loader, model, device)
                val_embeddings = embed(val_loader, model, device)
                
                knn_labels = get_knn_labels(
                    train_embeddings['embeddings'],
                    val_embeddings['embeddings'],
                    train_embeddings['labels'],
                    k=k
                )
                
                metric.reset()
                metric.update(knn_labels, val_embeddings['labels'])
                retrieval_score = metric.compute().item()
                history['val_acc'].append(retrieval_score)
                
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Retrieval@{k}={retrieval_score*100:.2f}%")
                
                if retrieval_score > best_score:
                    best_score = retrieval_score
                    torch.save({
                        'model': model.state_dict(),
                        'arc_margin': arc_margin.state_dict() if arc_margin else None,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                    }, save_path)
        else:
            history['val_acc'].append(None)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

    return history

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = "/root/tammathon/data" # Change this to your data path

    train_csv = pd.read_csv(os.path.join(path, "train.csv"))
    valid_csv = pd.read_csv(os.path.join(path, "val.csv"))

    train_csv, valid_csv = redistribute_classes(train_csv, valid_csv, test_size=0.2, random_state=42)

    config = {
        'batch_size': 128,
        'embedding_size': 512,
        'epochs': 30,
        'lr': 0.1,
        'margin': 0.3,
        'scale': 30.0,
        'weight_decay': 1e-4,
        'validate_every': 2,
        'use_arc_margin': True,
        'k': 3,
    }

    os.makedirs("output", exist_ok=True)

    train_loader, val_loader, num_classes = prepare_dataloaders(path, train_csv, valid_csv, 
                                                                config['batch_size'], config['use_arc_margin'])

    print("Unique labels:", torch.unique(torch.tensor(train_loader.dataset.dataframe['label'])))

    model = ResNetFE(config['embedding_size'], weights=ResNet50_Weights.IMAGENET1K_V2).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # ArcMargin + CrossEntropy Loss or Triplet Loss
    if config['use_arc_margin']:
        arc_margin = ArcMarginProduct(config['embedding_size'], num_classes, margin=config['margin'], scale=config['scale']).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        arc_margin = None
        distance = distances.CosineSimilarity()
        loss_fn = losses.TripletMarginLoss(margin=0.2, distance=distance)

    # optimizer = torch.optim.AdamW(
    #     list(model.parameters()) + (list(arc_margin.parameters()) if arc_margin else []),
    #     lr=1e-4, weight_decay=1e-5
    # )

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(arc_margin.parameters()),
        lr=config['lr'],  # Start higher
        momentum=0.9,
        weight_decay=1e-4, 
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs']
    )

    history = run_training_loop(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epochs=config['epochs'],
        k=config['k'],
        arc_margin=arc_margin,
        save_path=os.path.join("output", "best_model.pt"),
        validate_every=2
    )

    history_df = pd.DataFrame({
        'epoch': range(1, config['epochs'] + 1),
        'loss': history['loss'],
        f"retrieval@{config['k']}": history['val_acc']
    })

    history_csv_path = os.path.join("new_output", "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"\nTraining history saved to {history_csv_path}")

    print("Saving final embeddings...")
    final_train_embeddings = embed(train_loader, model, device)
    final_val_embeddings = embed(val_loader, model, device)
    
    torch.save({
        'train_embeddings': final_train_embeddings,
        'val_embeddings': final_val_embeddings,
        'model_state_dict': model.state_dict()
    }, os.path.join("new_output", "final_embeddings.pt"))
    print(f"Embeddings and model saved to new_output/final_embeddings.pt")

if __name__ == "__main__":
    main()
