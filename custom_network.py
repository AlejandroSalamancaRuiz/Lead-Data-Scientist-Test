import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import torchmetrics
from torch import nn
from sklearn.model_selection import train_test_split
import torchmetrics.classification

############################################ Custom DataLoader ############################################
############################################ ############## ############################################


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        # Convert pandas DataFrames/Series or numpy arrays to tensors
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            self.X = torch.tensor(X.values, dtype=torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
        
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            self.y = torch.tensor(y.values, dtype=torch.float32)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.y)

    def __getitem__(self, idx):
        # Retrieve the sample and label at the specified index
        return self.X[idx], self.y[idx]


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_test, y_test, n_workers, batch_size=32):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = n_workers

    def setup(self, stage=None):
        # Split the training data into training and validation datasets (80/20 split)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        # Create datasets for training, validation, and testing
        self.train_dataset = CustomDataset(X_train_split, y_train_split)
        self.val_dataset = CustomDataset(X_val_split, y_val_split)
        self.test_dataset = CustomDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        # Return DataLoader for training dataset
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Return DataLoader for validation dataset
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # Return DataLoader for test dataset
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


############################################ Trainer ############################################
############################################ ############## ############################################

class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.7/0.3))
        
        # Metrics for training, validation, and testing
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()

        self.test_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_auc = torchmetrics.AUROC(task="binary")
        self.test_prec = torchmetrics.classification.BinaryPrecision()
        self.test_rec = torchmetrics.classification.BinaryRecall()
        self.conf_mtx = torchmetrics.classification.BinaryConfusionMatrix()


    def forward(self, x):
        # Forward pass through the model
        return self.model(x)
    
    def _shared_step(self, batch):
        # Shared step for training, validation, and testing
        features, true_labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, true_labels)
        predicted_labels = torch.where(logits > 0, 1, 0)
        return loss, predicted_labels, true_labels

    def training_step(self, batch, batch_idx):
        # Training step
        loss, predicted_labels, true_labels = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Test step
        _, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.test_auc(predicted_labels, true_labels)
        self.test_prec(predicted_labels, true_labels)
        self.test_rec(predicted_labels, true_labels)
        self.conf_mtx(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)
        self.log("test_auc", self.test_auc)
        self.log("test_prec", self.test_prec)
        self.log("test_rec", self.test_rec)

        


    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

############################################ Custom Network ############################################
############################################ ############## ############################################


# Config class
class Config:
        def __init__(self, input_dim=32, n_embd=256, num_blocks=4, dropout=0.0, bias=False):
            self.input_dim = input_dim
            self.n_embd = n_embd    
            self.num_blocks = num_blocks
            self.dropout = dropout
            self.bias = bias


# MLP (Multi-Layer Perceptron) class
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # First linear layer: projects from embedding dimension to 4 times the embedding dimension
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # GELU activation function
        self.gelu = nn.GELU()
        # Second linear layer: projects back from 4 times the embedding dimension to the embedding dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Forward pass through the first linear layer and GELU activation
        x = self.c_fc(x)
        x = self.gelu(x)
        # Forward pass through the second linear layer and dropout
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# Network Block class
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Layer normalization before the first MLP
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # First MLP layer
        self.mlp_1 = MLP(config)
        # Layer normalization before the second MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # Second MLP layer
        self.mlp_2 = MLP(config)

    def forward(self, x):
        # Add residual connection and forward pass through the first MLP
        x = x + self.mlp_1(self.ln_1(x))
        # Add residual connection and forward pass through the second MLP
        x = x + self.mlp_2(self.ln_2(x))
        return x


# Custom Network
class Custom_Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Store the configuration
        self.config = config

        # First layer: Linear transformation from input dimension to embedding dimension
        self.first_layer = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)

        # Stack of blocks, each containing two MLP layers with layer normalization
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_blocks)])

        # Output layer: Linear transformation from embedding dimension to a single output
        self.out_layer = nn.Linear(config.n_embd, 1, bias=False)


    def forward(self, X_):
        # Forward pass through the first layer
        x = self.first_layer(X_)

        # Forward pass through each block
        for block in self.blocks:
            x = block(x)
        
        # Forward pass through the output layer and apply sigmoid activation
        logits = self.out_layer(x).squeeze(-1)

        return logits
    

############################################ ############## ############################################


