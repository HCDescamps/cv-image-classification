import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl



class Classifier(pl.LightningModule):
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 32, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Linear(
            64 * 7 * 7, 
            num_classes
        )

    def forward(self, x):
        
        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x= self.classifier(x)
      
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(
            logits,
            y
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
         
        x, y = batch
         
        logits = self(x)
         
        loss = F.cross_entropy(
            logits,
            y
        )
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log(
            'val_loss',
            loss,
            on_epoch=True
        )
        self.log(
            'val_accuracy',
            acc,
            on_epoch=True
        ) 

    def test_step(self, batch, batch_idx):
        
        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(
            logits,
            y
        )
        
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log(
            "test_accuracy",
            acc,
            on_epoch=True
        )
        self.log(
            'test_loss', 
            loss,
            on_epoch=True
        )
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3
        )
        return optimizer