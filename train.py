from model import Classifier
from data import ImageDataModule

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger

datamodule = ImageDataModule()
datamodule.setup()

model = Classifier.load_from_checkpoint(
    "lightning_logs/mnist_cnn/version_1/checkpoints/epoch=4-step=8595.ckpt"
)
# model.eval()
# print(model)

checkpoint = ModelCheckpoint(
        monitor='val_accuracy',  
        mode='max'
)

early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=2,
        mode='max'
)   

logger = TensorBoardLogger(
    "lightning_logs",
    name="mnist_cnn"
)

trainer = Trainer(
    max_epochs=5, 
    callbacks=[
        checkpoint, 
        early_stopping
    ],
    logger=logger,
    num_sanity_val_steps=0
)

trainer.fit(
    model, 
    datamodule
)

trainer.validate(
    model, 
    datamodule
)

trainer.test(
    model,
    datamodule
)