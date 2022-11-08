import torch

import flash
from flash.core.data.utils import download_data
from flash.video import VideoClassificationData, VideoClassifier

# 1. Create the DataModule
# Find more datasets at https://pytorchvideo.readthedocs.io/en/latest/data.html
print("#########before download")
download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip", "./data")
print("#########after download")

datamodule = VideoClassificationData.from_folders(
    train_folder="data/kinetics/train",
    val_folder="data/kinetics/val",
    clip_sampler="uniform",
    clip_duration=1,
    decode_audio=False,
    batch_size=1,
)
print("#########created datamodule")

# 2. Build the task
model = VideoClassifier(backbone="x3d_xs", labels=datamodule.labels, pretrained=False)

print("#########c built task")

print("######### training")

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(
    max_epochs=1, gpus=torch.cuda.device_count(), strategy="ddp" if torch.cuda.device_count() > 1 else None
)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

print("######### done!")


# 4. Make a prediction
datamodule = VideoClassificationData.from_folders(predict_folder="data/kinetics/predict", batch_size=1)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("video_classification.pt")