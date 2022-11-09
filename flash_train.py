import os.path
import lightning as L
import logging
import torch


from quick_start.components import ImageServeGradio

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set, List, Tuple

import requests
import urllib3

from tqdm.auto import tqdm as tq

import sys
import traceback
import zipfile

class FlashTrainer(L.LightningWork):

    def run(self):
        print("######### FlashTrainer1")
        print("######### FlashTrainer2")
        import flash
        from flash.core.data.utils import download_data
        from flash.video import VideoClassificationData, VideoClassifier

        url = "https://pl-flash-data.s3.amazonaws.com/kinetics.zip"
        path = "/data"

        if os.path.exists(path):
            print("#########Has mount!")
            path = "/data/kinetics-flash-test/kinetics/"
            # files = os.listdir("/data/kinetics-flash-test/kinetics/train")
            # for file in files:
            #     print("########train: " + file)
            #     if os.path.isdir(file):  
            #         print(file + " is a directory")
            #         ds = os.listdir(file)
            #         for d in ds:
            #             print(d)
            #     else: 
            #         print(file + " is a normal file")

            # files = os.listdir("/data/kinetics-flash-test/kinetics/val")
            # for file in files:
            #     print("######### val: " + file)
            #     if os.path.isdir(file):  
            #         print(file + " is a directory")
            #         ds = os.listdir(file)
            #         for d in ds:
            #             print(d)
            #     else: 
            #         print(file + " is a normal file")
        else:
            path = "./data/kinetics/"
            print("######### need to download data")
            os.makedirs(path)
            local_filename = os.path.join(path, url.split("/")[-1])
            # Find more datasets at https://pytorchvideo.readthedocs.io/en/latest/data.html

            r = requests.get(url, stream=True, verify=False)
            file_size = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else 0
            chunk_size = 1024

            num_bars = int(file_size / chunk_size)

            if not os.path.exists(local_filename):
              with open(local_filename, "wb") as fp:
                for chunk in tq(
                    r.iter_content(chunk_size=chunk_size),
                    total=num_bars,
                    unit="KB",
                    desc=local_filename,
                    leave=True,  # progressbar stays
                ):
                    fp.write(chunk)  # type: ignore

            with zipfile.ZipFile("./data/kinetics/kinetics.zip", "r") as zip_ref:
                zip_ref.extractall("./data")
            print("#########after download")

        datamodule = VideoClassificationData.from_folders(
            train_folder= path + "train",
            val_folder=path + "val",
            clip_sampler="uniform",
            clip_duration=1,
            decode_audio=False,
            batch_size=1,
        )
        print("#########created datamodule")

        # 2. Build the task
        model = VideoClassifier(backbone="x3d_xs", labels=datamodule.labels, pretrained=False)

        print("######### built task")

        print("######### training")

        # 3. Create the trainer and finetune the model
        trainer = flash.Trainer(
            max_epochs=1, gpus=torch.cuda.device_count(), strategy="ddp" if torch.cuda.device_count() > 1 else None
        )
        trainer.finetune(model, datamodule=datamodule, strategy="freeze")

        print("######### done training!")


        # 4. Make a prediction
        datamodule = VideoClassificationData.from_folders(predict_folder="data/kinetics/predict", batch_size=1)
        predictions = trainer.predict(model, datamodule=datamodule, output="labels")
        print(predictions)

        # 5. Save the model!
        trainer.save_checkpoint("video_classification.pt")
        print("######### save the model!")

class TrainDeploy(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train_work = FlashTrainer(
           cloud_compute=L.CloudCompute(
               mounts=L.storage.Mount(
                   source="s3://kinetics-flash-test/",
               ),
           )
       )

    def run(self):
        # 1. Run the python script that trains the model
        self.train_work.run()

app = L.LightningApp(TrainDeploy())