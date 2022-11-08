import os

import lightning as L

import os.path
import lightning as L
import logging


from quick_start.components import ImageServeGradio

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set, List, Tuple

import requests
import urllib3

from tqdm.auto import tqdm as tq

import sys
import traceback
import zipfile
import flash
from flash.core.data.utils import download_data
from flash.video import VideoClassificationData, VideoClassifier


class FlashTrainer(L.LightningWork):
   def run(self):
       # Print a list of files stored in the mounted S3 Bucket.
       print("######### run work")
       # files = os.listdir("/data/")
       # for file in files:
       #     print(file)

       # # Read the contents of a particular file in the bucket "esRedditJson1"
       # with open("/data", "r") as f:
       #     some_data = f.read()
           # do something with "some_data"...

class Flow(L.LightningFlow):
   def __init__(self):
       print("######### init flow")
       super().__init__()
       self.my_work = FlashTrainer(
           cloud_compute=L.CloudCompute(
               mounts=L.storage.Mount(
                   source="s3://kinetics-flash-test/",
                   mount_path="/data/",
               ),
           )
       )

   def run(self):
       self.my_work.run()

# Note: You can also pass multiple mounts to a single work by passing a 
#        List[Mount(...), ...] to the CloudCompute(mounts=...) argument.
app = L.LightningApp(Flow())