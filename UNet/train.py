from system import UNetSystem
from modelCheckpoint import BestAndLatestModelCheckpoint
import pytorch_lightning as pl

dataset_path = "/home/vmlab/Desktop/data/patch/label3d/image"
criteria = {
        "train" : ["000", "001", "003", "006"], 
        "val" : ["002", "004", "007", "009"]
        }
in_channel = 5
num_class = 3
batch_size = 18
num_workers = 6
epoch = 3
log = "log"
savepath = "modelweight"
#inital_filepath = log / "initial.cpkt"

#torch.manual_seed(0)

system = UNetSystem(
        dataset_path = dataset_path,
        criteria = criteria,
        in_channel = in_channel,
        num_class = num_class, 
        batch_size = batch_size,
        num_workers = num_workers,
        checkpoint = BestAndLatestModelCheckpoint(savepath)
        )

#trainer = pl.Trainer(num_sanity_val_steps=5)
trainer = pl.Trainer(num_sanity_val_steps=0, max_epochs=epoch)
trainer.fit(system)

