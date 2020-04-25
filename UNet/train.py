from system import UNetSystem
import pytorch_lightning as pl

dataset_path = "/Users/tanimotoryou/Documents/lab/kidney/aligned/test/image"
criteria = {
        "train" : ["000", "040", "135"], 
        "val" : ["002", "056"], 
        }
in_channel = 3
num_class = 3
batch_size = 15


system = UNetSystem(
        dataset_path = dataset_path,
        criteria = criteria,
        in_channel = in_channel,
        num_class = num_class, 
        batch_size = batch_size
        )

trainer = pl.Trainer()
trainer.fit(system)

