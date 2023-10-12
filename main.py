from models.xBD_data import xBDDataModule, xBDDataset
from models.netlit import DAHiTraLit
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

batch_size = 2
num_workers = 2
train_path = './data/xbd/train'

if __name__ == '__main__':
    # data_module = xBDDataModule(batch_size, train_path)
    dataloader = DataLoader(xBDDataset(dataset_path = train_path), batch_size=batch_size)
    model = DAHiTraLit()
    summary = ModelSummary(model)
    print(summary)
    trainer = pl.Trainer()
    trainer.fit(model, dataloader)
