from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.x)


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        print(f"Train loss {float(loss)}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == '__main__':
    pl.seed_everything(42)
    dataset = CustomDataset(x=torch.rand(100, 5), y=torch.rand(100))
    mlp = MLP()
    trainer = pl.Trainer(deterministic=True, max_epochs=3)
    trainer.fit(mlp, DataLoader(dataset, batch_size=50, shuffle=True))

# Losses
# Train loss 0.5009271502494812
# Train loss 0.4296863377094269
# Train loss 0.421977162361145
# Train loss 0.5070599317550659
# Train loss 0.3938509225845337
# Train loss 0.5337120890617371
