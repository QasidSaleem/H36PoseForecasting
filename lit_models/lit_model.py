"""Lightning Model."""
import pytorch_lightning as pl
import torch


class LitModule(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        optimizer = self.args.get("optimizer")
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr")

        loss = self.args.get("loss")
        # print("loss func", loss)
        self.loss_fn = getattr(torch.nn, loss)()

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps")
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.model(x)
    
    def _run_on_batch(self, batch):
        pred = self(batch)
        y = batch[:, self.model.n_seeds:, :]
        loss = self.loss_fn(pred, y)
        
        return pred, y, loss
    
    def training_step(self, batch, batch_idx):
        pred, y, loss = self._run_on_batch(batch)
        # self.log("train_loss", loss, batch_size=pred.shape[0], on_step=True, on_epoch=True)
        logs = {}
        logs["loss"] = loss

        for k,v in logs.items():
            self.log(f"train_{k}", v, batch_size=pred.shape[0], on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred, y, loss = self._run_on_batch(batch)
        logs = {}
        logs["loss"] = loss
        for k,v in logs.items():
            prog_bar = True if k == "loss" else False
            self.log(f"val_{k}", v, batch_size=pred.shape[0], on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        pred, y, loss = self._run_on_batch(batch)
        logs = {}
        logs["loss"] = loss
        for k,v in logs.items():
            self.log(f"test_{k}", v, batch_size=pred.shape[0], on_step=False, on_epoch=True, prog_bar=False)
        
        return loss