from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/tensorboard")

def on_train_epoch_end(trainer):
    metrics = trainer.metrics
    epoch = trainer.epoch

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(k, v, epoch)