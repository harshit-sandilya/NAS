import torch
import time
from model import Transformer
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def measure_time(start_time=None):
    if start_time is None:
        return time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def train_model(action, config, dataModule):
    torch.set_float32_matmul_precision("high")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    config.train["num_heads"] = int(action[1])
    config.train["num_layers"] = int(action[0])
    config.train["embedding_dimension"] = config.train["num_heads"] * 64

    print("+++++++++++++++++CONFIG+++++++++++++++++")
    print(config.train)
    print(len(dataModule.train))
    print("++++++++++++++++++++++++++++++++++++++++")

    start_time = measure_time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TensorBoardLogger("logs/", name="transformer")
    if config.deepspeed is not None:
        strategy = DeepSpeedStrategy(config=config.deepspeed)
    else:
        strategy = "ddp"
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"logs/checkpoints/",
        filename="checkpoint-step-{step:08d}",
        save_top_k=-1,
        mode="min",
    )

    trainer = Trainer(
        accelerator="auto",
        devices=config.train["gpu_cores"],
        max_epochs=1,
        precision=config.train["precision"],
        log_every_n_steps=config.train["log_steps"],
        strategy=strategy,
        logger=logger,
        callbacks=[lr_monitor, checkpoint],
        gradient_clip_val=config.train["gradient_clip_val"],
    )

    print(f"[{measure_time(start_time)}]Initializing model on {trainer.global_rank}...")
    model = (
        Transformer(config.train, dataModule.vocab_size, config.dtype)
        .to(device)
        .to(config.dtype)
    )
    print(f"[{measure_time(start_time)}]Model initialized on {trainer.global_rank}.")

    print(f"[{measure_time(start_time)}]Starting training on {trainer.global_rank}...")
    trainer.fit(model, dataModule)
    print(f"[{measure_time(start_time)}]Training complete on {trainer.global_rank}.")

    with torch.no_grad():
        torch.cuda.empty_cache()

    return 0
