import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.optimization import AdamW, get_scheduler

from utils.dataset_utils import CircuitDataset
from utils.train_utils import sanitize_config, custom_mape_loss
from model.circuitformer import CircuitTransformer
from test import validation


logger = get_logger(__name__, log_level="INFO")


@hydra.main(version_base=None, config_path="configs", config_name="config_new.yaml")
def train(cfg: DictConfig):
    logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)

    # Create an Accelerator instance to handle DDP training.
    accelerator = Accelerator(
        project_config=accelerator_project_config,
        log_with="wandb"
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # Build training and validation datasets.
    topology_file = cfg.dataset.topology_file
    max_nodes = cfg.dataset.max_nodes
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers

    print("1")

    vocab_path = os.path.join("configs", "device_vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            device2idx = json.load(f)
        
        train_dataset = CircuitDataset(
            topology_file, 
            cfg.dataset.train_sample_files, 
            max_nodes=max_nodes, 
            device2idx=device2idx
        )
    else:
        train_dataset = CircuitDataset(
            topology_file, 
            cfg.dataset.train_sample_files, 
            max_nodes=max_nodes
        )
        device2idx = train_dataset.device2idx
        # Save the device vocabulary for later use
        with open(vocab_path, "w") as f:
            json.dump(device2idx, f)

    # Save the normalization stats from the training set.
    train_perf_mean = train_dataset.performance_mean.tolist()
    train_perf_std = train_dataset.performance_std.tolist()
    print("2")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("3")
    val_loader, new_val_loader = None, None
    if cfg.dataset.val_sample_files is not None:
        val_dataset = CircuitDataset(
            topology_file,
            cfg.dataset.val_sample_files, 
            max_nodes=max_nodes, 
            device2idx=device2idx,
            performance_mean=train_perf_mean,
            performance_std=train_perf_std
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # if cfg.dataset.new_val_sample_files is not None:
    #     new_val_dataset = CircuitDataset(
    #         topology_file, 
    #         cfg.dataset.new_val_sample_files, 
    #         max_nodes=max_nodes, 
    #         device2idx=device2idx,
    #         performance_mean=train_perf_mean,
    #         performance_std=train_perf_std
    #     )
    #     new_val_loader = DataLoader(new_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("4")
    # Create the model. 
    model = CircuitTransformer(max_nodes=max_nodes, num_device_types=len(device2idx), **cfg.model)
    print("5")
    # Define loss, optimizer, and learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=cfg.optimizer.lr, eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes)
    num_training_steps = cfg.training.num_epochs * num_update_steps_per_epoch
    num_training_steps_for_scheduler = cfg.training.num_epochs * num_update_steps_per_epoch * accelerator.num_processes

    scheduler = get_scheduler(
        name=cfg.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.scheduler.num_warmup_steps,
        num_training_steps=num_training_steps_for_scheduler
    )
    
    # Prepare objects with Accelerator.
    to_prepare = [model, optimizer, train_loader, scheduler]
    if val_loader is not None:
        to_prepare.append(val_loader)
    if new_val_loader is not None:
        to_prepare.append(new_val_loader)
        
    prepared = accelerator.prepare(*to_prepare)
    # Unpack in the same order
    if val_loader is not None and new_val_loader is not None:
        model, optimizer, train_loader, scheduler, val_loader, new_val_loader = prepared
    elif val_loader is not None:
        model, optimizer, train_loader, scheduler, val_loader = prepared
    else:
        model, optimizer, train_loader, scheduler = prepared

    if accelerator.is_main_process:
        tracker_config = {
            "optimizer": sanitize_config(cfg["optimizer"]),
            "model": sanitize_config(cfg["model"]),
            "scheduler": sanitize_config(cfg["scheduler"]),
            "training": sanitize_config(cfg["training"]),
        }
        accelerator.init_trackers(cfg.wandb.project,)
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb.define_metric("global_step")
            wandb.define_metric("train_loss*", step_metric="global_step")
            wandb.define_metric("global_step")
            wandb.define_metric("loss", step_metric="global_step")
            wandb.define_metric("val_loss", step_metric="global_step")
            wandb.define_metric("new_topology_loss", step_metric="global_step")

    # Train!
    num_epochs = cfg.training.num_epochs
    total_batch_size = cfg.training.batch_size * accelerator.num_processes
    torch.autograd.set_detect_anomaly(True)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    global_step = 0
    custom_step = 0

    progress_bar = tqdm(
        range(0, num_training_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            nodes = batch["nodes"]            # shape: [B, max_nodes]
            params = batch["params"]          # shape: [B, max_nodes, max_param_len]
            attn_mask = batch["attn_mask"]    # shape: [B, max_nodes, max_nodes]
            performance = batch["performance"]  # shape: [B, num_metrics]
            
            outputs = model(nodes, params, attn_mask=attn_mask)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN or Inf in outputs:", outputs)
            if torch.isnan(performance).any() or torch.isinf(performance).any():
                print("NaN or Inf in performance:", performance)

            # loss = F.mse_loss(outputs, performance, reduction="mean")
            loss = custom_mape_loss(outputs, performance)
            
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            for tracker in accelerator.trackers:
                tracker.log({"custom_step": custom_step,
                             "loss": loss}, step=custom_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"global_step": global_step,
                                 "train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            # Update progress bar
            custom_step += 1
            logs = {"step_loss": loss.detach().item(), "lr_unet": scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if (epoch + 1) % cfg.training.ckpt_epochs == 0: 
                save_path = os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                # Convert configuration to a standard dict.
                config_dict = OmegaConf.to_container(cfg, resolve=True)
                accelerator.unwrap_model(model).save_checkpoint(optimizer, epoch+1, config_dict, save_path)

            # validation
            if val_loader is not None:
                val_loss = validation(model, val_loader)
                accelerator.log({"val_loss": val_loss}, step=custom_step)
            if new_val_loader is not None:
                new_val_loss = validation(model, new_val_loader)
                accelerator.log({"new_topology_loss": new_val_loss}, step=custom_step)

    accelerator.end_training()

    
if __name__ == "__main__":
    train()
