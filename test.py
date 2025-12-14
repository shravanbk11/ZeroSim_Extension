import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from utils.dataset_utils import CircuitDataset
from model.circuitformer import CircuitTransformer


def validation(model, data_loader):
    model.eval()
    val_loss = 0.0
    nb_val_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            nodes = batch["nodes"]
            params = batch["params"]
            attn_mask = batch["attn_mask"]
            performance = batch["performance"]
            
            outputs = model(nodes, params, attn_mask=attn_mask)
            loss = F.mse_loss(outputs, performance, reduction="mean")
            batch_size_val = nodes.size(0)
            val_loss += loss.item() * batch_size_val
            nb_val_samples += batch_size_val

        val_loss = val_loss / nb_val_samples

    return val_loss


@hydra.main(version_base=None, config_path="configs", config_name="config_new.yaml")
def test(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # Initialize wandb (if you want to log test metrics).
    # wandb.init(project=cfg.wandb.project, name="test_run", reinit=True)
    
    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running on device: %s", device)
    
    # Build the test dataset.
    topology_file = cfg.dataset.topology_file
    max_nodes = cfg.dataset.max_nodes
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    
    test_dataset = CircuitDataset(topology_file, cfg.dataset.test_sample_files, max_nodes=max_nodes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Instantiate the model using Hydra configuration and move it to the chosen device.
    vocab_path = os.path.join("configs", "device_vocab.json")
    with open(vocab_path, "r") as f:
        device2idx = json.load(f)

    model = CircuitTransformer(max_nodes=max_nodes, num_device_types=len(device2idx), **cfg.model).to(device)
    
    # Load checkpoint if a checkpoint_path is provided.
    if cfg.checkpoint_path:
        saved_config = model.load_checkpoint(cfg.checkpoint_path, filter_func=None)
        logger.info("Loaded checkpoint with config: %s", saved_config)
    
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0

    # Retrieve the normalization statistics from the dataset and move them to device.
    performance_mean = torch.tensor(test_dataset.performance_mean, dtype=torch.float).to(device)
    performance_std = torch.tensor(test_dataset.performance_std, dtype=torch.float).to(device)

    # Initialize accumulators for MAPE computation per performance metric.
    num_metrics = len(test_dataset.metric_keys)
    metric_errors_sum = torch.zeros(num_metrics, device=device)
    performance_avg = torch.zeros(num_metrics, device=device)
    outputs_avg = torch.zeros(num_metrics, device=device)
    num_samples = 0
    epsilon = 1e-8  # Small value to avoid division by zero
    
    with torch.no_grad():
        for batch in test_loader:
            nodes = batch["nodes"].to(device)
            params = batch["params"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            performance = batch["performance"].to(device)
            
            outputs = model(nodes, params, attn_mask=attn_mask)
            loss = criterion(outputs, performance)
            test_loss += loss.item() * nodes.size(0)

            # Restore predictions and targets to their original scale.
            outputs_orig = outputs * performance_std + performance_mean
            performance_orig = performance * performance_std + performance_mean
            
            # Compute the absolute percentage error per metric for each sample.
            batch_errors = torch.abs((performance_orig - outputs_orig) / (performance_orig + epsilon))
            # Sum errors for each performance metric.
            metric_errors_sum += torch.sum(batch_errors, dim=0)
            
            num_samples += nodes.size(0)

            # Just for debugging, once per batch or at the end:
            # for i, key in enumerate(test_dataset.metric_keys):
            #     gt_vals = performance_orig[:, i]   # ground truth for this metric in the batch
            #     pred_vals = outputs_orig[:, i]     # predictions for this metric in the batch

            #     # Print or log min/max/mean to see if ground truths are near zero:
            #     logger.info(
            #         "[%s] GT min=%.5g, max=%.5g, mean=%.5g | Pred min=%.5g, max=%.5g, mean=%.5g",
            #         key,
            #         gt_vals.min().item(),
            #         gt_vals.max().item(),
            #         gt_vals.mean().item(),
            #         pred_vals.min().item(),
            #         pred_vals.max().item(),
            #         pred_vals.mean().item(),
            #     )
    
    test_loss /= len(test_dataset)
    # Compute MAPE for each performance metric (in percentage).
    mape_per_metric = metric_errors_sum / num_samples * 100

    logger.info("Test Loss: %.4f", test_loss)
    # wandb.log({"test_loss": test_loss})
    for i, key in enumerate(test_dataset.metric_keys):
        logger.info("MAPE for %s: %.2f%%", key, mape_per_metric[i].item())
    
if __name__ == "__main__":
    test()
