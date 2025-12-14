import torch


def custom_mape_loss(outputs, targets):
    loss = torch.abs((targets - outputs) / (targets + 1e-8))
    return loss.mean()


def sanitize_config(config):
    """Recursively sanitize the config, replacing None with 'None' for JSON compatibility."""
    if isinstance(config, dict):
        return {k: sanitize_config(v) for k, v in config.items() if v is not None}
    elif isinstance(config, list):
        return [sanitize_config(v) for v in config if v is not None]
    elif config is None:
        return "None"
    else:
        return config
