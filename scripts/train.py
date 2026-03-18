#!/usr/bin/env python3
"""
CLI entry point for training the GO fusion policy.

Usage:
    python scripts/train.py [--config configs/default.yaml] [--resume path/to/checkpoint.pt]

Examples:
    # Train with default config
    python scripts/train.py

    # Train with custom config
    python scripts/train.py --config configs/my_config.yaml

    # Resume training from checkpoint
    python scripts/train.py --resume checkpoints/checkpoint_000200.pt
"""

import argparse
import os
import sys

import yaml

# Add project root to path so imports work when running from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GO fusion policy with PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"),
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--hlo-dir",
        type=str,
        default=None,
        help="Override HLO directory from config",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Override target GPU (e.g., 'v100', 'a100')",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=None,
        help="Override number of training updates",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration parameters.
    """
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Apply CLI overrides
    if args.hlo_dir is not None:
        config["paths"]["hlo_dir"] = args.hlo_dir
    if args.gpu is not None:
        config["environment"]["gpu_target"] = args.gpu
    if args.num_updates is not None:
        config["training"]["num_updates"] = args.num_updates
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    # Resolve relative paths relative to project root
    for path_key in ["hlo_dir", "checkpoint_dir", "log_dir"]:
        path_val = config["paths"][path_key]
        if not os.path.isabs(path_val):
            config["paths"][path_key] = os.path.normpath(
                os.path.join(PROJECT_ROOT, path_val)
            )

    # Print config summary
    print("\nConfiguration:")
    print(f"  Model: hidden_dim={config['model']['hidden_dim']}, "
          f"num_priorities={config['model']['num_priorities']}, "
          f"iterations={config['model']['num_iterations']}")
    print(f"  Training: lr={config['training']['learning_rate']}, "
          f"updates={config['training']['num_updates']}, "
          f"rollouts/update={config['training']['rollouts_per_update']}")
    print(f"  Environment: GPU={config['environment']['gpu_target']}, "
          f"max_cluster={config['environment']['max_cluster_size']}")
    print(f"  HLO dir: {config['paths']['hlo_dir']}")
    print(f"  Checkpoints: {config['paths']['checkpoint_dir']}")
    print(f"  Logs: {config['paths']['log_dir']}")

    # Create trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Checkpoint not found: {args.resume}")
            sys.exit(1)
        trainer.load_checkpoint(args.resume)

    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
