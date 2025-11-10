"""
Training tab for Orpheus Gradio GUI.

This tab provides a complete interface for AI model training:
- Dataset configuration
- Hyperparameter tuning
- Training progress monitoring
- Checkpoint management
"""

import sys
import gradio as gr
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.state import app_state
from pytorch.config.training_config import TrainingConfig, get_config_by_name
from pytorch.model.transformer import create_model
from pytorch.data.dataloader import create_dataloaders
from pytorch.training import GradioTrainer, TrainingMetrics
from pytorch.utils.checkpoint_utils import get_latest_checkpoint
from pytorch.data.split import split_dataset, save_split_manifest


# ============================================================================
# Backend Functions
# ============================================================================

def validate_model_name(model_name: str) -> Tuple[bool, str]:
    """
    Validate model name for filesystem compatibility.

    Args:
        model_name: Proposed model name

    Returns:
        Tuple of (is_valid, error_message)
    """
    import re

    if not model_name or model_name.strip() == "":
        return False, "Model name cannot be empty"

    model_name = model_name.strip()

    # Only allow alphanumeric, underscores, and hyphens
    if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
        return False, "Model name can only contain letters, numbers, underscores, and hyphens"

    # Check length (reasonable filesystem limits)
    if len(model_name) > 100:
        return False, "Model name is too long (max 100 characters)"

    # Avoid reserved names
    reserved_names = ['con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
                     'lpt1', 'lpt2', 'lpt3']
    if model_name.lower() in reserved_names:
        return False, f"'{model_name}' is a reserved name"

    return True, ""


def auto_generate_split_manifest(
    processed_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[bool, str]:
    """
    Automatically generate split manifest if it doesn't exist.

    Args:
        processed_dir: Directory containing processed JSON files
        output_dir: Directory to save split manifest
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        seed: Random seed for reproducibility

    Returns:
        Tuple of (success, message)
    """
    try:
        # Check if processed directory exists and has files
        if not processed_dir.exists():
            return False, f"❌ Processed directory not found: {processed_dir}"

        json_files = list(processed_dir.glob("*.json"))
        if not json_files:
            return False, f"❌ No JSON files found in {processed_dir}"

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Perform split
        train_files, val_files, test_files = split_dataset(
            processed_dir=processed_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )

        # Save manifest
        save_split_manifest(train_files, val_files, test_files, output_dir)

        return True, (
            f"✅ Auto-generated dataset split:\n"
            f"  • Training: {len(train_files)} files\n"
            f"  • Validation: {len(val_files)} files\n"
            f"  • Test: {len(test_files)} files"
        )

    except Exception as e:
        return False, f"❌ Error generating split manifest: {str(e)}"


def load_training_config(preset_name: str) -> Tuple[Dict[str, Any], str]:
    """
    Load training configuration from preset.

    Args:
        preset_name: Config preset name

    Returns:
        Tuple of (config_dict, status_message)
    """
    try:
        config = get_config_by_name(preset_name)
        config_dict = {
            # Main hyperparameters
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'warmup_steps': config.warmup_steps,

            # Basic advanced settings
            'gradient_accumulation': config.gradient_accumulation_steps,
            'use_mixed_precision': config.mixed_precision,
            'early_stopping': config.early_stopping,
            'validation_interval': config.validation_interval,

            # Model architecture
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'ff_dim': config.ff_dim,
            'context_length': config.context_length,
            'dropout': config.dropout,
            'use_track_embeddings': config.use_track_embeddings,

            # Optimizer settings
            'weight_decay': config.weight_decay,
            'adam_beta1': config.adam_beta1,
            'adam_beta2': config.adam_beta2,
            'adam_epsilon': config.adam_epsilon,
            'max_grad_norm': config.max_grad_norm,

            # Learning rate schedule
            'lr_scheduler_type': config.lr_scheduler_type,
            'min_lr_ratio': config.min_lr_ratio,

            # Training loop settings
            'max_steps': config.max_steps,

            # Loss settings
            'loss_type': config.loss_type,
            'label_smoothing': config.label_smoothing,
            'melody_violation_weight': config.melody_violation_weight,
            'chord_violation_weight': config.chord_violation_weight,

            # Data settings
            'num_workers': config.num_workers,
            'use_cache': config.use_cache,

            # Validation settings
            'do_validation': config.do_validation,
            'validation_batches': config.validation_batches,

            # Logging settings
            'log_interval': config.log_interval,
            'use_tensorboard': config.use_tensorboard,
            'use_wandb': config.use_wandb,
            'wandb_project': config.wandb_project if config.wandb_project else "Orpheus",
            'wandb_run_name': config.wandb_run_name if config.wandb_run_name else "",

            # Checkpointing settings
            'checkpoint_interval': config.checkpoint_interval,
            'save_best_only': config.save_best_only,
            'max_checkpoints_to_keep': config.max_checkpoints_to_keep,

            # Early stopping settings
            'early_stopping_patience': config.early_stopping_patience,
            'early_stopping_min_delta': config.early_stopping_min_delta,

            # Hardware settings
            'device': config.device,
            'compile_model': config.compile_model,

            # Reproducibility settings
            'seed': config.seed,
            'deterministic': config.deterministic,

            # Debugging settings
            'debug': config.debug,
            'overfit_batch': config.overfit_batch,
            'profile': config.profile,
        }
        return config_dict, f"✅ Loaded '{preset_name}' configuration"
    except Exception as e:
        return {}, f"❌ Error loading config: {str(e)}"


def start_training_session(
    model_name: str,
    preset_name: str,
    # Main hyperparameters
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    warmup_steps: int,
    # Basic advanced settings
    gradient_accumulation: int,
    use_mixed_precision: bool,
    early_stopping: bool,
    validation_interval: int,
    # Model architecture
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    context_length: int,
    dropout: float,
    use_track_embeddings: bool,
    # Optimizer settings
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    max_grad_norm: float,
    # Learning rate schedule
    lr_scheduler_type: str,
    min_lr_ratio: float,
    # Training loop settings
    max_steps: Optional[int],
    # Loss settings
    loss_type: str,
    label_smoothing: float,
    melody_violation_weight: float,
    chord_violation_weight: float,
    # Data settings
    num_workers: int,
    use_cache: bool,
    # Validation settings
    do_validation: bool,
    validation_batches: Optional[int],
    # Logging settings
    log_interval: int,
    use_tensorboard: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str,
    # Checkpointing settings
    checkpoint_interval: int,
    save_best_only: bool,
    max_checkpoints_to_keep: int,
    # Early stopping settings
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    # Hardware settings
    device: str,
    compile_model: bool,
    # Reproducibility settings
    seed: int,
    deterministic: bool,
    # Debugging settings
    debug: bool,
    overfit_batch: bool,
    profile: bool,
) -> Tuple[str, str]:
    """
    Initialize and start training session.

    Returns:
        Tuple of (status_message, button_state)
    """
    try:
        # Validate model name
        is_valid, error_msg = validate_model_name(model_name)
        if not is_valid:
            return f"❌ Invalid model name: {error_msg}", "Start Training"

        model_name = model_name.strip()

        # Check if training is already running
        if app_state.trainer is not None and app_state.trainer.is_running():
            return "⚠️ Training is already running", "Start Training"

        # Clean up any existing trainer before creating new one
        if app_state.trainer is not None:
            try:
                app_state.clear_training_state()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                return f"❌ Error cleaning up previous training session: {str(e)}", "Start Training"

        # Setup model-specific paths
        project_root = Path(__file__).parent.parent.parent
        processed_dir = project_root / 'processed'

        # Model-specific split directory
        split_dir = project_root / 'pytorch' / 'data' / 'splits' / model_name
        manifest_path = split_dir / 'split_manifest.json'

        # Model-specific checkpoint directory
        checkpoint_dir = project_root / 'pytorch' / 'checkpoints' / model_name

        # Always regenerate split manifest on training start
        status_msg = f"📋 Generating dataset split for model '{model_name}'...\n"
        success, message = auto_generate_split_manifest(processed_dir, split_dir)

        if not success:
            return f"{status_msg}\n{message}", "Start Training"

        status_msg += f"{message}\n\n"

        # Verify manifest was created
        if not manifest_path.exists():
            return f"{status_msg}❌ Failed to create manifest at {manifest_path}", "Start Training"

        # Load configuration
        config = get_config_by_name(preset_name)

        # Override with GUI values - Main hyperparameters
        config.batch_size = batch_size
        config.learning_rate = learning_rate
        config.num_epochs = num_epochs
        config.warmup_steps = warmup_steps

        # Basic advanced settings
        config.gradient_accumulation_steps = gradient_accumulation
        config.mixed_precision = use_mixed_precision
        config.early_stopping = early_stopping
        config.validation_interval = validation_interval

        # Model architecture
        config.hidden_dim = hidden_dim
        config.num_layers = num_layers
        config.num_heads = num_heads
        config.ff_dim = ff_dim
        config.context_length = context_length
        config.dropout = dropout
        config.use_track_embeddings = use_track_embeddings

        # Optimizer settings
        config.weight_decay = weight_decay
        config.adam_beta1 = adam_beta1
        config.adam_beta2 = adam_beta2
        config.adam_epsilon = adam_epsilon
        config.max_grad_norm = max_grad_norm

        # Learning rate schedule
        config.lr_scheduler_type = lr_scheduler_type
        config.min_lr_ratio = min_lr_ratio

        # Training loop settings
        config.max_steps = max_steps

        # Loss settings
        config.loss_type = loss_type
        config.label_smoothing = label_smoothing
        config.melody_violation_weight = melody_violation_weight
        config.chord_violation_weight = chord_violation_weight

        # Data settings
        config.num_workers = num_workers
        config.use_cache = use_cache

        # Validation settings
        config.do_validation = do_validation
        config.validation_batches = validation_batches

        # Logging settings
        config.log_interval = log_interval
        config.use_tensorboard = use_tensorboard
        config.use_wandb = use_wandb
        config.wandb_project = wandb_project if wandb_project else None
        config.wandb_run_name = wandb_run_name if wandb_run_name else None

        # Checkpointing settings
        config.checkpoint_interval = checkpoint_interval
        config.save_best_only = save_best_only
        config.max_checkpoints_to_keep = max_checkpoints_to_keep

        # Early stopping settings
        config.early_stopping_patience = early_stopping_patience
        config.early_stopping_min_delta = early_stopping_min_delta

        # Hardware settings
        config.device = device
        config.compile_model = compile_model

        # Reproducibility settings
        config.seed = seed
        config.deterministic = deterministic

        # Debugging settings
        config.debug = debug
        config.overfit_batch = overfit_batch
        config.profile = profile

        # Path overrides
        config.split_manifest_path = manifest_path
        config.checkpoint_dir = checkpoint_dir

        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            split_manifest_path=config.split_manifest_path,
            batch_size=config.batch_size,
            max_length=config.context_length,
            num_workers=config.num_workers,
            use_cache=config.use_cache
        )

        if train_loader is None:
            return "❌ No training data found in split manifest", "Start Training"

        # Create model
        model = create_model(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            max_len=config.context_length,
            dropout=config.dropout
        )

        # Create GradioTrainer
        trainer = GradioTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # Initialize in app state
        app_state.initialize_trainer(trainer)

        # Define callbacks
        def progress_callback(metrics: TrainingMetrics):
            app_state.update_training_metrics(metrics)

        def log_callback(message: str):
            app_state.add_training_log(message)

        # Start training
        success = trainer.start_training(
            progress_callback=progress_callback,
            log_callback=log_callback
        )

        if success:
            final_msg = status_msg + "🚀 Training started successfully!"
            return final_msg, "⏸️ Pause"
        else:
            return status_msg + "❌ Failed to start training", "Start Training"

    except Exception as e:
        import traceback
        error_msg = f"❌ Error starting training: {str(e)}\n{traceback.format_exc()}"
        return error_msg, "Start Training"


def pause_training() -> Tuple[str, str]:
    """
    Pause current training session.

    Returns:
        Tuple of (status_message, button_state)
    """
    if app_state.trainer is None:
        return "⚠️ No training session active", "Start Training"

    if app_state.trainer.pause_training():
        return "⏸️ Training paused", "▶️ Resume"
    else:
        return "⚠️ Could not pause training", "⏸️ Pause"


def resume_training() -> Tuple[str, str]:
    """
    Resume paused training session.

    Returns:
        Tuple of (status_message, button_state)
    """
    if app_state.trainer is None:
        return "⚠️ No training session active", "Start Training"

    if app_state.trainer.resume_training():
        return "▶️ Training resumed", "⏸️ Pause"
    else:
        return "⚠️ Could not resume training", "▶️ Resume"


def stop_training() -> Tuple[str, str]:
    """
    Stop current training session.

    Returns:
        Tuple of (status_message, button_state)
    """
    if app_state.trainer is None:
        return "⚠️ No training session active", "Start Training"

    if app_state.trainer.stop_training(save_checkpoint=True):
        app_state.clear_training_state()
        return "🛑 Training stopped and checkpoint saved", "Start Training"
    else:
        return "⚠️ Could not stop training", "⏸️ Pause"


def handle_training_button(button_text: str, model_name: str, *config_args) -> Tuple[str, str]:
    """
    Handle training button click based on current state.

    Args:
        button_text: Current button text
        model_name: Name of the model to train
        *config_args: Configuration arguments

    Returns:
        Tuple of (status_message, new_button_text)
    """
    if button_text == "Start Training":
        return start_training_session(model_name, *config_args)
    elif button_text == "⏸️ Pause":
        return pause_training()
    elif button_text == "▶️ Resume":
        return resume_training()
    else:
        return "⚠️ Unknown button state", button_text


def update_training_progress() -> Dict[str, Any]:
    """
    Update training progress displays.

    Returns:
        Dictionary with updated component values
    """
    if app_state.trainer is None:
        return {
            'status_text': "No training session active",
            'epoch_text': "Epoch: -",
            'step_text': "Step: -",
            'loss_text': "Loss: -",
            'val_loss_text': "Val Loss: -",
            'lr_text': "LR: -",
            'time_text': "Time: -",
            'logs': "No logs available",
            'loss_plot': None,
            'lr_plot': None,
            'button_state': "Start Training",
        }

    # Get current metrics
    metrics = app_state.trainer.get_metrics()

    # Format time
    elapsed = int(metrics.elapsed_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Get logs
    logs = app_state.get_recent_training_logs(50)

    # Create loss plot data
    loss_plot_data = None
    if len(app_state.training_metrics_history['steps']) > 0:
        loss_df = pd.DataFrame({
            'Step': app_state.training_metrics_history['steps'],
            'Train Loss': app_state.training_metrics_history['train_loss'],
        })
        if len(app_state.training_metrics_history['val_loss']) > 0:
            # Align validation loss with steps
            val_steps = app_state.training_metrics_history['steps'][-len(app_state.training_metrics_history['val_loss']):]
            val_df = pd.DataFrame({
                'Step': val_steps,
                'Val Loss': app_state.training_metrics_history['val_loss']
            })
            loss_plot_data = pd.concat([loss_df, val_df], axis=1)
        else:
            loss_plot_data = loss_df

    # Create LR plot data
    lr_plot_data = None
    if len(app_state.training_metrics_history['steps']) > 0:
        lr_plot_data = pd.DataFrame({
            'Step': app_state.training_metrics_history['steps'],
            'Learning Rate': app_state.training_metrics_history['learning_rate'],
        })

    # Determine epoch count display
    # The trainer config may have num_epochs set, so we try to get it
    total_epochs = "?"
    if app_state.trainer and hasattr(app_state.trainer, 'config'):
        total_epochs = app_state.trainer.config.num_epochs

    # Determine button state based on training status
    button_state = "Start Training"
    if metrics.status == "running":
        button_state = "⏸️ Pause"
    elif metrics.status == "paused":
        button_state = "▶️ Resume"
    elif metrics.status in ["completed", "error", "idle"]:
        button_state = "Start Training"

    return {
        'status_text': f"Status: {metrics.status.upper()} - {metrics.message}",
        'epoch_text': f"Epoch: {metrics.epoch}/{total_epochs}",
        'step_text': f"Step: {metrics.step}/{metrics.total_steps}",
        'loss_text': f"Loss: {metrics.train_loss:.4f}",
        'val_loss_text': f"Val Loss: {metrics.val_loss:.4f}" if metrics.val_loss else "Val Loss: -",
        'lr_text': f"LR: {metrics.learning_rate:.2e}",
        'time_text': f"Time: {time_str}",
        'logs': logs if logs else "No logs yet...",
        'loss_plot': loss_plot_data,
        'lr_plot': lr_plot_data,
        'button_state': button_state,
    }


def list_checkpoints(checkpoint_dir: str) -> str:
    """
    List available checkpoints.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Formatted string with checkpoint list
    """
    try:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return "Checkpoint directory does not exist"

        checkpoints = list(checkpoint_path.glob("*.pt"))
        if not checkpoints:
            return "No checkpoints found"

        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        lines = ["Available Checkpoints:\n"]
        for i, ckpt in enumerate(checkpoints, 1):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            lines.append(f"{i}. {ckpt.name} ({size_mb:.1f} MB)")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing checkpoints: {str(e)}"


def get_checkpoint_dropdown_choices(checkpoint_dir: str) -> list:
    """
    Get list of checkpoint names for dropdown.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        List of checkpoint names
    """
    try:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return []

        checkpoints = list(checkpoint_path.glob("*.pt"))
        if not checkpoints:
            return []

        # Sort by modification time (most recent first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return [ckpt.name for ckpt in checkpoints]

    except Exception as e:
        return []


def load_checkpoint_for_training(
    checkpoint_name: str,
    checkpoint_dir: str,
    preset_name: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Load a checkpoint to resume training from it.

    Args:
        checkpoint_name: Name of checkpoint file
        checkpoint_dir: Directory containing checkpoints
        preset_name: Training preset to use as base config

    Returns:
        Tuple of (status_message, config_updates_dict)
    """
    try:
        if not checkpoint_name:
            return "⚠️ No checkpoint selected", {}

        checkpoint_path = Path(checkpoint_dir) / checkpoint_name

        if not checkpoint_path.exists():
            return f"❌ Checkpoint not found: {checkpoint_name}", {}

        # Load configuration from checkpoint
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract training state
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('best_val_loss', 'N/A')

        # Get config if available
        config_updates = {}
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            config_updates = {
                # Main hyperparameters
                'batch_size': saved_config.get('batch_size', 8),
                'learning_rate': saved_config.get('learning_rate', 1e-4),
                'num_epochs': saved_config.get('num_epochs', 50),
                'warmup_steps': saved_config.get('warmup_steps', 1000),
                # Basic advanced settings
                'gradient_accumulation': saved_config.get('gradient_accumulation_steps', 1),
                'use_mixed_precision': saved_config.get('mixed_precision', True),
                'early_stopping': saved_config.get('early_stopping', True),
                'validation_interval': saved_config.get('validation_interval', 500),
                # Model architecture
                'hidden_dim': saved_config.get('hidden_dim', 512),
                'num_layers': saved_config.get('num_layers', 8),
                'num_heads': saved_config.get('num_heads', 8),
                'ff_dim': saved_config.get('ff_dim', 2048),
                'context_length': saved_config.get('context_length', 2048),
                'dropout': saved_config.get('dropout', 0.1),
                'use_track_embeddings': saved_config.get('use_track_embeddings', True),
                # Optimizer settings
                'weight_decay': saved_config.get('weight_decay', 0.01),
                'adam_beta1': saved_config.get('adam_beta1', 0.9),
                'adam_beta2': saved_config.get('adam_beta2', 0.999),
                'adam_epsilon': saved_config.get('adam_epsilon', 1e-8),
                'max_grad_norm': saved_config.get('max_grad_norm', 1.0),
                # Learning rate schedule
                'lr_scheduler_type': saved_config.get('lr_scheduler_type', 'cosine'),
                'min_lr_ratio': saved_config.get('min_lr_ratio', 0.1),
                # Training loop settings
                'max_steps': saved_config.get('max_steps', None),
                # Loss settings
                'loss_type': saved_config.get('loss_type', 'weighted'),
                'label_smoothing': saved_config.get('label_smoothing', 0.0),
                'melody_violation_weight': saved_config.get('melody_violation_weight', 10.0),
                'chord_violation_weight': saved_config.get('chord_violation_weight', 5.0),
                # Data settings
                'num_workers': saved_config.get('num_workers', 0),
                'use_cache': saved_config.get('use_cache', False),
                # Validation settings
                'do_validation': saved_config.get('do_validation', True),
                'validation_batches': saved_config.get('validation_batches', None),
                # Logging settings
                'log_interval': saved_config.get('log_interval', 50),
                'use_tensorboard': saved_config.get('use_tensorboard', True),
                'use_wandb': saved_config.get('use_wandb', False),
                'wandb_project': saved_config.get('wandb_project', 'Orpheus'),
                'wandb_run_name': saved_config.get('wandb_run_name', ''),
                # Checkpointing settings
                'checkpoint_interval': saved_config.get('checkpoint_interval', 1000),
                'save_best_only': saved_config.get('save_best_only', False),
                'max_checkpoints_to_keep': saved_config.get('max_checkpoints_to_keep', 5),
                # Early stopping settings
                'early_stopping_patience': saved_config.get('early_stopping_patience', 10),
                'early_stopping_min_delta': saved_config.get('early_stopping_min_delta', 0.001),
                # Hardware settings
                'device': saved_config.get('device', 'cuda'),
                'compile_model': saved_config.get('compile_model', False),
                # Reproducibility settings
                'seed': saved_config.get('seed', 42),
                'deterministic': saved_config.get('deterministic', False),
                # Debugging settings
                'debug': saved_config.get('debug', False),
                'overfit_batch': saved_config.get('overfit_batch', False),
                'profile': saved_config.get('profile', False),
            }

        status_msg = (
            f"✅ Checkpoint loaded: {checkpoint_name}\n"
            f"  • Epoch: {epoch}\n"
            f"  • Step: {step}\n"
            f"  • Best Val Loss: {best_val_loss if isinstance(best_val_loss, str) else f'{best_val_loss:.4f}'}\n\n"
            f"⚠️ To resume training from this checkpoint, you must use the command line:\n"
            f"python pytorch/scripts/train.py --checkpoint {checkpoint_path}"
        )

        return status_msg, config_updates

    except Exception as e:
        import traceback
        error_msg = f"❌ Error loading checkpoint: {str(e)}\n{traceback.format_exc()}"
        return error_msg, {}


# ============================================================================
# Gradio Interface
# ============================================================================

def create_training_tab() -> gr.Tab:
    """
    Create the training tab with complete UI.

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Training") as tab:

        gr.Markdown("""
        ## AI Model Training

        Train your music generation model with real-time monitoring and control.
        """)

        with gr.Row():
            # ===== LEFT COLUMN: Configuration =====
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")

                # Model name input
                model_name_input = gr.Textbox(
                    label="Model Name",
                    value="model_1",
                    info="Name for this model",
                    placeholder="e.g., model_1, baseline_v2, experimental-lstm"
                )

                # Preset selector
                preset_dropdown = gr.Dropdown(
                    label="Training Preset",
                    choices=["default", "quick_test", "overfit", "production", "optimized_default", "low_memory"],
                    value="default",
                    info="Select a preset configuration (use 'low_memory' for 4GB VRAM or less)"
                )

                load_preset_btn = gr.Button("Load Preset", size="sm")

                config_status = gr.Textbox(
                    label="Status",
                    value="Select a preset and click 'Load Preset'",
                    interactive=False,
                    lines=2
                )

                gr.Markdown("### 🎛️ Hyperparameters")

                with gr.Row():
                    batch_size_slider = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=64,
                        value=8,
                        step=1,
                        info="Number of samples per batch"
                    )

                    learning_rate_input = gr.Number(
                        label="Learning Rate",
                        value=1e-4,
                        info="Optimizer learning rate"
                    )

                with gr.Row():
                    num_epochs_slider = gr.Slider(
                        label="Num Epochs",
                        minimum=1,
                        maximum=500,
                        value=50,
                        step=1,
                        info="Number of training epochs"
                    )

                    warmup_steps_slider = gr.Slider(
                        label="Warmup Steps",
                        minimum=0,
                        maximum=5000,
                        value=1000,
                        step=100,
                        info="LR warmup steps"
                    )

                with gr.Accordion("Advanced Settings", open=False):
                    gradient_accumulation_slider = gr.Slider(
                        label="Gradient Accumulation Steps",
                        minimum=1,
                        maximum=16,
                        value=1,
                        step=1,
                        info="Simulate larger batches"
                    )

                    use_mixed_precision_checkbox = gr.Checkbox(
                        label="Use Mixed Precision (FP16)",
                        value=True,
                        info="Faster training with FP16"
                    )

                    early_stopping_checkbox = gr.Checkbox(
                        label="Early Stopping",
                        value=True,
                        info="Stop if validation loss plateaus"
                    )

                    validation_interval_slider = gr.Slider(
                        label="Validation Interval (steps)",
                        minimum=50,
                        maximum=2000,
                        value=500,
                        step=50,
                        info="How often to validate"
                    )

                    # ===== Model Architecture =====
                    with gr.Accordion("Model Architecture", open=False):
                        hidden_dim_slider = gr.Slider(
                            label="Hidden Dimension",
                            minimum=128,
                            maximum=1024,
                            value=512,
                            step=64,
                            info="Size of hidden layers"
                        )

                        num_layers_slider = gr.Slider(
                            label="Number of Layers",
                            minimum=2,
                            maximum=16,
                            value=8,
                            step=1,
                            info="Number of transformer layers"
                        )

                        num_heads_slider = gr.Slider(
                            label="Number of Attention Heads",
                            minimum=2,
                            maximum=16,
                            value=8,
                            step=1,
                            info="Number of attention heads"
                        )

                        ff_dim_slider = gr.Slider(
                            label="Feed-Forward Dimension",
                            minimum=512,
                            maximum=4096,
                            value=2048,
                            step=256,
                            info="Size of feed-forward layers"
                        )

                        context_length_slider = gr.Slider(
                            label="Context Length",
                            minimum=256,
                            maximum=4096,
                            value=2048,
                            step=256,
                            info="Maximum sequence length"
                        )

                        dropout_slider = gr.Slider(
                            label="Dropout",
                            minimum=0.0,
                            maximum=0.5,
                            value=0.1,
                            step=0.05,
                            info="Dropout probability"
                        )

                        use_track_embeddings_checkbox = gr.Checkbox(
                            label="Use Track Embeddings",
                            value=True,
                            info="Use separate embeddings for melody vs chord tracks"
                        )

                    # ===== Optimizer Settings =====
                    with gr.Accordion("Optimizer Settings", open=False):
                        weight_decay_input = gr.Number(
                            label="Weight Decay",
                            value=0.01,
                            info="L2 regularization strength"
                        )

                        adam_beta1_slider = gr.Slider(
                            label="Adam Beta1",
                            minimum=0.8,
                            maximum=0.999,
                            value=0.9,
                            step=0.001,
                            info="First moment decay"
                        )

                        adam_beta2_slider = gr.Slider(
                            label="Adam Beta2",
                            minimum=0.9,
                            maximum=0.9999,
                            value=0.999,
                            step=0.0001,
                            info="Second moment decay"
                        )

                        adam_epsilon_input = gr.Number(
                            label="Adam Epsilon",
                            value=1e-8,
                            info="Numerical stability constant"
                        )

                        max_grad_norm_input = gr.Number(
                            label="Max Gradient Norm",
                            value=1.0,
                            info="Gradient clipping threshold"
                        )

                    # ===== Learning Rate Schedule =====
                    with gr.Accordion("Learning Rate Schedule", open=False):
                        lr_scheduler_type_dropdown = gr.Dropdown(
                            label="LR Scheduler Type",
                            choices=["linear", "cosine", "constant"],
                            value="cosine",
                            info="Type of learning rate schedule"
                        )

                        min_lr_ratio_slider = gr.Slider(
                            label="Min LR Ratio",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.05,
                            info="Minimum LR as ratio of max LR"
                        )

                    # ===== Training Loop Settings =====
                    with gr.Accordion("Training Loop Settings", open=False):
                        max_steps_input = gr.Number(
                            label="Max Steps (optional)",
                            value=None,
                            info="If set, overrides num_epochs"
                        )

                    # ===== Loss Settings =====
                    with gr.Accordion("Loss Settings", open=False):
                        loss_type_dropdown = gr.Dropdown(
                            label="Loss Type",
                            choices=["weighted", "constraint_aware", "track_aware"],
                            value="weighted",
                            info="Type of loss function"
                        )

                        label_smoothing_slider = gr.Slider(
                            label="Label Smoothing",
                            minimum=0.0,
                            maximum=0.2,
                            value=0.0,
                            step=0.01,
                            info="Label smoothing factor"
                        )

                        melody_violation_weight_slider = gr.Slider(
                            label="Melody Violation Weight",
                            minimum=1.0,
                            maximum=20.0,
                            value=10.0,
                            step=1.0,
                            info="Weight for melody constraint violations"
                        )

                        chord_violation_weight_slider = gr.Slider(
                            label="Chord Violation Weight",
                            minimum=1.0,
                            maximum=20.0,
                            value=5.0,
                            step=1.0,
                            info="Weight for chord constraint violations"
                        )

                    # ===== Data Settings =====
                    with gr.Accordion("Data Settings", open=False):
                        num_workers_slider = gr.Slider(
                            label="Number of Workers",
                            minimum=0,
                            maximum=8,
                            value=0,
                            step=1,
                            info="Number of data loading workers (0=main process)"
                        )

                        use_cache_checkbox = gr.Checkbox(
                            label="Use Cache",
                            value=False,
                            info="Cache dataset in memory"
                        )

                    # ===== Validation Settings =====
                    with gr.Accordion("Validation Settings", open=False):
                        do_validation_checkbox = gr.Checkbox(
                            label="Do Validation",
                            value=True,
                            info="Perform validation during training"
                        )

                        validation_batches_input = gr.Number(
                            label="Validation Batches (optional)",
                            value=None,
                            info="Limit number of validation batches"
                        )

                    # ===== Logging Settings =====
                    with gr.Accordion("Logging Settings", open=False):
                        log_interval_slider = gr.Slider(
                            label="Log Interval (steps)",
                            minimum=10,
                            maximum=500,
                            value=50,
                            step=10,
                            info="How often to log training metrics"
                        )

                        use_tensorboard_checkbox = gr.Checkbox(
                            label="Use TensorBoard",
                            value=True,
                            info="Enable TensorBoard logging"
                        )

                        use_wandb_checkbox = gr.Checkbox(
                            label="Use Weights & Biases",
                            value=False,
                            info="Enable W&B logging"
                        )

                        wandb_project_input = gr.Textbox(
                            label="W&B Project Name",
                            value="Orpheus",
                            info="Weights & Biases project name"
                        )

                        wandb_run_name_input = gr.Textbox(
                            label="W&B Run Name (optional)",
                            value="",
                            info="Custom run name for W&B"
                        )

                    # ===== Checkpointing Settings =====
                    with gr.Accordion("Checkpointing Settings", open=False):
                        checkpoint_interval_slider = gr.Slider(
                            label="Checkpoint Interval (steps)",
                            minimum=100,
                            maximum=5000,
                            value=1000,
                            step=100,
                            info="How often to save checkpoints"
                        )

                        save_best_only_checkbox = gr.Checkbox(
                            label="Save Best Only",
                            value=False,
                            info="Only save the best model checkpoint"
                        )

                        max_checkpoints_to_keep_slider = gr.Slider(
                            label="Max Checkpoints to Keep",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            info="Maximum number of checkpoints to retain"
                        )

                    # ===== Early Stopping Settings =====
                    with gr.Accordion("Early Stopping Settings", open=False):
                        early_stopping_patience_slider = gr.Slider(
                            label="Early Stopping Patience",
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            info="Validation intervals without improvement"
                        )

                        early_stopping_min_delta_input = gr.Number(
                            label="Early Stopping Min Delta",
                            value=0.001,
                            info="Minimum improvement to reset patience"
                        )

                    # ===== Hardware Settings =====
                    with gr.Accordion("Hardware Settings", open=False):
                        device_dropdown = gr.Dropdown(
                            label="Device",
                            choices=["auto", "cuda", "cpu"],
                            value="cuda",
                            info="Device to use for training"
                        )

                        compile_model_checkbox = gr.Checkbox(
                            label="Compile Model",
                            value=False,
                            info="Use torch.compile (PyTorch 2.0+)"
                        )

                    # ===== Reproducibility Settings =====
                    with gr.Accordion("Reproducibility Settings", open=False):
                        seed_slider = gr.Slider(
                            label="Random Seed",
                            minimum=0,
                            maximum=10000,
                            value=42,
                            step=1,
                            info="Random seed for reproducibility"
                        )

                        deterministic_checkbox = gr.Checkbox(
                            label="Deterministic Mode",
                            value=False,
                            info="More reproducible but slower"
                        )

                    # ===== Debugging Settings =====
                    with gr.Accordion("Debugging Settings", open=False):
                        debug_checkbox = gr.Checkbox(
                            label="Debug Mode",
                            value=False,
                            info="Enable debug mode (more logging)"
                        )

                        overfit_batch_checkbox = gr.Checkbox(
                            label="Overfit Batch",
                            value=False,
                            info="Train on single batch (for debugging)"
                        )

                        profile_checkbox = gr.Checkbox(
                            label="Enable Profiling",
                            value=False,
                            info="Enable performance profiling"
                        )

                gr.Markdown("### 🎮 Training Controls")

                with gr.Row():
                    training_button = gr.Button(
                        "Start Training",
                        variant="primary",
                        size="lg"
                    )

                    stop_button = gr.Button(
                        "🛑 Stop",
                        variant="stop",
                        size="lg"
                    )

            # ===== RIGHT COLUMN: Monitoring =====
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Live Monitoring")

                # Status displays
                status_display = gr.Textbox(
                    label="Training Status",
                    value="No training session active",
                    interactive=False,
                    lines=2
                )

                with gr.Row():
                    epoch_display = gr.Textbox(
                        label="Epoch",
                        value="Epoch: -",
                        interactive=False
                    )

                    step_display = gr.Textbox(
                        label="Step",
                        value="Step: -",
                        interactive=False
                    )

                    time_display = gr.Textbox(
                        label="Elapsed Time",
                        value="Time: -",
                        interactive=False
                    )

                with gr.Row():
                    loss_display = gr.Textbox(
                        label="Training Loss",
                        value="Loss: -",
                        interactive=False
                    )

                    val_loss_display = gr.Textbox(
                        label="Validation Loss",
                        value="Val Loss: -",
                        interactive=False
                    )

                    lr_display = gr.Textbox(
                        label="Learning Rate",
                        value="LR: -",
                        interactive=False
                    )

                # Loss plot
                loss_plot = gr.LinePlot(
                    x="Step",
                    y="Train Loss",
                    title="Training & Validation Loss",
                    height=300,
                    width=800
                )

                # Learning rate plot
                lr_plot = gr.LinePlot(
                    x="Step",
                    y="Learning Rate",
                    title="Learning Rate Schedule",
                    height=200,
                    width=800
                )

                # Training logs
                gr.Markdown("### Training Logs")

                logs_display = gr.Textbox(
                    label="Recent Logs",
                    value="No logs yet...",
                    interactive=False,
                    lines=12,
                    max_lines=20,
                    autoscroll=True
                )

                # Checkpoint management
                with gr.Accordion("💾 Checkpoint Management", open=False):
                    checkpoint_dir_input = gr.Textbox(
                        label="Checkpoint Directory",
                        value="pytorch/checkpoints",
                        interactive=True
                    )

                    checkpoint_dropdown = gr.Dropdown(
                        label="Select Checkpoint",
                        choices=[],
                        value=None,
                        info="Choose a checkpoint to load its configuration"
                    )

                    with gr.Row():
                        load_checkpoint_btn = gr.Button("Load Checkpoint", size="sm", variant="primary")
                        refresh_checkpoint_dropdown_btn = gr.Button("🔄 Refresh List", size="sm")

                    checkpoint_load_status = gr.Textbox(
                        label="Load Status",
                        value="Select a checkpoint and click 'Load Checkpoint'",
                        interactive=False,
                        lines=8
                    )

                    gr.Markdown("---")
                    gr.Markdown("**View All Checkpoints**")

                    with gr.Row():
                        list_checkpoints_btn = gr.Button("List Checkpoints", size="sm")
                        refresh_checkpoints_btn = gr.Button("🔄 Refresh", size="sm")

                    checkpoints_display = gr.Textbox(
                        label="Available Checkpoints",
                        value="Click 'List Checkpoints' to view",
                        interactive=False,
                        lines=8
                    )

        # ===== Event Handlers =====

        # Load preset configuration
        def on_load_preset(preset_name):
            config_dict, status = load_training_config(preset_name)
            if config_dict:
                return [
                    status,
                    # Main hyperparameters
                    config_dict['batch_size'],
                    config_dict['learning_rate'],
                    config_dict['num_epochs'],
                    config_dict['warmup_steps'],
                    # Basic advanced settings
                    config_dict['gradient_accumulation'],
                    config_dict['use_mixed_precision'],
                    config_dict['early_stopping'],
                    config_dict['validation_interval'],
                    # Model architecture
                    config_dict['hidden_dim'],
                    config_dict['num_layers'],
                    config_dict['num_heads'],
                    config_dict['ff_dim'],
                    config_dict['context_length'],
                    config_dict['dropout'],
                    config_dict['use_track_embeddings'],
                    # Optimizer settings
                    config_dict['weight_decay'],
                    config_dict['adam_beta1'],
                    config_dict['adam_beta2'],
                    config_dict['adam_epsilon'],
                    config_dict['max_grad_norm'],
                    # Learning rate schedule
                    config_dict['lr_scheduler_type'],
                    config_dict['min_lr_ratio'],
                    # Training loop settings
                    config_dict['max_steps'],
                    # Loss settings
                    config_dict['loss_type'],
                    config_dict['label_smoothing'],
                    config_dict['melody_violation_weight'],
                    config_dict['chord_violation_weight'],
                    # Data settings
                    config_dict['num_workers'],
                    config_dict['use_cache'],
                    # Validation settings
                    config_dict['do_validation'],
                    config_dict['validation_batches'],
                    # Logging settings
                    config_dict['log_interval'],
                    config_dict['use_tensorboard'],
                    config_dict['use_wandb'],
                    config_dict['wandb_project'],
                    config_dict['wandb_run_name'],
                    # Checkpointing settings
                    config_dict['checkpoint_interval'],
                    config_dict['save_best_only'],
                    config_dict['max_checkpoints_to_keep'],
                    # Early stopping settings
                    config_dict['early_stopping_patience'],
                    config_dict['early_stopping_min_delta'],
                    # Hardware settings
                    config_dict['device'],
                    config_dict['compile_model'],
                    # Reproducibility settings
                    config_dict['seed'],
                    config_dict['deterministic'],
                    # Debugging settings
                    config_dict['debug'],
                    config_dict['overfit_batch'],
                    config_dict['profile'],
                ]
            else:
                return [status] + [gr.update()] * 50

        load_preset_btn.click(
            fn=on_load_preset,
            inputs=[preset_dropdown],
            outputs=[
                config_status,
                # Main hyperparameters
                batch_size_slider,
                learning_rate_input,
                num_epochs_slider,
                warmup_steps_slider,
                # Basic advanced settings
                gradient_accumulation_slider,
                use_mixed_precision_checkbox,
                early_stopping_checkbox,
                validation_interval_slider,
                # Model architecture
                hidden_dim_slider,
                num_layers_slider,
                num_heads_slider,
                ff_dim_slider,
                context_length_slider,
                dropout_slider,
                use_track_embeddings_checkbox,
                # Optimizer settings
                weight_decay_input,
                adam_beta1_slider,
                adam_beta2_slider,
                adam_epsilon_input,
                max_grad_norm_input,
                # Learning rate schedule
                lr_scheduler_type_dropdown,
                min_lr_ratio_slider,
                # Training loop settings
                max_steps_input,
                # Loss settings
                loss_type_dropdown,
                label_smoothing_slider,
                melody_violation_weight_slider,
                chord_violation_weight_slider,
                # Data settings
                num_workers_slider,
                use_cache_checkbox,
                # Validation settings
                do_validation_checkbox,
                validation_batches_input,
                # Logging settings
                log_interval_slider,
                use_tensorboard_checkbox,
                use_wandb_checkbox,
                wandb_project_input,
                wandb_run_name_input,
                # Checkpointing settings
                checkpoint_interval_slider,
                save_best_only_checkbox,
                max_checkpoints_to_keep_slider,
                # Early stopping settings
                early_stopping_patience_slider,
                early_stopping_min_delta_input,
                # Hardware settings
                device_dropdown,
                compile_model_checkbox,
                # Reproducibility settings
                seed_slider,
                deterministic_checkbox,
                # Debugging settings
                debug_checkbox,
                overfit_batch_checkbox,
                profile_checkbox,
            ]
        )

        # Training button (start/pause/resume)
        training_button.click(
            fn=handle_training_button,
            inputs=[
                training_button,  # Current button text
                model_name_input,  # Model name
                preset_dropdown,
                # Main hyperparameters
                batch_size_slider,
                learning_rate_input,
                num_epochs_slider,
                warmup_steps_slider,
                # Basic advanced settings
                gradient_accumulation_slider,
                use_mixed_precision_checkbox,
                early_stopping_checkbox,
                validation_interval_slider,
                # Model architecture
                hidden_dim_slider,
                num_layers_slider,
                num_heads_slider,
                ff_dim_slider,
                context_length_slider,
                dropout_slider,
                use_track_embeddings_checkbox,
                # Optimizer settings
                weight_decay_input,
                adam_beta1_slider,
                adam_beta2_slider,
                adam_epsilon_input,
                max_grad_norm_input,
                # Learning rate schedule
                lr_scheduler_type_dropdown,
                min_lr_ratio_slider,
                # Training loop settings
                max_steps_input,
                # Loss settings
                loss_type_dropdown,
                label_smoothing_slider,
                melody_violation_weight_slider,
                chord_violation_weight_slider,
                # Data settings
                num_workers_slider,
                use_cache_checkbox,
                # Validation settings
                do_validation_checkbox,
                validation_batches_input,
                # Logging settings
                log_interval_slider,
                use_tensorboard_checkbox,
                use_wandb_checkbox,
                wandb_project_input,
                wandb_run_name_input,
                # Checkpointing settings
                checkpoint_interval_slider,
                save_best_only_checkbox,
                max_checkpoints_to_keep_slider,
                # Early stopping settings
                early_stopping_patience_slider,
                early_stopping_min_delta_input,
                # Hardware settings
                device_dropdown,
                compile_model_checkbox,
                # Reproducibility settings
                seed_slider,
                deterministic_checkbox,
                # Debugging settings
                debug_checkbox,
                overfit_batch_checkbox,
                profile_checkbox,
            ],
            outputs=[status_display, training_button]
        )

        # Stop button
        stop_button.click(
            fn=stop_training,
            outputs=[status_display, training_button]
        )

        # Refresh checkpoint dropdown
        def on_refresh_checkpoint_dropdown(checkpoint_dir):
            choices = get_checkpoint_dropdown_choices(checkpoint_dir)
            return gr.Dropdown(choices=choices, value=None)

        refresh_checkpoint_dropdown_btn.click(
            fn=on_refresh_checkpoint_dropdown,
            inputs=[checkpoint_dir_input],
            outputs=[checkpoint_dropdown]
        )

        # Load checkpoint
        def on_load_checkpoint(checkpoint_name, checkpoint_dir, preset_name):
            status_msg, config_updates = load_checkpoint_for_training(
                checkpoint_name, checkpoint_dir, preset_name
            )

            if config_updates:
                return [
                    status_msg,
                    # Main hyperparameters
                    config_updates['batch_size'],
                    config_updates['learning_rate'],
                    config_updates['num_epochs'],
                    config_updates['warmup_steps'],
                    # Basic advanced settings
                    config_updates['gradient_accumulation'],
                    config_updates['use_mixed_precision'],
                    config_updates['early_stopping'],
                    config_updates['validation_interval'],
                    # Model architecture
                    config_updates['hidden_dim'],
                    config_updates['num_layers'],
                    config_updates['num_heads'],
                    config_updates['ff_dim'],
                    config_updates['context_length'],
                    config_updates['dropout'],
                    config_updates['use_track_embeddings'],
                    # Optimizer settings
                    config_updates['weight_decay'],
                    config_updates['adam_beta1'],
                    config_updates['adam_beta2'],
                    config_updates['adam_epsilon'],
                    config_updates['max_grad_norm'],
                    # Learning rate schedule
                    config_updates['lr_scheduler_type'],
                    config_updates['min_lr_ratio'],
                    # Training loop settings
                    config_updates['max_steps'],
                    # Loss settings
                    config_updates['loss_type'],
                    config_updates['label_smoothing'],
                    config_updates['melody_violation_weight'],
                    config_updates['chord_violation_weight'],
                    # Data settings
                    config_updates['num_workers'],
                    config_updates['use_cache'],
                    # Validation settings
                    config_updates['do_validation'],
                    config_updates['validation_batches'],
                    # Logging settings
                    config_updates['log_interval'],
                    config_updates['use_tensorboard'],
                    config_updates['use_wandb'],
                    config_updates['wandb_project'],
                    config_updates['wandb_run_name'],
                    # Checkpointing settings
                    config_updates['checkpoint_interval'],
                    config_updates['save_best_only'],
                    config_updates['max_checkpoints_to_keep'],
                    # Early stopping settings
                    config_updates['early_stopping_patience'],
                    config_updates['early_stopping_min_delta'],
                    # Hardware settings
                    config_updates['device'],
                    config_updates['compile_model'],
                    # Reproducibility settings
                    config_updates['seed'],
                    config_updates['deterministic'],
                    # Debugging settings
                    config_updates['debug'],
                    config_updates['overfit_batch'],
                    config_updates['profile'],
                ]
            else:
                return [status_msg] + [gr.update()] * 50

        load_checkpoint_btn.click(
            fn=on_load_checkpoint,
            inputs=[checkpoint_dropdown, checkpoint_dir_input, preset_dropdown],
            outputs=[
                checkpoint_load_status,
                # Main hyperparameters
                batch_size_slider,
                learning_rate_input,
                num_epochs_slider,
                warmup_steps_slider,
                # Basic advanced settings
                gradient_accumulation_slider,
                use_mixed_precision_checkbox,
                early_stopping_checkbox,
                validation_interval_slider,
                # Model architecture
                hidden_dim_slider,
                num_layers_slider,
                num_heads_slider,
                ff_dim_slider,
                context_length_slider,
                dropout_slider,
                use_track_embeddings_checkbox,
                # Optimizer settings
                weight_decay_input,
                adam_beta1_slider,
                adam_beta2_slider,
                adam_epsilon_input,
                max_grad_norm_input,
                # Learning rate schedule
                lr_scheduler_type_dropdown,
                min_lr_ratio_slider,
                # Training loop settings
                max_steps_input,
                # Loss settings
                loss_type_dropdown,
                label_smoothing_slider,
                melody_violation_weight_slider,
                chord_violation_weight_slider,
                # Data settings
                num_workers_slider,
                use_cache_checkbox,
                # Validation settings
                do_validation_checkbox,
                validation_batches_input,
                # Logging settings
                log_interval_slider,
                use_tensorboard_checkbox,
                use_wandb_checkbox,
                wandb_project_input,
                wandb_run_name_input,
                # Checkpointing settings
                checkpoint_interval_slider,
                save_best_only_checkbox,
                max_checkpoints_to_keep_slider,
                # Early stopping settings
                early_stopping_patience_slider,
                early_stopping_min_delta_input,
                # Hardware settings
                device_dropdown,
                compile_model_checkbox,
                # Reproducibility settings
                seed_slider,
                deterministic_checkbox,
                # Debugging settings
                debug_checkbox,
                overfit_batch_checkbox,
                profile_checkbox,
            ]
        )

        # List checkpoints
        list_checkpoints_btn.click(
            fn=list_checkpoints,
            inputs=[checkpoint_dir_input],
            outputs=[checkpoints_display]
        )

        refresh_checkpoints_btn.click(
            fn=list_checkpoints,
            inputs=[checkpoint_dir_input],
            outputs=[checkpoints_display]
        )

        # Auto-update progress every 2 seconds using Gradio 5.x Timer
        def update_all_displays():
            updates = update_training_progress()
            return [
                updates['status_text'],
                updates['epoch_text'],
                updates['step_text'],
                updates['loss_text'],
                updates['val_loss_text'],
                updates['lr_text'],
                updates['time_text'],
                updates['logs'],
                updates['loss_plot'],
                updates['lr_plot'],
                updates['button_state'],
            ]

        # Manual refresh button
        with gr.Row():
            refresh_metrics_btn = gr.Button("🔄 Refresh Metrics Now", size="sm")

        # Manual refresh on button click
        refresh_metrics_btn.click(
            fn=update_all_displays,
            outputs=[
                status_display,
                epoch_display,
                step_display,
                loss_display,
                val_loss_display,
                lr_display,
                time_display,
                logs_display,
                loss_plot,
                lr_plot,
                training_button,
            ]
        )

        # Auto-refresh every 2 seconds using Timer (Gradio 5.x)
        timer = gr.Timer(2)
        timer.tick(
            fn=update_all_displays,
            outputs=[
                status_display,
                epoch_display,
                step_display,
                loss_display,
                val_loss_display,
                lr_display,
                time_display,
                logs_display,
                loss_plot,
                lr_plot,
                training_button,
            ]
        )

        # Also refresh when tab is selected
        tab.select(
            fn=update_all_displays,
            outputs=[
                status_display,
                epoch_display,
                step_display,
                loss_display,
                val_loss_display,
                lr_display,
                time_display,
                logs_display,
                loss_plot,
                lr_plot,
                training_button,
            ]
        )

    return tab
