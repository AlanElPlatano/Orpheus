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
from typing import Tuple, Dict, Any

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
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'warmup_steps': config.warmup_steps,
            'gradient_accumulation': config.gradient_accumulation_steps,
            'use_mixed_precision': config.mixed_precision,
            'early_stopping': config.early_stopping,
            'validation_interval': config.validation_interval,
        }
        return config_dict, f"✅ Loaded '{preset_name}' configuration"
    except Exception as e:
        return {}, f"❌ Error loading config: {str(e)}"


def start_training_session(
    model_name: str,
    preset_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    warmup_steps: int,
    gradient_accumulation: int,
    use_mixed_precision: bool,
    early_stopping: bool,
    validation_interval: int,
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

        # Override with GUI values
        config.batch_size = batch_size
        config.learning_rate = learning_rate
        config.num_epochs = num_epochs
        config.warmup_steps = warmup_steps
        config.gradient_accumulation_steps = gradient_accumulation
        config.mixed_precision = use_mixed_precision
        config.early_stopping = early_stopping
        config.validation_interval = validation_interval
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
                'batch_size': saved_config.get('batch_size', 8),
                'learning_rate': saved_config.get('learning_rate', 1e-4),
                'num_epochs': saved_config.get('num_epochs', 50),
                'warmup_steps': saved_config.get('warmup_steps', 1000),
                'gradient_accumulation': saved_config.get('gradient_accumulation_steps', 1),
                'use_mixed_precision': saved_config.get('mixed_precision', True),
                'early_stopping': saved_config.get('early_stopping', True),
                'validation_interval': saved_config.get('validation_interval', 500),
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
                    info="Name for this model (used for checkpoints and splits)",
                    placeholder="e.g., model_1, baseline_v2, experimental-lstm"
                )

                # Preset selector
                preset_dropdown = gr.Dropdown(
                    label="Training Preset",
                    choices=["default", "quick_test", "overfit", "production"],
                    value="default",
                    info="Select a preset configuration"
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
                    config_dict['batch_size'],
                    config_dict['learning_rate'],
                    config_dict['num_epochs'],
                    config_dict['warmup_steps'],
                    config_dict['gradient_accumulation'],
                    config_dict['use_mixed_precision'],
                    config_dict['early_stopping'],
                    config_dict['validation_interval'],
                ]
            else:
                return [status] + [gr.update()] * 8

        load_preset_btn.click(
            fn=on_load_preset,
            inputs=[preset_dropdown],
            outputs=[
                config_status,
                batch_size_slider,
                learning_rate_input,
                num_epochs_slider,
                warmup_steps_slider,
                gradient_accumulation_slider,
                use_mixed_precision_checkbox,
                early_stopping_checkbox,
                validation_interval_slider,
            ]
        )

        # Training button (start/pause/resume)
        training_button.click(
            fn=handle_training_button,
            inputs=[
                training_button,  # Current button text
                model_name_input,  # Model name
                preset_dropdown,
                batch_size_slider,
                learning_rate_input,
                num_epochs_slider,
                warmup_steps_slider,
                gradient_accumulation_slider,
                use_mixed_precision_checkbox,
                early_stopping_checkbox,
                validation_interval_slider,
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
                    config_updates['batch_size'],
                    config_updates['learning_rate'],
                    config_updates['num_epochs'],
                    config_updates['warmup_steps'],
                    config_updates['gradient_accumulation'],
                    config_updates['use_mixed_precision'],
                    config_updates['early_stopping'],
                    config_updates['validation_interval'],
                ]
            else:
                return [status_msg] + [gr.update()] * 8

        load_checkpoint_btn.click(
            fn=on_load_checkpoint,
            inputs=[checkpoint_dropdown, checkpoint_dir_input, preset_dropdown],
            outputs=[
                checkpoint_load_status,
                batch_size_slider,
                learning_rate_input,
                num_epochs_slider,
                warmup_steps_slider,
                gradient_accumulation_slider,
                use_mixed_precision_checkbox,
                early_stopping_checkbox,
                validation_interval_slider,
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
