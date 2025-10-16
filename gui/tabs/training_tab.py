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
from typing import Optional, Tuple, Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.state import app_state
from pytorch.config.training_config import TrainingConfig, get_config_by_name
from pytorch.model.transformer import create_model
from pytorch.data.dataloader import create_dataloaders
from pytorch.training import GradioTrainer, TrainingMetrics
from pytorch.utils.checkpoint_utils import get_latest_checkpoint


# ============================================================================
# Backend Functions
# ============================================================================

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
    preset_name: str,
    split_manifest_path: str,
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
        # Check if training is already running
        if app_state.trainer is not None and app_state.trainer.is_running():
            return "⚠️ Training is already running", "Start Training"

        # Validate split manifest path
        manifest_path = Path(split_manifest_path)
        if not manifest_path.exists():
            return f"❌ Split manifest not found: {split_manifest_path}", "Start Training"

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
            return "🚀 Training started successfully!", "⏸️ Pause"
        else:
            return "❌ Failed to start training", "Start Training"

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


def handle_training_button(button_text: str, *config_args) -> Tuple[str, str]:
    """
    Handle training button click based on current state.

    Args:
        button_text: Current button text
        *config_args: Configuration arguments

    Returns:
        Tuple of (status_message, new_button_text)
    """
    if button_text == "Start Training":
        return start_training_session(*config_args)
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

    return {
        'status_text': f"Status: {metrics.status.upper()} - {metrics.message}",
        'epoch_text': f"Epoch: {metrics.epoch}/{metrics.total_steps // 100 if metrics.total_steps > 0 else '?'}",
        'step_text': f"Step: {metrics.step}/{metrics.total_steps}",
        'loss_text': f"Loss: {metrics.train_loss:.4f}",
        'val_loss_text': f"Val Loss: {metrics.val_loss:.4f}" if metrics.val_loss else "Val Loss: -",
        'lr_text': f"LR: {metrics.learning_rate:.2e}",
        'time_text': f"Time: {time_str}",
        'logs': logs if logs else "No logs yet...",
        'loss_plot': loss_plot_data,
        'lr_plot': lr_plot_data,
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
        ## 🤖 AI Model Training

        Train your music generation model with real-time monitoring and control.
        """)

        with gr.Row():
            # ===== LEFT COLUMN: Configuration =====
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")

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

                gr.Markdown("### 📁 Dataset")

                split_manifest_input = gr.Textbox(
                    label="Split Manifest Path",
                    value="pytorch/data/splits/split_manifest.json",
                    info="Path to train/val/test split manifest"
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
                gr.Markdown("### 📝 Training Logs")

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
                preset_dropdown,
                split_manifest_input,
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
            ]
        )

    return tab
