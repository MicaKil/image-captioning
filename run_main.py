import click
import wandb

from constants import COCO_IMGS_DIR, COCO_TRAIN_PKL, COCO_VAL_PKL, COCO_TEST_PKL, FLICKR8K_DIR, FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, FLICKR_TEST_CSV, \
    FLICKR8K_IMG_DIR, COCO_DIR
from runner.config import PROJECT, TAGS, CONFIG, DEVICE, NUM_WORKERS, PIN_MEMORY
from runner.runner import Runner

COLOR_INFO = "bright_blue"
COLOR_SUCCESS = "bright_green"
COLOR_WARNING = "yellow"
COLOR_ERROR = "red"
COLOR_HIGHLIGHT = "bright_white"


# checkpoint_ = "checkpoints/transformer/BEST_2025-03-23_20-58_2-4162.pt"

def print_run_config(config: dict):
    """Display the run configuration in a structured format"""
    click.secho("\n🔧 Current Run Configuration:", fg=COLOR_INFO, bold=True)

    # Model Configuration
    click.secho("Model Architecture:", fg=COLOR_HIGHLIGHT)
    click.echo(f"  • Type: {click.style(config['model'], fg=COLOR_INFO)}")
    click.echo(f"  • Encoder: {click.style(config['encoder'], fg=COLOR_INFO)}")
    click.echo(f"  • Decoder: {click.style(config['decoder'], fg=COLOR_INFO)}")
    click.echo(f"  • Hidden Size: {click.style(config['hidden_size'], fg=COLOR_INFO)}")
    click.echo(f"  • Layers: {click.style(config['num_layers'], fg=COLOR_INFO)}")

    # Training Parameters
    click.secho("\nTraining Setup:", fg=COLOR_HIGHLIGHT)
    click.echo(f"  • Batch Size: {click.style(config['batch_size'], fg=COLOR_INFO)}")
    click.echo(f"  • Max Epochs: {click.style(config['max_epochs'], fg=COLOR_INFO)}")
    click.echo(f"  • Optimizer: {click.style(config['optimizer'], fg=COLOR_INFO)}")
    click.echo(f"  • Encoder LR: {click.style(config['encoder_lr'], fg=COLOR_INFO)}")
    click.echo(f"  • Decoder LR: {click.style(config['decoder_lr'], fg=COLOR_INFO)}")

    # Dataset Info
    click.secho("\nDataset Configuration:", fg=COLOR_HIGHLIGHT)
    click.echo(f"  • Name: {click.style(config['dataset']['name'], fg=COLOR_INFO)}")
    click.echo(f"  • Version: {click.style(config['dataset']['version'], fg=COLOR_INFO)}")
    click.echo("  • Splits:")
    click.echo(f"    - Train: {click.style(config['dataset']['split']['train'], fg=COLOR_INFO)}%")
    click.echo(f"    - Val: {click.style(config['dataset']['split']['val'], fg=COLOR_INFO)}%")
    click.echo(f"    - Test: {click.style(config['dataset']['split']['test'], fg=COLOR_INFO)}%")

    # Advanced Settings
    click.secho("\nAdvanced Parameters:", fg=COLOR_HIGHLIGHT)
    click.echo(f"  • Dropout: {click.style(config['dropout'], fg=COLOR_INFO)}")
    click.echo(f"  • Gradient Clip: {click.style(config['gradient_clip'], fg=COLOR_INFO)}")
    click.echo(f"  • Patience: {click.style(config['patience'], fg=COLOR_INFO)} epochs")

    # Feature Toggles
    click.secho("\nFeature Flags:", fg=COLOR_HIGHLIGHT)
    click.echo(f"  • Scheduler: {click.style('Enabled' if config['scheduler'] else 'Disabled',
                                             fg=COLOR_SUCCESS if config['scheduler'] else COLOR_WARNING)}")
    if config['scheduler']:
        click.echo(f"    - Type: {config['scheduler']['type']}")
        click.echo(f"    - Factor: {config['scheduler']['factor']}")
        click.echo(f"    - Patience: {config['scheduler']['patience']} epochs")

    click.echo(f"  • BLEU-4 Evaluation: {click.style('Enabled' if config['eval_bleu4'] else 'Disabled',
                                                     fg=COLOR_SUCCESS if config['eval_bleu4'] else COLOR_WARNING)}")

    # Hardware Info
    click.secho("\nHardware Setup:", fg=COLOR_HIGHLIGHT)
    click.echo(f"  • Device: {click.style(str(DEVICE), fg=COLOR_INFO)}")
    click.echo(f"  • Workers: {click.style(NUM_WORKERS, fg=COLOR_INFO)}")
    click.echo(f"  • Pin Memory: {click.style(PIN_MEMORY, fg=COLOR_INFO)}")


def print_banner():
    """Display stylized application banner"""
    click.secho("╭─────────────────────────────────────╮", fg=COLOR_INFO)
    click.secho("│          Image Captioning           │", fg=COLOR_INFO)
    click.secho("│        Train & Test Pipeline        │", fg=COLOR_INFO)
    click.secho("╰─────────────────────────────────────╯", fg=COLOR_INFO)


@click.command()
@click.option("--use-wandb", is_flag=True, help="Enable Weights & Biases logging")
@click.option("--checkpoint", type=click.Path(exists=True), help="Path to checkpoint file for resuming training")
@click.option("--train/--no-train", default=True, help="Enable/disable training")
@click.option("--test/--no-test", default=True, help="Enable/disable testing")
@click.option("--save-ds", is_flag=True, help="Save processed dataset")
@click.option("--create-ds", is_flag=True, help="Create new dataset splits")
def run_cli(use_wandb, checkpoint, train, test, save_ds, create_ds):
    print_banner()

    if use_wandb:
        wandb.teardown()
    else:
        click.secho("\nℹ️ Weights & Biases disabled", fg=COLOR_WARNING)

    match CONFIG["dataset"]["name"]:
        case "flickr8k":
            ds_dir = FLICKR8K_DIR
            img_dir = FLICKR8K_IMG_DIR
            ds_splits = (FLICKR_TRAIN_CSV, FLICKR_VAL_CSV, FLICKR_TEST_CSV)
        case "coco":
            ds_dir = COCO_DIR
            img_dir = COCO_IMGS_DIR
            ds_splits = (COCO_TRAIN_PKL, COCO_VAL_PKL, COCO_TEST_PKL)
        case _:
            raise ValueError("Dataset not recognized")

    runner = Runner(use_wandb=use_wandb,
                    create_ds=create_ds,
                    save_ds=save_ds,
                    train_model=train,
                    test_model=test,
                    checkpoint_pth=checkpoint,
                    img_dir=img_dir,
                    ds_splits=ds_splits,
                    ds_dir=ds_dir,
                    project=PROJECT,
                    run_tags=TAGS,
                    run_config=CONFIG)

    # Execute pipeline
    try:
        click.secho("\nStarting training pipeline with this configuration...", fg=COLOR_INFO)
        print_run_config(CONFIG)
        runner.run()
        click.secho("\n✅ Pipeline completed successfully!", fg=COLOR_SUCCESS, bold=True)
    except KeyboardInterrupt:
        click.secho("\n❌ Pipeline interrupted by user!", fg=COLOR_WARNING)
    except Exception as e:
        click.secho(f"\n❌ Pipeline failed:\n", fg=COLOR_ERROR)
        raise e


if __name__ == "__main__":
    run_cli()
