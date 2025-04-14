import click

from attn_plotter import plot_attn_cli
from run_cli import run_cli


@click.group()
def cli():
    """CLI for the application."""
    pass


cli.add_command(plot_attn_cli)
cli.add_command(run_cli)
