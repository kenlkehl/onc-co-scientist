"""Embed the user-requested schematics and export them with report artifacts."""

from pathlib import Path
import shutil


ASSETS = Path(__file__).with_name("assets")


def draw_custom_workflow(canvas, out):
    draw_figure(canvas, out, ASSETS / "custom_runner_claim_states.png")


def draw_dataset_construction(canvas, out):
    draw_figure(canvas, out, ASSETS / "dataset_construction_compact.png")


def draw_figure(canvas, out, asset):
    """Place the revised diagram on the 1152 x 648 report page without cropping."""
    from reportlab.lib.utils import ImageReader

    image = ImageReader(str(asset))
    source_width, source_height = image.getSize()
    left, bottom, available_width, available_height = 28, 30, 1096, 600
    scale = min(available_width / source_width, available_height / source_height)
    width, height = source_width * scale, source_height * scale
    canvas.drawImage(image, left + (available_width - width) / 2,
                     bottom + (available_height - height) / 2,
                     width=width, height=height, mask="auto")
    destination = out / "figures" / asset.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(asset, destination)
