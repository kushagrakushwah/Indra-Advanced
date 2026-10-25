"""
Command Line Interface for Indra-Advanced Pipeline.
Run document intelligence and table extraction directly from the terminal.
"""

import sys
from pathlib import Path
from typing import Optional
import json
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from indra_advanced.config import (
    PipelineConfig,
    OCRModelTier,
    GLiNERModelVariant,
    SerializationMode,
)
from indra_advanced.pipeline import IndraAdvancedPipeline
from indra_advanced.schemas.domain_schemas import (
    INVOICE_SCHEMA,
    FINANCIAL_SCHEMA,
    MEDICAL_LAB_SCHEMA,
)
from indra_advanced.utils.table_renderer import SyntheticTableGenerator

app = typer.Typer(
    name="indra-advanced",
    help="Indra-Advanced: Lightweight High-Throughput Table Structure & Zero-Shot Schema Extraction CLI",
    add_completion=False,
)
console = Console()


@app.command()
def extract(
    image_path: Path = typer.Argument(..., help="Path to document image (PNG/JPG) or PDF"),
    schema_type: str = typer.Option(
        "invoice",
        "--schema",
        "-s",
        help="Schema to extract: 'invoice', 'financial', 'medical', or path to custom schema JSON",
    ),
    ocr_tier: str = typer.Option(
        "small",
        "--ocr-tier",
        help="PP-OCRv4 tier: 'tiny' (~1.5M), 'small' (~7.7M), 'medium' (~34.5M)",
    ),
    gliner_model: str = typer.Option(
        "gliner_2_5_small",
        "--gliner",
        help="GLiNER variant: 'gliner_2_5_small' (74M boundary), 'gliner_bi_edge' (60M), 'gliner_bi_small' (108M)",
    ),
    mode: str = typer.Option(
        "header_value",
        "--mode",
        "-m",
        help="Serialization mode: 'header_value', 'markdown_table', 'coordinate_annotated', 'hierarchical'",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Directory to save output JSON and Markdown",
    ),
):
    """Run full 4-stage document extraction on a file."""
    if not image_path.exists():
        console.print(f"[bold red]Error: File not found: {image_path}[/bold red]")
        raise typer.Exit(code=1)

    # Resolve schema
    selected_schema = None
    if schema_type == "invoice":
        selected_schema = INVOICE_SCHEMA
    elif schema_type in ("financial", "finance"):
        selected_schema = FINANCIAL_SCHEMA
    elif schema_type in ("medical", "lab"):
        selected_schema = MEDICAL_LAB_SCHEMA
    elif Path(schema_type).exists():
        with open(schema_type, "r", encoding="utf-8") as f:
            custom_data = json.load(f)
            from indra_advanced.schemas.base_schema import ExtractionSchema
            selected_schema = ExtractionSchema(
                name=custom_data.get("name", "custom"),
                entity_labels=custom_data.get("entity_labels", list(custom_data.keys())),
            )
    else:
        # Treat as comma-separated labels
        labels = [l.strip() for l in schema_type.split(",") if l.strip()]
        from indra_advanced.schemas.base_schema import ExtractionSchema
        selected_schema = ExtractionSchema.from_labels("user_schema", labels)

    # Configure pipeline
    cfg = PipelineConfig()
    try:
        cfg.stage2.tier = OCRModelTier(ocr_tier.lower())
    except ValueError:
        pass

    try:
        cfg.stage4.model_variant = GLiNERModelVariant(gliner_model.lower())
    except ValueError:
        pass

    try:
        cfg.stage3.primary_mode = SerializationMode(mode.lower())
    except ValueError:
        pass

    console.print(Panel(
        f"[bold cyan]Indra-Advanced Pipeline[/bold cyan]\n"
        f"Input: [green]{image_path}[/green]\n"
        f"Schema: [yellow]{selected_schema.name}[/yellow]\n"
        f"Footprint: [magenta]{cfg.total_estimated_parameters}[/magenta]",
        title="Processing Document",
    ))

    pipeline = IndraAdvancedPipeline(cfg)
    result = pipeline.process(image_path, schema=selected_schema)

    # Display timings
    timing_table = Table(title="Execution Performance Breakdown")
    timing_table.add_column("Pipeline Stage", style="cyan")
    timing_table.add_column("Latency (ms)", justify="right", style="green")
    for k, v in result.timing_breakdown_ms.items():
        timing_table.add_row(k.replace("_", " ").title(), f"{v:.2f} ms")
    timing_table.add_row("[bold]Total Latency[/bold]", f"[bold]{result.total_elapsed_ms:.2f} ms[/bold]")
    console.print(timing_table)

    # Display Serialized Text
    console.print("\n[bold yellow]Serialized Structured Text (Stage 3):[/bold yellow]")
    console.print(Panel(result.serialized_text or "(No text extracted)", border_style="yellow"))

    # Display Extracted Schema JSON
    console.print("\n[bold green]Extracted Schema JSON (Stage 4):[/bold green]")
    formatted_json = json.dumps(result.structured_json, indent=2)
    console.print(Syntax(formatted_json, "json", theme="monokai", line_numbers=True))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{image_path.stem}_extracted.json"
        md_path = output_dir / f"{image_path.stem}_table.md"
        result.save_json(json_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.markdown_table)
        console.print(f"\n[bold green]Outputs saved to {output_dir}:[/bold green]")
        console.print(f"  • JSON: {json_path}")
        console.print(f"  • Markdown: {md_path}")


@app.command()
def demo():
    """Run an interactive self-contained demo using a synthetic document invoice."""
    console.print("[bold cyan]Generating synthetic invoice document...[/bold cyan]")
    img_arr, expected_md = SyntheticTableGenerator.create_invoice_table_image()
    
    demo_img_path = Path("demo_invoice.png")
    from PIL import Image
    Image.fromarray(img_arr).save(demo_img_path)
    console.print(f"Saved demo invoice to [green]{demo_img_path}[/green]")

    extract(
        image_path=demo_img_path,
        schema_type="invoice",
        ocr_tier="small",
        gliner_model="gliner_2_5_small",
        mode="header_value",
        output_dir=Path("demo_results"),
    )


def main():
    app()


if __name__ == "__main__":
    main()
