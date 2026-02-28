#!/usr/bin/env python
"""Run complete workflow: train models, generate visualizations, update presentation."""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print("=" * 60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        return False
    return True


def main():
    print("Starting complete workflow...")

    steps = [
        ("python train_model.py", "Training models"),
        ("python train_nn.py", "Training neural network"),
        ("python visualize_confusion_matrices.py", "Generating confusion matrices"),
        (
            "python visualize.py",
            "Generating visualizations (LDA, t-SNE, feature importance)",
        ),
        ("python visualize_pca.py", "Generating PCA visualization"),
    ]

    for cmd, desc in steps:
        if not run_command(cmd, desc):
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Compiling LaTeX presentation...")
    print("=" * 60)
    result = subprocess.run(
        "cd presentation && pdflatex -interaction=nonstopmode presentation.tex",
        shell=True,
    )

    if result.returncode == 0:
        print("\n✓ Workflow complete!")
        print("  - Models trained")
        print("  - Visualizations generated")
        print("  - Presentation compiled")
    else:
        print("\n✗ Presentation compilation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
