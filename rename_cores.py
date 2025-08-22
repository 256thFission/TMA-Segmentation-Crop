#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import List, Tuple

import typer

app = typer.Typer(help="Rename all image files in a directory using row-wise IDs from a CSV grid.")


DIGITS_RE = re.compile(r"^\d+$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_csv_grid(csv_path: Path) -> List[List[str]]:
    """
    Parse a CSV that may contain header row/col of numeric indices.
    Returns a 2D grid (list of rows) after removing header row/col and empty trailing columns.
    """
    rows: List[List[str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for raw in reader:
            # Normalize and keep as-is (including possible empty strings)
            rows.append([cell.strip() for cell in raw])

    # Drop completely empty rows
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        raise typer.BadParameter(f"CSV '{csv_path}' is empty.")

    # Determine if the first row is a header of pure digits (or first cell empty + rest digits)
    first = rows[0]
    def is_digits_or_empty(s: str) -> bool:
        return s == "" or bool(DIGITS_RE.match(s))

    header_row = all(is_digits_or_empty(c) for c in first) and any(c != "" for c in first)

    if header_row:
        rows = rows[1:]

    # Determine if first column is a header of pure digits/empty
    if rows and all(is_digits_or_empty(r[0]) for r in rows):
        rows = [r[1:] for r in rows]

    # Remove any residual empty columns at the end that are entirely empty
    if rows:
        # Trim trailing empty columns across all rows
        max_len = max(len(r) for r in rows)
        # Find rightmost column index that is not entirely empty
        rightmost_nonempty = -1
        for j in range(max_len):
            if any(j < len(r) and r[j].strip() != "" for r in rows):
                rightmost_nonempty = j
        if rightmost_nonempty >= 0:
            rows = [r[: rightmost_nonempty + 1] for r in rows]

    return rows


def flatten_rowwise(grid: List[List[str]]) -> List[str]:
    flat: List[str] = []
    for r in grid:
        for c in r:
            c2 = c.strip()
            if c2:
                flat.append(c2)
    return flat


def list_images_sorted(dir_path: Path) -> List[Path]:
    """List image files in dir sorted by numeric index found in filename, else by name."""
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    def key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else 10**9, p.name.lower())

    files.sort(key=key)
    return files


def make_dst_name(csv_stem: str, core_id: str, ext: str) -> str:
    # Preserve CSV case; ensure filesystem-safe name
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "", core_id)
    return f"{csv_stem}_{safe_id}{ext}"


@app.command()
def rename(
    csv_path: Path = typer.Option(..., exists=True, readable=True, help="Path to CSV grid with core IDs."),
    directory: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="Directory containing images to rename."),
    row_size: int = typer.Option(..., min=1, help="Number of cores per row in the CSV grid (for validation/warnings)."),
    apply: bool = typer.Option(False, "--apply", help="Apply changes. Default is dry-run."),
    overwrite: bool = typer.Option(False, help="Overwrite existing destination files if present."),
):
    """
    Rename files in 'directory' using row-wise IDs from csv_path.
    Output filenames: {csv_stem}_{ID}{ext}
    """
    csv_stem = csv_path.stem
    grid = parse_csv_grid(csv_path)

    # Validate row sizes
    row_lengths = [len(r) for r in grid]
    if any(L != row_size for L in row_lengths if L > 0):
        typer.secho(
            f"WARNING: Some CSV rows have length {set(row_lengths)} which differs from --row-size={row_size}.",
            fg=typer.colors.YELLOW,
        )

    core_ids = flatten_rowwise(grid)
    image_files = list_images_sorted(directory)

    n_ids = len(core_ids)
    n_imgs = len(image_files)

    if n_imgs == 0:
        typer.secho("No image files found in the given directory.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    if n_ids != n_imgs:
        typer.secho(
            f"WARNING: CSV IDs ({n_ids}) != images ({n_imgs}). Will map up to min count.",
            fg=typer.colors.YELLOW,
        )

    n = min(n_ids, n_imgs)

    moves: List[Tuple[Path, Path]] = []
    for i in range(n):
        src = image_files[i]
        dst_name = make_dst_name(csv_stem, core_ids[i], src.suffix)
        dst = src.with_name(dst_name)
        moves.append((src, dst))

    typer.echo(f"CSV: {csv_path}")
    typer.echo(f"Directory: {directory}")
    typer.echo(f"Planning to rename {n} files. Dry-run={not apply}")

    # Conflicts
    conflicts = []
    for src, dst in moves:
        if dst.exists() and not overwrite:
            conflicts.append((src, dst))

    if conflicts and not apply:
        typer.secho(
            f"NOTE: {len(conflicts)} destination files already exist. They would be skipped unless --overwrite.",
            fg=typer.colors.YELLOW,
        )

    # Print preview
    for src, dst in moves:
        mark = "!" if dst.exists() and not overwrite else "->"
        try:
            src_rel = src.relative_to(directory)
            dst_rel = dst.relative_to(directory)
        except Exception:
            src_rel = src.name
            dst_rel = dst.name
        typer.echo(f"{src_rel} {mark} {dst_rel}")

    if not apply:
        typer.secho("Dry-run complete. Use --apply to perform renames.", fg=typer.colors.GREEN)
        raise typer.Exit(code=0)

    # Execute
    skipped = 0
    done = 0
    for src, dst in moves:
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        try:
            os.replace(src, dst)
            done += 1
        except Exception as e:
            typer.secho(f"ERROR renaming {src} -> {dst}: {e}", fg=typer.colors.RED)

    typer.secho(f"Renamed {done} files. Skipped {skipped}.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
