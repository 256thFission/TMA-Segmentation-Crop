#!/usr/bin/env python3
"""
Subprocess runner for a single-core VALIS alignment.

Usage:
  python -m myvalis.valis_subprocess --stain_path <path> --dapi_path <path> \
      --centroids_tsv <path> --workdir <dir> [--scale_factor 0.1] [--flip_axis y] \
      [--microns_per_pixel 0.5] [--weights indoor] [--keypoint_threshold 0.005] \
      [--match_threshold 0.2] [--save_overlays true] [--overlay_alpha 0.5] \
      [--result_json <file>]

This runner isolates native resources (e.g., JVM, OpenCV threads) per-core by
invoking VALIS in a fresh Python process. It writes a small JSON result file with:
  {
    "ok": true/false,
    "output_tsv_path": "/abs/path/to/centroids_transformed.tsv",
    "error": "...optional error message...",
    "rss_mb": <current RSS MB if available>,
    "max_rss_mb": <peak RSS MB if available>
  }
"""

import os
import sys
import json
import argparse
from pathlib import Path
import threading
import time

# Conservative thread limits to reduce memory pressure
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

try:
    import resource  # POSIX
except Exception:
    resource = None

# Import after env is set
from myvalis.valis_alignment_clean import AlignmentConfig, run_alignment_pipeline  # noqa: E402


def _to_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):  # common truthy tokens
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _rss_mb_pair():
    cur = -1.0
    mx = -1.0
    try:
        if resource is not None:
            # ru_maxrss is kilobytes on Linux
            mx = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        mx = -1.0
    try:
        import psutil  # optional
        p = psutil.Process(os.getpid())
        cur = float(p.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        cur = -1.0
    return cur, mx


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run VALIS alignment for a single core in a subprocess.")
    parser.add_argument("--stain_path", required=True)
    parser.add_argument("--dapi_path", required=True)
    parser.add_argument("--centroids_tsv", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--scale_factor", type=float, default=0.1022)
    parser.add_argument("--flip_axis", type=str, default="y", choices=["none", "x", "y"]) 
    parser.add_argument("--microns_per_pixel", type=float, default=0.5072)
    # SuperGlue
    parser.add_argument("--weights", type=str, default="indoor")
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--match_threshold", type=float, default=0.2)
    parser.add_argument("--force_cpu", type=str, default="true")
    # Overlays
    parser.add_argument("--save_overlays", type=str, default="true")
    parser.add_argument("--overlay_alpha", type=float, default=0.5)
    # Output
    parser.add_argument("--result_json", type=str, default=None)
    # Heartbeat (seconds); if provided and > 0, emit periodic status to STDERR
    parser.add_argument("--heartbeat_s", type=float, default=None)

    args = parser.parse_args(argv)

    result = {
        "ok": False,
        "output_tsv_path": None,
        "error": None,
        "rss_mb": -1.0,
        "max_rss_mb": -1.0,
    }

    # Heartbeat setup
    stop_evt: "threading.Event" = threading.Event()
    hb_thread: "threading.Thread | None" = None

    def _heartbeat(period_s: float, stop_event: "threading.Event") -> None:
        """Periodically print memory stats to STDERR to indicate liveness."""
        try:
            while not stop_event.wait(period_s):
                cur, mx = _rss_mb_pair()
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[HB] {ts} rss={cur:.1f}MB max={mx:.1f}MB", file=sys.stderr, flush=True)
        except Exception:
            # Never fail the main task because of heartbeat issues
            pass

    try:
        # Build config object for the alignment
        cfg = AlignmentConfig()
        cfg.stain_path = Path(args.stain_path)
        cfg.dapi_path = Path(args.dapi_path)
        cfg.centroids_tsv = Path(args.centroids_tsv)
        cfg.workdir = Path(args.workdir)
        cfg.scale_factor = float(args.scale_factor)
        cfg.flip_axis = str(args.flip_axis)
        cfg.microns_per_pixel = float(args.microns_per_pixel)
        cfg.superglue_config.update({
            "weights": str(args.weights),
            "keypoint_threshold": float(args.keypoint_threshold),
            "match_threshold": float(args.match_threshold),
            "force_cpu": _to_bool(args.force_cpu, True),
        })
        cfg.save_overlays = _to_bool(args.save_overlays, True)
        cfg.overlay_alpha = float(args.overlay_alpha)

        # Ensure workdir exists
        cfg.workdir.mkdir(parents=True, exist_ok=True)

        # Start heartbeat if requested
        try:
            if args.heartbeat_s is not None and float(args.heartbeat_s) > 0:
                hb_thread = threading.Thread(target=_heartbeat, args=(float(args.heartbeat_s), stop_evt), daemon=True)
                hb_thread.start()
        except Exception:
            hb_thread = None

        # Run the pipeline
        results = run_alignment_pipeline(cfg)
        out_path = None
        try:
            out_path = results.get("output_tsv_path") if isinstance(results, dict) else None
        except Exception:
            out_path = None

        if out_path is not None:
            out_path = str(Path(out_path).resolve())
            result["output_tsv_path"] = out_path
            result["ok"] = True
        else:
            result["error"] = "Missing output_tsv_path in results"

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Stop heartbeat
        try:
            stop_evt.set()
            if hb_thread is not None:
                hb_thread.join(timeout=2.0)
        except Exception:
            pass

        cur, mx = _rss_mb_pair()
        result["rss_mb"] = cur
        result["max_rss_mb"] = mx

        # Write result JSON to file if requested; always also print to stdout
        try:
            payload = json.dumps(result)
            if args.result_json:
                try:
                    out_file = Path(args.result_json)
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_text(payload)
                except Exception:
                    pass
            # Print as the last line to make it easy to scrape from stdout if needed
            print(payload)
        except Exception:
            pass

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
