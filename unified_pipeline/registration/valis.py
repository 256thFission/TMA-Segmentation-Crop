"""VALIS registration engine for per-core alignment."""

import json
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Optional
import gc
import os
import subprocess
import sys
import shlex
import time
try:
    import resource  # POSIX memory usage
except Exception:
    resource = None
try:
    import torch  # optional, for emptying CUDA cache if used elsewhere
except Exception:
    torch = None
try:
    import psutil  # optional, for current RSS
except Exception:
    psutil = None

# Import VALIS components (kept for compatibility, but subprocess runner is used)
try:
    from myvalis.valis_alignment_clean import (
        AlignmentConfig, 
        run_alignment_pipeline
    )
except ImportError:
    AlignmentConfig = None
    run_alignment_pipeline = None


class ValisEngine:
    """Handles VALIS registration for core pairs"""
    
    def __init__(self, config, utils):
        self.config = config
        self.utils = utils
        self.valis_workdir = config.get_output_dir('valis_workdir')
        
        # Ensure VALIS working directory exists
        self.valis_workdir.mkdir(parents=True, exist_ok=True)
    
    def run_per_core_valis(self, dapi_cores: List, stain_cores: List, preprocessed_centroids: pd.DataFrame,
                           dapi_report_path: str, stain_report_path: str,
                           dapi_grid: List[List[str]], stain_grid: List[List[str]]) -> List[Path]:
        """
        Run VALIS registration for each core pair
        
        Args:
            dapi_cores: List of DAPI core coordinates
            stain_cores: List of stain core coordinates  
            preprocessed_centroids: DataFrame with preprocessed centroids
            dapi_report_path: Path to DAPI detection report
            stain_report_path: Path to stain detection report
            dapi_grid: DAPI core naming grid
            stain_grid: Stain core naming grid
            
        Returns:
            List of paths to per-core transformed centroid files
        """
        # Subprocess runner will be used regardless of in-process availability
        
        self.utils.log("Starting per-core VALIS registration...")
        
        # Load detection reports to get core crop paths
        with open(dapi_report_path, 'r') as f:
            dapi_report = json.load(f)
        with open(stain_report_path, 'r') as f:
            stain_report = json.load(f)
        
        transformed_files: List[Path] = []

        def _rss_mb() -> float:
            if resource is None:
                return -1.0
            try:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                # ru_maxrss is kilobytes on Linux
                return float(ru.ru_maxrss) / 1024.0
            except Exception:
                return -1.0
        
        def _mem_usage_mb() -> tuple:
            """Return (current_rss_mb, max_rss_mb). -1.0 if unavailable."""
            cur = -1.0
            mx = -1.0
            try:
                if psutil is not None:
                    p = psutil.Process(os.getpid())
                    cur = float(p.memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                cur = -1.0
            try:
                mx = _rss_mb()
            except Exception:
                mx = -1.0
            return cur, mx
        
        # Match cores between modalities (same order from row-wise sorting)
        min_cores = min(len(dapi_cores), len(stain_cores))
        # Optional quick-test limit from config
        max_cores_cfg = self.config.get('global.max_cores', None)
        if isinstance(max_cores_cfg, int) and max_cores_cfg > 0:
            min_cores = min(min_cores, max_cores_cfg)
        self.utils.log(f"Processing {min_cores} core pairs (DAPI: {len(dapi_cores)}, Stain: {len(stain_cores)})")
        
        for i in range(min_cores):
            # Get canonical core names from grids
            dapi_core_name = self.utils.get_canonical_core_name(i, dapi_grid)
            stain_core_name = self.utils.get_canonical_core_name(i, stain_grid)
            
            self.utils.log(f"Processing core pair {i+1}/{min_cores}: DAPI={dapi_core_name}, Stain={stain_core_name}")
            
            # Create per-core working directory using DAPI name (primary reference)
            core_workdir = self.valis_workdir / f"core_{dapi_core_name}"
            core_workdir.mkdir(parents=True, exist_ok=True)
            
            # Get core crop image paths using canonical names (robust to report format)
            dapi_crop_path = self.utils.resolve_crop_path_from_report(dapi_report, dapi_core_name)
            stain_crop_path = self.utils.resolve_crop_path_from_report(stain_report, stain_core_name)

            if dapi_crop_path is None or stain_crop_path is None:
                self.utils.log(f"Missing crop images for DAPI={dapi_core_name}, Stain={stain_core_name}, skipping", "warning")
                continue
            
            if not dapi_crop_path.exists() or not stain_crop_path.exists():
                self.utils.log(f"Crop images not found for DAPI={dapi_core_name}, Stain={stain_core_name}, skipping", "warning")
                continue
            
            # Extract per-core centroids using spatial filtering
            dapi_x, dapi_y, dapi_r = dapi_cores[i]
            core_centroids = self.utils.extract_core_centroids(
                preprocessed_centroids, dapi_x, dapi_y, dapi_r * 1.2  # 20% padding
            )
            
            if len(core_centroids) == 0:
                self.utils.log(f"No centroids found for {dapi_core_name}, skipping", "warning")
                continue
            
            # Save per-core centroids
            core_centroids_path = core_workdir / f"{dapi_core_name}_centroids.tsv"
            core_centroids.to_csv(core_centroids_path, sep='\t', index=False)
            # Capture count for later logging, then free per-core DataFrame before heavy VALIS step
            try:
                n_core_centroids = int(len(core_centroids))
            except Exception:
                n_core_centroids = -1
            try:
                del core_centroids
            except Exception:
                pass
            gc.collect()
            
            # Run VALIS alignment for this core pair
            try:
                output_path = core_workdir / f"{dapi_core_name}_transformed.tsv"
                cur_mb, max_mb = _mem_usage_mb()
                if cur_mb >= 0 or max_mb >= 0:
                    self.utils.log(f"[MEM] Before core {dapi_core_name}: rss={cur_mb:.1f} MB, max={max_mb:.1f} MB", "debug")
                
                # Build subprocess command to isolate VALIS run
                project_root = Path(__file__).resolve().parents[2]
                result_json_path = core_workdir / 'valis_result.json'
                stdout_log_path = core_workdir / 'valis_stdout.log'
                stderr_log_path = core_workdir / 'valis_stderr.log'
                cmd_txt_path = core_workdir / 'command.txt'
                scale_factor = self.config.get('valis.preprocessing.scale_factor', 0.1022)
                flip_axis = self.config.get('global.flip_axis', 'y')
                um_per_px = self.config.get('global.microns_per_pixel', 0.5072)
                weights = self.config.get('valis.superglue.weights', 'indoor')
                kp_thresh = self.config.get('valis.superglue.keypoint_threshold', 0.005)
                match_thresh = self.config.get('valis.superglue.match_threshold', 0.2)
                save_overlays = self.config.get('valis.save_overlays', True)
                overlay_alpha = self.config.get('valis.overlay_alpha', 0.5)
                timeout_s = self.config.get('valis.registration.timeout_s', None)
                retry_on_to = bool(self.config.get('valis.registration.retry_on_timeout_cpu', True))
                heartbeat_s = self.config.get('valis.registration.heartbeat_s', None)

                base_cmd = [
                    sys.executable, '-m', 'myvalis.valis_subprocess',
                    '--stain_path', str(stain_crop_path),
                    '--dapi_path', str(dapi_crop_path),
                    '--centroids_tsv', str(core_centroids_path),
                    '--workdir', str(core_workdir),
                    '--scale_factor', str(scale_factor),
                    '--flip_axis', str(flip_axis),
                    '--microns_per_pixel', str(um_per_px),
                    '--weights', str(weights),
                    '--keypoint_threshold', str(kp_thresh),
                    '--match_threshold', str(match_thresh),
                    '--force_cpu', 'true',
                    '--save_overlays', 'true' if save_overlays else 'false',
                    '--overlay_alpha', str(overlay_alpha),
                    '--result_json', str(result_json_path),
                ]

                # Optional heartbeat
                try:
                    if isinstance(heartbeat_s, (int, float)) and heartbeat_s > 0:
                        base_cmd.extend(['--heartbeat_s', str(heartbeat_s)])
                except Exception:
                    pass

                # Persist invoked command for reproducibility
                try:
                    cmd_str = ' '.join(shlex.quote(s) for s in base_cmd)
                    cmd_txt_path.write_text(cmd_str + "\n")
                except Exception:
                    pass

                env = os.environ.copy()
                try:
                    env['PYTHONPATH'] = str(project_root) + (os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env and env['PYTHONPATH'] else '')
                except Exception:
                    pass

                def _run_once(cmd, attempt: int):
                    # Append headers and stream outputs to log files
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        with open(stdout_log_path, 'a') as out_fh, open(stderr_log_path, 'a') as err_fh:
                            out_fh.write(f"=== Attempt {attempt} start: {ts} ===\n")
                            out_fh.flush()
                            err_fh.write(f"=== Attempt {attempt} start: {ts} ===\n")
                            err_fh.flush()
                            return subprocess.run(
                                cmd,
                                cwd=str(project_root),
                                env=env,
                                stdout=out_fh,
                                stderr=err_fh,
                                text=True,
                                timeout=timeout_s if isinstance(timeout_s, (int, float)) and timeout_s > 0 else None,
                            )
                    except Exception as e:
                        # Also record exceptions to stderr log
                        try:
                            with open(stderr_log_path, 'a') as err_fh:
                                err_fh.write(f"[parent] Exception launching subprocess: {e}\n")
                        except Exception:
                            pass
                        raise

                completed = None
                try:
                    completed = _run_once(base_cmd, attempt=1)
                except subprocess.TimeoutExpired:
                    self.utils.log(f"VALIS subprocess timed out for {dapi_core_name}", "warning")
                    if retry_on_to:
                        try:
                            completed = _run_once(base_cmd, attempt=2)
                        except Exception as e:
                            self.utils.log(f"Retry failed for {dapi_core_name}: {e}", "error")
                            completed = None
                except Exception as e:
                    self.utils.log(f"VALIS subprocess failed to start for {dapi_core_name}: {e}", "error")

                # Parse result
                out_from_valis = None
                if result_json_path.exists():
                    try:
                        payload = json.loads(result_json_path.read_text())
                        if isinstance(payload, dict) and payload.get('ok'):
                            out_from_valis = payload.get('output_tsv_path')
                    except Exception:
                        pass
                if not out_from_valis:
                    # Try parsing last line of stdout log as JSON
                    try:
                        if stdout_log_path.exists():
                            lines = stdout_log_path.read_text().strip().splitlines()
                            if lines:
                                maybe_json = lines[-1]
                                payload2 = json.loads(maybe_json)
                                if isinstance(payload2, dict) and payload2.get('ok'):
                                    out_from_valis = payload2.get('output_tsv_path')
                    except Exception:
                        pass
                    # If still missing, log return code (if available) and tail of stderr log for debugging
                    try:
                        if completed is not None and hasattr(completed, 'returncode') and completed.returncode != 0:
                            self.utils.log(f"VALIS subprocess nonzero exit (rc={completed.returncode}) for {dapi_core_name}", "warning")
                    except Exception:
                        pass
                    try:
                        if stderr_log_path.exists():
                            err_tail = stderr_log_path.read_text().strip().splitlines()[-20:]
                            if err_tail:
                                self.utils.log("STDERR tail:\n" + "\n".join(err_tail), "debug")
                    except Exception:
                        pass

                # Copy/move output to canonical per-core path
                if out_from_valis and Path(out_from_valis).exists():
                    try:
                        shutil.copy2(out_from_valis, output_path)
                    except Exception:
                        try:
                            shutil.move(out_from_valis, output_path)
                        except Exception:
                            pass
                cur_mb2, max_mb2 = _mem_usage_mb()
                if cur_mb2 >= 0 or max_mb2 >= 0:
                    self.utils.log(f"[MEM] After core {dapi_core_name}: rss={cur_mb2:.1f} MB, max={max_mb2:.1f} MB", "debug")
                
                if output_path.exists():
                    transformed_files.append(output_path)
                    if n_core_centroids >= 0:
                        self.utils.log(f"Completed {dapi_core_name}: {n_core_centroids} centroids transformed", "success")
                    else:
                        self.utils.log(f"Completed {dapi_core_name}", "success")
                else:
                    self.utils.log(f"VALIS alignment failed for {dapi_core_name}", "warning")
                    
            except Exception as e:
                self.utils.log(f"Error processing {dapi_core_name}: {e}", "error")
                continue
        
        self.utils.log(f"VALIS registration completed: {len(transformed_files)} cores processed successfully")
        return transformed_files
    
    def aggregate_results(self, transformed_files: List[Path]) -> Path:
        """
        Aggregate per-core transformed centroids into final output file
        """
        self.utils.log("Aggregating transformed centroids...")
        
        final_output_path = self.config.get_output_dir('final_output')
        
        if not transformed_files:
            self.utils.log("No transformed files to aggregate", "warning")
            # Create empty output file
            pd.DataFrame().to_csv(final_output_path, sep='\t', index=False)
            return final_output_path
        
        # Combine all per-core results
        all_centroids = []
        for file_path in transformed_files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                all_centroids.append(df)
            except Exception as e:
                self.utils.log(f"Failed to read {file_path}: {e}", "error")
        
        if all_centroids:
            combined_df = pd.concat(all_centroids, ignore_index=True)
            combined_df.to_csv(final_output_path, sep='\t', index=False)
            self.utils.log(f"Aggregated {len(combined_df)} centroids to {final_output_path}", "success")
        else:
            self.utils.log("No valid transformed files found", "error")
            pd.DataFrame().to_csv(final_output_path, sep='\t', index=False)
        
        return final_output_path
