"""
Avalanche Runout Envelope - Main Orchestrator

Author: Valerie Foley
Last Updated: 4/24/2026

Description:
    Top-level workflow for the avalanche runout envelope. Reads N release
    scenarios (each with its own snowpack model output), runs each through
    AvaFrame com1DFA, stacks the per-scenario reach masks into a probability
    raster, and answers: does the slide reach the road?
    
    Per-scenario inputs live in scenarios/scenario_NNN/inputs/.
    Shared inputs (DEM, forest, road) live in shared/.
    Per-scenario outputs go to scenarios/scenario_NNN/outputs/.
    Ensemble outputs go to outputs/.

Three Stages:
    Stage 1: Validation
        - Discover scenarios, validate CRS consistency across all inputs
    Stage 2: Per-scenario sims (parallel)
        - For each scenario: build AvaFrame dir, run com1DFA, post-process
    Stage 3: Ensemble stacking
        - Stack reach masks -> probability + envelope + contours + report

Outputs:
    - Per scenario: peak rasters (COG), reach mask, flowpath, summary.json
    - Ensemble:
        * probability.tif
        * envelope.geojson
        * contours.geojson
        * most_likely_path.geojson
        * road_impact_summary.json
        * scenarios_summary.csv
        * report.html
        * viewer.html

Usage:
    - Validate inputs only:    python avalanche_runout.py --dry-run
    - Run all scenarios:       python avalanche_runout.py
    - Re-run post only:        python avalanche_runout.py --post-only
    - Run one scenario:        python avalanche_runout.py --scenario scenario_001
    - Sequential (debug):      python avalanche_runout.py --no-parallel
    - Verbose logging:         python avalanche_runout.py --verbose

Requirements:
    - avaframe, rasterio, fiona, geopandas, shapely, pyproj, numpy, scipy,
      pandas, matplotlib, jinja2, folium, branca, scikit-image, pillow, pyyaml
"""

# --------- Load Libraries --------
import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import avalanche_helpers as h
import avalanche_simulation as sim
import avalanche_stack as stack
import build_report
import build_viewer


__version__ = '0.1.0'




def setup_logging(verbose=False):
    # Configure logging for the workflow
    # @param verbose: If True, set logging to DEBUG level, otherwise INFO
    # @returns: None
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def load_config(config_path):
    # Load and validate the YAML config
    # @param config_path: Path to config.yaml
    # @returns: dict of configuration parameters
    
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        # throw error if no config file found
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required = ['paths', 'com1dfa', 'entrainment', 'reach',
                'probability', 'road', 'execution', 'output']
    for key in required:
        if key not in config:
            # throw error if required config section is missing
            raise ValueError(f"Missing required config section: {key}")
    
    logging.info(f"Loaded config: {config_path}")
    return config


def parse_args():
    # Parse command-line arguments
    # @param: None
    # @returns: argparse.Namespace with parsed arguments
    #     Arguments:
    #       - config: Path to config YAML
    #       - dry_run: Validate only, don't run sims
    #       - post_only: Skip sims, redo post-processing
    #       - scenario: Run only specified scenarios (can give multiple)
    #       - no_parallel: Force sequential execution
    #       - verbose: DEBUG-level logging
    
    parser = argparse.ArgumentParser(
        description='Avalanche Runout Envelope Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs only; do not run simulations')
    parser.add_argument('--post-only', action='store_true', help='Skip sims; re-run post-processing on existing outputs')
    parser.add_argument('--scenario', action='append', default=None, help='Run only this scenario (can give multiple times)')
    parser.add_argument('--no-parallel', action='store_true', help='Force sequential execution (easier debugging)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable DEBUG-level logging')
    
    return parser.parse_args()


def load_scenario(scenario_dir):
    # Load a single scenario's input files into a dict
    # @param scenario_dir: Path to scenarios/scenario_NNN/
    # @returns: dict with all per-scenario data + paths
    # note:
    #   - Required files: release.geojson, depth.asc/.tif, density.json, params.json
    #   - Optional files: metadata.json, snow_depth.tif (future)
    #   - Files can live in scenario_dir/inputs/ or directly in scenario_dir/
    #     (whichever the upstream model wrote)
    
    sid = scenario_dir.name
    
    # Auto-detect input directory: prefer inputs/ subfolder if present,
    # otherwise treat the scenario folder itself as the input directory
    if (scenario_dir / 'inputs').is_dir():
        inp = scenario_dir / 'inputs'
    else:
        inp = scenario_dir
    
    # Required: release polygon
    release_gj = inp / 'release.geojson'
    if not release_gj.exists():
        # throw error if no release polygon
        raise FileNotFoundError(f"{sid}: missing release.geojson")
    
    # Required: depth raster (.asc preferred, .tif accepted)
    depth_asc = inp / 'depth.asc'
    depth_tif = inp / 'depth.tif'
    depth = depth_asc if depth_asc.exists() else depth_tif
    if not depth.exists():
        # throw error if no depth raster
        raise FileNotFoundError(f"{sid}: missing depth.asc or depth.tif")
    
    # Required: density.json
    density_path = inp / 'density.json'
    if not density_path.exists():
        # throw error if no density file
        raise FileNotFoundError(f"{sid}: missing density.json")
    density = json.loads(density_path.read_text())
    
    # Required: params.json
    params_path = inp / 'params.json'
    if not params_path.exists():
        # throw error if no params file
        raise FileNotFoundError(f"{sid}: missing params.json")
    params = json.loads(params_path.read_text())
    
    # Validate required param keys
    for required_key in ('mu', 'xi'):
        if required_key not in params:
            # throw error if required friction param missing
            raise ValueError(
                f"{sid}: params.json missing required key '{required_key}'"
            )
    
    # Optional: metadata
    meta_path = inp / 'metadata.json'
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    
    # Optional: snow depth raster (for future entrainment upgrades)
    snow_depth = inp / 'snow_depth.tif'
    snow_depth = snow_depth if snow_depth.exists() else None
    
    return {
        'scenario_id': sid,
        'scenario_dir': scenario_dir,
        'release_geojson': release_gj,
        'depth_raster': depth,
        'density': density,
        'params': params,
        'metadata': metadata,
        'snow_depth_raster': snow_depth
    }


def discover_scenarios(scenarios_dir):
    # Find all scenario folders and load their inputs
    # @param scenarios_dir: Path to scenarios/ directory
    # @returns: list of scenario dicts (one per scenario)
    # note:
    #   - Looks for subfolders matching scenario_*
    #   - Each must have a release.geojson (in inputs/ or scenario folder)
    
    scenarios_dir = Path(scenarios_dir)
    if not scenarios_dir.exists():
        # throw error if scenarios dir is missing
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")
    
    scenarios = []
    for sub in sorted(scenarios_dir.iterdir()):
        # Skip non-scenario folders
        if not sub.is_dir() or not sub.name.startswith('scenario_'):
            continue
        
        # Determine input dir (inputs/ subfolder OR scenario folder itself)
        # Skip if neither layout has a release.geojson
        inp = sub / 'inputs' if (sub / 'inputs').is_dir() else sub
        if not (inp / 'release.geojson').exists():
            logging.warning(f"Skipping {sub.name}: no release.geojson in {inp}")
            continue
        
        scenarios.append(load_scenario(sub))
    
    if not scenarios:
        # throw error if no scenarios found at all
        raise RuntimeError(f"No valid scenarios found in {scenarios_dir}")
    
    logging.info(f"Discovered {len(scenarios)} scenario(s)")
    return scenarios


def validate_all_crs(scenarios, shared_paths):
    # Ensure all inputs share the same CRS (anchored on the DEM)
    # @param scenarios: list of scenario dicts
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @returns: rasterio CRS object (the reference CRS)
    # note:
    #   - DEM is treated as authoritative
    #   - Aborts on any mismatch (per config flag in main)
    
    # Reference CRS from DEM
    dem_crs = h.get_crs_of(shared_paths['dem'])
    if dem_crs is None:
        # throw error if DEM has no CRS
        raise ValueError(
            f"DEM {shared_paths['dem']} has no CRS. "
            f"Assign one (e.g., gdal_translate -a_srs EPSG:XXXX) and retry."
        )
    
    logging.info(f"Reference CRS (from DEM): {dem_crs}")
    
    # Build (path, label) list for everything that needs to match
    paths_and_labels = [
        (shared_paths['forest'], 'forest'),
        (shared_paths['road'], 'road'),
    ]
    for sc in scenarios:
        paths_and_labels.append(
            (sc['release_geojson'], f"{sc['scenario_id']}/release")
        )
        paths_and_labels.append(
            (sc['depth_raster'], f"{sc['scenario_id']}/depth")
        )
    
    # Validate
    success, mismatches = h.validate_crs(paths_and_labels, dem_crs)
    
    if not success:
        # throw error on any CRS mismatch
        msg = (
            "CRS validation failed. All inputs must share the DEM's CRS.\n"
            + "\n".join(f"  - {m}" for m in mismatches)
            + "\nFix with gdalwarp/ogr2ogr before re-running."
        )
        raise ValueError(msg)
    
    logging.info(f"CRS validation passed (all inputs in {dem_crs})")
    return dem_crs


def run_scenarios_parallel(scenarios, shared_paths, config, skip_sim, max_workers):
    # Run scenarios in parallel via ProcessPoolExecutor
    # @param scenarios: list of scenario dicts
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param config: top-level config dict
    # @param skip_sim: bool - skip com1DFA call (post-only mode)
    # @param max_workers: int or None - process pool size
    # @returns: list of summary dicts (one per successful scenario)
    
    summaries = []
    
    logging.info(
        f"Running {len(scenarios)} scenarios in parallel "
        f"(max_workers={max_workers or 'auto'})"
    )
    
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Submit all scenarios
        futures = {
            ex.submit(sim.run_one_scenario, sc, shared_paths, config, skip_sim):
                sc['scenario_id']
            for sc in scenarios
        }
        
        # Collect as they finish
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                summaries.append(fut.result())
            except Exception as e:
                logging.error(f"[{sid}] pipeline failed: {e}", exc_info=True)
    
    return summaries


def run_scenarios_sequential(scenarios, shared_paths, config, skip_sim):
    # Run scenarios one at a time (easier to debug)
    # @param scenarios: list of scenario dicts
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param config: top-level config dict
    # @param skip_sim: bool - skip com1DFA call (post-only mode)
    # @returns: list of summary dicts (one per successful scenario)
    
    summaries = []
    for sc in scenarios:
        try:
            summaries.append(
                sim.run_one_scenario(sc, shared_paths, config, skip_sim)
            )
        except Exception as e:
            logging.error(f"[{sc['scenario_id']}] pipeline failed: {e}", exc_info=True)
    
    return summaries


def main():
    # Main workflow function
    # @param: None (args from CLI, params from config.yaml)
    # @returns: None (exits with status code)
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Discover scenarios
        scenarios = discover_scenarios(config['paths']['scenarios_dir'])
        
        # Optional filter to specific scenarios
        if args.scenario:
            requested = set(args.scenario)
            scenarios = [s for s in scenarios if s['scenario_id'] in requested]
            if not scenarios:
                # throw error if filter excluded everything
                raise ValueError(f"No scenarios match: {args.scenario}")
        
        # Validate shared inputs exist
        shared_paths = {
            'dem': Path(config['paths']['dem']),
            'forest': Path(config['paths']['forest']),
            'road': Path(config['paths']['road'])
        }
        for label, path in shared_paths.items():
            if not path.exists():
                # throw error if shared input is missing
                raise FileNotFoundError(f"Missing shared {label}: {path}")
        
        # Validate CRS consistency across all inputs
        if config['execution']['abort_on_crs_mismatch']:
            validate_all_crs(scenarios, shared_paths)
        
        # Stop here if --dry-run
        if args.dry_run:
            logging.info(f"Dry run complete. {len(scenarios)} scenarios validated.")
            return
        
        # Stage 2: per-scenario sims (parallel or sequential)
        use_parallel = (
            config['execution']['parallel']
            and not args.no_parallel
            and len(scenarios) > 1
        )
        
        if use_parallel:
            summaries = run_scenarios_parallel(
                scenarios=scenarios,
                shared_paths=shared_paths,
                config=config,
                skip_sim=args.post_only,
                max_workers=config['execution'].get('max_workers')
            )
        else:
            logging.info(f"Running {len(scenarios)} scenarios sequentially")
            summaries = run_scenarios_sequential(
                scenarios=scenarios,
                shared_paths=shared_paths,
                config=config,
                skip_sim=args.post_only
            )
        
        if not summaries:
            # throw error if everything failed
            raise RuntimeError(
                "No scenarios completed successfully; aborting ensemble stage"
            )
        
        # Stage 3: ensemble stacking
        logging.info("Building ensemble outputs")
        ensemble_paths = stack.run_ensemble_stage(
            scenario_summaries=summaries,
            scenarios=scenarios,
            shared_paths=shared_paths,
            config=config
        )
        
        # Stage 4: reports
        logging.info("Building HTML report")
        build_report.build_report(
            outputs_dir=Path(config['paths']['outputs_dir']),
            scenario_summaries=summaries,
            road_impact=ensemble_paths['road_summary_dict'],
            shared_paths=shared_paths,
            config=config,
            wrapper_version=__version__
        )
        
        logging.info("Building Leaflet viewer")
        build_viewer.build_viewer(
            outputs_dir=Path(config['paths']['outputs_dir']),
            scenarios=scenarios,
            shared_paths=shared_paths,
            impact_points=ensemble_paths['road_summary_dict']['impact_points'],
            config=config
        )
        
        # woohoo success
        logging.info("\n" + "="*70)
        logging.info(f" Workflow Complete - {len(summaries)}/{len(scenarios)} scenarios")
        logging.info("="*70)
    
    except Exception as e:
        # Error handling
        logging.error(f"\nWorkflow Failed: {e}")
        
        # Print full traceback if verbose
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
