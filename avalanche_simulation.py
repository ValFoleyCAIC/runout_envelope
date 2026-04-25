"""
Avalanche Runout Envelope - Simulation Module

Author: Valerie Foley
Last Updated: 4/24/2026

Description:
    Per-scenario simulation pipeline. For one scenario:
        1. Materialize an AvaFrame-layout directory under scenarios/scenario_NNN/avaframe/
        2. Auto-detect entrainment area (delegated to avalanche_helpers)
        3. Save the steepest-descent flowline for the viewer
        4. Write the local com1DFA config (.ini) for this scenario
        5. Invoke AvaFrame's com1DFA directly via Python imports
        6. Locate peak rasters and copy them into scenario outputs/
        7. Build reach mask (flow depth >= threshold)
        8. Analyze road intersection
        9. Write summary.json
    
    No CLI here. Imported by avalanche_runout.py.

Requirements:
    - avaframe (separate install per their docs)
    - rasterio, fiona, geopandas, shapely, numpy
"""

# --------- Load Libraries --------
import os
import json
import time
import logging
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, Point

import avalanche_helpers as h




def materialize_ava_dir(scenario, shared_paths, config):
    # Build the AvaFrame Inputs/ folder for one scenario
    # @param scenario: dict with scenario_id, scenario_dir, release_geojson,
    #                  depth_raster, density, params (loaded from inputs/)
    # @param shared_paths: dict with 'dem', 'forest', 'road' Path objects
    # @param config: top-level config dict
    # @returns: tuple (ava_dir_path, entrainment_info_dict_or_None)
    # note:
    #   - Wipes any prior avaframe/ subdirectory before rebuilding
    #   - Layout: avaframe/Inputs/{DEM.tif, REL/, RELTH/, ENT/, ENTH/, RES/}
    #   - Also saves the per-scenario flowpath to outputs/flowpath.geojson
    #     for the viewer
    
    sid = scenario['scenario_id']
    ava_dir = Path(scenario['scenario_dir']) / 'avaframe'
    
    # Wipe and rebuild
    if ava_dir.exists():
        import shutil
        shutil.rmtree(ava_dir)
    inputs_dir = ava_dir / 'Inputs'
    for sub in ('REL', 'RELTH', 'RES'):
        (inputs_dir / sub).mkdir(parents=True)
    
    # Per-scenario outputs dir (created early so flowpath has somewhere to go)
    scenario_outputs = Path(scenario['scenario_dir']) / 'outputs'
    scenario_outputs.mkdir(exist_ok=True)
    
    # ---- DEM (shared across scenarios) ----
    # Convert to GeoTIFF rather than .asc. Two reasons:
    #   1. GeoTIFF stores CRS internally - no .prj sidecar needed.
    #   2. AvaFrame's searchRemeshedRaster globs Inputs/remeshedRasters/
    #      with an extension filter that catches .prj files too. When the
    #      DEM has a .prj, AvaFrame remeshes it (writing both .asc and .prj
    #      to the cache), then crashes trying to open the cached .prj as
    #      a raster. Using .tif sidesteps the bug entirely.
    h.convert_raster_format(shared_paths['dem'], inputs_dir / 'DEM.tif')
    
    # ---- Release polygon: GeoJSON -> Shapefile ----
    # AvaFrame validates that every release polygon feature has a 'thickness'
    # attribute, even when relThFromFile=True (raster). The value isn't used
    # by the physics in that mode - the raster overrides it - but the field
    # must exist to pass validation. Inject the scenario's mean depth as a
    # placeholder.
    rel_thickness = scenario['params'].get(
        'mean_depth_m',
        config['entrainment']['depth_m']
    )
    
    # Check for and log holes before fill_holes silently fixes them
    release_check = gpd.read_file(scenario['release_geojson'])
    n_holes = sum(
        len(g.interiors) for g in release_check.geometry
        if g is not None and g.geom_type == 'Polygon'
    )
    if n_holes > 0:
        logging.info(
            f"  Release polygon has {n_holes} hole(s) - filled for com1DFA. "
            "(AvaFrame rejects polygons with interior rings.)"
        )
    
    h.geojson_to_shapefile(
        scenario['release_geojson'],
        inputs_dir / 'REL' / 'release_01.shp',
        attributes={'thickness': float(rel_thickness)},
        fill_holes=True
    )
    
    # ---- Release thickness raster ----
    # AvaFrame requires DEM and RELTH to share an identical grid (lower-left
    # cell centers within 3 cells, same cell size). Ron's depth.asc is on a
    # different grid (different extent and possibly different cell size), so
    # resample it onto the DEM grid here. Cells outside the source extent
    # get 0.0 (no release thickness), which matches "outside the release
    # polygon" semantics.
    h.resample_to_reference_grid(
        src_path=scenario['depth_raster'],
        ref_path=inputs_dir / 'DEM.tif',
        dst_path=inputs_dir / 'RELTH' / 'release_01.asc',
        resampling_method='bilinear',
        fill_value=0.0
    )
    
    # ---- Forest as resistance area ----
    # Clip to DEM extent - com1DFA aborts if any resistance feature exceeds
    # the DEM bounds. Real-world forest layers often cover much wider areas
    # than a study-area DEM.
    h.geojson_to_shapefile(
        shared_paths['forest'],
        inputs_dir / 'RES' / 'forest.shp',
        clip_to_raster=inputs_dir / 'DEM.tif'
    )
    
    # ---- Auto-detect entrainment (only if sim_type='ent') ----
    # Always save the flowpath even if we're not running entrainment, so the
    # viewer can show the path for every scenario regardless of sim_type
    ent_info = None
    flowpath_out = scenario_outputs / 'flowpath.geojson'
    
    # Load the release polygon (single feature) once for downstream use
    release_gdf = gpd.read_file(scenario['release_geojson'])
    release_polygon = release_gdf.geometry.iloc[0]
    
    if config['com1dfa']['sim_type'] == 'ent':
        (inputs_dir / 'ENT').mkdir()
        (inputs_dir / 'ENTH').mkdir()
        
        ab_cfg = config['entrainment']['auto_detect']['alpha_beta']
        fa_cfg = config['entrainment']['auto_detect']['flow_accumulation']
        
        ent_info = h.auto_detect_entrainment(
            release_polygon=release_polygon,
            dem_path=inputs_dir / 'DEM.tif',
            polygon_out=inputs_dir / 'ENT' / 'entrainment.shp',
            thickness_out=inputs_dir / 'ENTH' / 'entrainment.asc',
            crs=h.get_crs_of(shared_paths['dem']),
            depth_m=config['entrainment']['depth_m'],
            alpha_min=ab_cfg['min_alpha_deg'],
            alpha_max=ab_cfg['max_alpha_deg'],
            fallback_buffer_m=fa_cfg['buffer_m'],
            flowline_out=flowpath_out
        )
    else:
        # No entrainment - still trace and save the flowline for the viewer
        flowline = h.steepest_descent_flowline(
            release_polygon, inputs_dir / 'DEM.tif'
        )
        flow_gdf = gpd.GeoDataFrame(
            {'name': ['flowpath']}, geometry=[flowline],
            crs=h.get_crs_of(shared_paths['dem'])
        )
        flow_gdf.to_file(flowpath_out, driver='GeoJSON')
    
    return ava_dir, ent_info


def write_com1dfa_config(scenario, config, out_path):
    # Write the per-scenario com1DFA config (.ini)
    # @param scenario: scenario dict (with params, density)
    # @param config: top-level config dict
    # @param out_path: Path for the .ini file
    # @returns: Path to written config
    # note:
    #   - Default friction model is Voellmy. Voellmy takes per-scenario
    #     mu and xi from params.json directly (via muvoellmy / xsivoellmy
    #     config keys). samosAT-family models use their own internal
    #     calibrated parameters (musamosat etc.) and ignore mu/xi.
    
    cfg_cfg = config['com1dfa']
    ent_cfg = config['entrainment']
    
    # Density: prefer density.json mean, fall back to params.json rho_kgm3
    rho = scenario['density'].get(
        'mean_kgm3', scenario['params'].get('rho_kgm3', 300.0)
    )
    
    # Build base GENERAL section (model-agnostic keys)
    # AvaFrame's config reader uses CASE-SENSITIVE camelCase keys
    # (e.g. cfg['GENERAL']['frictModel']). Python's default ConfigParser
    # lowercases all option names on write, which means AvaFrame can't
    # find our values and silently falls back to defaults. Subclass with
    # optionxform=str to preserve case.
    class _CaseSensitiveCfg(ConfigParser):
        optionxform = staticmethod(str)
    
    cp = _CaseSensitiveCfg()
    cp['GENERAL'] = {
        'simTypeList': cfg_cfg['sim_type'],
        'frictModel': cfg_cfg['friction_model'],
        'rho': str(rho),
        'meshCellSize': str(cfg_cfg['mesh_cell_size_m']),
        'stopCrit': str(cfg_cfg['stop_crit']),
        'resType': '|'.join(cfg_cfg['result_types']),
        # Use our own release thickness raster from RELTH/
        'relThFromShp': 'False',
        'relThFromFile': 'True',
        # Force AvaFrame to wipe Inputs/remeshedRasters/ before remeshing.
        # Defensive: prevents stale .prj sidecars from a previous run from
        # tripping the buggy searchRemeshedRaster glob (matches both .asc
        # and .prj in the cache directory).
        'cleanRemeshedRasters': 'True',
    }
    
    # Friction-model-specific parameters
    # Voellmy uses muvoellmy + xsivoellmy. samosAT-family uses internal
    # calibration (musamosat, tau0samosat, etc.) and won't read these.
    fm = cfg_cfg['friction_model'].lower()
    if fm == 'voellmy':
        cp['GENERAL']['muvoellmy'] = str(scenario['params']['mu'])
        cp['GENERAL']['xsivoellmy'] = str(scenario['params']['xi'])
    else:
        # For samosAT/samosATAuto/samosATSmall/samosATMedium, AvaFrame uses
        # its own calibrated parameters - log that we're not using ron's mu/xi
        logging.warning(
            f"  Friction model '{cfg_cfg['friction_model']}' uses internal "
            f"calibration; per-scenario mu/xi from params.json are NOT used."
        )
    
    # Entrainment-specific keys (only if sim_type='ent')
    if cfg_cfg['sim_type'] == 'ent':
        # Use polygon-attribute thickness (entThFromShp=True) instead of
        # raster (entThFromFile=True). For constant depth this is functionally
        # identical and follows AvaFrame's documented happy path. The raster
        # route triggers a bug where AvaFrame still tries to read the polygon
        # thickness attribute and crashes on empty config-key conversion.
        cp['GENERAL']['entThFromShp'] = 'True'
        cp['GENERAL']['entThFromFile'] = 'False'
        # Safety net: if the polygon thickness attribute is somehow missing,
        # fall back to this value (matches our config default)
        cp['GENERAL']['entThIfMissingInShp'] = str(ent_cfg['depth_m'])
        # Entrained snow density - use release density for v1
        cp['GENERAL']['rhoEnt'] = str(rho)
        cp['GENERAL']['entEroEnergy'] = str(ent_cfg['ent_ero_energy_jkg'])
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        cp.write(f)
    
    return out_path


def run_com1dfa(ava_dir, local_cfg_path):
    # Run AvaFrame com1DFA on a materialized scenario directory
    # @param ava_dir: Path to the scenario's avaframe/ directory
    # @param local_cfg_path: Path to the local_com1DFACfg.ini for this scenario
    # @returns: int - number of simulations completed (typically 1)
    # note:
    #   - Imports AvaFrame inline so the module loads even when AvaFrame
    #     isn't installed (helpful for unit testing post-processing alone)
    #   - When called inside a multiprocessing worker, each worker gets its
    #     own AvaFrame process state - no GIL contention
    #   - Pattern follows com6RockAvalanche / com3Hybrid: load the config
    #     via fileOverride, pass it to com1DFAMain as cfgInfo
    
    # Import inside the function so this module can be loaded without
    # AvaFrame present (used for testing / dry-run validation)
    from avaframe.in3Utils import cfgUtils
    from avaframe.com1DFA import com1DFA
    
    # Set up the AvaFrame "main" config to point at this scenario's avaDir
    cfg_main = cfgUtils.getGeneralConfig()
    cfg_main['MAIN']['avalancheDir'] = str(ava_dir)
    
    # Load com1DFA config with our per-scenario overrides as a configparser
    # object (not just a side-effect; we need to pass it to com1DFAMain)
    com1dfa_cfg = cfgUtils.getModuleConfig(
        com1DFA, fileOverride=str(local_cfg_path),
        modInfo=False, toPrint=False
    )
    
    # Run the simulation (cfgInfo = our config object)
    dem, plot_dict, report_dict, sim_df = com1DFA.com1DFAMain(
        cfg_main, cfgInfo=com1dfa_cfg
    )
    
    return len(sim_df)


def find_peak_rasters(ava_dir):
    # Locate com1DFA peak output rasters from the AvaFrame outputs
    # @param ava_dir: Path to the scenario's avaframe/ directory
    # @returns: dict keyed by result type ('ppr', 'pft', 'pfv', 'pta')
    #           with Path values; missing types are absent from dict
    # note:
    #   - AvaFrame typically writes to Outputs/com1DFA/peakFiles/, but
    #     newer versions may nest under a simHash subfolder. We do a
    #     recursive search and accept both .asc and .tif.
    #   - Filename patterns vary across AvaFrame versions:
    #       <simHash>_pft.asc
    #       <simHash>_<scenarioName>_<simType>_<frictModel>_pft.asc
    #     We just look for files containing the result type code.
    #   - We pick the most recent file per resType so a re-run with a
    #     new hash still resolves correctly.
    
    peak_dir = Path(ava_dir) / 'Outputs' / 'com1DFA' / 'peakFiles'
    if not peak_dir.exists():
        # throw error if AvaFrame produced no peakFiles dir
        raise FileNotFoundError(f"No peakFiles directory at {peak_dir}")
    
    # Collect all raster files anywhere under peak_dir (recursive)
    all_rasters = list(peak_dir.rglob('*.asc')) + list(peak_dir.rglob('*.tif'))
    
    if not all_rasters:
        # throw error if peakFiles dir exists but is empty
        raise FileNotFoundError(
            f"No raster files found under {peak_dir}. "
            "Check Outputs/com1DFA/ for the actual location of peak fields."
        )
    
    rasters = {}
    for res_type in ('ppr', 'pft', 'pfv', 'pta'):
        # Match final peak files only - filename must END with _<res_type>
        # (e.g. abc_pft.tif). This excludes time-step snapshots in
        # peakFiles/timeSteps/ that look like abc_pft_t0.00.tif.
        suffix = f'_{res_type}'
        matches = [p for p in all_rasters if p.stem.endswith(suffix)]
        if matches:
            # Sort by modification time, newest first
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            rasters[res_type] = matches[0]
            logging.debug(f"  Found {res_type}: {matches[0].name}")
        else:
            logging.warning(f"  No {res_type} peak raster found")
    
    return rasters


def restype_to_human_name(res_type):
    # Map com1DFA result type codes to human-readable names
    # @param res_type: 'ppr', 'pft', 'pfv', or 'pta'
    # @returns: str - readable name (or original code if unknown)
    
    return {
        'ppr': 'pressure',
        'pft': 'flow_depth',
        'pfv': 'velocity',
        'pta': 'travel_angle'
    }.get(res_type, res_type)


def build_reach_mask(pft_path, out_path, depth_threshold_m,
                     pfv_path=None, velocity_threshold_ms=0.0):
    # Build a binary reach mask from peak flow depth (and optionally velocity)
    # @param pft_path: Path to peak flow thickness raster
    # @param out_path: Path to write binary reach mask GeoTIFF
    # @param depth_threshold_m: minimum flow depth to count as "reached"
    # @param pfv_path: Optional path to peak velocity (only used if threshold>0)
    # @param velocity_threshold_ms: minimum velocity to count (0 = no constraint)
    # @returns: Path to written mask
    # note:
    #   - Output is uint8 (0/1) GeoTIFF aligned to pft grid
    #   - Reach criterion is flow depth alone unless velocity_threshold > 0
    
    depth, meta = h.read_raster(pft_path)
    reach = (depth >= depth_threshold_m)
    
    if velocity_threshold_ms > 0 and pfv_path is not None:
        vel, _ = h.read_raster(pfv_path)
        reach = reach & (vel >= velocity_threshold_ms)
    
    # Write as uint8 GeoTIFF (0/1 binary)
    profile = meta['profile']
    profile['nodata'] = 0
    return h.write_raster(
        out_path, reach.astype('uint8'), profile,
        driver='GTiff', cog=True
    )


def analyze_road_intersection(reach_mask_path, pft_path, pfv_path,
                              road_path, release_polygon_path,
                              road_buffer_m=5.0,
                              chainage_origin='far_from_release'):
    # Did the scenario reach the road? If yes, where, depth, velocity?
    # @param reach_mask_path: Path to binary reach mask
    # @param pft_path: Path to peak flow depth raster (for sampling at impact)
    # @param pfv_path: Path to peak velocity raster (for sampling at impact)
    # @param road_path: Path to road centerline polyline
    # @param release_polygon_path: Path to release polygon (for chainage origin)
    # @param road_buffer_m: buffer width around centerline = "road corridor"
    # @param chainage_origin: 'far_from_release', 'first_vertex', 'last_vertex'
    # @returns: dict with reached, intersection_xy, flow_depth_at_road_m,
    #           velocity_at_road_ms, distance_to_road_m, chainage_m
    # note:
    #   - "Distance to road" is positive if scenario fell short, 0 if reached
    #   - For multi-segment polylines, segments are concatenated (chainage
    #     order matters)
    
    # Load and merge road segments
    # Real road shapefiles often have multiple features (one per highway
    # segment) and may use MultiLineString geometry. Flatten everything
    # to a single LineString for chainage and buffering.
    road_gdf = gpd.read_file(road_path)
    road_lines = []
    for geom in road_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        # MultiLineString -> list of LineStrings
        if geom.geom_type == 'MultiLineString':
            road_lines.extend(list(geom.geoms))
        elif geom.geom_type == 'LineString':
            road_lines.append(geom)
        else:
            logging.warning(f"Skipping unexpected road geom type: {geom.geom_type}")
    
    if not road_lines:
        # throw error if road file has no usable geometry
        raise ValueError(f"Road file {road_path} has no usable LineString geometry")
    
    if len(road_lines) == 1:
        road = road_lines[0]
    else:
        # Concatenate all segment coords into a single polyline
        # Order roughly along the road by sorting on first vertex
        road_lines.sort(key=lambda g: (g.coords[0][0], g.coords[0][1]))
        all_coords = []
        for g in road_lines:
            all_coords.extend(list(g.coords))
        road = LineString(all_coords)
    
    # Buffer road -> "road corridor" polygon for the reached test
    corridor = road.buffer(road_buffer_m, cap_style=2)
    
    # Test if reach mask intersects corridor
    reach, meta = h.read_raster(reach_mask_path)
    transform = meta['transform']
    
    # Burn corridor into a mask aligned to reach raster, then intersect
    corridor_mask = h.polygon_to_raster_mask(
        corridor, reach.shape, transform, fill_value=1.0
    )
    overlap = reach.astype(bool) & corridor_mask.astype(bool)
    
    # Compute chainage origin point
    release_gdf = gpd.read_file(release_polygon_path)
    release_centroid = release_gdf.geometry.iloc[0].centroid
    
    if chainage_origin == 'far_from_release':
        # Whichever end of the polyline is farther from the release
        pt_first = Point(road.coords[0])
        pt_last = Point(road.coords[-1])
        if pt_first.distance(release_centroid) < pt_last.distance(release_centroid):
            origin_pt = pt_last
        else:
            origin_pt = pt_first
    elif chainage_origin == 'first_vertex':
        origin_pt = Point(road.coords[0])
    elif chainage_origin == 'last_vertex':
        origin_pt = Point(road.coords[-1])
    else:
        # Default to first vertex on unknown option
        origin_pt = Point(road.coords[0])
    
    # If origin is the last vertex, reverse the line so project() measures
    # from origin going along the road
    if origin_pt.equals(Point(road.coords[-1])):
        road_for_chainage = LineString(list(road.coords)[::-1])
    else:
        road_for_chainage = road
    
    # Case 1: Did NOT reach. Compute shortest distance to road.
    if not overlap.any():
        if reach.any():
            # Flow exists but didn't hit road - measure distance from any
            # reach cell to road centerline
            ys, xs = np.where(reach.astype(bool))
            wx, wy = h.pixels_to_world(ys, xs, transform)
            
            # Subsample for speed if huge
            if len(wx) > 5000:
                idx = np.linspace(0, len(wx) - 1, 5000, dtype=int)
                wx, wy = wx[idx], wy[idx]
            
            min_dist = min(
                road.distance(Point(x, y))
                for x, y in zip(wx, wy)
            )
        else:
            # No reach at all - distance from release centroid to road
            min_dist = float(release_centroid.distance(road))
        
        return {
            'reached': False,
            'intersection_xy': None,
            'flow_depth_at_road_m': None,
            'velocity_at_road_ms': None,
            'distance_to_road_m': float(min_dist),
            'chainage_m': None
        }
    
    # Case 2: Reached. Find first impact point + sample conditions.
    # First impact = overlap cell closest to the release
    ys, xs = np.where(overlap)
    wx, wy = h.pixels_to_world(ys, xs, transform)
    dists = np.hypot(wx - release_centroid.x, wy - release_centroid.y)
    first_idx = int(np.argmin(dists))
    impact_pt = Point(wx[first_idx], wy[first_idx])
    
    # Sample flow depth and velocity at the impact point
    depth_val = h.sample_raster_at_xy(pft_path, [(impact_pt.x, impact_pt.y)])[0]
    vel_val = h.sample_raster_at_xy(pfv_path, [(impact_pt.x, impact_pt.y)])[0]
    
    # Chainage along road
    chainage = road_for_chainage.project(impact_pt)
    
    return {
        'reached': True,
        'intersection_xy': [float(impact_pt.x), float(impact_pt.y)],
        'flow_depth_at_road_m': float(depth_val),
        'velocity_at_road_ms': float(vel_val),
        'distance_to_road_m': 0.0,
        'chainage_m': float(chainage)
    }


def build_scenario_summary(scenario, peak_rasters, reach_mask_path,
                           road_result, entrainment_info, runtime_s):
    # Aggregate per-scenario metrics into a single dict (-> JSON later)
    # @param scenario: scenario dict
    # @param peak_rasters: dict from find_peak_rasters
    # @param reach_mask_path: Path to reach mask raster
    # @param road_result: dict from analyze_road_intersection
    # @param entrainment_info: dict from auto_detect_entrainment (or None)
    # @param runtime_s: float - total scenario runtime in seconds
    # @returns: dict ready to JSON-dump
    
    # Peak max stats
    pft_data, pft_meta = h.read_raster(peak_rasters['pft'])
    pfv_data, _ = h.read_raster(peak_rasters['pfv'])
    ppr_data, _ = h.read_raster(peak_rasters['ppr'])
    
    cell_area = abs(pft_meta['transform'].a * pft_meta['transform'].e)
    
    # Reach mask + max runout
    reach, reach_meta = h.read_raster(reach_mask_path)
    reach_area = float(reach.sum()) * cell_area
    
    # Max runout = farthest reach cell from release centroid
    release_gdf = gpd.read_file(scenario['release_geojson'])
    centroid = release_gdf.geometry.iloc[0].centroid
    
    if reach.any():
        ys, xs = np.where(reach.astype(bool))
        wx, wy = h.pixels_to_world(ys, xs, reach_meta['transform'])
        dists = np.hypot(wx - centroid.x, wy - centroid.y)
        max_runout = float(dists.max())
    else:
        max_runout = 0.0
    
    return {
        'scenario_id': scenario['scenario_id'],
        'params': scenario['params'],
        'entrainment': entrainment_info or {},
        'runout_max_distance_m': max_runout,
        'reach_area_m2': reach_area,
        'peak_velocity_max_ms': float(np.nanmax(pfv_data)),
        'peak_flow_depth_max_m': float(np.nanmax(pft_data)),
        'peak_pressure_max_pa': float(np.nanmax(ppr_data)),
        'road': road_result,
        'runtime_s': runtime_s
    }


def run_one_scenario(scenario, shared_paths, config, skip_sim=False):
    # Run a single scenario through the full pipeline
    # @param scenario: scenario dict (from load_scenarios in main)
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param config: top-level config dict
    # @param skip_sim: If True, skip com1DFA call (use existing peak rasters)
    # @returns: dict - the scenario summary (also written to summary.json)
    # note:
    #   - Used both directly (sequential mode) and as multiprocessing worker
    
    sid = scenario['scenario_id']
    logging.info(f"[{sid}] starting")
    start_t = time.time()
    
    scenario_outputs = Path(scenario['scenario_dir']) / 'outputs'
    scenario_outputs.mkdir(exist_ok=True)
    
    # 1. Materialize AvaFrame directory + auto-detect entrainment + save flowpath
    ava_dir, ent_info = materialize_ava_dir(scenario, shared_paths, config)
    
    # 2. Write com1DFA config
    local_cfg = Path(scenario['scenario_dir']) / 'scenario.ini'
    write_com1dfa_config(scenario, config, local_cfg)
    
    # 3. Run com1DFA (or skip if --post-only)
    if not skip_sim:
        logging.info(f"[{sid}] running com1DFA")
        n_sims = run_com1dfa(ava_dir, local_cfg)
        logging.info(f"[{sid}] com1DFA finished ({n_sims} sim)")
    else:
        logging.info(f"[{sid}] skipping simulation (--post-only)")
    
    # 4. Find peak rasters and copy to scenario outputs as COGs
    peak_rasters = find_peak_rasters(ava_dir)
    for res_type, src_path in peak_rasters.items():
        dst = scenario_outputs / f"peak_{restype_to_human_name(res_type)}.tif"
        h.convert_raster_format(src_path, dst)
    
    # 5. Reach mask (flow depth >= threshold)
    reach_path = scenario_outputs / 'reach_mask.tif'
    build_reach_mask(
        pft_path=peak_rasters['pft'],
        out_path=reach_path,
        depth_threshold_m=config['reach']['flow_depth_threshold_m'],
        pfv_path=peak_rasters.get('pfv'),
        velocity_threshold_ms=config['reach']['min_velocity_ms']
    )
    
    # 6. Reach polygon (vector form of reach mask, for the viewer)
    reach_data, reach_meta = h.read_raster(reach_path)
    reach_polys = h.mask_to_polygons(reach_data, reach_meta['transform'])
    if reach_polys:
        reach_gdf = gpd.GeoDataFrame(
            {'scenario': [sid] * len(reach_polys)},
            geometry=reach_polys,
            crs=reach_meta['crs']
        )
        reach_gdf.to_file(
            scenario_outputs / 'reach_polygon.geojson', driver='GeoJSON'
        )
    
    # 7. Road intersection analysis
    road_result = analyze_road_intersection(
        reach_mask_path=reach_path,
        pft_path=peak_rasters['pft'],
        pfv_path=peak_rasters['pfv'],
        road_path=shared_paths['road'],
        release_polygon_path=scenario['release_geojson'],
        road_buffer_m=config['road']['buffer_m'],
        chainage_origin=config['road']['chainage_origin']
    )
    
    # 8. Build summary and write to summary.json
    summary = build_scenario_summary(
        scenario=scenario,
        peak_rasters=peak_rasters,
        reach_mask_path=reach_path,
        road_result=road_result,
        entrainment_info=ent_info,
        runtime_s=time.time() - start_t
    )
    with open(scenario_outputs / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(
        f"[{sid}] done in {summary['runtime_s']:.1f}s | "
        f"reached_road={road_result['reached']} | "
        f"runout={summary['runout_max_distance_m']:.1f} m"
    )
    return summary
