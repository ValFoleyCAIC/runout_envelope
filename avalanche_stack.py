"""
Avalanche Runout Envelope - Probability Stacking Module

Author: Valerie Foley
Last Updated: 4/24/2026

Description:
    After all per-scenario sims complete, stack their reach masks into
    ensemble-level outputs:
        - probability.tif: P(reach) = mean of N binary reach masks
        - envelope.geojson: outer boundary at P >= envelope_min_probability
        - contours.geojson: polygons at each contour level
        - road_impact_summary.json: headline N-of-M reached + impact points
        - scenarios_summary.csv: one row per scenario with params and outcomes
        - most_likely_path.geojson: medial-axis line through the high-P core
    
    No CLI. Imported by avalanche_runout.py.

References:
    - Bevilacqua et al. (2019), NHESS - probabilistic hazard map theory
    - Hungr (2014), Can. Geotech. J. - landslide runout exceedance practice
    - Zhang et al. (2024), Landslides - debris flow exceedance probabilities
    - AvaFrame ana4Stats.probAna - stacking convention for com1DFA

Requirements:
    - rasterio, geopandas, shapely, numpy, pandas, scikit-image (optional)
"""

# --------- Load Libraries --------
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import LineString

import avalanche_helpers as h




def stack_reach_masks(mask_paths, out_path):
    # Stack N binary reach masks into a probability raster (mean across stack)
    # @param mask_paths: list of Paths to reach mask GeoTIFFs
    # @param out_path: Path to write probability raster
    # @returns: Path to written raster
    # note:
    #   - All masks must be raster-aligned (same shape, transform, CRS).
    #     Since they come from com1DFA on the same DEM with the same cell
    #     size, this is satisfied for our pipeline.
    #   - Output is float32; nodata = -1.0
    
    if not mask_paths:
        # throw error if no masks supplied
        raise ValueError("No reach masks provided to stack")
    
    # Establish reference grid from first mask
    _, ref_meta = h.read_raster(mask_paths[0])
    ref_shape = ref_meta['shape']
    accum = np.zeros(ref_shape, dtype='float32')
    
    # Accumulate
    for p in mask_paths:
        data, meta = h.read_raster(p)
        if meta['shape'] != ref_shape:
            # throw error if grids don't align
            raise ValueError(
                f"Mask {p} shape {meta['shape']} != reference {ref_shape}. "
                "All scenario reach masks must share the same grid."
            )
        accum += data.astype('float32')
    
    # Mean -> probability
    probability = accum / len(mask_paths)
    
    # Write with -1 nodata (so true zeros stay distinct from "no data")
    profile = ref_meta['profile']
    profile['nodata'] = -1.0
    h.write_raster(out_path, probability, profile, driver='GTiff', cog=True)
    
    logging.info(f"Stacked {len(mask_paths)} masks -> {out_path}")
    return out_path


def polygonize_probability(probability_path, contour_levels,
                           envelope_path, contours_path,
                           envelope_threshold):
    # Polygonize the probability raster at each contour level
    # @param probability_path: Path to probability raster
    # @param contour_levels: list of P thresholds (e.g. [0.05, 0.25, 0.5, 0.75, 1.0])
    # @param envelope_path: Path to write envelope GeoJSON (single threshold)
    # @param contours_path: Path to write contours GeoJSON (multiple thresholds)
    # @param envelope_threshold: P-threshold for the outer envelope (e.g. 0.05)
    # @returns: None
    
    prob, meta = h.read_raster(probability_path)
    transform = meta['transform']
    crs = meta['crs']
    
    # Envelope: outer boundary above threshold
    envelope_mask = (prob >= envelope_threshold)
    env_polys = h.mask_to_polygons(envelope_mask, transform)
    
    env_gdf = gpd.GeoDataFrame(
        {'min_prob': [envelope_threshold] * len(env_polys)},
        geometry=env_polys,
        crs=crs
    )
    env_gdf.to_file(envelope_path, driver='GeoJSON')
    
    # Contours: one feature per (level, polygon)
    rows = []
    for level in contour_levels:
        mask = (prob >= level)
        polys = h.mask_to_polygons(mask, transform)
        for poly in polys:
            rows.append({'level': level, 'geometry': poly})
    
    if rows:
        contour_gdf = gpd.GeoDataFrame(rows, crs=crs)
    else:
        # Empty case - still write valid GeoJSON with empty features
        contour_gdf = gpd.GeoDataFrame({'level': []}, geometry=[], crs=crs)
    
    contour_gdf.to_file(contours_path, driver='GeoJSON')
    
    logging.info(
        f"Wrote envelope ({len(env_polys)} polys) and "
        f"contours ({len(rows)} features)"
    )


def extract_most_likely_path(probability_path, threshold, out_path):
    # Extract the medial axis of the P >= threshold region as a polyline
    # @param probability_path: Path to probability raster
    # @param threshold: P value defining the "high probability core"
    # @param out_path: Path to write GeoJSON polyline
    # @returns: Path or None (if scikit-image missing or region too small)
    # note:
    #   - "Most likely path" = skeleton of high-P region, NOT statistical
    #     mode of trajectories. Branching skeletons indicate multi-path
    #     terrain (which is correct behavior, not a bug).
    
    # Soft dependency on scikit-image
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        logging.warning("scikit-image not installed; skipping most_likely_path")
        return None
    
    prob, meta = h.read_raster(probability_path)
    transform = meta['transform']
    crs = meta['crs']
    
    mask = (prob >= threshold)
    if mask.sum() < 10:
        logging.warning(
            f"High-P region (P >= {threshold:.2f}) has < 10 cells; "
            "skipping most_likely_path"
        )
        return None
    
    # Skeletonize and convert to world coords
    skeleton = skeletonize(mask)
    ys, xs = np.where(skeleton)
    
    if len(xs) < 2:
        return None
    
    wx, wy = h.pixels_to_world(ys, xs, transform)
    
    # Order by scanline (top -> bottom, left -> right). Not a true graph
    # traversal but works for visualization. A proper skeleton-graph
    # walker would be a future upgrade.
    coords = sorted(zip(wx, wy), key=lambda p: (-p[1], p[0]))
    line = LineString(coords)
    
    gdf = gpd.GeoDataFrame(
        {'p_threshold': [threshold]}, geometry=[line], crs=crs
    )
    gdf.to_file(out_path, driver='GeoJSON')
    return out_path


def build_road_impact_summary(scenario_summaries, out_path):
    # Aggregate road-reach results across all scenarios
    # @param scenario_summaries: list of summary dicts (one per scenario)
    # @param out_path: Path to write JSON
    # @returns: dict (also written to disk)
    # note:
    #   - Wilson 95% CI on reach fraction handles small-n properly (n=20)
    #   - "Nearest miss" = scenario that fell shortest distance to road
    
    n_total = len(scenario_summaries)
    reached = [s for s in scenario_summaries if s['road']['reached']]
    n_reached = len(reached)
    frac = n_reached / n_total if n_total > 0 else 0.0
    ci_low, ci_high = h.wilson_ci(n_reached, n_total)
    
    # Impact points list
    impact_points = []
    for s in reached:
        impact_points.append({
            'scenario': s['scenario_id'],
            'xy': s['road']['intersection_xy'],
            'depth_m': s['road']['flow_depth_at_road_m'],
            'velocity_ms': s['road']['velocity_at_road_ms'],
            'chainage_m': s['road']['chainage_m']
        })
    
    # Nearest miss
    misses = [s for s in scenario_summaries if not s['road']['reached']]
    if misses:
        nearest = min(misses, key=lambda s: s['road']['distance_to_road_m'])
        nearest_miss = {
            'scenario': nearest['scenario_id'],
            'distance_short_m': nearest['road']['distance_to_road_m']
        }
    else:
        nearest_miss = None
    
    summary = {
        'n_scenarios': n_total,
        'n_reached': n_reached,
        'fraction_reached': round(frac, 4),
        'wilson_95_ci': [round(ci_low, 4), round(ci_high, 4)],
        'impact_points': impact_points,
        'nearest_miss': nearest_miss
    }
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def write_scenarios_csv(scenario_summaries, out_path):
    # Flatten scenario summaries into a CSV - one row per scenario
    # @param scenario_summaries: list of summary dicts
    # @param out_path: Path to write CSV
    # @returns: Path to written CSV
    # note:
    #   - Sorted by max runout descending (most extreme scenarios first)
    
    rows = []
    for s in scenario_summaries:
        # Extract optional road sub-fields safely
        depth_at_road = s['road']['flow_depth_at_road_m']
        vel_at_road = s['road']['velocity_at_road_ms']
        chainage = s['road']['chainage_m']
        
        rows.append({
            'scenario_id': s['scenario_id'],
            'mu': s['params'].get('mu'),
            'xi': s['params'].get('xi'),
            'rho_kgm3': s['params'].get('rho_kgm3'),
            'release_volume_m3': s['params'].get('total_volume_m3'),
            'runout_max_m': round(s['runout_max_distance_m'], 1),
            'peak_velocity_ms': round(s['peak_velocity_max_ms'], 2),
            'peak_flow_depth_m': round(s['peak_flow_depth_max_m'], 2),
            'peak_pressure_kpa': round(s['peak_pressure_max_pa'] / 1000, 2),
            'reach_area_m2': round(s['reach_area_m2'], 1),
            'reached_road': s['road']['reached'],
            'distance_to_road_m': round(s['road']['distance_to_road_m'], 1),
            'depth_at_road_m': round(depth_at_road, 2) if depth_at_road else None,
            'velocity_at_road_ms': round(vel_at_road, 2) if vel_at_road else None,
            'chainage_m': round(chainage, 1) if chainage else None,
            'entrainment_method': s['entrainment'].get('method_used'),
            'entrainment_area_m2': s['entrainment'].get('area_m2'),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('runout_max_m', ascending=False)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    return out_path


def run_ensemble_stage(scenario_summaries, scenarios, shared_paths, config):
    # Run the full ensemble post-processing (call after all sims complete)
    # @param scenario_summaries: list of dicts (one per completed scenario)
    # @param scenarios: list of scenario dicts (for paths back to inputs)
    # @param shared_paths: dict with 'dem', 'forest', 'road'
    # @param config: top-level config dict
    # @returns: dict with paths to all generated outputs
    
    outputs_dir = Path(config['paths']['outputs_dir'])
    outputs_dir.mkdir(exist_ok=True)
    
    # 1. Collect reach mask paths
    mask_paths = []
    for s in scenario_summaries:
        # Find this scenario's reach mask via its scenario_id
        sid = s['scenario_id']
        mask_path = (Path(config['paths']['scenarios_dir'])
                     / sid / 'outputs' / 'reach_mask.tif')
        if mask_path.exists():
            mask_paths.append(mask_path)
        else:
            logging.warning(f"Missing reach mask for {sid}: {mask_path}")
    
    if not mask_paths:
        # throw error if nothing to stack
        raise RuntimeError("No reach masks found; cannot build probability raster")
    
    # 2. Probability raster
    prob_path = outputs_dir / 'probability.tif'
    stack_reach_masks(mask_paths, prob_path)
    
    # 3. Envelope + contours
    polygonize_probability(
        probability_path=prob_path,
        contour_levels=config['probability']['contour_levels'],
        envelope_path=outputs_dir / 'envelope.geojson',
        contours_path=outputs_dir / 'contours.geojson',
        envelope_threshold=config['probability']['envelope_min_probability']
    )
    
    # 4. Most-likely-path (medial axis at P >= 0.5)
    extract_most_likely_path(
        probability_path=prob_path,
        threshold=0.5,
        out_path=outputs_dir / 'most_likely_path.geojson'
    )
    
    # 5. Road impact summary
    road_summary = build_road_impact_summary(
        scenario_summaries,
        outputs_dir / 'road_impact_summary.json'
    )
    
    # 6. Scenarios CSV
    write_scenarios_csv(
        scenario_summaries,
        outputs_dir / 'scenarios_summary.csv'
    )
    
    logging.info(f"Ensemble outputs written to {outputs_dir}")
    
    return {
        'probability': prob_path,
        'envelope': outputs_dir / 'envelope.geojson',
        'contours': outputs_dir / 'contours.geojson',
        'most_likely_path': outputs_dir / 'most_likely_path.geojson',
        'road_summary': outputs_dir / 'road_impact_summary.json',
        'scenarios_csv': outputs_dir / 'scenarios_summary.csv',
        # In-memory copy for the report builder
        'road_summary_dict': road_summary
    }
