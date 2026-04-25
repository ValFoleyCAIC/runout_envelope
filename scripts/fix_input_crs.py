"""
Avalanche Runout Envelope - CRS Metadata Fix Utility

Author: Valerie Foley
Last Updated: 4/24/2026

Description:
    Walks all scenarios in a directory and adds correct CRS metadata to
    files that are missing or wrongly tagged. Does NOT reproject - just
    declares the CRS the coordinates are already in.
    
    Two fixes per scenario:
        1. release.geojson: re-saves with EPSG:6342 in the metadata
           (was incorrectly tagged EPSG:4326 by ron's writer)
        2. depth.asc: writes a sidecar .prj file declaring EPSG:6342
           (was missing entirely)

Usage:
    python fix_ron_crs.py --scenarios-dir /path/to/scenarios --epsg 6342
    python fix_ron_crs.py --scenarios-dir /path/to/scenarios --epsg 6342 --dry-run
    python fix_ron_crs.py --scenarios-dir /path/to/scenarios --epsg 6342 --force

Requirements:
    - geopandas, pyproj
"""

# --------- Load Libraries --------
import argparse
import sys
import logging
from pathlib import Path

import geopandas as gpd
from pyproj import CRS




def fix_release_crs(release_path, target_epsg):
    # Re-tag a release.geojson with the correct EPSG (no reprojection)
    # @param release_path: Path to release.geojson
    # @param target_epsg: int - EPSG code the coordinates are actually in
    # @returns: bool - True if file was modified, False if already correct
    
    gdf = gpd.read_file(release_path)
    current = gdf.crs
    
    # Check if already correct
    if current is not None and current.to_epsg() == target_epsg:
        return False
    
    # Override CRS metadata without reprojecting coordinates
    gdf = gdf.set_crs(f'EPSG:{target_epsg}', allow_override=True)
    gdf.to_file(release_path, driver='GeoJSON')
    
    return True


def fix_depth_prj(depth_path, target_epsg, force=False):
    # Write a .prj sidecar file for an .asc raster
    # @param depth_path: Path to depth.asc
    # @param target_epsg: int - EPSG code the coordinates are in
    # @param force: bool - if True, overwrite an existing .prj
    # @returns: bool - True if .prj was written, False if skipped
    
    prj_path = depth_path.with_suffix('.prj')
    
    # Skip if already present (unless --force)
    if prj_path.exists() and not force:
        return False
    
    # Write WKT1 (ESRI flavor) - more reliable than WKT2 for .asc files.
    # Rasterio uses GDAL which prefers WKT1 for AAIGrid sidecars.
    wkt = CRS.from_epsg(target_epsg).to_wkt(version='WKT1_ESRI')
    prj_path.write_text(wkt)
    
    return True


def parse_args():
    # Parse command-line arguments
    # @param: None
    # @returns: argparse.Namespace with parsed arguments
    
    parser = argparse.ArgumentParser(
        description='Fix CRS metadata on scenario release/depth files'
    )
    parser.add_argument('--scenarios-dir', required=True, help='Path to scenarios/ directory')
    parser.add_argument('--epsg', type=int, required=True, help='Correct EPSG code (e.g. 6342)')
    parser.add_argument('--dry-run', action='store_true', help='Report what would be changed without changing anything')
    parser.add_argument('--force', action='store_true', help='Overwrite existing .prj files (use if previous run wrote bad WKT that rasterio cannot read)')
    
    return parser.parse_args()


def main():
    # Walk scenarios directory and fix CRS on all files
    # @param: None (args from command line)
    # @returns: None (exits with status code)
    
    # Parse args
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    scenarios_dir = Path(args.scenarios_dir)
    if not scenarios_dir.exists():
        # throw error if directory doesn't exist
        logging.error(f"Scenarios directory not found: {scenarios_dir}")
        sys.exit(1)
    
    # Track stats
    n_scenarios = 0
    n_release_fixed = 0
    n_depth_fixed = 0
    n_skipped = 0
    
    # Walk each scenario folder
    for sub in sorted(scenarios_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith('scenario_'):
            continue
        n_scenarios += 1
        
        # Determine input dir (inputs/ subfolder OR scenario folder itself)
        inp = sub / 'inputs' if (sub / 'inputs').is_dir() else sub
        
        # Fix release.geojson
        release_path = inp / 'release.geojson'
        if release_path.exists():
            if args.dry_run:
                gdf = gpd.read_file(release_path)
                current = gdf.crs.to_epsg() if gdf.crs else None
                if current != args.epsg:
                    logging.info(
                        f"  [dry-run] {sub.name}: would re-tag release "
                        f"({current} -> {args.epsg})"
                    )
                    n_release_fixed += 1
            else:
                if fix_release_crs(release_path, args.epsg):
                    logging.info(f"  {sub.name}: re-tagged release.geojson")
                    n_release_fixed += 1
        else:
            logging.warning(f"  {sub.name}: no release.geojson")
        
        # Fix depth.asc (write .prj sidecar)
        depth_path = inp / 'depth.asc'
        if depth_path.exists():
            prj_path = depth_path.with_suffix('.prj')
            if args.dry_run:
                if not prj_path.exists():
                    logging.info(f"  [dry-run] {sub.name}: would write depth.prj")
                    n_depth_fixed += 1
                elif args.force:
                    logging.info(f"  [dry-run] {sub.name}: would overwrite depth.prj")
                    n_depth_fixed += 1
            else:
                if fix_depth_prj(depth_path, args.epsg, force=args.force):
                    action = "overwrote" if prj_path.exists() and args.force else "wrote"
                    logging.info(f"  {sub.name}: {action} depth.prj")
                    n_depth_fixed += 1
        else:
            n_skipped += 1
            logging.warning(f"  {sub.name}: no depth.asc")
    
    # woohoo success
    logging.info("")
    logging.info("=" * 60)
    logging.info(f" Scenarios processed: {n_scenarios}")
    logging.info(f" release.geojson re-tagged: {n_release_fixed}")
    logging.info(f" depth.prj written:         {n_depth_fixed}")
    if args.dry_run:
        logging.info(" (dry-run - nothing actually changed)")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
