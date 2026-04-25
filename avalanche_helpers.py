"""
Avalanche Runout Envelope - Helpers Module

Author: Valerie Foley
Last Updated: 4/24/2026

Description:
    Shared utilities for the avalanche runout pipeline. Contains:
        - Raster I/O (read, write, format conversion, COG output)
        - Vector I/O (GeoJSON <-> Shapefile)
        - CRS validation across mixed inputs
        - Pixel <-> world coordinate helpers
        - Statistics helpers (Wilson CI, residual stats)
        - Entrainment area auto-detection (alpha-beta corridor + fallback)
    
    This module has no command-line interface. It's imported by
    avalanche_runout.py, avalanche_simulation.py, avalanche_stack.py,
    build_report.py, and build_viewer.py.

Requirements:
    - rasterio, fiona, geopandas, shapely, pyproj, numpy, scipy
"""

# --------- Load Libraries --------
import os
import json
import shutil
import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes
import fiona
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, mapping, shape as shp_shape
from scipy import stats as scipy_stats




def read_raster(path):
    # Read first band of a raster + key metadata
    # @param path: Path to raster file (.tif, .asc, etc.)
    # @returns: tuple (data, meta) where:
    #   - data: 2D numpy array
    #   - meta: dict with transform, crs, nodata, shape, dtype, profile
    
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'nodata': src.nodata,
            'shape': data.shape,
            'dtype': data.dtype,
            'profile': src.profile.copy()
        }
    
    logging.debug(f"Read raster: {Path(path).name}")
    return data, meta


def write_raster(path, data, profile, driver='GTiff', cog=True):
    # Write a raster to disk, optionally as Cloud-Optimized GeoTIFF
    # @param path: Output file path
    # @param data: 2D numpy array to write
    # @param profile: rasterio profile dict (will be modified for driver)
    # @param driver: 'GTiff' or 'AAIGrid'
    # @param cog: If True and driver=GTiff, write tiled+compressed
    # @returns: Path to written file
    # note:
    #   - AAIGrid (.asc) doesn't support compression or tiling - those
    #     options get stripped automatically
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build clean output profile based on driver
    out_profile = profile.copy()
    out_profile.update(driver=driver, count=1, dtype=data.dtype)
    
    if driver == 'GTiff' and cog:
        # COG-friendly settings
        out_profile.update(
            compress='deflate', tiled=True,
            blockxsize=256, blockysize=256
        )
    else:
        # AAIGrid rejects these keys - strip them
        for key in ('compress', 'tiled', 'blockxsize', 'blockysize',
                    'interleave', 'cellsize'):
            out_profile.pop(key, None)
    
    with rasterio.open(path, 'w', **out_profile) as dst:
        dst.write(data, 1)
    
    return path


def convert_raster_format(src_path, dst_path):
    # Convert a raster between formats based on file extensions
    # @param src_path: Source raster (.tif or .asc)
    # @param dst_path: Destination raster (extension determines format)
    # @returns: Path to written file
    # note:
    #   - If src and dst are both same format, copies file directly
    #   - .prj sidecars only get copied for .asc files. GeoTIFF embeds CRS
    #     internally; copying a .prj alongside a .tif is unnecessary and
    #     would re-trigger AvaFrame's buggy remeshedRasters glob.
    #   - .tif <-> .asc conversion goes through rasterio
    
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    
    # Same format - just copy
    if src_path.suffix.lower() == dst_path.suffix.lower():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        # Copy .prj sidecar only for .asc files (GeoTIFF embeds CRS internally)
        if dst_path.suffix.lower() == '.asc':
            prj = src_path.with_suffix('.prj')
            if prj.exists():
                shutil.copy2(prj, dst_path.with_suffix('.prj'))
        return dst_path
    
    # Format conversion via rasterio
    data, meta = read_raster(src_path)
    driver = 'AAIGrid' if dst_path.suffix.lower() == '.asc' else 'GTiff'
    
    # Force float32 for AAIGrid compatibility
    return write_raster(
        dst_path, data.astype('float32'), meta['profile'],
        driver=driver, cog=(driver == 'GTiff')
    )


def resample_to_reference_grid(src_path, ref_path, dst_path,
                               resampling_method='bilinear',
                               fill_value=0.0):
    # Resample a raster onto another raster's exact grid (cell size, origin, extent)
    # @param src_path: Source raster to resample
    # @param ref_path: Reference raster whose grid to align to
    # @param dst_path: Output raster path
    # @param resampling_method: 'bilinear' (default), 'nearest', or 'cubic'
    # @param fill_value: Value to assign cells outside the source extent
    # @returns: Path to written raster
    # note:
    #   - Use this for AvaFrame inputs where DEM and release-thickness must
    #     share an identical grid. AvaFrame's grid-alignment check requires
    #     LL-corner coords within 3 cells and identical cell size.
    #   - Cells in the reference grid that fall outside the source raster's
    #     coverage are set to fill_value (default 0 = no release/no entrainment).
    #   - bilinear is the right default for continuous fields like depth or
    #     elevation. Use 'nearest' for categorical/binary data.
    
    from rasterio.warp import reproject, Resampling
    
    # Map method strings to rasterio Resampling enum
    methods = {
        'bilinear': Resampling.bilinear,
        'nearest': Resampling.nearest,
        'cubic': Resampling.cubic,
    }
    if resampling_method not in methods:
        # throw error if invalid method
        raise ValueError(
            f"resampling_method must be one of {list(methods.keys())}, "
            f"got '{resampling_method}'"
        )
    
    # Open reference to get target grid
    with rasterio.open(ref_path) as ref:
        target_profile = ref.profile.copy()
        target_transform = ref.transform
        target_crs = ref.crs
        target_shape = (ref.height, ref.width)
    
    # Open source and reproject onto target grid
    with rasterio.open(src_path) as src:
        # Output array, pre-filled with fill_value for cells outside source
        out_data = np.full(target_shape, fill_value, dtype='float32')
        
        reproject(
            source=rasterio.band(src, 1),
            destination=out_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=methods[resampling_method],
            dst_nodata=fill_value
        )
    
    # Write with reference grid's profile (driver determined by dst extension)
    dst_path = Path(dst_path)
    driver = 'AAIGrid' if dst_path.suffix.lower() == '.asc' else 'GTiff'
    target_profile.update(
        driver=driver, count=1, dtype='float32', nodata=fill_value
    )
    
    return write_raster(
        dst_path, out_data, target_profile,
        driver=driver, cog=(driver == 'GTiff')
    )


def geojson_to_shapefile(src_path, dst_path, attributes=None, fill_holes=False,
                         clip_to_raster=None):
    # Convert a GeoJSON to ESRI Shapefile (com1DFA expects .shp)
    # @param src_path: Path to .geojson (or any geopandas-readable vector file)
    # @param dst_path: Path to .shp
    # @param attributes: Optional dict of {field_name: value} to inject into
    #                    every feature (used to add fields AvaFrame requires
    #                    even when their values aren't physically used,
    #                    e.g. 'thickness' when relTh comes from a raster)
    # @param fill_holes: If True, drop interior rings from polygons. Use for
    #                    com1DFA release polygons since AvaFrame rejects any
    #                    feature with holes (asserts in checkForMultiplePartsShpArea).
    #                    Filling holes is the conservative interpretation - tiny
    #                    gaps in a release area get treated as part of the slab,
    #                    which slightly enlarges the release (errs toward bigger
    #                    runout, the protective direction for road risk).
    # @param clip_to_raster: Optional Path to a raster - clip output features
    #                        to the raster's bounds (shrunk slightly to ensure
    #                        no feature touches the very edge). Use for forest
    #                        and other area shapefiles that may extend beyond
    #                        the DEM coverage; com1DFA aborts if any feature
    #                        exceeds DEM extent.
    # @returns: Path to written shapefile
    # note:
    #   - Shapefile field names are limited to 10 chars - column names get
    #     truncated to avoid fiona warnings
    
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    gdf = gpd.read_file(src_path)
    
    # Clip to raster extent (shrunk by safety margin to keep features
    # well inside DEM coverage). AvaFrame checks against cell-center
    # bounds, not cell-corner bounds, and rejects features that touch
    # the edge. 3 cells gives reliable margin even with sub-pixel
    # geometric jitter from intersection.
    if clip_to_raster is not None:
        from shapely.geometry import box
        with rasterio.open(clip_to_raster) as src:
            cell = max(abs(src.transform.a), abs(src.transform.e))
            margin = 3 * cell
            l, b, r, t = src.bounds
            clip_box = box(l + margin, b + margin, r - margin, t - margin)
        
        # Clip in the raster's CRS - reproject if needed, then return to original
        original_crs = gdf.crs
        target_crs = src.crs
        if original_crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        gdf['geometry'] = gdf.geometry.intersection(clip_box)
        
        # Drop empty geometries that fell entirely outside
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
        if original_crs != target_crs:
            gdf = gdf.to_crs(original_crs)
        
        if len(gdf) == 0:
            # throw error if clipping removed everything
            raise ValueError(
                f"Clipping {src_path.name} to {Path(clip_to_raster).name} "
                "extent removed all features. Check that your shapefile "
                "actually covers the DEM area."
            )
    
    # Fill holes by replacing each polygon with one made from its exterior only
    if fill_holes:
        new_geoms = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                new_geoms.append(geom)
                continue
            if geom.geom_type == 'Polygon':
                # Drop interiors - keep only the outer ring
                new_geoms.append(Polygon(geom.exterior.coords))
            elif geom.geom_type == 'MultiPolygon':
                # Same for each part
                new_parts = [Polygon(p.exterior.coords) for p in geom.geoms]
                from shapely.geometry import MultiPolygon
                new_geoms.append(MultiPolygon(new_parts))
            else:
                # Lines or points - leave alone
                new_geoms.append(geom)
        gdf = gdf.set_geometry(new_geoms)
    
    # Inject required attributes (e.g. thickness for AvaFrame validation)
    if attributes:
        for field, value in attributes.items():
            gdf[field] = value
    
    # Truncate column names to 10 chars for shapefile compliance
    gdf.columns = [c[:10] for c in gdf.columns]
    gdf.to_file(dst_path, driver='ESRI Shapefile')
    
    return dst_path


def get_crs_of(path):
    # Get the CRS of any geospatial file (raster or vector)
    # @param path: Path to .tif/.asc/.shp/.geojson/.gpkg
    # @returns: rasterio CRS object, or None if no CRS set
    
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix in ('.tif', '.tiff', '.asc'):
        with rasterio.open(path) as src:
            return src.crs
    
    if suffix in ('.shp', '.geojson', '.gpkg', '.json'):
        with fiona.open(path) as src:
            if src.crs:
                return CRS.from_user_input(src.crs)
        return None
    
    # throw error if file extension not recognized
    raise ValueError(f"Unknown geospatial extension: {suffix}")


def validate_crs(paths_and_labels, reference_crs):
    # Validate that all input files share the reference CRS
    # @param paths_and_labels: list of (path, label) tuples for error messages
    # @param reference_crs: rasterio CRS to compare against
    # @returns: tuple (success, mismatches) where:
    #   - success: Boolean
    #   - mismatches: list of human-readable mismatch descriptions
    
    mismatches = []
    for path, label in paths_and_labels:
        c = get_crs_of(path)
        if c is None:
            mismatches.append(f"{label} ({path}) has no CRS")
        elif c != reference_crs:
            mismatches.append(
                f"{label} ({path}) CRS = {c}, expected {reference_crs}"
            )
    
    return (len(mismatches) == 0), mismatches


def pixels_to_world(rows, cols, transform):
    # Convert pixel (row, col) to world (x, y) coordinates - cell centers
    # @param rows: array of row indices
    # @param cols: array of column indices
    # @param transform: rasterio Affine transform
    # @returns: tuple (xs, ys) of world coordinates
    # note:
    #   - Uses cell centers (+0.5 offset), not cell corners
    
    xs = transform.a * (np.asarray(cols) + 0.5) + transform.c
    ys = transform.e * (np.asarray(rows) + 0.5) + transform.f
    return xs, ys


def world_to_pixel(x, y, transform):
    # Convert single world (x, y) point to pixel (row, col)
    # @param x: world x coordinate
    # @param y: world y coordinate
    # @param transform: rasterio Affine transform
    # @returns: tuple (row, col) of integer pixel indices
    
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def sample_raster_at_xy(raster_path, xy_pairs):
    # Sample raster values at world coordinates (interpolation = nearest)
    # @param raster_path: Path to raster
    # @param xy_pairs: list of (x, y) tuples
    # @returns: list of sampled float values (NaN if outside raster)
    
    with rasterio.open(raster_path) as src:
        return [float(v[0]) for v in src.sample(xy_pairs)]


def polygon_to_raster_mask(polygon, raster_shape, transform, fill_value=1.0):
    # Burn a polygon into a raster mask aligned to a reference grid
    # @param polygon: shapely Polygon
    # @param raster_shape: tuple (height, width)
    # @param transform: rasterio Affine
    # @param fill_value: value to assign inside polygon (outside = 0)
    # @returns: 2D numpy array
    
    mask = geometry_mask(
        [mapping(polygon)],
        out_shape=raster_shape,
        transform=transform,
        invert=True
    )
    return np.where(mask, fill_value, 0.0).astype('float32')


def mask_to_polygons(mask, transform):
    # Polygonize a binary raster (value=1 -> polygon)
    # @param mask: 2D uint8/bool array
    # @param transform: rasterio Affine
    # @returns: list of shapely Polygons
    
    polys = []
    for geom, val in shapes(mask.astype('uint8'),
                            mask=mask.astype(bool),
                            transform=transform):
        if val == 1:
            polys.append(shp_shape(geom))
    return polys


def calculate_statistics(values):
    # Calculate basic stats from an array of values (NaN-safe)
    # @param values: numpy array
    # @returns: dict with n, mean, min, max, std, or None if no valid data
    
    clean = values[~np.isnan(values)]
    
    if len(clean) == 0:
        return None
    
    return {
        'n': int(len(clean)),
        'mean': float(np.mean(clean)),
        'min': float(np.min(clean)),
        'max': float(np.max(clean)),
        'std': float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    }


def wilson_ci(k, n, alpha=0.05):
    # Wilson score confidence interval for a binomial proportion
    # @param k: number of successes
    # @param n: number of trials
    # @param alpha: significance level (0.05 for 95% CI)
    # @returns: tuple (low, high) of CI bounds in [0, 1]
    # note:
    #   - More robust than normal approximation at small n (our n=20 regime)
    #   - Returns (0, 0) if n=0 to avoid div-by-zero
    
    if n == 0:
        return (0.0, 0.0)
    
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    
    return (max(0.0, centre - half), min(1.0, centre + half))


def steepest_descent_flowline(release_polygon, dem_path, max_steps=5000):
    # Walk downhill from the release centroid using D8-style steepest descent
    # @param release_polygon: shapely Polygon of the release area
    # @param dem_path: Path to DEM raster
    # @param max_steps: Maximum number of descent steps before giving up
    # @returns: shapely LineString of the descent path
    # note:
    #   - Step size = DEM cell size
    #   - Terminates when slope < 3 deg over a 3-step rolling window
    #   - References: McClung & Lied (1987), AvaFrame com2AB module
    
    # Load DEM
    dem, meta = read_raster(dem_path)
    transform = meta['transform']
    nodata = meta['nodata']
    step_m = float(min(abs(transform.a), abs(transform.e)))
    
    # Start at release centroid
    start = release_polygon.centroid
    x, y = start.x, start.y
    path = [(x, y)]
    
    # Helper: sample DEM at (x, y), returns None if outside or nodata
    def sample(px, py):
        row, col = world_to_pixel(px, py, transform)
        if row < 0 or row >= dem.shape[0] or col < 0 or col >= dem.shape[1]:
            return None
        v = dem[row, col]
        if nodata is not None and v == nodata:
            return None
        return float(v)
    
    # 8 compass directions
    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    recent_drops = []
    
    for _ in range(max_steps):
        z_here = sample(x, y)
        if z_here is None:
            break
        
        # Find direction of steepest descent
        best = None
        best_slope = 0.0
        for dx, dy in dirs:
            nx, ny = x + dx * step_m, y + dy * step_m
            z_next = sample(nx, ny)
            if z_next is None:
                continue
            length = math.hypot(dx * step_m, dy * step_m)
            slope = (z_here - z_next) / length if length > 0 else 0
            if slope > best_slope:
                best_slope = slope
                best = (nx, ny)
        
        # Stop if no downhill direction
        if best is None or best_slope <= 0:
            break
        
        x, y = best
        path.append((x, y))
        
        # Stop when 3-step average slope < 3 degrees (flow has flattened out)
        recent_drops.append(best_slope)
        if len(recent_drops) > 3:
            recent_drops.pop(0)
        if len(recent_drops) == 3 and np.mean(recent_drops) < math.tan(math.radians(3)):
            break
    
    # Degenerate fallback - shouldn't happen with real terrain
    if len(path) < 2:
        path.append((x + step_m, y - step_m))
    
    return LineString(path)


def compute_alpha_angle(flowline, dem_path):
    # Compute alpha-angle runout via the McClung-Lied statistical model
    # @param flowline: shapely LineString from release downhill
    # @param dem_path: Path to DEM raster
    # @returns: tuple (alpha_deg, alpha_point) where:
    #   - alpha_deg: alpha angle in degrees
    #   - alpha_point: shapely Point on flowline at the stopping location
    # note:
    #   - Beta point = first point along flowline where slope drops below 10 deg
    #   - Alpha = beta - 5 deg (default offset; would be regional in calibration)
    
    dem, meta = read_raster(dem_path)
    transform = meta['transform']
    
    # Helper: sample DEM at a Point
    def z_at(pt):
        row, col = world_to_pixel(pt.x, pt.y, transform)
        if row < 0 or row >= dem.shape[0] or col < 0 or col >= dem.shape[1]:
            return None
        return float(dem[row, col])
    
    coords = list(flowline.coords)
    start = Point(coords[0])
    z_start = z_at(start)
    if z_start is None:
        # throw error if start is outside DEM
        raise ValueError("Flowline start is outside DEM coverage")
    
    # Walk along flowline; beta = first point where slope-from-start < 10 deg
    beta_idx = None
    for i in range(5, len(coords)):
        pt = Point(coords[i])
        z = z_at(pt)
        if z is None:
            break
        dh = start.distance(pt)
        if dh < 1e-6:
            continue
        slope_deg = math.degrees(math.atan2(z_start - z, dh))
        if slope_deg < 10.0:
            beta_idx = i
            break
    
    # If slope never flattens, use end of flowline
    if beta_idx is None:
        beta_idx = len(coords) - 1
    
    beta_pt = Point(coords[beta_idx])
    z_beta = z_at(beta_pt) or z_start
    
    dh_total = start.distance(beta_pt)
    if dh_total < 1e-6:
        # throw error if beta point coincides with start
        raise ValueError("Degenerate flowline: beta point coincides with start")
    
    beta_deg = math.degrees(math.atan2(z_start - z_beta, dh_total))
    
    # Statistical offset: alpha = beta - 5 deg (generic; calibrate regionally)
    alpha_deg = max(beta_deg - 5.0, 10.0)
    
    # Project alpha-point on flowline by walking until profile rises above
    # the alpha-angle line from start
    tan_alpha = math.tan(math.radians(alpha_deg))
    last_valid = Point(coords[-1])
    for i in range(1, len(coords)):
        pt = Point(coords[i])
        z = z_at(pt)
        if z is None:
            break
        required_drop = start.distance(pt) * tan_alpha
        if (z_start - z) >= required_drop:
            last_valid = pt
            continue
        return alpha_deg, last_valid
    
    return alpha_deg, last_valid


def build_entrainment_polygon(release_polygon, flowline, alpha_point=None,
                              buffer_m=80.0):
    # Build the entrainment polygon as a corridor along the flowline
    # @param release_polygon: shapely Polygon of release area
    # @param flowline: shapely LineString from release downhill
    # @param alpha_point: shapely Point where flow stops (alpha-beta mode);
    #                     if None, use full flowline (flow-accumulation mode)
    # @param buffer_m: corridor half-width in meters
    # @returns: shapely Polygon (release polygon subtracted)
    # note:
    #   - Subtracting the release means entrainment starts downslope of the crown
    
    coords = list(flowline.coords)
    
    # Truncate to alpha-point if provided
    if alpha_point is not None:
        d_alpha = Point(coords[0]).distance(alpha_point)
        truncated = []
        cum = 0.0
        prev = None
        for c in coords:
            p = Point(c)
            if prev is not None:
                cum += prev.distance(p)
            truncated.append(c)
            if cum >= d_alpha:
                break
            prev = p
        if len(truncated) < 2:
            truncated = coords[:2]
        line = LineString(truncated)
    else:
        line = flowline
    
    # Buffer to corridor, subtract release area
    corridor = line.buffer(buffer_m, cap_style=2, join_style=2)
    entrainment = corridor.difference(release_polygon)
    
    # Pick largest piece if multipolygon
    if hasattr(entrainment, 'geoms'):
        entrainment = max(entrainment.geoms, key=lambda g: g.area)
    
    return entrainment


def auto_detect_entrainment(release_polygon, dem_path, polygon_out, thickness_out,
                            crs, depth_m, alpha_min=15.0, alpha_max=45.0,
                            fallback_buffer_m=50.0, flowline_out=None):
    # Auto-detect entrainment area for one scenario, write polygon + thickness raster
    # @param release_polygon: shapely Polygon of release area
    # @param dem_path: Path to DEM raster (also defines output grid)
    # @param polygon_out: Path to write entrainment polygon shapefile
    # @param thickness_out: Path to write entrainment thickness .asc
    # @param crs: CRS to assign to outputs
    # @param depth_m: constant entrainment depth (meters)
    # @param alpha_min: lower bound on plausible alpha angle (deg)
    # @param alpha_max: upper bound on plausible alpha angle (deg)
    # @param fallback_buffer_m: corridor half-width if alpha-beta fails
    # @param flowline_out: Optional Path to also write the steepest-descent
    #                      flowline as GeoJSON (used by viewer for per-scenario paths)
    # @returns: dict with method_used, alpha_deg, area_m2, depth_m
    
    # Step 1: trace flowline by steepest descent
    flowline = steepest_descent_flowline(release_polygon, dem_path)
    
    # Optionally save the flowline for the viewer
    if flowline_out is not None:
        flowline_out = Path(flowline_out)
        flowline_out.parent.mkdir(parents=True, exist_ok=True)
        flow_gdf = gpd.GeoDataFrame(
            {'name': ['flowpath']}, geometry=[flowline], crs=crs
        )
        flow_gdf.to_file(flowline_out, driver='GeoJSON')
    
    # Step 2: try alpha-beta first, fall back to flow-accumulation buffer
    method_used = 'alpha_beta'
    alpha_deg = None
    polygon = None
    
    try:
        alpha_deg, alpha_pt = compute_alpha_angle(flowline, dem_path)
        
        # Sanity check the computed alpha
        if alpha_deg < alpha_min or alpha_deg > alpha_max:
            # throw error to trigger fallback
            raise ValueError(
                f"alpha={alpha_deg:.1f} deg outside plausible "
                f"range [{alpha_min}, {alpha_max}]"
            )
        
        polygon = build_entrainment_polygon(
            release_polygon, flowline, alpha_point=alpha_pt
        )
        logging.info(f"  Entrainment: alpha-beta corridor, alpha={alpha_deg:.1f} deg")
    
    except Exception as e:
        # Fallback: flow-accumulation buffer (no alpha truncation)
        logging.warning(f"  alpha-beta failed ({e}); using flow-accumulation buffer")
        method_used = 'flow_accumulation'
        alpha_deg = None
        polygon = build_entrainment_polygon(
            release_polygon, flowline, alpha_point=None,
            buffer_m=fallback_buffer_m
        )
    
    # Step 3: write polygon as shapefile
    # AvaFrame validates that entrainment polygons have a 'thickness'
    # attribute, even though we provide the actual values via raster
    # (entThFromFile=True). Field must exist to pass validation.
    polygon_out = Path(polygon_out)
    polygon_out.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {'name': ['entrainment'], 'thickness': [float(depth_m)]},
        geometry=[polygon],
        crs=crs
    )
    gdf.to_file(polygon_out, driver='ESRI Shapefile')
    
    # Step 4: write thickness raster (constant depth inside polygon, 0 outside)
    _, dem_meta = read_raster(dem_path)
    mask = polygon_to_raster_mask(
        polygon, dem_meta['shape'], dem_meta['transform'],
        fill_value=depth_m
    )
    write_raster(thickness_out, mask, dem_meta['profile'], driver='AAIGrid')
    
    return {
        'method_used': method_used,
        'alpha_deg': alpha_deg,
        'area_m2': float(polygon.area),
        'depth_m': depth_m
    }
