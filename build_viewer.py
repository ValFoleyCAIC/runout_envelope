"""
Avalanche Runout Envelope - Leaflet Viewer Builder

Author: Valerie Foley
Last Updated: 4/24/2026

Description:
    Renders an interactive Leaflet viewer (viewer.html) from the ensemble
    outputs. The basemap is a hillshade rendered from the project DEM -
    no external tile server (avoids OSM rate-limit / referer-policy errors).
    
    Layers (toggleable):
        - Hillshade basemap (from project DEM, always on)
        - Probability raster (image overlay, semi-transparent)
        - Each contour level as its own outline layer
        - Envelope outline
        - Most-likely-path centerline (medial axis at P>=0.5)
        - Road centerline (always on, red)
        - Impact points (clickable markers with per-scenario popups)
        - Per-scenario release polygons (each scenario its own toggle)
        - Per-scenario flowpaths (each scenario its own toggle)
        - Per-scenario reach polygons (each scenario its own toggle)
        - Forest resistance polygon (off by default)
    
    All raster overlays are reprojected to EPSG:4326 and embedded as
    base64-encoded PNG data URIs so viewer.html is fully self-contained.
    
    No CLI. Imported by avalanche_runout.py.

Requirements:
    - folium, branca, geopandas, rasterio, pillow, pyproj, matplotlib
"""

# --------- Load Libraries --------
import base64
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from PIL import Image
from pyproj import Transformer

import avalanche_helpers as h


# Colors for per-scenario layers (cycled if more scenarios than colors).
# Picked to be distinguishable on a grayscale hillshade and on the viridis
# probability heatmap.
SCENARIO_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#1b9e77', '#d95f02',
    '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d',
    '#666666', '#1f78b4', '#33a02c', '#fb9a99', '#ff99cc'
]




def reproject_raster_to_wgs84(src_path, dst_path, resampling=Resampling.nearest):
    # Reproject a raster to EPSG:4326 (WGS84) for Leaflet image overlay
    # @param src_path: Path to source raster (any CRS)
    # @param dst_path: Path to write WGS84 raster
    # @param resampling: rasterio Resampling enum (default: nearest)
    # @returns: Path to written raster
    # note:
    #   - Default uses nearest-neighbor (preserves probability values
    #     without smoothing). Use bilinear for elevation/hillshade.
    
    # If already in WGS84, just copy
    with rasterio.open(src_path) as src:
        if src.crs.to_string() == 'EPSG:4326':
            import shutil
            shutil.copy2(src_path, dst_path)
            return dst_path
        
        # Compute target transform/dimensions
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs='EPSG:4326', transform=transform,
            width=width, height=height
        )
        
        # Reproject
        with rasterio.open(dst_path, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs='EPSG:4326',
                resampling=resampling
            )
    
    return dst_path


def array_to_data_uri(rgba_array):
    # Encode an RGBA numpy array as a base64 PNG data URI
    # @param rgba_array: numpy array, shape (H, W, 4), dtype uint8
    # @returns: str - 'data:image/png;base64,...' suitable for Leaflet
    
    img = Image.fromarray(rgba_array, mode='RGBA')
    buf = BytesIO()
    img.save(buf, format='PNG')
    png_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{png_b64}'


def raster_bounds_wgs84(raster_path):
    # Compute Leaflet imageOverlay bounds [[south, west], [north, east]]
    # @param raster_path: Path to a raster in EPSG:4326
    # @returns: list of two [lat, lon] pairs
    
    with rasterio.open(raster_path) as src:
        t = src.transform
        left, top = t.c, t.f
        right = left + src.width * t.a
        bottom = top + src.height * t.e
    
    return [[bottom, left], [top, right]]


def hillshade_array(elevation, cell_size_m, azdeg=315, altdeg=45):
    # Compute a hillshade from an elevation array
    # @param elevation: 2D float numpy array
    # @param cell_size_m: float - DEM cell size in meters (for dx, dy scaling)
    # @param azdeg: float - sun azimuth in degrees (default 315 = NW)
    # @param altdeg: float - sun altitude in degrees (default 45)
    # @returns: 2D float array in [0, 1]
    # note:
    #   - Uses matplotlib's LightSource which is Horn's algorithm
    #   - Lower altdeg gives more dramatic shading; 45 is a good default
    
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    return ls.hillshade(elevation, dx=cell_size_m, dy=cell_size_m)


def render_hillshade_data_uri(dem_path, outputs_dir, vert_exag=1.0):
    # Render a DEM as a hillshade PNG data URI (for Leaflet image overlay)
    # @param dem_path: Path to DEM raster
    # @param outputs_dir: Path - workspace for the WGS84 reprojection
    # @param vert_exag: float - vertical exaggeration multiplier
    # @returns: tuple (data_uri, bounds) where:
    #   - data_uri: base64 PNG data URI
    #   - bounds: [[south, west], [north, east]] for Leaflet
    # note:
    #   - Reprojects DEM to EPSG:4326 with bilinear resampling (smooth
    #     elevation interpolation, since hillshade reads gradients)
    #   - NoData / negative elevations rendered transparent
    
    # Reproject DEM to WGS84
    dem_wgs = outputs_dir / '_tmp_dem_wgs84.tif'
    reproject_raster_to_wgs84(dem_path, dem_wgs, resampling=Resampling.bilinear)
    
    # Read reprojected DEM and compute approximate cell size in meters
    # (latitude-dependent; use mid-latitude approximation)
    elevation, meta = h.read_raster(dem_wgs)
    nodata = meta['nodata']
    
    # Replace nodata with neighborhood-mean fill so hillshade gradients
    # don't include huge artificial jumps at edges
    if nodata is not None:
        bad = (elevation == nodata)
        if bad.any():
            valid_mean = elevation[~bad].mean() if (~bad).any() else 0.0
            elevation = np.where(bad, valid_mean, elevation)
    else:
        bad = np.zeros_like(elevation, dtype=bool)
    
    # Estimate cell size in meters: deg -> m at the raster's mid latitude
    t = meta['transform']
    mid_lat_rad = np.radians((t.f + (elevation.shape[0] / 2) * t.e))
    deg_to_m_x = 111_320 * np.cos(mid_lat_rad)
    deg_to_m_y = 111_320
    cell_m = (abs(t.a) * deg_to_m_x + abs(t.e) * deg_to_m_y) / 2
    
    # Apply vertical exaggeration and compute hillshade
    shade = hillshade_array(elevation * vert_exag, cell_size_m=cell_m)
    
    # Convert to grayscale RGBA: dark hillshade = low shade value
    rgba = np.zeros((*shade.shape, 4), dtype=np.uint8)
    gray = (shade * 255).astype(np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    
    # Alpha: opaque inside DEM, transparent outside (where data was nodata)
    rgba[..., 3] = np.where(bad, 0, 255)
    
    # Encode and clean up
    data_uri = array_to_data_uri(rgba)
    bounds = raster_bounds_wgs84(dem_wgs)
    dem_wgs.unlink(missing_ok=True)
    
    return data_uri, bounds


def render_probability_data_uri(prob_path, outputs_dir, colormap='viridis'):
    # Render a probability raster as a colored semi-transparent PNG data URI
    # @param prob_path: Path to probability raster
    # @param outputs_dir: Path - workspace for the WGS84 reprojection
    # @param colormap: matplotlib colormap name
    # @returns: tuple (data_uri, bounds) where:
    #   - data_uri: base64 PNG data URI
    #   - bounds: [[south, west], [north, east]] for Leaflet
    # note:
    #   - Alpha scales with sqrt(probability) so high-P areas pop
    #   - Cells with nodata or P<0 are fully transparent
    
    # Reproject to WGS84
    prob_wgs = outputs_dir / '_tmp_probability_wgs84.tif'
    reproject_raster_to_wgs84(prob_path, prob_wgs, resampling=Resampling.nearest)
    
    # Read raster
    data, meta = h.read_raster(prob_wgs)
    nodata = meta['nodata']
    
    # Mask: nodata and negative probabilities are transparent
    if nodata is not None:
        mask = (data == nodata) | (data < 0)
    else:
        mask = data < 0
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)
    rgba = cmap(norm(data))
    
    # Set alpha based on probability (sqrt for stronger visual)
    rgba[..., 3] = np.where(mask, 0, np.clip(data, 0, 1) ** 0.5)
    rgba[mask] = 0
    
    # Convert to 8-bit
    rgba_8bit = (rgba * 255).astype(np.uint8)
    
    # Encode and clean up
    data_uri = array_to_data_uri(rgba_8bit)
    bounds = raster_bounds_wgs84(prob_wgs)
    prob_wgs.unlink(missing_ok=True)
    
    return data_uri, bounds


def add_hillshade_basemap(m, dem_path, outputs_dir):
    # Add a hillshade DEM as the basemap (replaces OSM tiles)
    # @param m: folium Map
    # @param dem_path: Path to DEM raster
    # @param outputs_dir: Path - workspace for temporary reprojections
    # @returns: bounds of the hillshade in WGS84 (used to fit the map)
    
    data_uri, bounds = render_hillshade_data_uri(dem_path, outputs_dir)
    
    folium.raster_layers.ImageOverlay(
        name='Hillshade (basemap)',
        image=data_uri,
        bounds=bounds,
        opacity=1.0,
        z_index=1
    ).add_to(m)
    
    return bounds


def add_probability_overlay(m, prob_path, outputs_dir, colormap):
    # Add the probability raster as a base64-embedded image overlay
    # @param m: folium Map
    # @param prob_path: Path to probability raster
    # @param outputs_dir: Path - workspace for temporary reprojections
    # @param colormap: matplotlib colormap name
    # @returns: None
    
    data_uri, bounds = render_probability_data_uri(
        prob_path, outputs_dir, colormap=colormap
    )
    
    folium.raster_layers.ImageOverlay(
        name='Probability P(reach)',
        image=data_uri,
        bounds=bounds,
        opacity=0.65,
        z_index=2
    ).add_to(m)


def add_contour_layers(m, contours_path):
    # Add contour outlines, one Leaflet layer per probability level
    # @param m: folium Map
    # @param contours_path: Path to contours.geojson
    # @returns: None
    
    try:
        contours_gdf = gpd.read_file(contours_path).to_crs('EPSG:4326')
    except Exception as e:
        logging.debug(f"No contours in viewer: {e}")
        return
    
    # Group by level - one Leaflet layer per level
    for level in sorted(contours_gdf['level'].unique()):
        subset = contours_gdf[contours_gdf['level'] == level]
        
        # Style: thicker line + more opacity at higher P
        folium.GeoJson(
            subset.to_json(),
            name=f"P >= {level:.2f}",
            style_function=lambda feat, lvl=level: {
                'color': 'black',
                'weight': 1 + lvl * 2,
                'fill': False,
                'opacity': 0.5 + lvl * 0.4
            },
            show=(level >= 0.5)
        ).add_to(m)


def add_envelope_layer(m, envelope_path, threshold):
    # Add envelope polygon outline (single layer, always visible)
    # @param m: folium Map
    # @param envelope_path: Path to envelope.geojson
    # @param threshold: float - probability threshold for envelope label
    # @returns: GeoDataFrame in WGS84 (used for centering the map)
    
    env_wgs = gpd.read_file(envelope_path).to_crs('EPSG:4326')
    folium.GeoJson(
        env_wgs.to_json(),
        name=f"Envelope (P >= {threshold})",
        style_function=lambda _: {
            'color': '#c0392b',
            'weight': 2,
            'fill': False,
            'opacity': 0.9
        }
    ).add_to(m)
    return env_wgs


def add_most_likely_path_layer(m, mlp_path):
    # Add the most-likely-path polyline (medial axis at P >= 0.5)
    # @param m: folium Map
    # @param mlp_path: Path to most_likely_path.geojson
    # @returns: None
    
    if not Path(mlp_path).exists():
        return
    
    try:
        mlp = gpd.read_file(mlp_path).to_crs('EPSG:4326')
    except Exception as e:
        logging.debug(f"Could not read most_likely_path: {e}")
        return
    
    if len(mlp) == 0:
        return
    
    folium.GeoJson(
        mlp.to_json(),
        name='Most-likely path (P >= 0.5)',
        style_function=lambda _: {
            'color': '#000000',
            'weight': 3,
            'opacity': 0.85,
            'dashArray': '6, 4'
        }
    ).add_to(m)


def add_road_layer(m, road_path):
    # Add road centerline (always visible, red)
    # @param m: folium Map
    # @param road_path: Path to road shapefile/geojson
    # @returns: GeoDataFrame in WGS84 (also used for transformer setup)
    
    road_wgs = gpd.read_file(road_path).to_crs('EPSG:4326')
    folium.GeoJson(
        road_wgs.to_json(),
        name='Road (centerline)',
        style_function=lambda _: {
            'color': '#e60000',
            'weight': 3,
            'opacity': 1.0
        }
    ).add_to(m)
    return road_wgs


def add_forest_layer(m, forest_path):
    # Add forest resistance area (off by default)
    # @param m: folium Map
    # @param forest_path: Path to forest geojson
    # @returns: None
    
    if not Path(forest_path).exists():
        return
    
    forest_wgs = gpd.read_file(forest_path).to_crs('EPSG:4326')
    folium.GeoJson(
        forest_wgs.to_json(),
        name='Forest (resistance)',
        style_function=lambda _: {
            'color': '#27ae60',
            'weight': 1,
            'fillOpacity': 0.25
        },
        show=False
    ).add_to(m)


def add_impact_points(m, impact_points, source_crs):
    # Add clickable impact-point markers
    # @param m: folium Map
    # @param impact_points: list of dicts (xy, depth_m, velocity_ms, chainage_m)
    # @param source_crs: CRS of the impact_points coordinates
    # @returns: None
    # note:
    #   - Coords come in projected CRS; transform to WGS84 for Leaflet
    
    if not impact_points:
        return
    
    # Coordinate transformer (projected -> WGS84)
    transformer = Transformer.from_crs(source_crs, 'EPSG:4326', always_xy=True)
    
    # FeatureGroup so all markers toggle together
    fg = folium.FeatureGroup(name='Impact points', show=True)
    
    for pt in impact_points:
        if not pt['xy']:
            continue
        
        # Transform coords
        lon, lat = transformer.transform(pt['xy'][0], pt['xy'][1])
        
        # Popup HTML
        popup_html = (
            f"<b>{pt['scenario']}</b><br>"
            f"Flow depth: {pt['depth_m']:.2f} m<br>"
            f"Velocity: {pt['velocity_ms']:.2f} m/s<br>"
            f"Chainage: {pt['chainage_m']:.0f} m"
        )
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color='#e60000',
            weight=2,
            fill=True,
            fill_color='white',
            fill_opacity=1.0,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(fg)
    
    fg.add_to(m)


def add_per_scenario_layers(m, scenarios, show_default=False):
    # Add per-scenario release/flowpath/reach layers (one toggle per scenario)
    # @param m: folium Map
    # @param scenarios: list of scenario dicts (with scenario_id, release_geojson,
    #                    scenario_dir for finding outputs)
    # @param show_default: bool - whether to show all scenarios by default
    # @returns: None
    # note:
    #   - Each scenario gets ONE FeatureGroup containing its release, flowpath,
    #     and reach polygon, all in the same color. This way the user can
    #     toggle a single scenario and see all of its geometries together.
    #   - Color is cycled from SCENARIO_COLORS by scenario index
    
    for i, sc in enumerate(scenarios):
        sid = sc['scenario_id']
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]
        
        # FeatureGroup that holds all this scenario's geometries
        fg = folium.FeatureGroup(name=f"{sid}", show=show_default)
        added_anything = False
        
        # Release polygon
        try:
            rel_gdf = gpd.read_file(sc['release_geojson']).to_crs('EPSG:4326')
            folium.GeoJson(
                rel_gdf.to_json(),
                style_function=lambda _, c=color: {
                    'color': c,
                    'weight': 2,
                    'fillColor': c,
                    'fillOpacity': 0.35
                },
                tooltip=f"{sid} release"
            ).add_to(fg)
            added_anything = True
        except Exception as e:
            logging.debug(f"Skip release for {sid}: {e}")
        
        # Flowpath (steepest descent) - written to outputs/flowpath.geojson
        flow_path = Path(sc['scenario_dir']) / 'outputs' / 'flowpath.geojson'
        if flow_path.exists():
            try:
                flow_gdf = gpd.read_file(flow_path).to_crs('EPSG:4326')
                folium.GeoJson(
                    flow_gdf.to_json(),
                    style_function=lambda _, c=color: {
                        'color': c,
                        'weight': 2,
                        'opacity': 0.85,
                        'dashArray': '4, 3'
                    },
                    tooltip=f"{sid} flowpath"
                ).add_to(fg)
                added_anything = True
            except Exception as e:
                logging.debug(f"Skip flowpath for {sid}: {e}")
        
        # Reach polygon - written to outputs/reach_polygon.geojson
        reach_path = Path(sc['scenario_dir']) / 'outputs' / 'reach_polygon.geojson'
        if reach_path.exists():
            try:
                reach_gdf = gpd.read_file(reach_path).to_crs('EPSG:4326')
                folium.GeoJson(
                    reach_gdf.to_json(),
                    style_function=lambda _, c=color: {
                        'color': c,
                        'weight': 1,
                        'fillColor': c,
                        'fillOpacity': 0.15
                    },
                    tooltip=f"{sid} reach"
                ).add_to(fg)
                added_anything = True
            except Exception as e:
                logging.debug(f"Skip reach for {sid}: {e}")
        
        if added_anything:
            fg.add_to(m)


def add_legend(m, scenarios):
    # Add a small HTML legend in the corner explaining colors
    # @param m: folium Map
    # @param scenarios: list of scenario dicts
    # @returns: None
    # note:
    #   - Uses folium's MacroElement to inject a static HTML div
    
    from branca.element import MacroElement, Template
    
    # Build per-scenario rows (limit to first 10 for legend space)
    n_show = min(10, len(scenarios))
    swatches = ""
    for i in range(n_show):
        sid = scenarios[i]['scenario_id']
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]
        swatches += (
            f'<div><span style="background:{color};width:14px;height:14px;'
            f'display:inline-block;border:1px solid #444;margin-right:6px;'
            f'vertical-align:middle;"></span>{sid}</div>'
        )
    if len(scenarios) > n_show:
        swatches += f'<div><em>+ {len(scenarios) - n_show} more...</em></div>'
    
    # Build legend HTML
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed; bottom: 30px; left: 12px; z-index: 9999;
        background: rgba(255,255,255,0.92); padding: 10px 12px;
        border: 1px solid #888; border-radius: 4px; font-size: 12px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-height: 300px; overflow-y: auto;">
      <div style="font-weight:bold; margin-bottom:6px;">Scenarios</div>
      {swatches}
      <div style="margin-top:8px; padding-top:6px; border-top:1px solid #ccc;">
        <div><span style="background:#e60000;width:14px;height:3px;
             display:inline-block;vertical-align:middle;margin-right:6px;"></span>Road</div>
        <div><span style="background:#c0392b;width:14px;height:14px;
             display:inline-block;border:2px solid #c0392b;background:transparent;
             margin-right:6px;vertical-align:middle;"></span>Envelope</div>
      </div>
    </div>
    {{% endmacro %}}
    """
    
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)


def build_viewer(outputs_dir, scenarios, shared_paths, impact_points, config):
    # Build viewer.html with all toggleable layers
    # @param outputs_dir: Path to ensemble outputs directory
    # @param scenarios: list of scenario dicts (each with scenario_id,
    #                    release_geojson, scenario_dir)
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param impact_points: list of impact point dicts
    # @param config: top-level config dict
    # @returns: Path to written viewer.html
    
    outputs_dir = Path(outputs_dir)
    prob_path = outputs_dir / 'probability.tif'
    envelope_path = outputs_dir / 'envelope.geojson'
    contours_path = outputs_dir / 'contours.geojson'
    mlp_path = outputs_dir / 'most_likely_path.geojson'
    
    # Determine map center from envelope bounds (fall back to road if empty)
    env_wgs = gpd.read_file(envelope_path).to_crs('EPSG:4326')
    if len(env_wgs) > 0:
        bounds = env_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    else:
        # Fall back to road center
        road_wgs = gpd.read_file(shared_paths['road']).to_crs('EPSG:4326')
        b = road_wgs.total_bounds
        center = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
    
    # Initialize Leaflet map with NO basemap tiles - hillshade will be added
    # below. Setting tiles=None avoids fetching from OSM (which blocks
    # requests without a Referer header) and keeps the viewer offline-safe.
    m = folium.Map(
        location=center,
        zoom_start=15,
        tiles=None,
        control_scale=True
    )
    
    # Hillshade DEM basemap (replaces external tiles entirely)
    add_hillshade_basemap(m, shared_paths['dem'], outputs_dir)
    
    # Probability heatmap on top of hillshade
    add_probability_overlay(
        m, prob_path, outputs_dir,
        colormap=config['output']['probability_colormap']
    )
    
    # Contour outlines (one layer per level)
    add_contour_layers(m, contours_path)
    
    # Envelope outline
    add_envelope_layer(
        m, envelope_path,
        threshold=config['probability']['envelope_min_probability']
    )
    
    # Most-likely path (medial axis at P >= 0.5)
    add_most_likely_path_layer(m, mlp_path)
    
    # Road (always visible)
    add_road_layer(m, shared_paths['road'])
    
    # Forest (off by default)
    add_forest_layer(m, shared_paths['forest'])
    
    # Impact points (clickable, on by default)
    # Get source CRS from the road file (all inputs share CRS)
    source_crs = gpd.read_file(shared_paths['road']).crs
    add_impact_points(m, impact_points, source_crs)
    
    # Per-scenario release + flowpath + reach polygon (off by default,
    # since N=20 layers all on at once would be visually overwhelming)
    add_per_scenario_layers(m, scenarios, show_default=False)
    
    # Static HTML legend (scenario colors + key annotations)
    add_legend(m, scenarios)
    
    # Layer control panel
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save viewer.html
    out_path = outputs_dir / 'viewer.html'
    m.save(str(out_path))
    logging.info(f"Wrote viewer: {out_path}")
    
    return out_path
