"""
Avalanche Runout Envelope - HTML Report Builder

Author: Valerie Foley
Last Updated: 4/24/2026

Description:
    Renders a static HTML report from the ensemble outputs:
        - Headline: N of M scenarios reached the road, with Wilson 95% CI
        - Probability map figure (raster + road + contours + impact points)
        - Chainage histogram of road impact points
        - Runout distance histogram (stacked: reached vs missed)
        - Parameter sensitivity scatter (runout vs mu, xi, rho, volume)
        - Per-scenario table sorted by runout distance
        - Methodology + references + limitations sections
    
    Plots are embedded as base64-encoded PNGs so the report is a single
    self-contained file (no external image references).
    
    No CLI. Imported by avalanche_runout.py.

Requirements:
    - matplotlib, jinja2, geopandas, rasterio, pandas
"""

# --------- Load Libraries --------
import base64
import logging
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd
from jinja2 import Template

import avalanche_helpers as h




def fig_to_base64(fig, dpi=120):
    # Encode a matplotlib figure as base64 PNG (for inline HTML embedding)
    # @param fig: matplotlib Figure
    # @param dpi: int - rendering DPI
    # @returns: str - base64-encoded PNG
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_probability_map(probability_path, road_path, contours_path,
                         impact_points, dpi=120):
    # Plot the probability raster with road, contours, and impact points
    # @param probability_path: Path to probability raster
    # @param road_path: Path to road shapefile/geojson
    # @param contours_path: Path to contours geojson
    # @param impact_points: list of dicts with 'xy' field
    # @param dpi: int - rendering DPI
    # @returns: str - base64 PNG
    
    # Load probability raster
    prob, meta = h.read_raster(probability_path)
    
    # Compute extent from transform + shape for imshow
    t = meta['transform']
    left, top = t.c, t.f
    right = left + meta['shape'][1] * t.a
    bottom = top + meta['shape'][0] * t.e
    extent = [left, right, bottom, top]
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Probability heatmap (mask out nodata = -1)
    masked = np.where(prob >= 0, prob, np.nan)
    im = ax.imshow(masked, extent=extent, origin='upper',
                   cmap='viridis', vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(im, ax=ax, label='P(reach)')
    
    # Road overlay
    road_gdf = gpd.read_file(road_path)
    road_gdf.plot(ax=ax, color='red', linewidth=2, label='Road')
    
    # Contour outlines (if any)
    try:
        contours = gpd.read_file(contours_path)
        for level in sorted(contours['level'].unique()):
            subset = contours[contours['level'] == level]
            subset.boundary.plot(
                ax=ax, linewidth=0.6 + level, edgecolor='black',
                alpha=0.5 + level * 0.4
            )
    except Exception as e:
        logging.debug(f"No contours drawn: {e}")
    
    # Impact points
    if impact_points:
        xs = [p['xy'][0] for p in impact_points if p['xy']]
        ys = [p['xy'][1] for p in impact_points if p['xy']]
        ax.scatter(xs, ys, c='white', edgecolors='red', s=80, zorder=5,
                   label=f"Impact points (n={len(xs)})")
    
    # Labels and finish
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Runout probability across ensemble')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    return fig_to_base64(fig, dpi)


def plot_chainage_histogram(impact_points, dpi=120):
    # Histogram of road chainage at impact points
    # @param impact_points: list of dicts with 'chainage_m' field
    # @param dpi: int - rendering DPI
    # @returns: str (base64 PNG) or None if no impact points
    
    chainages = [p['chainage_m'] for p in impact_points
                 if p['chainage_m'] is not None]
    
    if not chainages:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(chainages, bins=20, color='steelblue', edgecolor='black')
    ax.set_xlabel('Chainage along road (m)')
    ax.set_ylabel('Number of scenarios reaching')
    ax.set_title('Distribution of road impact points')
    return fig_to_base64(fig, dpi)


def plot_runout_histogram(df, dpi=120):
    # Stacked histogram of max runout, split by reached/missed
    # @param df: DataFrame from scenarios_summary.csv
    # @param dpi: int - rendering DPI
    # @returns: str - base64 PNG
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Split by reach status
    reached = df[df['reached_road'] == True]
    missed = df[df['reached_road'] == False]
    
    ax.hist(
        [reached['runout_max_m'], missed['runout_max_m']],
        bins=15, stacked=True,
        label=['Reached road', 'Did not reach'],
        color=['#c0392b', '#3498db'],
        edgecolor='black'
    )
    ax.set_xlabel('Max runout distance (m)')
    ax.set_ylabel('Number of scenarios')
    ax.set_title('Runout distribution')
    ax.legend()
    
    return fig_to_base64(fig, dpi)


def plot_parameter_sensitivity(df, dpi=120):
    # 4-panel scatter: runout vs each input parameter
    # @param df: DataFrame from scenarios_summary.csv
    # @param dpi: int - rendering DPI
    # @returns: str - base64 PNG
    
    # Parameter columns to plot, with display labels
    params = [
        ('mu', 'Friction μ'),
        ('xi', 'Turbulent ξ (m/s²)'),
        ('rho_kgm3', 'Release density (kg/m³)'),
        ('release_volume_m3', 'Release volume (m³)')
    ]
    
    fig, axes = plt.subplots(1, len(params), figsize=(16, 4))
    
    for ax, (key, label) in zip(axes, params):
        # Skip params with no data
        if key not in df.columns or df[key].isna().all():
            ax.set_visible(False)
            continue
        
        # Color by reach status
        colors = ['#c0392b' if r else '#3498db' for r in df['reached_road']]
        ax.scatter(df[key], df['runout_max_m'],
                   c=colors, s=60, edgecolor='black')
        ax.set_xlabel(label)
        ax.set_ylabel('Runout (m)')
    
    fig.suptitle('Parameter sensitivity (red = reached road)')
    
    # Legend (manual since we used per-point colors)
    legend_handles = [
        Patch(facecolor='#c0392b', label='Reached road'),
        Patch(facecolor='#3498db', label='Did not reach')
    ]
    fig.legend(handles=legend_handles, loc='upper right')
    
    return fig_to_base64(fig, dpi)




# Single Jinja2 template for the whole report. Plots embedded as base64 PNGs.
REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Avalanche Runout Envelope - Report</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
    max-width: 1100px;
    margin: 2em auto;
    padding: 0 1.5em 3em;
    color: #1f2933;
    line-height: 1.55;
    background: #fafafa;
  }
  h1 {
    border-bottom: 3px solid #c0392b;
    padding-bottom: 0.3em;
    margin-bottom: 0.2em;
    font-weight: 600;
  }
  h2 {
    margin-top: 2em;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.25em;
    font-weight: 600;
    color: #2c3e50;
  }
  .meta {
    color: #7f8c8d;
    font-size: 0.85em;
    margin-bottom: 2em;
  }
  .headline {
    background: linear-gradient(to right, #fdf2f0, #fafafa);
    border-left: 4px solid #c0392b;
    padding: 1.2em 1.5em;
    font-size: 1.05em;
    margin: 1.5em 0;
    border-radius: 0 4px 4px 0;
  }
  .headline .big {
    font-size: 2.2em;
    font-weight: 700;
    color: #c0392b;
    line-height: 1;
  }
  .headline p {
    margin: 0.5em 0;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.88em;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  th, td {
    border: 1px solid #e1e4e8;
    padding: 0.5em 0.7em;
    text-align: right;
  }
  th {
    background: #2c3e50;
    color: white;
    font-weight: 500;
    border-color: #34495e;
  }
  td.id, th.id {
    text-align: left;
    font-family: 'SF Mono', Monaco, Menlo, monospace;
    font-size: 0.92em;
  }
  .reached-yes { background: #fdf2f0; }
  .reached-yes td.id { color: #c0392b; font-weight: 600; }
  tr:hover { background: #f5f5f5; }
  img {
    max-width: 100%;
    height: auto;
    margin: 1em 0;
    border: 1px solid #e1e4e8;
    border-radius: 3px;
    background: white;
  }
  .caveat {
    background: #fef9e7;
    border-left: 4px solid #f1c40f;
    padding: 1em 1.2em;
    border-radius: 0 4px 4px 0;
  }
  .caveat ul {
    margin: 0.5em 0;
    padding-left: 1.5em;
  }
  .caveat li { margin: 0.6em 0; }
  code {
    background: #ecf0f1;
    padding: 0.1em 0.4em;
    border-radius: 3px;
    font-size: 0.92em;
    font-family: 'SF Mono', Monaco, Menlo, monospace;
  }
  .ref {
    font-size: 0.88em;
    color: #555;
    line-height: 1.7;
  }
</style>
</head>
<body>
<h1>Avalanche Runout Envelope</h1>
<p class="meta">Generated {{ timestamp }} &middot;
   Wrapper v{{ version }} &middot;
   {{ n_scenarios }} scenarios</p>

<div class="headline">
  <p><span class="big">{{ n_reached }} / {{ n_scenarios }}</span>
     scenarios reach the road</p>
  <p>Fraction reached: <b>{{ "%.1f"|format(fraction * 100) }}%</b><br>
     Wilson 95% CI: [<b>{{ "%.1f"|format(ci_low * 100) }}%</b>,
                     <b>{{ "%.1f"|format(ci_high * 100) }}%</b>]</p>
  {% if nearest_miss %}
  <p>Nearest miss: <code>{{ nearest_miss.scenario }}</code> stopped
     <b>{{ "%.1f"|format(nearest_miss.distance_short_m) }} m</b>
     short of the road.</p>
  {% endif %}
</div>

<h2>Probability map</h2>
<img src="data:image/png;base64,{{ map_b64 }}" alt="Probability map">

<h2>Road impact distribution</h2>
{% if chainage_b64 %}
<img src="data:image/png;base64,{{ chainage_b64 }}" alt="Chainage histogram">
<p class="meta">Chainage origin: far end of road polyline from release centroid.</p>
{% else %}
<p><em>No scenarios reached the road.</em></p>
{% endif %}

<h2>Runout distance distribution</h2>
<img src="data:image/png;base64,{{ runout_b64 }}" alt="Runout histogram">

<h2>Parameter sensitivity</h2>
<img src="data:image/png;base64,{{ sens_b64 }}" alt="Parameter sensitivity">

<h2>Per-scenario results</h2>
<table>
<thead>
<tr>
  <th class="id">Scenario</th>
  <th>μ</th><th>ξ</th><th>ρ</th><th>Vol (m³)</th>
  <th>Runout (m)</th>
  <th>Peak v (m/s)</th>
  <th>Peak depth (m)</th>
  <th>Reached</th>
  <th>Dist to road (m)</th>
  <th>Depth @ road</th>
  <th>v @ road</th>
</tr>
</thead>
<tbody>
{% for row in scenarios %}
<tr class="{{ 'reached-yes' if row.reached_road else 'reached-no' }}">
  <td class="id">{{ row.scenario_id }}</td>
  <td>{{ row.mu }}</td>
  <td>{{ row.xi }}</td>
  <td>{{ row.rho_kgm3 }}</td>
  <td>{{ row.release_volume_m3 or "—" }}</td>
  <td>{{ row.runout_max_m }}</td>
  <td>{{ row.peak_velocity_ms }}</td>
  <td>{{ row.peak_flow_depth_m }}</td>
  <td>{{ "✓" if row.reached_road else "—" }}</td>
  <td>{{ row.distance_to_road_m }}</td>
  <td>{{ row.depth_at_road_m or "—" }}</td>
  <td>{{ row.velocity_at_road_ms or "—" }}</td>
</tr>
{% endfor %}
</tbody>
</table>

<h2>Methodology</h2>
<p>Each scenario runs through AvaFrame com1DFA (Voellmy dense-flow DFA model)
with per-scenario friction (μ, ξ) and release density (ρ) from its
<code>params.json</code>. Entrainment is handled as a constant-depth layer
(<b>{{ ent_depth }} m</b>) applied over an auto-detected α–β corridor
downslope of the release. Forest is represented as a resistance area.
Reach masks (flow depth ≥ <b>{{ reach_threshold }} m</b>) are stacked
cell-by-cell to produce the probability raster.</p>

<p>The probability value at any cell is the fraction of the {{ n_scenarios }}
scenarios whose reach mask includes that cell. <b>This is a conditional
probability</b>: P(reach | this set of plausible release scenarios). It is
not an annual or return-period probability and cannot be interpreted as
such without external frequency information.</p>

<h2>Limitations</h2>
<div class="caveat">
<ul>
  <li><b>Small ensemble (n = {{ n_scenarios }}).</b> With 20 scenarios,
      probability granularity is 5% and CIs on low-probability contours are wide.</li>
  <li><b>Entrainment as constant depth ({{ ent_depth }} m).</b>
      Defensible starting value for dry surface slab
      (Vera Valero et al. 2016; Bartelt et al. 2018 Vallée de la Sionne
      calibrations), but not spatially varying. Upgrade path: layered
      snowpack model output.</li>
  <li><b>Friction is a placeholder, not derived from snowpack state.</b>
      The Voellmy friction parameters (μ, ξ) in <code>params.json</code>
      are currently identical across all scenarios — they are not yet
      derived from the upstream snowpack model. The values used
      (μ={{ scenarios[0].mu if scenarios else "—" }},
       ξ={{ scenarios[0].xi if scenarios else "—" }})
      are on the lower end of typical dry-snow Voellmy ranges
      (μ ~ 0.15–0.30, ξ ~ 1000–4000), which biases toward longer runouts
      — a conservative direction for road-impact analysis.
      Upgrade path: link μ/ξ to per-scenario snowpack state (density,
      temperature, layer structure).</li>
  <li><b>Road buffer = {{ road_buffer }} m.</b> Road is treated as
      centerline buffered by this distance. Three-lane centerlines with
      different road widths warrant per-segment half-widths
      (future upgrade).</li>
  <li><b>α–β fallback.</b> When α outside [{{ ab_min }}°, {{ ab_max }}°],
      entrainment area reverts to flow-accumulation buffer. Logged per
      scenario in <code>scenarios_summary.csv</code>.</li>
</ul>
</div>

<h2>References</h2>
<p class="ref">
Bevilacqua, A. et al. (2019). <em>Statistical theory of probabilistic hazard
maps.</em> NHESS 19, 1347.<br>
Hungr, O. (2014). <em>2014 Canadian Geotechnical Colloquium: Landslide runout
analysis.</em> Can. Geotech. J.<br>
Zhang, Y. et al. (2024). <em>Uncertainty characterization, propagation, and
evaluation in debris flow run-out hazard assessment.</em> Landslides.<br>
Vera Valero, C. et al. (2016). <em>Release temperature, snow-cover entrainment
and thermal flow regime of snow avalanches.</em> J. Glaciol.<br>
Bartelt, P. et al. (2018). <em>Snow entrainment: avalanche interaction with
an erodible substrate.</em> ISSW 2018.<br>
McClung, D.M. & Lied, K. (1987). <em>Statistical and geometrical definition
of snow avalanche runout.</em> Cold Reg. Sci. Technol.<br>
AvaFrame documentation (<a href="https://docs.avaframe.org">docs.avaframe.org</a>).
</p>

</body>
</html>
"""




def build_report(outputs_dir, scenario_summaries, road_impact, shared_paths,
                 config, wrapper_version):
    # Render the full HTML report and save to outputs_dir/report.html
    # @param outputs_dir: Path to ensemble outputs directory
    # @param scenario_summaries: list of summary dicts (one per scenario)
    # @param road_impact: dict from build_road_impact_summary
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param config: top-level config dict
    # @param wrapper_version: str - version string for the report header
    # @returns: Path to written HTML file
    
    outputs_dir = Path(outputs_dir)
    
    # Read scenarios CSV (already written by stack module)
    csv_path = outputs_dir / 'scenarios_summary.csv'
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    
    dpi = config['output']['report_plots_dpi']
    
    # Build all four plots
    map_b64 = plot_probability_map(
        probability_path=outputs_dir / 'probability.tif',
        road_path=shared_paths['road'],
        contours_path=outputs_dir / 'contours.geojson',
        impact_points=road_impact['impact_points'],
        dpi=dpi
    )
    chainage_b64 = plot_chainage_histogram(
        road_impact['impact_points'], dpi=dpi
    )
    runout_b64 = plot_runout_histogram(df, dpi=dpi)
    sens_b64 = plot_parameter_sensitivity(df, dpi=dpi)
    
    # Render template
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=wrapper_version,
        n_scenarios=road_impact['n_scenarios'],
        n_reached=road_impact['n_reached'],
        fraction=road_impact['fraction_reached'],
        ci_low=road_impact['wilson_95_ci'][0],
        ci_high=road_impact['wilson_95_ci'][1],
        nearest_miss=road_impact['nearest_miss'],
        scenarios=df.to_dict(orient='records'),
        map_b64=map_b64,
        chainage_b64=chainage_b64,
        runout_b64=runout_b64,
        sens_b64=sens_b64,
        ent_depth=config['entrainment']['depth_m'],
        reach_threshold=config['reach']['flow_depth_threshold_m'],
        road_buffer=config['road']['buffer_m'],
        ab_min=config['entrainment']['auto_detect']['alpha_beta']['min_alpha_deg'],
        ab_max=config['entrainment']['auto_detect']['alpha_beta']['max_alpha_deg']
    )
    
    # Write to disk
    out_path = outputs_dir / 'report.html'
    out_path.write_text(html)
    logging.info(f"Wrote report: {out_path}")
    
    return out_path
