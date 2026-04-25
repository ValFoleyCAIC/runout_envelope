# Avalanche Runout Envelope

A wrapper around AvaFrame's com1DFA dense-flow avalanche kernel that runs an ensemble of plausible release scenarios for a single path, stacks the per-scenario reach masks into a probability surface, and answers the operational question: **does the slide reach the road, and how often?**

## What it does

For one avalanche path, the pipeline takes:

- A digital elevation model
- A road centerline
- A forest polygon (resistance area)
- N release scenarios, each with its own release polygon, snow depth raster, density, and friction parameters (μ, ξ)

It runs each scenario through AvaFrame com1DFA in parallel, builds a binary "did the flow get here" reach mask per scenario, and stacks those masks cell-by-cell to produce a conditional probability surface P(reach | this set of plausible releases). Outputs include the probability raster, an outer envelope polygon, contour polygons at each P level, a most-likely-path centerline (medial axis at P ≥ 0.5), per-scenario impact-point statistics, and a self-contained Leaflet viewer rendered over a hillshade basemap derived from the project DEM.

## How to run

```bash
# Validate inputs only
python avalanche_runout.py --dry-run

# Full run (parallel by default)
python avalanche_runout.py

# Run a single scenario for debugging
python avalanche_runout.py --scenario scenario_001 --no-parallel --verbose
```

All paths and parameters live in `config.yaml`. Per-scenario parameters (μ, ξ, ρ, release polygon, release depth) live under each `scenarios/scenario_NNN/` folder.

## Methods

### Dynamics: AvaFrame com1DFA

The dense-flow kernel `com1DFA` is a thickness-integrated SPH-style particle–grid model (Tonnel et al. 2023). The default friction model used here is **Voellmy** (Voellmy 1955), in which the basal shear stress combines a Mohr-Coulomb dry-friction term with a turbulent term:

$$\tau^{(b)} = \mu \sigma^{(b)} + \frac{g}{\xi} \rho \bar{u}^2$$

The Voellmy parameters μ and ξ are passed in per-scenario from `params.json`. Forest is implemented as a com1DFA *resistance area*, particles passing through it experience added drag (Christen et al. 2010). com1DFA also supports samosAT-family friction models (Sampl & Zwinger 2004), but those use internal volume-keyed Austrian calibrations that ignore externally-supplied μ/ξ, so we chose to use Voellmy.

Reach is defined as flow depth ≥ 0.1 m, following the AvaFrame edge convention. An optional minimum velocity gate is available in the config.

### Entrainment: α–β corridor with flow-accumulation fallback

Each scenario's entrainment polygon is auto-detected by tracing a steepest-descent flowline downhill from the release centroid (D8, terminating at <3° rolling slope) and computing an α-angle stopping point following the **McClung & Lied (1987)** statistical runout model — β = first point where the slope-from-start drops below 10°, α = β − 5°. The flowline buffered to a corridor and truncated at α gives the entrainment polygon. If α falls outside the plausible range [15°, 45°], the pipeline falls back to a flow-accumulation buffer (Horton et al. 2013).

Entrainment depth is currently a constant (0.15 m default). Selected from dry surface-slab releases per the Vallée de la Sionne calibrations of **Vera Valero et al. (2016)** and **Bartelt et al. (2018)**. A to-do is to drive entrainment depth from snowpack-state output.

### Probability stacking

Each scenario produces a binary reach raster Bᵢ(x), and the ensemble probability surface is:

$$P(x) = \frac{1}{N} \sum_{i=1}^{N} B_i(x)$$

This is the indicator-function definition formalized by **Hyman, Bevilacqua, & Bursik (2019)** for probabilistic hazard maps. The probability at each cell is interpretable as P(reach | this set of plausible scenarios), explicitly **conditional**, not annual. Frequency information would have to come from outside the model. AvaFrame's `ana4Stats.probAna` module follows the same stacking convention.

The outer envelope is the polygonization of P ≥ 0.05 (configurable). Contours are drawn at the levels in `config.yaml` (default: 0.05, 0.25, 0.50, 0.75, 1.00). The most-likely-path is a skeleton (medial axis via Zhang-Suen thinning) of the P ≥ 0.5 region.

### Confidence intervals on small ensembles

For "N of M scenarios reached the road" we report a **Wilson score 95% CI** rather than a normal-approximation one — Wilson is well-behaved at small N and at edge cases (k=0 or k=N) where the normal interval degenerates. With N=20, our probability granularity is 5%, and CI half-widths on low-P contours are wide; the reported CIs make this explicit.

### Viewer

The viewer is a single self-contained `viewer.html`. Folium is initialized with `tiles=None`, and the basemap is a hillshade computed from the project DEM via Horn's algorithm (matplotlib `LightSource`, NW sun at 45°), reprojected to WGS84 and embedded as a base64 PNG image overlay. The probability raster sits on top with sqrt-scaled alpha. Per-scenario release polygons, steepest-descent flowpaths, and reach polygons are each toggleable.

## References

### Avalanche dynamics

- Tonnel, M., Wirbel, A., Oesterle, F., & Fischer, J.-T. (2023). AvaFrame com1DFA (v1.3): a thickness-integrated computational avalanche module — theory, numerics, and testing. *Geosci. Model Dev.* 16, 7013–7035.
- Voellmy, A. (1955). Über die Zerstörungskraft von Lawinen. *Schweizerische Bauzeitung* 73.
- Sampl, P., & Zwinger, T. (2004). Avalanche simulation with SAMOS. *Annals of Glaciology* 38, 393–398.
- Christen, M., Kowalski, J., & Bartelt, P. (2010). RAMMS: numerical simulation of dense snow avalanches in three-dimensional terrain. *Cold Reg. Sci. Technol.* 63, 1–14.
- Salm, B. (2004). A short and personal history of snow avalanche dynamics. *Cold Reg. Sci. Technol.* 39.

### Runout statistics & entrainment

- McClung, D. M., & Lied, K. (1987). Statistical and geometrical definition of snow avalanche runout. *Cold Reg. Sci. Technol.* 13, 107–119.
- Vera Valero, C., Wikstroem Jones, K., Bühler, Y., & Bartelt, P. (2016). Release temperature, snow-cover entrainment and the thermal flow regime of snow avalanches. *J. Glaciol.* 62, 277–288.
- Bartelt, P., Buser, O., Vera Valero, C., & Bühler, Y. (2018). Configurational energy and the formation of mixed flowing/powder snow and ice avalanches. *Annals of Glaciology* 59.
- Horton, P., Jaboyedoff, M., Rudaz, B., & Zimmermann, M. (2013). Flow-R, a model for susceptibility mapping of debris flows and other gravitational hazards at a regional scale. *Nat. Hazards Earth Syst. Sci.* 13, 869–885.

### Probabilistic hazard mapping (cross-field)

- **Hyman, D. M., Bevilacqua, A., & Bursik, M. I. (2019).** Statistical theory of probabilistic hazard maps: a probability distribution for the hazard boundary location. *Nat. Hazards Earth Syst. Sci.* 19, 1347–1363. — Foundational PHM theory; our stacking method is the indicator-function definition from this paper, applied to avalanches.
- Bevilacqua, A., Patra, A. K., Bursik, M. I., et al. (2019). Probabilistic forecasting of plausible debris flows from Nevado de Colima (Mexico) using data from the Atenquique 1955 debris flow. *Nat. Hazards Earth Syst. Sci.* 19, 791–820. — Same stacking approach for volcaniclastic debris flows.
- Rutarindwa, R., Spiller, E. T., Bevilacqua, A., Bursik, M. I., & Patra, A. K. (2019). Dynamic probabilistic hazard mapping in the Long Valley volcanic region. *J. Geophys. Res. Solid Earth* 124, 9600–9621. — Statistical surrogates for expensive physical models, an upgrade direction worth tracking.
- Aravena, A., Tadini, A., Bevilacqua, A., et al. (2024). Probabilistic, scenario-based hazard assessment for pyroclastic density currents at Tungurahua volcano. *Bull. Volcanol.* 86. — Scenario-based ensemble strategy, same conceptual framework as ours.

### Landslide & rock-avalanche runout

- McDougall, S. (2017). 2014 Canadian Geotechnical Colloquium: Landslide runout analysis — current practice and challenges. *Can. Geotech. J.* 54, 605–620. — Survey of the equivalent-fluid calibration approach we use; explicitly recommends probabilistic calibration over single deterministic runs.
- Hungr, O. (1995). A model for the runout analysis of rapid flow slides, debris flows, and avalanches. *Can. Geotech. J.* 32, 610–623.
- McDougall, S., & Hungr, O. (2004). A model for the analysis of rapid landslide motion across three-dimensional terrain. *Can. Geotech. J.* 41, 1084–1097.
- Mergili, M., Fischer, J.-T., Krenn, J., & Pudasaini, S. P. (2017). r.avaflow v1, an advanced open-source computational framework for the propagation and interaction of two-phase mass flows. *Geosci. Model Dev.* 10, 553–569.

### Debris flow & flood

- Zhang, Y., Yu, B., Zhu, Y., et al. (2024). Uncertainty characterization, propagation, and evaluation in debris-flow run-out hazard assessment. *Landslides* 21, 2841–2860. — Probabilistic envelope construction for FLO-2D debris flows; same logic as ours but driven by a different physical model.
- O'Brien, J. S., Julien, P. Y., & Fullerton, W. T. (1993). Two-dimensional water flood and mudflow simulation. *J. Hydraul. Eng.* 119, 244–261. — FLO-2D foundation paper.
- Iverson, R. M. (1997). The physics of debris flows. *Rev. Geophys.* 35, 245–296.

### Statistics & numerics

- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *J. Am. Stat. Assoc.* 22, 209–212. — Wilson score interval used for our N-of-M binomial CIs.
- Horn, B. K. P. (1981). Hill shading and the reflectance map. *Proc. IEEE* 69, 14–47. — Algorithm used for the viewer's hillshade basemap.
- Zhang, T. Y., & Suen, C. Y. (1984). A fast parallel algorithm for thinning digital patterns. *Communications of the ACM* 27, 236–239. — Skeletonization for the most-likely-path layer.

## Where this approach sits in the broader hazard-modelling landscape

The "run an ensemble, stack the indicator functions, polygonize the result" pattern is not specific to snow avalanches. It's the de facto standard for hazard maps of geophysical mass flows when the underlying physical model is too complex to invert analytically and parameter uncertainty matters more than physical-process detail:

- **Volcanology** uses it for pyroclastic density currents (Bevilacqua et al. 2017; Aravena et al. 2024; Tierz et al. 2018), lava flows (Connor et al. 2012; Gallant et al. 2018), and lahars.
- **Debris-flow/landslide engineering** uses it for FLO-2D ensembles (Zhang et al. 2024) and DAN3D Monte Carlo (McDougall 2017).
- **Tsunami hazard** uses run-up exceedance thresholds in PTHA (Geist & Parsons 2006; Grezio et al. 2017).
- **Seismology** uses ground-shaking thresholds in PSHA.
- **Flood mapping** uses inundation indicator stacking across hydrograph ensembles.

The shared mathematical structure — propagate a parameter PDF through a deterministic mapping, integrate the indicator against the PDF to get cell-wise impact probability — is detailed in **Hyman, Bevilacqua, & Bursik (2019)**, which is the citation to reach for when defending the methodology to a reviewer who's used to one of the above adjacent fields. Each community calibrates its own "equivalent fluid" parameters to its own dataset of past events, but the probabilistic-stacking machinery on top is identical.

## Notes on current iteration

- **Small ensemble (N = 20).** Probability granularity is 5%; CIs on low-P contours are wide. Larger ensembles or surrogate-model emulators (Rutarindwa et al. 2019) would be the upgrade path. We chose a small ensemble for initial pipeline testing. 
- **Friction is a placeholder.** Per-scenario μ/ξ are currently identical across scenarios. Values bias toward longer runouts.
- **Entrainment is constant-depth.** Spatially varying entrainment from snowpack output is the next upgrade.
- **Conditional probability only.** Outputs are P(reach | scenarios).
- **Road buffer is uniform.** Current road layer is the centerline of the road sourced from CDOT. We set a 1m buffer on the centerline. Future versions should take in a road width attribute or dynamically adjust the buffer based on the number of lanes if information is available. 
