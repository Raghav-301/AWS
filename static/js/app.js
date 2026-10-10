// ─── PawanSthal Application Logic ───

let map;
let existingTurbinesLayer = L.layerGroup();
let repoweredTurbinesLayer = L.layerGroup();
let substationsLayer = L.layerGroup();
let windHeatmapLayer = L.layerGroup();

let currentSelectedZone = "Zone-M4";
let clustersData = [];

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadData();
  setupEventListeners();
});

function initMap() {
  // Center on Muppandal Wind Farm & Aralvaimozhi Pass
  map = L.map('map', {
    center: [8.22, 77.53],
    zoom: 12,
    zoomControl: true
  });

  // Dark basemap (CartoDB Dark Matter)
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(map);

  existingTurbinesLayer.addTo(map);
  repoweredTurbinesLayer.addTo(map);
  windHeatmapLayer.addTo(map);
  substationsLayer.addTo(map);
}

async function loadData() {
  try {
    // 1. Load Summary Metrics
    const resSummary = await fetch('/api/summary');
    const summary = await resSummary.json();
    updateKPICards(summary);

    // 2. Load Classified Turbines
    const resTurbines = await fetch('/api/turbines/classified');
    const turbinesGeo = await resTurbines.json();
    renderExistingTurbines(turbinesGeo);

    // 3. Load Proposed Repowered Layout
    const resRepowered = await fetch('/api/turbines/repowered');
    const repoweredGeo = await resRepowered.json();
    renderRepoweredTurbines(repoweredGeo);

    // 4. Load Wind Heatmap
    const resHeatmap = await fetch('/api/wind/heatmap');
    const heatmapData = await resHeatmap.json();
    renderWindHeatmap(heatmapData);

    // 5. Load Infrastructure
    const resInfra = await fetch('/api/infrastructure');
    const infraData = await resInfra.json();
    renderInfrastructure(infraData);

    // 6. Load Clusters
    const resClusters = await fetch('/api/clusters');
    const clustersPayload = await resClusters.json();
    clustersData = clustersPayload.clusters || [];
    renderClusterList(clustersData);

  } catch (err) {
    console.error("Failed to load initial data:", err);
  }
}

function updateKPICards(summary) {
  if (!summary) return;
  document.getElementById('kpi-turbines').textContent = Number(summary.total_turbines_mapped).toLocaleString();
  document.getElementById('kpi-eligible').textContent = Number(summary.repowering_eligible_turbines).toLocaleString();
  document.getElementById('kpi-eligible-pct').textContent = `${summary.eligible_pct}%`;
  document.getElementById('kpi-multiplier').textContent = `${summary.mean_generation_multiplier}×`;
  document.getElementById('kpi-aep').innerHTML = `+${summary.total_delta_aep_gwh} <span style="font-size: 1rem; color: var(--text-muted);">GWh/yr</span>`;
  document.getElementById('kpi-co2').innerHTML = `${Number(summary.total_co2_avoided_tonnes).toLocaleString()} <span style="font-size: 0.95rem; color: var(--text-muted);">t CO₂</span>`;
  document.getElementById('kpi-homes').textContent = Number(summary.total_homes_powered).toLocaleString();
}

function renderExistingTurbines(geojson) {
  existingTurbinesLayer.clearLayers();
  if (!geojson.features) return;

  geojson.features.forEach(feat => {
    const coords = feat.geometry.coordinates;
    const p = feat.properties;
    const color = p.marker_color || "#ef4444";

    const circle = L.circleMarker([coords[1], coords[0]], {
      radius: 3.5,
      fillColor: color,
      color: color,
      weight: 1,
      opacity: 0.85,
      fillOpacity: 0.65
    });

    const popupHtml = `
      <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; line-height: 1.4; color: #111;">
        <strong style="font-size: 0.85rem; color: #000;">Turbine #${p.turbine_id}</strong><br>
        <strong>Class:</strong> ${p.turbine_class.toUpperCase()} (${p.capacity_kw} kW)<br>
        <strong>Spacing to 1st NN:</strong> ${p.nn1_dist_m} m<br>
        <strong>Mean 3-NN:</strong> ${p.mean_nn3_m} m<br>
        <strong>Eligible:</strong> ${p.repowering_eligible ? 'YES (< 2MW)' : 'NO'}<br>
        <strong>Dist to Substation:</strong> ${(p.dist_substation_m/1000).toFixed(2)} km
      </div>
    `;
    circle.bindPopup(popupHtml);
    existingTurbinesLayer.addLayer(circle);
  });
}

function renderRepoweredTurbines(geojson) {
  repoweredTurbinesLayer.clearLayers();
  if (!geojson.features) return;

  geojson.features.forEach(feat => {
    const coords = feat.geometry.coordinates;
    const p = feat.properties;

    const circle = L.circleMarker([coords[1], coords[0]], {
      radius: 6,
      fillColor: '#10b981',
      color: '#34d399',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.85
    });

    const popupHtml = `
      <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; line-height: 1.4; color: #111;">
        <strong style="font-size: 0.88rem; color: #047857;">Proposed Modern Turbine</strong><br>
        <strong>ID:</strong> ${p.turbine_id}<br>
        <strong>Model:</strong> ${p.turbine_model}<br>
        <strong>Rating:</strong> ${(p.capacity_kw/1000).toFixed(1)} MW<br>
        <strong>Hub Height:</strong> ${p.hub_height_m} m | <strong>Rotor:</strong> ${p.rotor_diameter_m} m<br>
        <strong>Zone:</strong> ${p.zone_name}
      </div>
    `;
    circle.bindPopup(popupHtml);
    repoweredTurbinesLayer.addLayer(circle);
  });
}

function renderWindHeatmap(heatmapData) {
  windHeatmapLayer.clearLayers();
  if (!heatmapData.points) return;

  heatmapData.points.forEach(pt => {
    const ws = pt.ws_120m;
    // Color scale: 5.5 m/s (blue/cyan) -> 7.0 m/s (emerald) -> 8.5+ m/s (amber/rose)
    let fill = '#0284c7';
    if (ws > 8.0) fill = '#f43f5e';
    else if (ws > 7.3) fill = '#f59e0b';
    else if (ws > 6.5) fill = '#10b981';
    else if (ws > 6.0) fill = '#06b6d4';

    const rect = L.circle([pt.lat, pt.lon], {
      radius: 400,
      fillColor: fill,
      color: 'transparent',
      weight: 0,
      fillOpacity: 0.18
    });
    rect.bindTooltip(`Wind @ 120m: ${ws} m/s`, { sticky: true });
    windHeatmapLayer.addLayer(rect);
  });
}

function renderInfrastructure(infra) {
  substationsLayer.clearLayers();
  if (!infra.substations || !infra.substations.features) return;

  infra.substations.features.forEach(feat => {
    const coords = feat.geometry.coordinates;
    const p = feat.properties;
    const name = p.name || "Grid Substation";

    const marker = L.circleMarker([coords[1], coords[0]], {
      radius: 7,
      fillColor: '#fbbf24',
      color: '#ffffff',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.95
    });

    marker.bindPopup(`
      <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #111;">
        <strong style="color: #b45309;">⚡ ${name}</strong><br>
        <strong>Voltage:</strong> ${p.voltage || '66 / 110 kV'}<br>
        <strong>Grid:</strong> TANGEDCO InSTS
      </div>
    `);
    substationsLayer.addLayer(marker);
  });
}

function renderClusterList(clusters) {
  const container = document.getElementById('cluster-list-container');
  container.innerHTML = '';

  clusters.forEach((c, idx) => {
    const item = document.createElement('div');
    item.className = `cluster-item ${c.zone_id === currentSelectedZone ? 'active' : ''}`;
    item.onclick = () => selectCluster(c.zone_id);

    item.innerHTML = `
      <div class="cluster-item-head">
        <div class="cluster-rank-tag">#${c.rank}</div>
        <div class="cluster-name">${c.name}</div>
        <div class="cluster-score-pill">${c.priority_score} pts</div>
      </div>
      <div class="cluster-metrics-row">
        <span>Capacity:</span>
        <span class="val">${c.old_capacity_mw} MW → ${c.repowered_capacity_mw} MW</span>
      </div>
      <div class="cluster-metrics-row">
        <span>Energy Gain:</span>
        <span class="val">+${c.delta_aep_gwh} GWh (${c.generation_multiplier}×)</span>
      </div>
      <div class="cluster-metrics-row">
        <span>Payback Period:</span>
        <span class="val">${c.payback_years} yrs (CAPEX ₹${c.total_investment_cr} Cr)</span>
      </div>
    `;
    container.appendChild(item);
  });
}

function selectCluster(zoneId) {
  currentSelectedZone = zoneId;
  const items = document.querySelectorAll('.cluster-item');
  items.forEach(el => el.classList.remove('active'));

  // Trigger simulation update
  triggerSimulation();
}

function setupEventListeners() {
  // Layer toggles
  document.getElementById('btn-toggle-existing').onclick = function() {
    toggleLayer(existingTurbinesLayer, this);
  };
  document.getElementById('btn-toggle-repowered').onclick = function() {
    toggleLayer(repoweredTurbinesLayer, this);
  };
  document.getElementById('btn-toggle-wind').onclick = function() {
    toggleLayer(windHeatmapLayer, this);
  };
  document.getElementById('btn-toggle-subs').onclick = function() {
    toggleLayer(substationsLayer, this);
  };

  // Slider events
  const sliderHub = document.getElementById('slider-hub');
  const sliderCap = document.getElementById('slider-cap');
  const sliderSpacing = document.getElementById('slider-spacing');

  sliderHub.oninput = () => {
    document.getElementById('val-hub').textContent = `${sliderHub.value} m`;
    debouncedSimulation();
  };
  sliderCap.oninput = () => {
    document.getElementById('val-cap').textContent = `${sliderCap.value} MW`;
    debouncedSimulation();
  };
  sliderSpacing.oninput = () => {
    document.getElementById('val-spacing').textContent = `${sliderSpacing.value} D`;
    debouncedSimulation();
  };
}

function toggleLayer(layer, btn) {
  if (map.hasLayer(layer)) {
    map.removeLayer(layer);
    btn.classList.remove('active');
  } else {
    map.addLayer(layer);
    btn.classList.add('active');
  }
}

function switchTab(tab) {
  const btnSim = document.getElementById('tab-btn-sim');
  const btnRank = document.getElementById('tab-btn-rank');
  const contentSim = document.getElementById('tab-sim-content');
  const contentRank = document.getElementById('tab-rank-content');

  if (tab === 'sim') {
    btnSim.classList.add('active');
    btnRank.classList.remove('active');
    contentSim.style.display = 'block';
    contentRank.style.display = 'none';
  } else {
    btnRank.classList.add('active');
    btnSim.classList.remove('active');
    contentRank.style.display = 'block';
    contentSim.style.display = 'none';
  }
}

let debounceTimer = null;
function debouncedSimulation() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(triggerSimulation, 250);
}

async function triggerSimulation() {
  const hub = parseFloat(document.getElementById('slider-hub').value);
  const cap = parseFloat(document.getElementById('slider-cap').value);
  const spacing = parseFloat(document.getElementById('slider-spacing').value);

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        zone_id: currentSelectedZone,
        hub_height_m: hub,
        turbine_capacity_mw: cap,
        spacing_diameters: spacing
      })
    });
    const data = await res.json();
    const r = data.scenario_results;

    document.getElementById('sim-net-aep').innerHTML = `${r.net_aep_gwh} <span style="font-size: 0.8rem;">GWh</span>`;
    document.getElementById('sim-wake').textContent = `${r.wake_loss_pct}%`;
    document.getElementById('sim-plf').textContent = `${r.plf_pct}%`;
    document.getElementById('sim-payback').innerHTML = `${r.payback_years} <span style="font-size: 0.8rem;">yrs</span>`;

    const badge = document.getElementById('sim-policy-badge');
    const multText = document.getElementById('sim-multiplier-text');
    if (r.policy_1_5x_pass) {
      badge.className = 'policy-badge-container';
      multText.textContent = `Generation: ${r.generation_multiplier}× baseline (Req: ≥ 1.5×)`;
    } else {
      badge.className = 'policy-badge-container fail';
      multText.textContent = `Generation: ${r.generation_multiplier}× baseline (Under target)`;
    }
  } catch (err) {
    console.error("Simulation request failed:", err);
  }
}
