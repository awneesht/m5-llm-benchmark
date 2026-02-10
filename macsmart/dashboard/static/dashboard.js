/* MacSmart Dashboard — fetch data and render charts + tables */

const CHART_COLORS = [
  '#4285f4', '#0f9d58', '#f4a142', '#ea4335',
  '#ab47bc', '#00acc1', '#ff7043', '#9ccc65',
];

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: '#e0e0e0' } },
  },
  scales: {
    x: { ticks: { color: '#8892a4' }, grid: { color: '#2a2a4a' } },
    y: { ticks: { color: '#8892a4' }, grid: { color: '#2a2a4a' } },
  },
};

/* ---- Data fetching ---- */

async function fetchJSON(url) {
  try {
    const resp = await fetch(url);
    return await resp.json();
  } catch {
    return [];
  }
}

/* ---- Chart rendering ---- */

function renderBarChart(canvasId, labels, data, label, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || typeof Chart === 'undefined') return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data,
        backgroundColor: labels.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
        borderRadius: 6,
      }],
    },
    options: CHART_DEFAULTS,
  });
}

function renderScatterChart(canvasId, points, labelX, labelY) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || typeof Chart === 'undefined') return;
  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: `${labelY} vs ${labelX}`,
        data: points,
        backgroundColor: CHART_COLORS[0],
        pointRadius: 6,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: labelX, color: '#8892a4' } },
        y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: labelY, color: '#8892a4' } },
      },
    },
  });
}

/* ---- Table rendering ---- */

function renderResultsTable(results) {
  const container = document.getElementById('results-table');
  if (!container) return;

  if (!results.length) {
    container.innerHTML = '<p class="no-data">No benchmark results found. Run `macsmart benchmark` first.</p>';
    return;
  }

  let html = `<table>
    <thead><tr>
      <th>Model</th><th>Quant</th>
      <th class="num">Tokens/s</th><th class="num">TTFT (ms)</th>
      <th class="num">Peak Mem (GB)</th><th class="num">Swap (GB)</th>
      <th class="num">Tokens</th><th class="num">Duration (s)</th>
    </tr></thead><tbody>`;

  for (const r of results) {
    const swapClass = r.swap_used_gb > 0.1 ? 'badge badge-warn' : '';
    html += `<tr>
      <td>${r.model_name}</td><td>${r.quantization}</td>
      <td class="num">${r.tokens_per_sec.toFixed(1)}</td>
      <td class="num">${r.ttft_ms.toFixed(1)}</td>
      <td class="num">${r.peak_memory_gb.toFixed(2)}</td>
      <td class="num"><span class="${swapClass}">${r.swap_used_gb.toFixed(2)}</span></td>
      <td class="num">${r.generation_tokens}</td>
      <td class="num">${r.duration_sec.toFixed(1)}</td>
    </tr>`;
  }

  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderEnergyTable(energyResults) {
  const container = document.getElementById('energy-table');
  if (!container) return;

  if (!energyResults.length) {
    container.innerHTML = '<p class="no-data">No energy results. Run `macsmart benchmark --energy` or `macsmart energy-compare`.</p>';
    return;
  }

  let html = `<table>
    <thead><tr>
      <th>Model</th><th>Source</th>
      <th class="num">Tokens/s</th><th class="num">TTFT (ms)</th>
      <th class="num">Power (W)</th><th class="num">Energy (J)</th>
      <th class="num">Peak Mem (GB)</th>
    </tr></thead><tbody>`;

  for (const r of energyResults) {
    const b = r.benchmark;
    const e = r.energy;
    const srcBadge = r.power_source === 'ac'
      ? '<span class="badge badge-ok">AC</span>'
      : '<span class="badge badge-warn">Battery</span>';
    html += `<tr>
      <td>${b.model_name}</td><td>${srcBadge}</td>
      <td class="num">${b.tokens_per_sec.toFixed(1)}</td>
      <td class="num">${b.ttft_ms.toFixed(1)}</td>
      <td class="num">${e.total_power_watts != null ? e.total_power_watts.toFixed(1) : 'N/A'}</td>
      <td class="num">${e.total_energy_joules != null ? e.total_energy_joules.toFixed(1) : 'N/A'}</td>
      <td class="num">${b.peak_memory_gb.toFixed(2)}</td>
    </tr>`;
  }

  html += '</tbody></table>';
  container.innerHTML = html;
}

/* ---- Main ---- */

async function main() {
  const [results, energyResults] = await Promise.all([
    fetchJSON('/api/results'),
    fetchJSON('/api/energy-results'),
  ]);

  // Render charts
  if (results.length) {
    const labels = results.map(r => `${r.model_name} (${r.quantization})`);
    renderBarChart('chart-tps', labels, results.map(r => r.tokens_per_sec), 'Tokens/s');
    renderBarChart('chart-ttft', labels, results.map(r => r.ttft_ms), 'TTFT (ms)');
    renderBarChart('chart-memory', labels, results.map(r => r.peak_memory_gb), 'Peak Memory (GB)');

    const effPoints = results.map(r => ({
      x: r.peak_memory_gb,
      y: r.peak_memory_gb > 0 ? r.tokens_per_sec / r.peak_memory_gb : 0,
    }));
    renderScatterChart('chart-efficiency', effPoints, 'Peak Memory (GB)', 'Tokens/s per GB');
  }

  // Render tables
  renderResultsTable(results);
  renderEnergyTable(energyResults);
}

document.addEventListener('DOMContentLoaded', main);
