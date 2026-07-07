'use strict';

// ---- State ----
let sessionId = null;
let ws = null;
let playing = false;
let stepCount = 0;
let driftHistory = [];
let biasAccum = new Array(10).fill(0);
let humanSessionId = null;
let humanTrialStart = null;
let humanTotal = 0;
let humanCurrent = 0;
let calibBestParams = null;

const MAX_DRIFT_HISTORY = 120;
const MAX_HISTORY_STRIP = 50;

// ---- Init ----
window.addEventListener('DOMContentLoaded', async () => {
  await initSession();
  setupKeyboard();
  renderProbBars(new Array(10).fill(0), new Array(10).fill(0), -1);
  drawDriftCanvas();
});

async function initSession() {
  const params = getParams();
  const res = await fetch('/api/session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ params })
  });
  const data = await res.json();
  sessionId = data.session_id;
  document.getElementById('session-id-label').textContent = sessionId.slice(0, 8) + '…';
  connectWS();
}

function connectWS() {
  if (ws) ws.close();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/play/${sessionId}`);

  ws.onopen = () => setWSStatus('connected');
  ws.onclose = () => { setWSStatus('disconnected'); setTimeout(connectWS, 2000); };
  ws.onerror = () => setWSStatus('error');
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'prediction') handlePrediction(msg);
    else if (msg.type === 'reset_ack') handleResetAck();
  };
}

// ---- Controls ----
function getParams() {
  const seqMode = document.getElementById('ctrl-seqmode').value;
  return {
    intensity: parseFloat(document.getElementById('ctrl-intensity').value),
    interval_ms: parseInt(document.getElementById('ctrl-interval').value),
    return_to_normal_enabled: document.getElementById('ctrl-decay').checked,
    return_rate: parseFloat(document.getElementById('ctrl-rate').value),
    confidence_mode: document.getElementById('ctrl-confmode').value,
    max_delta_norm: parseFloat(document.getElementById('ctrl-maxnorm').value),
    dynamic_layer_mode: document.getElementById('ctrl-layermode').value,
    sequence_mode: seqMode,
    class_a: parseInt(document.getElementById('ctrl-classa').value) || null,
    class_b: parseInt(document.getElementById('ctrl-classb').value) || null,
    show_true_label: document.getElementById('ctrl-truelabel').checked,
    noise_std: parseFloat(document.getElementById('ctrl-noise').value),
  };
}

function sendParamsUpdate() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: 'update_params', params: getParams() }));
}

function updateVal(inputId, badgeId, fmt) {
  const v = document.getElementById(inputId).value;
  document.getElementById(badgeId).textContent = fmt(v);
}

function onSeqModeChange() {
  const mode = document.getElementById('ctrl-seqmode').value;
  const show = ['custom_pair', 'only_a', 'only_b', '3_vs_8_ambiguous'].includes(mode);
  document.getElementById('pair-inputs').style.display = show ? 'flex' : 'none';
  document.getElementById('pair-inputs').style.flexDirection = 'column';
  sendParamsUpdate();
}

function toggleProbChart() {
  const show = document.getElementById('ctrl-probchart').checked;
  document.getElementById('prob-chart').style.display = show ? '' : 'none';
}

// ---- Playback ----
function togglePlay() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  playing = !playing;
  const btn = document.getElementById('btn-play');
  if (playing) {
    ws.send(JSON.stringify({ type: 'play', interval_ms: parseInt(document.getElementById('ctrl-interval').value) }));
    btn.textContent = '⏸ Pause';
    btn.classList.remove('btn-primary');
    document.getElementById('digit-card').closest('.prediction-area')?.parentElement?.classList.add('playing');
  } else {
    ws.send(JSON.stringify({ type: 'pause' }));
    btn.textContent = '▶ Play';
    btn.classList.add('btn-primary');
  }
}

function stepOnce() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (playing) { togglePlay(); }
  ws.send(JSON.stringify({ type: 'step' }));
}

function resetDelta() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: 'reset' }));
}

async function resetSession() {
  if (playing) togglePlay();
  ws?.close();
  stepCount = 0;
  driftHistory = [];
  biasAccum = new Array(10).fill(0);
  document.getElementById('history-strip').innerHTML = '';
  document.getElementById('step-counter').textContent = '0';
  document.getElementById('pred-digit').textContent = '—';
  document.getElementById('m-confidence').textContent = '—';
  document.getElementById('m-margin').textContent = '—';
  document.getElementById('m-deltanorm').textContent = '0.00';
  document.getElementById('drift-norm-display').textContent = '0.000';
  renderProbBars(new Array(10).fill(0), new Array(10).fill(0), -1);
  drawDriftCanvas();
  await initSession();
}

function handleResetAck() {
  driftHistory = [];
  biasAccum = new Array(10).fill(0);
  drawDriftCanvas();
  document.getElementById('m-deltanorm').textContent = '0.00';
  document.getElementById('drift-norm-display').textContent = '0.000';
}

// ---- Prediction rendering ----
function handlePrediction(msg) {
  stepCount++;
  document.getElementById('step-counter').textContent = stepCount;

  // Image
  const img = document.getElementById('digit-img');
  const placeholder = document.getElementById('digit-placeholder');
  img.src = 'data:image/png;base64,' + msg.image_base64;
  img.style.display = '';
  placeholder.style.display = 'none';
  img.classList.remove('flash');
  void img.offsetWidth;
  img.classList.add('flash');

  // Predicted label
  document.getElementById('pred-digit').textContent = msg.predicted_label;

  // True label
  const badge = document.getElementById('true-label-badge');
  const showTrue = document.getElementById('ctrl-truelabel').checked;
  if (showTrue) {
    badge.style.display = '';
    const correct = msg.predicted_label === msg.true_label;
    badge.textContent = `true: ${msg.true_label}`;
    badge.className = 'true-label-badge ' + (correct ? 'correct' : 'wrong');
  } else {
    badge.style.display = 'none';
  }

  // Metrics
  document.getElementById('m-confidence').textContent = (msg.confidence * 100).toFixed(1) + '%';
  document.getElementById('m-margin').textContent = (msg.margin * 100).toFixed(1) + '%';
  document.getElementById('m-deltanorm').textContent = msg.delta_norm.toFixed(3);
  document.getElementById('drift-norm-display').textContent = msg.delta_norm.toFixed(3);

  // Prob bars
  renderProbBars(msg.probabilities, msg.vanilla_probabilities || [], msg.predicted_label);

  // Drift history
  driftHistory.push(msg.delta_norm);
  if (driftHistory.length > MAX_DRIFT_HISTORY) driftHistory.shift();
  drawDriftCanvas();

  // Bias accum
  biasAccum[msg.predicted_label] += 1 - msg.confidence;
  renderBias();

  // History strip
  addHistoryItem(msg);
}

function renderProbBars(probs, vanillaProbs, topClass) {
  const container = document.getElementById('prob-bars');
  const hasVanilla = vanillaProbs && vanillaProbs.length === 10;
  const DELTA_THRESH = 0.005; // min delta to show label

  // Build or update columns (reuse DOM for smooth CSS transitions)
  if (container.children.length !== 10) {
    container.innerHTML = '';
    for (let i = 0; i < 10; i++) {
      container.insertAdjacentHTML('beforeend', `
        <div class="prob-col" id="pcol-${i}">
          <div class="prob-delta" id="pdelta-${i}"></div>
          <div class="prob-bar-stack">
            <div class="prob-bar-vanilla" id="pvan-${i}"></div>
            <div class="prob-bar-dyn neutral" id="pdyn-${i}"></div>
            <div class="prob-vanilla-line" id="pvline-${i}" style="display:none"></div>
          </div>
          <div class="prob-col-label">${i}</div>
        </div>
      `);
    }
  }

  const barAreaH = container.clientHeight - 24; // subtract label area
  const h = barAreaH > 0 ? barAreaH : 106;

  for (let i = 0; i < 10; i++) {
    const p = probs[i] ?? 0;
    const v = hasVanilla ? (vanillaProbs[i] ?? 0) : p;
    const isPred = i === topClass;
    const delta = p - v;

    const dynBar   = document.getElementById(`pdyn-${i}`);
    const vanBar   = document.getElementById(`pvan-${i}`);
    const vline    = document.getElementById(`pvline-${i}`);
    const deltaEl  = document.getElementById(`pdelta-${i}`);
    const col      = document.getElementById(`pcol-${i}`);

    // Heights as % of bar area
    const dynH = (p * h).toFixed(1);
    const vanH = (v * h).toFixed(1);

    dynBar.style.height = dynH + 'px';
    vanBar.style.height = vanH + 'px';

    // Color the dynamic bar
    dynBar.className = 'prob-bar-dyn ' + (
      isPred       ? 'predicted'  :
      delta > DELTA_THRESH  ? 'boosted'    :
      delta < -DELTA_THRESH ? 'suppressed' :
      'neutral'
    );

    // Vanilla marker line
    if (hasVanilla && Math.abs(delta) > DELTA_THRESH) {
      vline.style.display = '';
      vline.style.bottom = vanH + 'px';
    } else {
      vline.style.display = 'none';
    }

    // Delta label
    if (hasVanilla && Math.abs(delta) >= DELTA_THRESH) {
      const sign = delta > 0 ? '+' : '';
      deltaEl.textContent = sign + (delta * 100).toFixed(1) + '%';
      deltaEl.className = 'prob-delta visible ' + (delta > 0 ? 'up' : 'down');
    } else {
      deltaEl.className = 'prob-delta';
      deltaEl.textContent = '';
    }

    // Highlight predicted column label
    col.className = 'prob-col' + (isPred ? ' predicted' : '');
  }
}

function drawDriftCanvas() {
  const canvas = document.getElementById('drift-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (driftHistory.length < 2) return;

  const maxNorm = Math.max(...driftHistory, 0.1);
  ctx.strokeStyle = '#7c6af7';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = '#7c6af7';
  ctx.shadowBlur = 4;
  ctx.beginPath();

  driftHistory.forEach((v, i) => {
    const x = (i / (MAX_DRIFT_HISTORY - 1)) * W;
    const y = H - (v / maxNorm) * (H - 6) - 3;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // fill under curve
  ctx.shadowBlur = 0;
  ctx.lineTo((driftHistory.length - 1) / (MAX_DRIFT_HISTORY - 1) * W, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  ctx.fillStyle = 'rgba(124, 106, 247, 0.08)';
  ctx.fill();
}

function renderBias() {
  const total = biasAccum.reduce((a, b) => a + b, 0);
  if (total < 0.001) return;

  const sorted = biasAccum
    .map((v, i) => ({ cls: i, val: v }))
    .sort((a, b) => b.val - a.val);

  const top = sorted.slice(0, 3);
  const container = document.getElementById('bias-content');
  container.innerHTML = top.map(({ cls, val }) => {
    const pct = (val / total * 100).toFixed(1);
    return `<div class="bias-row">
      <span class="bias-arrow-up">↑</span>
      <span>Class <strong>${cls}</strong></span>
      <span style="color:var(--text3);margin-left:auto;font-family:var(--mono)">${pct}%</span>
    </div>`;
  }).join('');

  const biasTag = document.getElementById('bias-tag');
  biasTag.textContent = `bias → ${sorted[0].cls}`;
}

function addHistoryItem(msg) {
  const strip = document.getElementById('history-strip');
  const items = strip.querySelectorAll('.history-item');

  items.forEach(el => el.classList.remove('current'));

  const item = document.createElement('div');
  item.className = 'history-item current';
  const correct = msg.predicted_label === msg.true_label;
  if (document.getElementById('ctrl-truelabel').checked) {
    item.classList.add(correct ? 'correct' : 'wrong');
  }
  item.textContent = msg.predicted_label;
  item.title = `#${stepCount} pred=${msg.predicted_label} true=${msg.true_label} conf=${(msg.confidence*100).toFixed(0)}%`;
  strip.appendChild(item);

  while (strip.children.length > MAX_HISTORY_STRIP) {
    strip.removeChild(strip.firstChild);
  }

  strip.scrollLeft = strip.scrollWidth;
  document.getElementById('history-count').textContent = `${strip.children.length} shown`;
}

// ---- Screen navigation ----
function showScreen(name) {
  ['playback', 'human', 'calibration'].forEach(s => {
    document.getElementById(`screen-${s}`).classList.toggle('active', s === name);
    document.getElementById(`nav-${s}`).classList.toggle('active', s === name);
  });
  document.getElementById('pb-controls').style.display = name === 'playback' ? '' : 'none';
}

// ---- Human training ----
document.getElementById('h-mode').addEventListener('change', () => {
  const mode = document.getElementById('h-mode').value;
  document.getElementById('h-pair-inputs').style.display = mode === 'custom_pair' ? 'flex' : 'none';
  if (mode === 'custom_pair') document.getElementById('h-pair-inputs').style.flexDirection = 'column';
});

function humanTab(tab) {
  ['setup', 'trial', 'summary'].forEach(t => {
    document.getElementById(`human-${t}`).style.display = t === tab ? '' : 'none';
    document.getElementById(`htab-${t}`).classList.toggle('active', t === tab);
  });
}

async function startHumanSession() {
  const mode = document.getElementById('h-mode').value;
  const body = { mode };
  if (mode === 'custom_pair') {
    body.class_a = parseInt(document.getElementById('h-classa').value);
    body.class_b = parseInt(document.getElementById('h-classb').value);
  }
  const res = await fetch('/api/human/session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await res.json();
  humanSessionId = data.session_id;
  humanTotal = data.total_trials;
  humanCurrent = 0;
  humanTab('trial');
  await loadNextHumanTrial();
}

async function loadNextHumanTrial() {
  const res = await fetch(`/api/human/${humanSessionId}/next`, { method: 'POST' });
  if (!res.ok) {
    await showHumanSummary();
    return;
  }
  const data = await res.json();
  humanCurrent = data.trial_index + 1;
  humanTotal = data.total;

  const img = document.getElementById('h-digit-img');
  img.src = 'data:image/png;base64,' + data.image_base64;
  img.classList.remove('flash');
  void img.offsetWidth;
  img.classList.add('flash');

  const progress = (humanCurrent / humanTotal * 100).toFixed(0);
  document.getElementById('h-progress').style.width = progress + '%';
  document.getElementById('h-progress-label').textContent = `${humanCurrent} / ${humanTotal}`;
  document.getElementById('h-rt-display').textContent = 'Press a key 0–9';
  humanTrialStart = Date.now();
}

function humanKeyPress(digit) {
  if (!humanSessionId || document.getElementById('human-trial').style.display === 'none') return;
  const rt = humanTrialStart ? Date.now() - humanTrialStart : null;
  humanTrialStart = null;

  // Visual feedback
  const cap = document.querySelector(`.key-cap[data-key="${digit}"]`);
  if (cap) {
    cap.classList.add('pressed');
    setTimeout(() => cap.classList.remove('pressed'), 200);
  }

  fetch(`/api/human/${humanSessionId}/response`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ user_label: digit, response_time_ms: rt })
  }).then(() => loadNextHumanTrial());
}

async function humanSkip() {
  await fetch(`/api/human/${humanSessionId}/response`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ user_label: null, response_time_ms: null })
  });
  await loadNextHumanTrial();
}

async function endHumanSession() {
  await showHumanSummary();
}

async function showHumanSummary() {
  humanTab('summary');
  const res = await fetch(`/api/human/${humanSessionId}/summary`);
  const data = await res.json();
  const container = document.getElementById('summary-content');

  const topConf = Object.entries(data.confusion || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  container.innerHTML = `
    <div class="summary-grid">
      <div class="summary-card">
        <h3>Overview</h3>
        <div class="calib-row"><span>Total trials</span><span class="calib-val">${data.total}</span></div>
        <div class="calib-row"><span>Correct</span><span class="calib-val" style="color:var(--green)">${data.correct}</span></div>
        <div class="calib-row"><span>Accuracy</span><span class="calib-val">${(data.accuracy * 100).toFixed(1)}%</span></div>
        <div class="calib-row"><span>Avg RT</span><span class="calib-val">${data.avg_response_time_ms?.toFixed(0) ?? '—'}ms</span></div>
        <div class="calib-row" style="border:none;margin-top:8px">
          <span style="color:var(--text3);font-size:11px">Session ID</span>
          <span class="calib-val" style="font-size:10px">${humanSessionId?.slice(0,8)}…</span>
        </div>
      </div>
      <div class="summary-card">
        <h3>Top Confusions</h3>
        ${topConf.map(([k, v]) => `
          <div class="conf-entry">
            <span>${k.replace('->', ' → ')}</span>
            <span class="count">${v}×</span>
          </div>
        `).join('')}
      </div>
    </div>
    <div style="margin-top:12px">
      <button class="btn btn-sm" onclick="useForCalibration('${humanSessionId}')">
        Use for Calibration →
      </button>
    </div>
  `;
}

function useForCalibration(sid) {
  showScreen('calibration');
  document.getElementById('calib-session-id').value = sid;
}

// ---- Calibration ----
async function runCalibration() {
  const sid = document.getElementById('calib-session-id').value.trim();
  if (!sid) return;
  const btn = document.getElementById('calib-btn');
  btn.textContent = 'Running…';
  btn.disabled = true;
  document.getElementById('calib-result').style.display = 'none';
  document.getElementById('calib-error').style.display = 'none';

  try {
    const res = await fetch(`/api/calibrate/${sid}`, { method: 'POST' });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    calibBestParams = data.params;
    renderCalibResult(data.score, data.params);
  } catch (e) {
    document.getElementById('calib-error').style.display = '';
    document.getElementById('calib-error').textContent = 'Error: ' + e.message;
  } finally {
    btn.textContent = 'Run Calibration';
    btn.disabled = false;
  }
}

function renderCalibResult(score, params) {
  const container = document.getElementById('calib-params-list');
  const entries = [
    ['Score (NLL)', score.toFixed(4)],
    ['Intensity', params.intensity],
    ['Return rate', params.return_rate],
    ['Inhibition strength', params.inhibition_strength],
    ['Confidence mode', params.confidence_mode],
    ['Return to normal', params.return_to_normal_enabled ? 'yes' : 'no'],
  ];
  container.innerHTML = entries.map(([k, v]) =>
    `<div class="calib-row"><span>${k}</span><span class="calib-val">${v}</span></div>`
  ).join('');
  document.getElementById('calib-result').style.display = '';
}

function applyCalibParams() {
  if (!calibBestParams) return;
  const p = calibBestParams;
  document.getElementById('ctrl-intensity').value = p.intensity;
  updateVal('ctrl-intensity', 'val-intensity', v => parseFloat(v).toFixed(3));
  document.getElementById('ctrl-rate').value = p.return_rate;
  updateVal('ctrl-rate', 'val-rate', v => parseFloat(v).toFixed(2));
  document.getElementById('ctrl-decay').checked = p.return_to_normal_enabled;
  document.getElementById('ctrl-confmode').value = p.confidence_mode;
  showScreen('playback');
  sendParamsUpdate();
}

// ---- Status ----
function setWSStatus(s) {
  const dot = document.getElementById('ws-dot');
  const label = document.getElementById('ws-status');
  dot.className = 'status-dot';
  if (s === 'connected') { dot.classList.add('green'); label.textContent = 'Connected'; }
  else if (s === 'disconnected') { label.textContent = 'Disconnected'; }
  else { dot.classList.add('red'); label.textContent = 'Error'; }
}

// ---- Keyboard ----
function setupKeyboard() {
  document.addEventListener('keydown', (e) => {
    const tag = e.target.tagName.toLowerCase();
    if (tag === 'input' || tag === 'select') return;

    if (e.code === 'Space') {
      e.preventDefault();
      const screen = document.getElementById('screen-playback');
      if (screen.classList.contains('active')) togglePlay();
    }
    if (e.key === 'ArrowRight') stepOnce();
    if (e.key === 'r' || e.key === 'R') resetDelta();

    // human training digit keys
    if (/^[0-9]$/.test(e.key)) {
      const humanTrial = document.getElementById('human-trial');
      if (humanTrial.style.display !== 'none' && humanSessionId) {
        humanKeyPress(parseInt(e.key));
      }
    }
  });
}
