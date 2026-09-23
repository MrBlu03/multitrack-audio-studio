// ============================================================================
// MULTITRACK AUDIO STUDIO - CLIENT CONTROLLER (PROFESSIONAL DAW)
// High-efficiency WebSocket telemetry streaming & channel strip controller
// ============================================================================

let ws = null;
let appState = null;
let currentTab = 'auto-master';
let isProcessing = false;

// Initialize when DOM loads
document.addEventListener('DOMContentLoaded', () => {
  initWebSocket();
  fetchState();
  setupDragAndDrop();
});

// ---------------------------------------------------------------------------
// Real-Time WebSocket Telemetry
// ---------------------------------------------------------------------------
function initWebSocket() {
  const loc = window.location;
  const wsProto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProto}//${loc.host}/ws/live`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    appendConsole('[SYSTEM] WebSocket telemetry bridge established.', 'info');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleWsMessage(msg);
    } catch (e) {
      console.error('Error parsing WebSocket message:', e);
    }
  };

  ws.onclose = () => {
    appendConsole('[SYSTEM] Telemetry link offline. Reconnecting in 2s...', 'warn');
    setTimeout(initWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
  };
}

function handleWsMessage(msg) {
  if (msg.type === 'init') {
    appState = msg.state;
    renderUI();
  } else if (msg.type === 'state_update') {
    appState = msg.state;
    renderUI();
  } else if (msg.type === 'session_loaded') {
    if (msg.state) {
      appState = msg.state;
      renderUI();
    }
  } else if (msg.type === 'log') {

    appendConsole(msg.text, 'log');
  } else if (msg.type === 'progress') {
    const p = msg.data || msg;
    if (p && typeof p === 'object') {
      const totalPct = p.total_pct ?? p.total_percent ?? 0;
      document.getElementById('progressFill').style.width = `${totalPct}%`;
      const trackIdx = p.track_idx ?? 1;
      const totalTracks = p.total_tracks ?? p.num_tracks ?? 1;
      const trackName = p.track_name || '';
      const pct = (p.pct ?? p.track_percent ?? 0);
      const pctStr = typeof pct === 'number' ? pct.toFixed(1) : pct;
      const speed = p.speed ?? 1.0;
      const spdStr = typeof speed === 'number' ? speed.toFixed(1) : speed;
      const eta = p.eta || '--:--';
      const statusText = `[TRACK ${trackIdx}/${totalTracks}] ${trackName} | ${pctStr}% | ${spdStr}x | ETA ${eta}`;
      document.getElementById('dockStatusText').textContent = statusText;
    }
  } else if (msg.type === 'finish') {
    if (msg.success) {
      document.getElementById('progressFill').style.width = '100%';
      document.getElementById('dockStatusText').textContent = 'Processing finished successfully.';
      if (msg.output) {
        document.getElementById('btnOpenFolder').style.display = 'inline-flex';
      }
    } else {
      document.getElementById('dockStatusText').textContent = 'Processing failed: ' + (msg.message || 'Unknown error');
    }
    setProcessingUI(false);
  } else if (msg.type === 'video_progress') {
    const pct = msg.percent || 0;
    document.getElementById('progressFill').style.width = `${pct}%`;
    const speed = msg.speed ? `${msg.speed.toFixed(1)}x` : '~100x';
    document.getElementById('dockStatusText').textContent = `[EXTRACTING 4K VIDEO] ${pct.toFixed(1)}% | ${speed} real-time`;
  } else if (msg.type === 'video_extracted') {
    if (msg.success) {
      document.getElementById('progressFill').style.width = '100%';
      document.getElementById('dockStatusText').textContent = `[VIDEO READY] Extracted ${msg.count} audio tracks into pristine 24-bit PCM stems.`;
      if (msg.state) {
        appState = msg.state;
        renderUI();
      }
    } else {
      document.getElementById('dockStatusText').textContent = `Video extraction failed: ${msg.error || 'Unknown error'}`;
    }
  } else if (msg.type === 'transcribe_progress') {
    const d = msg.data || msg;
    if (d && typeof d === 'object') {
      const pct = d.progress_pct ?? d.percent ?? 0;
      document.getElementById('progressFill').style.width = `${pct}%`;
      const trackIdx = d.track_idx ?? 1;
      const totalTracks = d.total_tracks ?? d.num_tracks ?? 1;
      const spk = d.speaker || '';
      document.getElementById('dockStatusText').textContent = `[TRANSCRIBE ${trackIdx}/${totalTracks}] ${spk} | ${typeof pct === 'number' ? pct.toFixed(1) : pct}%`;
    }
  }
}

// ---------------------------------------------------------------------------
// REST API Calls
// ---------------------------------------------------------------------------
async function fetchState() {
  try {
    const res = await fetch('/api/state');
    appState = await res.json();
    renderUI();
  } catch (e) {
    console.error('Failed to fetch state:', e);
  }
}

async function switchCampaign(mode) {
  try {
    const res = await fetch('/api/campaign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    appState = await res.json();
    renderUI();
  } catch (e) {
    console.error('Failed to switch campaign:', e);
  }
}

async function browseVideo() {
  try {
    const res = await fetch('/api/browse-video', { method: 'POST' });
    appState = await res.json();
    renderUI();
  } catch (e) {
    console.error('Browse video error:', e);
  }
}

async function browseFolder() {
  try {
    const res = await fetch('/api/browse-folder', { method: 'POST' });
    appState = await res.json();
    renderUI();
  } catch (e) {
    console.error('Browse folder error:', e);
  }
}

async function browseFiles() {
  try {
    const res = await fetch('/api/browse-files', { method: 'POST' });
    appState = await res.json();
    renderUI();
  } catch (e) {
    console.error('Browse files error:', e);
  }
}

async function browseSingleFile(slot) {
  try {
    const res = await fetch(`/api/browse-single-file/${slot}`, { method: 'POST' });
    appState = await res.json();
    renderUI();
  } catch (e) {
    console.error(`Browse single file slot ${slot} error:`, e);
  }
}

async function browseOutputDir() {
  try {
    const res = await fetch('/api/browse-output-dir', { method: 'POST' });
    const data = await res.json();
    if (data.output_dir) {
      document.getElementById('inputOutputDir').value = data.output_dir;
      appState.output_dir = data.output_dir;
    }
  } catch (e) {
    console.error('Browse output dir error:', e);
  }
}

async function clearSlots() {
  try {
    const res = await fetch('/api/clear-slots', { method: 'POST' });
    appState = await res.json();
    renderUI();
    document.getElementById('btnOpenFolder').style.display = 'none';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('dockStatusText').textContent = 'Rack cleared. Load session folder or stems.';
  } catch (e) {
    console.error('Clear slots error:', e);
  }
}

async function updateSlotProfile(slot, profile_id) {
  try {
    await fetch('/api/update-slot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slot: parseInt(slot), profile_id })
    });
    if (appState && appState.slots && appState.slots[slot]) {
      appState.slots[slot].profile_id = profile_id;
      const choices = appState.profile_choices || [];
      const ch = choices.find(c => (typeof c === 'object' ? c.id : c) === profile_id);
      if (ch) {
        appState.slots[slot].profile_name = typeof ch === 'object' ? ch.name : ch;
      }
      renderUI();
    }

  } catch (e) {
    console.error('Update slot profile error:', e);
  }
}

async function toggleSlotActive(slot, active) {
  try {
    await fetch('/api/update-slot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slot: parseInt(slot), active })
    });
    if (appState && appState.slots && appState.slots[slot]) {
      appState.slots[slot].active = active;
      renderUI();
    }
  } catch (e) {
    console.error('Toggle slot active error:', e);
  }
}

async function updateSettings() {
  const payload = {
    export_format: document.getElementById('selectFormat').value,
    target_lufs: parseFloat(document.getElementById('sliderLufs').value),
    ai_denoise_strength: parseFloat(document.getElementById('sliderAi').value) / 100.0,
    whisper_model: document.getElementById('selectModel').value,
    merge_gap: parseFloat(document.getElementById('selectGap').value),
    transcribe_prompt: document.getElementById('inputPrompt').value,
    enable_moderation: document.getElementById('checkModeration').checked,
    export_srt: document.getElementById('checkExportSrt').checked,
    export_json: document.getElementById('checkExportJson').checked,
  };
  try {
    await fetch('/api/update-settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (e) {
    console.error('Update settings error:', e);
  }
}

async function startMaster() {
  if (currentTab === 'transcriber') {
    startTranscribe();
    return;
  }
  setProcessingUI(true);
  try {
    await fetch('/api/start-master', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preview_sec: null })
    });
  } catch (e) {
    console.error('Start master error:', e);
    setProcessingUI(false);
  }
}

async function startPreview() {
  setProcessingUI(true);
  try {
    await fetch('/api/start-master', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preview_sec: 120.0 })
    });
  } catch (e) {
    console.error('Start preview error:', e);
    setProcessingUI(false);
  }
}

async function startTranscribe() {
  setProcessingUI(true);
  try {
    await fetch('/api/start-transcribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preview_sec: null })
    });
  } catch (e) {
    console.error('Start transcribe error:', e);
    setProcessingUI(false);
  }
}

async function cancelProcessing() {
  try {
    await fetch('/api/cancel', { method: 'POST' });
  } catch (e) {
    console.error('Cancel error:', e);
  }
}

async function openOutputFolder() {
  try {
    await fetch('/api/open-output', { method: 'POST' });
  } catch (e) {
    console.error('Open output error:', e);
  }
}

// ---------------------------------------------------------------------------
// UI Rendering & Channel Strip Construction
// ---------------------------------------------------------------------------
function getDspTags(profile_id) {
  const map = {
    't5_bleed_gate': ['AUTO GATE', 'VOICE LEVELER', 'AI DENOISE', '-18 LUFS'],
    't6_laptop_fan': ['FAN NOTCH', 'DE-RUMBLE', 'PRESENCE', 'AI DENOISE'],
    't1_room_echo': ['ROOM EXPANDER', '500MS HOLD', 'AI DENOISE', '-18 LUFS'],
    't2_muffled': ['AIR SHELF', 'CLARITY EQ', 'AI DENOISE', '-18 LUFS'],
    't3_reference': ['STUDIO COMP', 'TRANSPARENT', 'AI DENOISE', '-18 LUFS'],
    't4_megaphone': ['DE-HARSH', 'WARMTH', 'AI DENOISE', '-18 LUFS'],
    't7_rati_clarity': ['AIR LIFT', 'PRESENCE POLISH', 'AI DENOISE', '-18 LUFS'],
    'ai_rnnoise': ['RNNoise AI', 'VOICE ISOLATOR', 'TRANSPARENT EQ', '-18 LUFS'],
    'silence_only': ['TRANSPARENT GATE', 'VOICE LEVELER', 'NO FILTER', '-18 LUFS'],
    'skip': ['MUTED', 'BYPASS', 'INACTIVE']
  };
  return map[profile_id] || ['RESTORE EQ', 'AI DENOISE', '-18 LUFS'];
}

function renderUI() {
  if (!appState) return;

  // 1. Campaign Switcher
  const btnSw5e = document.getElementById('btnCampaignSw5e');
  const btnRed = document.getElementById('btnCampaignRed');
  if (appState.campaign_mode === 'red') {
    btnRed.className = 'seg-btn active red';
    btnSw5e.className = 'seg-btn';
  } else {
    btnSw5e.className = 'seg-btn active sw5e';
    btnRed.className = 'seg-btn';
  }

  // 2. Hardware / GPU telemetry chip
  const textGpu = document.getElementById('textGpu');
  const dotGpu = document.getElementById('dotGpu');
  if (appState.cuda_available) {
    textGpu.textContent = `RTX GPU (${appState.cuda_device || 'CUDA'})`;
    dotGpu.style.background = '#10b981';
  } else {
    textGpu.textContent = 'Multi-Core CPU (int8)';
    dotGpu.style.background = '#0ea5e9';
  }

  // 3. Render Channel Rack Grid
  const grid = document.getElementById('slotsGrid');
  grid.innerHTML = '';
  let activeCount = 0;

  for (let s = 1; s <= 6; s++) {
    const slot = appState.slots[s];
    if (!slot) continue;
    if (slot.path) activeCount++;

    const strip = document.createElement('div');
    strip.className = `channel-strip ${slot.path ? 'has-audio' : ''} ${!slot.active ? 'disabled' : ''}`;

    const charTag = slot.character ? `<span class="ch-char-tag">${slot.character}</span>` : '';
    const fnDisplay = slot.filename ? slot.filename : 'No audio stem mapped';
    const fnClass = slot.filename ? '' : 'empty';
    const durDisplay = slot.duration_sec > 0 ? formatSec(slot.duration_sec) : '--:--';

    // Profile options dropdown
    let profileOpts = '';
    const choices = appState.profile_choices || [];
    for (const ch of choices) {
      let chId, chName;
      if (typeof ch === 'object' && ch !== null) {
        chId = ch.id;
        chName = ch.name;
      } else if (typeof ch === 'string') {
        chName = ch;
        chId = ch;
      }
      if (!chId) chId = chName;
      if (!chName) chName = chId;

      const selected = (slot.profile_id === chId || slot.profile_name === chName || slot.profile_id === chName) ? 'selected' : '';
      profileOpts += `<option value="${chId}" ${selected}>${chName}</option>`;
    }


    // Active DSP pipeline stages tags
    const tags = getDspTags(slot.profile_id);
    const tagHtml = tags.map(t => `<span class="dsp-tag active">${t}</span>`).join('');

    // Meter segments
    const meterActiveCount = slot.path ? (slot.active ? 5 : 2) : 0;
    let meterBars = '';
    for (let m = 0; m < 8; m++) {
      let colorClass = 'green';
      if (m >= 5) colorClass = 'amber';
      if (m >= 7) colorClass = 'red';
      const isActive = m < meterActiveCount ? 'active' : '';
      meterBars += `<div class="meter-segment ${colorClass} ${isActive}"></div>`;
    }

    strip.innerHTML = `
      <div class="strip-header">
        <div class="strip-id-group">
          <span class="ch-badge">CH 0${s}</span>
          <span class="ch-name">${slot.player}</span>
          ${charTag}
        </div>
        <div class="strip-mute-toggle">
          <button class="mute-btn ${slot.active ? 'active' : 'muted'}" onclick="toggleSlotActive(${s}, ${!slot.active})">
            ${slot.active ? 'ON' : 'M'}
          </button>
        </div>
      </div>

      <div class="strip-body">
        <div class="strip-file-area" onclick="browseSingleFile(${s})" title="Click to map audio file for Slot ${s}">
          <div class="strip-file-row">
            <div class="file-info">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 18V5l12-2v13M9 9l12-2M6 18a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM18 16a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/>
              </svg>
              <span class="file-text ${fnClass}">${fnDisplay}</span>
            </div>
            <span class="file-duration">${durDisplay}</span>
          </div>

          <div class="meter-rack">
            ${meterBars}
          </div>
        </div>

        <div class="dsp-selector-row">
          <div class="dsp-label-row">
            <span>Acoustic Profile</span>
            <span>DSP Pipeline</span>
          </div>
          <select class="dsp-select" onchange="updateSlotProfile(${s}, this.value)">
            ${profileOpts}
          </select>
        </div>

        <div class="dsp-stages-row">
          ${tagHtml}
        </div>
      </div>
    `;

    grid.appendChild(strip);
  }

  // 4. Session Ingestion Toolbar Labels
  const sName = appState.session_source_name ? appState.session_source_name : 'No Session Loaded';
  document.getElementById('sessionNameDisplay').textContent = sName;
  document.getElementById('sessionTrackCountTag').textContent = `${activeCount} Channels Loaded`;
  if (appState.output_dir) {
    document.getElementById('sessionPathDisplay').textContent = appState.output_dir;
    document.getElementById('inputOutputDir').value = appState.output_dir;
  }
  document.getElementById('sessionSummaryText').textContent = `${activeCount} / 6 Channels Active (${appState.campaign_name})`;

  // 5. Render Transcription Speaker Matrix
  renderTranscribeSlots();
}

function renderTranscribeSlots() {
  const tGrid = document.getElementById('transcribeSlotsGrid');
  if (!tGrid || !appState) return;
  tGrid.innerHTML = '';

  for (let s = 1; s <= 6; s++) {
    const slot = appState.slots[s];
    if (!slot) continue;

    const strip = document.createElement('div');
    strip.className = `channel-strip ${slot.path ? 'has-audio' : ''} ${!slot.active ? 'disabled' : ''}`;

    strip.innerHTML = `
      <div class="strip-header">
        <div class="strip-id-group">
          <span class="ch-badge">SPK 0${s}</span>
          <span class="ch-name">${slot.player}</span>
          ${slot.character ? `<span class="ch-char-tag">${slot.character}</span>` : ''}
        </div>
        <button class="mute-btn ${slot.active ? 'active' : 'muted'}" onclick="toggleSlotActive(${s}, ${!slot.active})">
          ${slot.active ? 'ON' : 'M'}
        </button>
      </div>
      <div class="strip-body">
        <div class="strip-file-area" onclick="browseSingleFile(${s})">
          <div class="strip-file-row">
            <span class="file-text ${slot.filename ? '' : 'empty'}">${slot.filename || 'No audio mapped'}</span>
            <span class="file-duration">${slot.duration_sec > 0 ? formatSec(slot.duration_sec) : '--:--'}</span>
          </div>
        </div>
      </div>
    `;
    tGrid.appendChild(strip);
  }
}

function formatSec(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const h = Math.floor(m / 60);
  if (h > 0) {
    const mm = m % 60;
    return `${h}:${mm.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------
// Tabs & UI State Controls
// ---------------------------------------------------------------------------
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.work-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

  if (tab === 'auto-master') {
    document.getElementById('tabAutoMaster').classList.add('active');
    document.getElementById('paneAutoMaster').classList.add('active');
    document.getElementById('btnProcess').innerHTML = `
      <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
      Master Session
    `;
    document.getElementById('btnPreview').style.display = 'inline-flex';
  } else {
    document.getElementById('tabTranscriber').classList.add('active');
    document.getElementById('paneTranscriber').classList.add('active');
    document.getElementById('btnProcess').innerHTML = `
      <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      Transcribe & Merge
    `;
    document.getElementById('btnPreview').style.display = 'none';
  }
}

function setProcessingUI(active) {
  isProcessing = active;
  document.getElementById('btnProcess').disabled = active;
  document.getElementById('btnPreview').disabled = active;
  document.getElementById('btnCancel').disabled = !active;

  const bullet = document.getElementById('statusBullet');
  if (active) {
    bullet.className = 'status-bullet active';
  } else {
    bullet.className = 'status-bullet idle';
  }
}

function updateLufsLabel(val) {
  document.getElementById('valLufs').textContent = `${parseFloat(val).toFixed(1)} LUFS`;
}

function updateAiLabel(val) {
  document.getElementById('valAiStrength').textContent = `${val}%`;
}

// ---------------------------------------------------------------------------
// Console Logging
// ---------------------------------------------------------------------------
function appendConsole(text, type = 'log') {
  const body = document.getElementById('consoleBody');
  if (!body) return;

  const line = document.createElement('div');
  line.className = `log-entry ${type}`;
  line.textContent = text;
  body.appendChild(line);

  while (body.children.length > 500) {
    body.removeChild(body.firstChild);
  }
  body.scrollTop = body.scrollHeight;
}

function clearConsoleLog() {
  const body = document.getElementById('consoleBody');
  if (body) body.innerHTML = '';
}

function copyConsoleLog() {
  const body = document.getElementById('consoleBody');
  if (body) {
    const text = body.innerText;
    navigator.clipboard.writeText(text);
  }
}

// ---------------------------------------------------------------------------
// Drag and Drop Handling
// ---------------------------------------------------------------------------
function setupDragAndDrop() {
  const bar = document.getElementById('sessionBar');
  if (!bar) return;

  ['dragenter', 'dragover'].forEach(name => {
    bar.addEventListener(name, (e) => {
      e.preventDefault();
      bar.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(name => {
    bar.addEventListener(name, (e) => {
      e.preventDefault();
      bar.classList.remove('drag-over');
    }, false);
  });

  bar.addEventListener('drop', async (e) => {
    e.preventDefault();
    bar.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const firstFile = files[0];
      const videoExts = ['.mkv', '.mp4', '.mov', '.webm', '.avi', '.m4v'];
      const isVideo = videoExts.some(ext => firstFile.name.toLowerCase().endsWith(ext));

      if (firstFile.path) {
        if (isVideo) {
          try {
            document.getElementById('dockStatusText').textContent = `[INGESTING 4K VIDEO] ${firstFile.name}...`;
            const res = await fetch('/api/ingest-video', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ video_path: firstFile.path })
            });
            const data = await res.json();
            if (data && data.state) {
              appState = data.state;
              renderUI();
            }
            return;
          } catch (err) {
            console.error('Video drop ingest error:', err);
          }

        }
      } else if (isVideo) {
        browseVideo();
        return;
      }
    }
    browseFolder();
  });
}
