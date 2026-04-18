// ── State ──
let stream = null;
let running = false;
let mode = 'local'; // 'local' | 'backend'
let ws = null;
let frameCount = 0;
let alertCount = 0;
let fireDetected = false;
let lastAlarmTime = 0;
let fps = 0;
let fpsFrameBuffer = 0;
let lastFpsTime = Date.now();
let backendConnected = false;
let confidence = 0;
let detectionInterval = null;
let videoDetectionInterval = null;
const BACKEND_URL = 'http://localhost:8000';

// ── ML Model ──
let model = null;
let modelLoaded = false;

async function loadModel() {
  try {
    model = await cocoSsd.load();
    modelLoaded = true;
    console.log('COCO-SSD model loaded');
  } catch (e) {
    console.error('Failed to load model:', e);
  }
}

// Load model on script load
loadModel();

// ── Audio Alarm ──
let alarmAudio = null;

// ── Clock ──
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toTimeString().slice(0,8);
}
setInterval(updateClock, 1000);
updateClock();

// ── Mode Toggle ──
function setMode(m) {
  mode = m;
  document.getElementById('btn-local').classList.toggle('active', m === 'local');
  document.getElementById('btn-backend').classList.toggle('active', m === 'backend');
  document.getElementById('mode-tag').textContent = m === 'local' ? 'LOCAL SIM' : 'BACKEND AI';
  document.getElementById('mode-tag').className = 'tag ' + (m === 'local' ? 'tag-warn' : 'tag-online');
}

// ── Backend Connection ──
async function tryConnectBackend() {
  try {
    const res = await fetch(`${BACKEND_URL}/status`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      backendConnected = true;
      document.getElementById('api-tag').textContent = 'ONLINE';
      document.getElementById('api-tag').className = 'tag tag-online';
      connectWebSocket();
    }
  } catch (e) {
    backendConnected = false;
    document.getElementById('api-tag').textContent = 'OFFLINE';
    document.getElementById('api-tag').className = 'tag tag-offline';
  }
}

// ── WebSocket Connection ──
function connectWebSocket() {
  if (ws) ws.close();
  try {
    ws = new WebSocket(`ws://localhost:8000/ws`);
    ws.onopen = () => {
      console.log('WebSocket Connected');
      document.getElementById('ws-tag').textContent = 'CONNECTED';
      document.getElementById('ws-tag').className = 'tag tag-online';
      document.getElementById('ws-dot').className = 'conn-dot connected';
      document.getElementById('ws-label').textContent = 'BACKEND LIVE';
    };
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (mode === 'backend') {
        processBackendResult(msg);
      }
    };
    ws.onclose = () => {
      document.getElementById('ws-tag').textContent = 'DISCONNECTED';
      document.getElementById('ws-tag').className = 'tag tag-offline';
      document.getElementById('ws-dot').className = 'conn-dot';
      document.getElementById('ws-label').textContent = 'WEBCAM ONLY';
    };
  } catch (e) {
    console.error('WS Error:', e);
  }
}

function processBackendResult(data) {
  const conf = data.confidence || 0;
  const detections = data.detections || [];
  
  if (data.fire_detected || data.smoke_detected) {
    triggerFireAlert(conf, 'BACKEND YOLOv8');
  } else {
    if (fireDetected) clearFireAlert();
  }
  
  updateConfidence(conf);
  
  // Draw all YOLO boxes
  if (detections.length > 0) {
    drawYOLOBoxes(detections);
  } else {
    clearOverlay();
  }
}

function drawYOLOBoxes(detections) {
  const canvas = document.getElementById('overlay-canvas');
  const video = document.getElementById('webcam');
  if (!canvas || !video) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Scaling factor if video display size != video intrinsic size
  const scaleX = canvas.width / video.videoWidth;
  const scaleY = canvas.height / video.videoHeight;

  detections.forEach(det => {
    let [x1, y1, x2, y2] = det.box;
    
    // Apply scaling
    x1 *= scaleX; y1 *= scaleY;
    x2 *= scaleX; y2 *= scaleY;

    const label = det.class.toUpperCase();
    const conf = Math.round(det.conf * 100);
    
    ctx.strokeStyle = label === "FIRE" ? "#ff2020" : "#aaaaaa";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "bold 14px monospace";
    // Draw background for text
    const text = `${label} ${conf}%`;
    const txtWidth = ctx.measureText(text).width;
    ctx.fillRect(x1, y1 - 20, txtWidth + 10, 20);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, x1 + 5, y1 - 5);
  });
}


// ── Local AI / Frame Analysis ──
async function analyzeFrame() {
  const video = document.getElementById('webcam');
  if (!video || video.readyState < 2) return;

  frameCount++;
  fpsFrameBuffer++;

  const now = Date.now();
  if (now - lastFpsTime >= 1000) {
    fps = fpsFrameBuffer;
    fpsFrameBuffer = 0;
    lastFpsTime = now;
  }

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  if (mode === 'backend') {
    // SEND TO BACKEND
    if (ws && ws.readyState === WebSocket.OPEN) {
      canvas.toBlob((blob) => {
        if (blob) ws.send(blob);
      }, 'image/jpeg', 0.8);
    } else {
      // Fallback to REST if WS is down
      try {
        canvas.toBlob(async (blob) => {
          if (!blob) return;
          try {
            const res = await fetch(`${BACKEND_URL}/detect`, {
              method: 'POST',
              body: blob
            });
            const data = await res.json();
            processBackendResult(data);
          } catch (e) {}
        }, 'image/jpeg', 0.7);
      } catch (e) {}
    }
  } else {
    // LOCAL AI SIMULATION
    let detected = false;
    let conf = 0;

    if (modelLoaded) {
      try {
        const predictions = await model.detect(canvas);
        const firePred = predictions.find(p => /fire|smoke/i.test(p.class));
        if (firePred && firePred.score > 0.4) {
          detected = true;
          conf = firePred.score;
          drawYOLOBoxes([{
            class: firePred.class,
            conf: firePred.score,
            box: [firePred.bbox[0], firePred.bbox[1], firePred.bbox[0]+firePred.bbox[2], firePred.bbox[1]+firePred.bbox[3]]
          }]);
        } else {
          const result = analyzeColor(canvas);
          detected = result.detected;
          conf = result.conf;
          if (detected) {
            drawDetectionBox(video, conf, result.type);
          } else {
            clearOverlay();
          }
        }
      } catch (e) {
        const result = analyzeColor(canvas);
        detected = result.detected;
        conf = detected ? 0.5 : 0;
      }
    } else {
      const result = analyzeColor(canvas);
      detected = result.detected;
      conf = detected ? 0.5 : 0;
      if (detected) drawDetectionBox(video, conf, result.type);
    }

    updateConfidence(conf);
    if (detected && mode === 'local') {
      if (!fireDetected) triggerFireAlert(conf, 'LOCAL AI');
    } else {
      if (fireDetected) clearFireAlert();
    }
  }

  updateStats();
  document.getElementById('frame-count').textContent = frameCount;
}



// Try backend on load
tryConnectBackend();

// ── Webcam ──
async function startStream() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
      audio: false
    });
    const video = document.getElementById('webcam');
    video.srcObject = stream;
    video.style.display = 'block';
    document.getElementById('video-idle').style.display = 'none';

    video.addEventListener('loadedmetadata', () => {
      const canvas = document.getElementById('overlay-canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    });

    running = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;

    startProcessing();
  } catch (err) {
    alert('Camera access denied or not available.\n\n' + err.message);
  }
}

function stopStream() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  running = false;
  const video = document.getElementById('webcam');
  video.style.display = 'none';
  video.srcObject = null;
  document.getElementById('video-idle').style.display = 'flex';
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;
  clearInterval(detectionInterval);
  clearFireAlert();
  frameCount = 0;
  updateStats();
}

// ── Local AI Simulation Logic ──
async function analyzeFrame_Deprecated() {

  const video = document.getElementById('webcam');
  if (!video || video.readyState < 2) return;

  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 240;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, 320, 240);

  frameCount++;
  fpsFrameBuffer++;

  const now = Date.now();
  if (now - lastFpsTime >= 1000) {
    fps = fpsFrameBuffer;
    fpsFrameBuffer = 0;
    lastFpsTime = now;
  }

  let detected = false;
  let conf = 0;

  if (modelLoaded) {
    try {
      const predictions = await model.detect(canvas);
      console.log('Predictions:', predictions); // Debug log
      // Only use model output if it actually contains fire/smoke classes
      const firePred = predictions.find(p => /fire|smoke/i.test(p.class));
      if (firePred && firePred.score > 0.4) {
        detected = true;
        conf = firePred.score;
        drawDetectionBox(video, conf);
      } else {
        // Do not treat human hands or people as fire; use color analysis instead
        const result = analyzeColor(canvas);
        detected = result.detected;
        conf = result.conf;

        if (detected) {
          drawDetectionBox(video, conf, result.type);
        } else {
      clearOverlay();
}
        
      }
    } catch (e) {
      console.error('Detection error:', e);
      detected = analyzeColor(canvas);
      conf = detected ? 0.5 : 0;
    }
  } else {
    console.log('Model not loaded, using color detection');
    detected = analyzeColor(canvas);
    conf = detected ? 0.5 : 0;
  }

  updateConfidence(conf);

  if (detected && mode === 'local') {
    if (!fireDetected) triggerFireAlert(conf, 'LOCAL AI');
  } else {
    if (fireDetected) clearFireAlert();
  }

  updateStats();
  document.getElementById('frame-count').textContent = frameCount;
}

function analyzeColor(canvas) {
  const ctx = canvas.getContext('2d');
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const d = imgData.data;

  let firePixels = 0;
  let smokePixels = 0;
  let total = 0;

  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i+1], b = d[i+2];
    total++;

    // 🔥 REALISTIC FIRE MODEL (channel relationship)
    const fireLike =
      r > 180 &&          // strong red
      g > 80 &&           // some green (yellow/orange)
      g < r &&            // red dominant
      b < g &&            // blue lowest
      (r - g) > 40;       // strong contrast

    // 🔥 BRIGHT FLAME (white/yellow core)
    const brightFlame =
      r > 50 &&
      g > 60 &&
      b < 30;

    // 🌫 SMOKE DETECTION (fixed)
    const brightness = (r + g + b) / 3;
    const colorDiff = Math.max(r, g, b) - Math.min(r, g, b);

    const smoke =
      brightness > 320 &&       // not too dark
      brightness < 430 &&       // not pure white
      colorDiff < 55 &&         // low saturation (gray)
      !(b > r && b > g);        // avoid sky/blue

    if (fireLike || brightFlame) firePixels++;
    if (smoke) smokePixels++;
  }

  const fireRatio = firePixels / total;
  const smokeRatio = smokePixels / total;

  // 🚫 IGNORE tiny noise (VERY IMPORTANT)
  if (firePixels < 50 && smokePixels < 50) {
    return { detected: false, type: "", conf: 0 };
  }

  // 🎯 Threshold tuning
  if (fireRatio > 0.03) {
    return {
      detected: true,
      type: "FIRE",
      conf: Math.min(1, fireRatio * 6)
    };
  }

  if (smokeRatio > 0.02) {
    return {
      detected: true,
      type: "SMOKE",
      conf: Math.min(1, smokeRatio * 5)
    };
  }

  return { detected: false, type: "", conf: 0 };
}

function drawDetectionBox(video, conf, type = "FIRE") {
  const canvas = document.getElementById('overlay-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const w = canvas.width, h = canvas.height;
  const bw = w * 0.4, bh = h * 0.4;
  const bx = w * 0.3, by = h * 0.3;

  ctx.strokeStyle = type === "SMOKE" ? '#aaaaaa' : '#ff2020';
  ctx.lineWidth = 2;
  ctx.strokeRect(bx, by, bw, bh);

  ctx.fillStyle = ctx.strokeStyle;
  ctx.font = 'bold 13px monospace';
  ctx.fillText(`${type} ${Math.round(conf * 100)}%`, bx + 4, by - 6);
}
function clearOverlay() {
  const canvas = document.getElementById('overlay-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function startProcessing() {
  detectionInterval = setInterval(async () => {
    if (running) await analyzeFrame();
  }, 200); // 5 fps for model inference
}

// ── Alert Management ──
function triggerFireAlert(conf, source) {
  fireDetected = true;
  alertCount++;

  const banner = document.getElementById('alert-banner');
  banner.className = 'alert-banner fire';

  document.getElementById('alert-label').textContent = '⚠ FIRE DETECTED — ALERT ACTIVE';
  document.getElementById('alert-sub').textContent = `Source: ${source} | Confidence: ${(conf * 100).toFixed(1)}%`;
  document.getElementById('conf-big').textContent = (conf * 100).toFixed(1) + '%';
  document.getElementById('alert-icon-svg').innerHTML = `
    <path d="M16 4C16 4 8 12 8 20C8 24.4183 11.5817 28 16 28C20.4183 28 24 24.4183 24 20C24 12 16 4 16 4Z" fill="#ff2020" opacity="0.9"/>
    <path d="M16 14C16 14 12 18 12 21C12 23.2091 13.7909 25 16 25C18.2091 25 20 23.2091 20 21C20 18 16 14 16 14Z" fill="#ffaa00"/>
  `;

  document.getElementById('fire-border').classList.add('active');
  document.getElementById('ws-dot').className = 'conn-dot fire';

  addAlertToLog(conf, source);
  playAlarm();
  document.getElementById('stat-alerts').textContent = alertCount;
}

function clearFireAlert() {
  fireDetected = false;
  const banner = document.getElementById('alert-banner');
  banner.className = 'alert-banner safe';
  document.getElementById('alert-label').textContent = 'ALL CLEAR — NO FIRE DETECTED';
  document.getElementById('alert-sub').textContent = 'System monitoring active';
  document.getElementById('fire-border').classList.remove('active');
  document.getElementById('ws-dot').className = backendConnected ? 'conn-dot connected' : 'conn-dot';
  document.getElementById('alert-icon-svg').innerHTML = `
    <circle cx="16" cy="16" r="13" stroke="#00e676" stroke-width="2"/>
    <path d="M10 16L14 20L22 12" stroke="#00e676" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  `;
  clearOverlay();
  stopAlarm();
}

function updateConfidence(conf) {
  confidence = conf;
  const pct = Math.round(conf * 100);
  document.getElementById('conf-big').textContent = pct + '%';
  document.getElementById('stat-conf').textContent = pct + '%';

  const bar = document.getElementById('conf-bar');
  bar.style.width = pct + '%';
  bar.className = 'conf-bar-fill' + (pct > 60 ? ' high' : pct > 30 ? ' mid' : '');
}

function updateStats() {
  document.getElementById('stat-frames').textContent = frameCount;
  document.getElementById('stat-fps').textContent = fps;
  const fpsPct = Math.min(100, (fps / 30) * 100);
  document.getElementById('fps-bar').style.width = fpsPct + '%';
  document.getElementById('fps-num').textContent = fps + ' fps';
}

function addAlertToLog(conf, source) {
  const list = document.getElementById('alerts-list');
  const noMsg = document.getElementById('no-alerts-msg');
  if (noMsg) noMsg.remove();

  const now = new Date();
  const timeStr = now.toTimeString().slice(0, 8);
  const item = document.createElement('div');
  item.className = 'alert-item';
  item.innerHTML = `
    <div class="alert-dot"></div>
    <div>
      <div class="alert-time">${timeStr} — ${source}</div>
      <div class="alert-conf-badge">Conf: ${(conf * 100).toFixed(1)}% | Frame #${frameCount}</div>
    </div>
  `;
  list.insertBefore(item, list.firstChild);

  // Keep max 30
  while (list.children.length > 30) list.removeChild(list.lastChild);
}

function clearAlerts() {
  const list = document.getElementById('alerts-list');
  list.innerHTML = '<div class="no-alerts" id="no-alerts-msg">— No alerts yet —</div>';
  alertCount = 0;
  document.getElementById('stat-alerts').textContent = 0;
}

// ── Alarm Sound ──
function playAlarm() {
  if (!alarmAudio) {
    alarmAudio = document.getElementById('alarm-audio');
  }
  if (alarmAudio && !alarmAudio.paused) return; // already playing

  alarmAudio.play().catch(err => {
    console.log('Audio play failed:', err);
  });
}

function stopAlarm() {
  if (alarmAudio) {
    alarmAudio.pause();
    alarmAudio.currentTime = 0;
  }
}
// ── Backend polling (when connected) ──
setInterval(async () => {
  if (!backendConnected || mode !== 'backend') return;
  try {
    const res = await fetch(`${BACKEND_URL}/status`, { signal: AbortSignal.timeout(1000) });
    if (res.ok) {
      const data = await res.json();
      if (data.fire_detected) {
        triggerFireAlert(data.confidence, 'BACKEND');
      } else {
        if (fireDetected) clearFireAlert();
        updateConfidence(data.confidence || 0);
      }
      if (data.frames_processed !== undefined) {
        document.getElementById('backend-frames').textContent = data.frames_processed;
      }
    }
  } catch (e) {
    backendConnected = false;
    document.getElementById('api-tag').textContent = 'OFFLINE';
    document.getElementById('api-tag').className = 'tag tag-offline';
  }
}, 1000);