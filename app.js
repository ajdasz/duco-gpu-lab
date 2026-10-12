import { GpuHasher } from "./gpu.js";
import { BatchTuner } from "./batch-tuner.js";

const ids = ["gpuSupport", "gpuName", "startButton", "stopButton", "statusDot", "statusText", "hashrate", "rateUnit", "totalHashes", "goodShares", "badShares", "rateChart", "chartEmpty", "peakRate", "formulaInput", "sampleHash", "targetHash", "matchBadge", "difficulty", "currentNonce", "lastMatch", "latency", "batchTime", "username", "miningKey", "rigName", "batchSize", "autoBatch", "batchHint", "poolUrl", "nodeName", "poolVersion", "uptime", "lastShare", "acceptance", "eventLog", "clearLog"];
const el = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
const numberFormat = new Intl.NumberFormat("en-US");
const credentialKeys = { username: "duco-gpu-lab.username", miningKey: "duco-gpu-lab.miningKey" };
const batchTuner = new BatchTuner([...el.batchSize.options].map(option => Number(option.value)));
let adapterPromise = null;
let webglModelCache = null;
const state = {
  running: false, session: 0, hasher: null, requests: new Set(),
  startedAt: 0, lastShareAt: 0,
  totalHashes: 0, good: 0, bad: 0, rate: 0, peak: 0,
  rateSamples: Array(60).fill(0), latestJob: null,
};

function getMiningAdapter() {
  if (!navigator.gpu) return Promise.resolve(null);
  if (!adapterPromise) {
    adapterPromise = navigator.gpu.requestAdapter({ powerPreference: "high-performance" })
      .catch(error => { adapterPromise = null; throw error; });
  }
  return adapterPromise;
}

function webglGpuModel(info) {
  const vendor = info?.vendor?.trim().toLowerCase();
  if (!vendor) return "";
  if (webglModelCache?.vendor === vendor) return webglModelCache.name;

  let name = "";
  try {
    const gl = document.createElement("canvas").getContext("webgl2", { powerPreference: "high-performance" });
    const extension = gl?.getExtension("WEBGL_debug_renderer_info");
    const renderer = extension ? String(gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) || "") : "";
    const aliases = vendor === "amd" ? ["amd", "ati"] : [vendor];
    if (aliases.some(alias => renderer.toLowerCase().includes(alias))) {
      const angleModel = renderer.match(/^ANGLE \([^,]+,\s*([^,]+)(?:,|\))/i);
      name = (angleModel?.[1] || renderer)
        .replace(/\s*\(0x[\da-f]+\).*$/i, "")
        .replace(/\s+(?:Direct3D\d*|D3D\d*|Vulkan|OpenGL).*$/i, "")
        .trim();
    }
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch { /* Browser may restrict renderer details. */ }
  webglModelCache = { vendor, name };
  return name;
}

function showGpuName(info) {
  const description = info?.description?.trim();
  const device = info?.device?.trim();
  const vendor = info?.vendor?.trim();
  const architecture = info?.architecture?.trim();
  const webgpuModel = description || (device && !/^0x[\da-f]+$/i.test(device) ? device : "");
  const webglModel = webgpuModel ? "" : webglGpuModel(info);
  const name = webgpuModel || webglModel || [vendor, architecture].filter(Boolean).join(" ") || "GPU model unavailable";
  el.gpuName.textContent = name;
  el.gpuName.title = webglModel ? `${name} (reported by WebGL; WebGPU: ${[vendor, architecture].filter(Boolean).join(" ")})` : name;
}

async function detectGpu() {
  if (!navigator.gpu) { showGpuName(null); return; }
  try {
    const adapter = await getMiningAdapter();
    if (state.hasher) return;
    if (!adapter) {
      showGpuName(null);
      el.gpuSupport.textContent = "NO GPU ADAPTER";
      el.gpuSupport.classList.remove("supported");
      return;
    }
    showGpuName(adapter.info);
  } catch {
    if (!state.hasher) showGpuName(null);
  }
}

function setStatus(text, kind = "idle") {
  el.statusText.textContent = text;
  el.statusDot.className = `status-dot ${kind}`;
}

function log(message, kind = "info") {
  const row = document.createElement("li");
  const time = document.createElement("time");
  const type = document.createElement("span");
  const detail = document.createElement("span");
  time.textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
  type.className = `log-type ${kind}`;
  type.textContent = kind.toUpperCase();
  detail.textContent = message;
  row.append(time, type, detail);
  el.eventLog.prepend(row);
  while (el.eventLog.children.length > 80) el.eventLog.lastElementChild.remove();
}

function shortRate(value) {
  if (value >= 1e9) return [value / 1e9, "GH/s"];
  if (value >= 1e6) return [value / 1e6, "MH/s"];
  if (value >= 1e3) return [value / 1e3, "kH/s"];
  return [value, "H/s"];
}

function formatRate(value) {
  const [amount, unit] = shortRate(value);
  return `${amount >= 100 ? amount.toFixed(0) : amount.toFixed(1)} ${unit}`;
}

function formatHashCount(value) {
  if (value < 1000) return numberFormat.format(value);
  const units = [[1e3, "K"], [1e6, "M"], [1e9, "B"], [1e12, "T"], [1e15, "Q"]];
  let index = units.findLastIndex(([scale]) => value >= scale);
  let amount = value / units[index][0];
  if (amount >= 999.5 && index < units.length - 1) amount = value / units[++index][0];
  return `${amount < 100 ? amount.toFixed(1) : amount.toFixed(0)}${units[index][1]}`;
}

function restoreCredentials() {
  try {
    for (const [field, key] of Object.entries(credentialKeys)) {
      const saved = localStorage.getItem(key);
      if (saved !== null) el[field].value = saved;
    }
  } catch { /* Mining can still run when browser storage is unavailable. */ }
}

function saveCredentials() {
  try {
    for (const [field, key] of Object.entries(credentialKeys)) localStorage.setItem(key, el[field].value);
  } catch { /* Mining can still run when browser storage is unavailable. */ }
}

function updateBatchHint() {
  el.batchHint.textContent = el.autoBatch.checked
    ? "Auto adjusts the selected size toward 35 to 65 ms per full batch. Stop waits for the current batch."
    : "Manual size. Check Batch time under Connection details. Stop waits for the current batch.";
}

function renderMetrics() {
  const [amount, unit] = shortRate(state.rate);
  el.hashrate.textContent = amount >= 100 ? amount.toFixed(0) : amount.toFixed(1);
  el.rateUnit.textContent = unit;
  el.totalHashes.textContent = formatHashCount(state.totalHashes);
  el.totalHashes.title = `${numberFormat.format(state.totalHashes)} hashes`;
  el.goodShares.textContent = numberFormat.format(state.good);
  el.badShares.textContent = numberFormat.format(state.bad);
  el.peakRate.textContent = `PEAK ${formatRate(state.peak)}`;
  const shares = state.good + state.bad;
  el.acceptance.textContent = shares ? `${(100 * state.good / shares).toFixed(1)}%` : "--";
}

function drawChart() {
  const canvas = el.rateChart;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * dpr);
  const height = Math.round(rect.height * dpr);
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(170, 180, 186, .16)";
  ctx.lineWidth = dpr;
  for (let i = 1; i < 4; i++) {
    const y = Math.round(height * i / 4);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  if (!state.rateSamples.some(Boolean)) return;
  const max = Math.max(...state.rateSamples, 1) * 1.18;
  const points = state.rateSamples.map((value, i) => ({ x: width * i / 59, y: height - 22 * dpr - (height - 44 * dpr) * value / max }));
  ctx.beginPath();
  points.forEach((point, i) => i ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  const fill = ctx.createLinearGradient(0, 0, 0, height);
  fill.addColorStop(0, "rgba(243, 163, 106, .13)");
  fill.addColorStop(1, "rgba(243, 163, 106, 0)");
  ctx.lineTo(width, height); ctx.lineTo(0, height); ctx.closePath();
  ctx.fillStyle = fill; ctx.fill();
  ctx.beginPath();
  points.forEach((point, i) => i ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.strokeStyle = "#f3a36a"; ctx.lineWidth = 2 * dpr; ctx.lineJoin = "round"; ctx.stroke();
  const last = points.at(-1);
  ctx.beginPath(); ctx.arc(last.x, last.y, 3 * dpr, 0, Math.PI * 2); ctx.fillStyle = "#f3a36a"; ctx.fill();
}

function tick() {
  if (state.startedAt && state.running) {
    const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
    el.uptime.textContent = [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60].map(value => String(value).padStart(2, "0")).join(":");
  }
  if (state.lastShareAt) el.lastShare.textContent = `${Math.floor((Date.now() - state.lastShareAt) / 1000)} s ago`;
  state.rateSamples.push(state.running ? state.rate : 0);
  state.rateSamples.shift();
  drawChart();
}

async function poolRequest(method, params) {
  const url = new URL("/legacy_job", el.poolUrl.value.trim());
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
  url.searchParams.set("nocache", Date.now());
  const controller = new AbortController();
  state.requests.add(controller);
  const timeout = setTimeout(() => controller.abort("Pool request timed out"), 20000);
  try {
    const response = await fetch(url, { method, signal: controller.signal, cache: "no-store" });
    if (!response.ok) throw new Error(`Pool returned HTTP ${response.status}.`);
    return (await response.text()).trim();
  } finally {
    clearTimeout(timeout);
    state.requests.delete(controller);
  }
}

function poolIdentity() {
  return { u: el.username.value.trim(), i: navigator.userAgent };
}

function parseJob(message) {
  const fields = message.split(",");
  if (fields.length !== 3 || !/^[0-9a-f]{40}$/i.test(fields[0]) ||
      !/^[0-9a-f]{40}$/i.test(fields[1]) || !/^\d+$/.test(fields[2])) {
    throw new Error(`Pool did not return a valid mining job: ${message.slice(0, 100)}`);
  }
  const difficulty = Number(fields[2]);
  if (!Number.isSafeInteger(difficulty) || difficulty < 1) throw new Error("Pool sent an invalid difficulty.");
  if (difficulty > Math.floor(GpuHasher.maxNonce / 100)) {
    throw new Error("Pool difficulty exceeds the supported GPU nonce range.");
  }
  return { previous: fields[0], target: fields[1].toLowerCase(), difficulty };
}

async function verifyMatch(prefix, nonce, target) {
  const bytes = new TextEncoder().encode(prefix + nonce);
  const digest = await crypto.subtle.digest("SHA-1", bytes);
  const hex = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, "0")).join("");
  return hex === target.toLowerCase();
}

async function mineJob(job, session) {
  const maxNonce = job.difficulty * 100;
  let nextNonce = 0;
  let attempted = 0;
  const started = performance.now();
  let lastDisplay = 0;
  setStatus("GPU searching nonces", "active");
  while (nextNonce <= maxNonce && state.running && session === state.session) {
    const selectedBatchSize = Number(el.batchSize.value);
    const count = Math.min(selectedBatchSize, maxNonce - nextNonce + 1);
    const batchStarted = performance.now();
    const result = await state.hasher.hashBatch(job.previous, job.target, nextNonce, count);
    if (!state.running || session !== state.session) return null;
    const batchMs = performance.now() - batchStarted;
    el.batchTime.textContent = `${Math.round(batchMs)} ms`;
    if (el.autoBatch.checked && Number(el.batchSize.value) === selectedBatchSize && count === selectedBatchSize) {
      el.batchSize.value = String(batchTuner.observe(selectedBatchSize, batchMs));
    }
    attempted += result.count;
    state.totalHashes += result.count;
    nextNonce += result.count;
    const seconds = Math.max((performance.now() - started) / 1000, 0.001);
    state.rate = attempted / seconds;
    state.peak = Math.max(state.peak, state.rate);
    const now = performance.now();
    if (result.found !== null || nextNonce > maxNonce || now - lastDisplay >= 200) {
      el.currentNonce.textContent = numberFormat.format(nextNonce - 1);
      el.formulaInput.textContent = `${job.previous} + ${result.start}`;
      el.sampleHash.textContent = result.sample;
      el.chartEmpty.hidden = true;
      renderMetrics();
      lastDisplay = now;
    }
    if (result.found !== null) {
      if (!await verifyMatch(job.previous, result.found, job.target)) throw new Error("GPU result failed independent SHA-1 verification.");
      if (!state.running || session !== state.session) return null;
      el.lastMatch.textContent = numberFormat.format(result.found);
      el.matchBadge.textContent = "MATCH FOUND";
      el.matchBadge.classList.add("matched");
      log(`GPU found nonce ${result.found}. SHA-1 matches the pool target.`, "good");
      return { nonce: result.found, rate: state.rate, seconds };
    }
    // mapAsync already yields to the browser between batches.
  }
  if (!state.running || session !== state.session) return null;
  log("No matching nonce within the full difficulty range. Requesting a new job.", "warn");
  return { exhausted: true };
}

async function runSession(session) {
  let retryCount = 0;
  while (state.running && session === state.session) {
    try {
      setStatus("Requesting a pool job", "working");
      const job = parseJob(await poolRequest("GET", poolIdentity()));
      if (!state.running || session !== state.session) return;
      retryCount = 0;
      state.latestJob = job;
      el.poolVersion.textContent = "Legacy HTTP";
      el.difficulty.textContent = numberFormat.format(job.difficulty);
      el.targetHash.textContent = job.target;
      el.formulaInput.textContent = `${job.previous} + nonce`;
      el.sampleHash.textContent = "Searching on GPU";
      el.matchBadge.textContent = "SEARCHING";
      el.matchBadge.classList.remove("matched");
      log(`New job received at difficulty ${numberFormat.format(job.difficulty)}.`);
      const solution = await mineJob(job, session);
      if (!solution || !state.running || session !== state.session) return;
      if (solution.exhausted) {
        await new Promise(resolve => setTimeout(resolve, 500));
        continue;
      }

      const params = {
        ...poolIdentity(), r: solution.nonce, k: el.miningKey.value,
        s: `WebGPU Miner 1.0 (${el.rigName.value.trim() || "WebGPU-Lab"})`,
        j: job.target, h: solution.rate.toFixed(2), b: solution.seconds.toFixed(2),
      };
      setStatus("Waiting for pool verdict", "working");
      log(`Submitted nonce ${solution.nonce} at ${formatRate(solution.rate)}.`, "submit");
      const submittedAt = performance.now();
      const verdict = await poolRequest("POST", params);
      if (!state.running || session !== state.session) return;
      el.latency.textContent = `${Math.round(performance.now() - submittedAt)} ms`;
      state.lastShareAt = Date.now();
      if (verdict === "GOOD" || verdict === "BLOCK") {
        state.good++;
        log(verdict === "BLOCK" ? "Pool accepted a block." : "Pool accepted the share.", "good");
      } else if (verdict.startsWith("BAD")) {
        state.bad++;
        log(`Pool rejected the share: ${verdict.slice(4) || "no reason given"}.`, "bad");
      } else {
        log(`Unexpected pool verdict: ${verdict.slice(0, 120)}`, "warn");
      }
      renderMetrics();
      await new Promise(resolve => setTimeout(resolve, 500));
    } catch (error) {
      if (!state.running || session !== state.session) return;
      if (/GPU|shader|invalid mining job|invalid difficulty|difficulty exceeds|SHA-1 verification/i.test(error.message)) { fail(error); return; }
      retryCount++;
      const delay = Math.min(retryCount * retryCount, 30) * 1000;
      state.rate = 0;
      renderMetrics();
      setStatus(`Pool error. Retry in ${delay / 1000} s`, "working");
      log(`Pool request failed: ${error.message}. Retrying.`, "warn");
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

function fail(error) {
  log(error.message || String(error), "bad");
  stop();
  setStatus(error.message || "Mining stopped after an error", "error");
}

async function start() {
  if (state.running) return;
  const username = el.username.value.trim();
  if (!/^[A-Za-z0-9_]{1,32}$/.test(username)) {
    setStatus("Enter a valid Duino-Coin username", "error");
    el.username.focus();
    return;
  }
  if (/[,\r\n]/.test(el.rigName.value)) { fail(new Error("Rig name cannot contain commas or line breaks.")); return; }
  try {
    const endpoint = new URL(el.poolUrl.value.trim());
    if (endpoint.protocol !== "https:" || endpoint.username || endpoint.password || endpoint.search || endpoint.hash) throw new Error();
  } catch { fail(new Error("Enter a valid pool API URL beginning with https://.")); return; }
  el.username.value = username;
  saveCredentials();
  state.running = true;
  document.body.classList.add("mining");
  const session = ++state.session;
  state.startedAt = Date.now();
  state.lastShareAt = 0;
  state.totalHashes = state.good = state.bad = state.rate = state.peak = 0;
  state.rateSamples.fill(0);
  batchTuner.reset();
  el.uptime.textContent = "00:00:00";
  el.lastShare.textContent = "--";
  el.latency.textContent = "-- ms";
  el.batchTime.textContent = "-- ms";
  el.difficulty.textContent = "0";
  el.currentNonce.textContent = "0";
  el.lastMatch.textContent = "None yet";
  el.chartEmpty.hidden = false;
  renderMetrics();
  el.nodeName.textContent = new URL(el.poolUrl.value.trim()).host;
  el.poolVersion.textContent = "--";
  el.startButton.disabled = true;
  el.stopButton.disabled = false;
  for (const field of [el.username, el.miningKey, el.rigName, el.poolUrl]) field.disabled = true;
  setStatus("Initializing WebGPU", "working");
  log("Requesting high-performance WebGPU adapter and running SHA-1 self-test.");
  try {
    if (!state.hasher) state.hasher = await GpuHasher.create(await getMiningAdapter());
    if (!state.running || session !== state.session) return;
    showGpuName(state.hasher.adapterInfo);
    el.gpuSupport.textContent = "GPU VERIFIED";
    el.gpuSupport.classList.add("supported");
    log("GPU SHA-1 self-tests passed, including a real pool job.", "good");
    state.hasher.device.lost.then(info => {
      state.hasher = null;
      adapterPromise = null;
      webglModelCache = null;
      if (state.running) fail(new Error(`GPU device lost: ${info.message || "unknown reason"}`));
    });
    log(`Connecting to ${el.nodeName.textContent} with the Mini Miner HTTP protocol.`);
    runSession(session);
  } catch (error) { if (state.running && session === state.session) fail(error); }
}

function stop() {
  state.running = false;
  document.body.classList.remove("mining");
  state.session++;
  for (const request of state.requests) request.abort("Mining stopped");
  state.rate = 0;
  renderMetrics();
  el.startButton.disabled = false;
  el.stopButton.disabled = true;
  for (const field of [el.username, el.miningKey, el.rigName, el.poolUrl]) field.disabled = false;
  setStatus("Stopped", "idle");
  log("Mining stopped.");
}

el.startButton.addEventListener("click", start);
el.stopButton.addEventListener("click", stop);
el.autoBatch.addEventListener("change", () => { batchTuner.reset(); updateBatchHint(); });
el.batchSize.addEventListener("change", () => { el.autoBatch.checked = false; batchTuner.reset(); updateBatchHint(); });
for (const field of [el.username, el.miningKey]) field.addEventListener("input", saveCredentials);
el.clearLog.addEventListener("click", () => { el.eventLog.replaceChildren(); log("Event log cleared."); });
window.addEventListener("resize", drawChart);
restoreCredentials();
updateBatchHint();
el.gpuSupport.textContent = navigator.gpu ? "WEBGPU AVAILABLE" : "WEBGPU UNAVAILABLE";
el.gpuSupport.classList.toggle("supported", Boolean(navigator.gpu));
detectGpu();
renderMetrics();
drawChart();
setInterval(tick, 1000);
