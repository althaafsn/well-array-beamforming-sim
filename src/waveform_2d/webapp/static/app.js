"use strict";

const view = document.getElementById("view");
const ctx = view.getContext("2d");
const statusEl = document.getElementById("status");
const dot = document.getElementById("dot");
const fpsEl = document.getElementById("fps");

// Offscreen buffer at simulation resolution; the visible canvas scales it up
// (browser bilinear smoothing -> continuous RippleGL-like field).
let grid = { w: 240, h: 240 };
let buf = ctx.createImageData(grid.w, grid.h);
const off = document.createElement("canvas");
const offCtx = off.getContext("2d");

let ws = null;
let frames = 0;
let lastFps = performance.now();

function setStatus(text, on) {
  statusEl.textContent = text;
  dot.classList.toggle("on", !!on);
}

function resizeBuffers(w, h) {
  grid = { w, h };
  view.width = w;
  view.height = h;
  off.width = w;
  off.height = h;
  buf = offCtx.createImageData(w, h);
}

function paint(bytes) {
  buf.data.set(new Uint8Array(bytes));
  offCtx.putImageData(buf, 0, 0);
  ctx.drawImage(off, 0, 0, grid.w, grid.h);
  frames++;
  const now = performance.now();
  if (now - lastFps > 1000) {
    fpsEl.textContent = (frames * 1000 / (now - lastFps)).toFixed(0);
    frames = 0;
    lastFps = now;
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("live", true);
  ws.onclose = () => { setStatus("disconnected — retrying…", false); setTimeout(connect, 1200); };
  ws.onerror = () => ws.close();

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.type === "init") {
        resizeBuffers(m.width, m.height);
        populate(m.scenes, m.colormaps);
      }
      return;
    }
    paint(ev.data);
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

// ---- controls ----
const sceneSel = document.getElementById("scene");
const cmapSel = document.getElementById("colormap");
const freq = document.getElementById("frequency");
const bright = document.getElementById("brightness");
const absorbing = document.getElementById("absorbing");

function populate(scenes, colormaps) {
  if (!sceneSel.options.length) {
    scenes.forEach((s) => sceneSel.add(new Option(s.replace("_", " "), s)));
    colormaps.forEach((c) => cmapSel.add(new Option(c, c)));
  }
}

sceneSel.onchange = () => send({ type: "scene", name: sceneSel.value });
cmapSel.onchange = () => send({ type: "param", key: "colormap", value: cmapSel.value });

freq.oninput = () => {
  document.getElementById("freqval").textContent = (+freq.value).toFixed(2);
  send({ type: "param", key: "frequency", value: +freq.value });
};
bright.oninput = () => {
  document.getElementById("brightval").textContent = (+bright.value).toFixed(2);
  send({ type: "param", key: "brightness", value: +bright.value });
};
absorbing.onchange = () => send({ type: "param", key: "absorbing", value: absorbing.checked });

let paused = false;
document.getElementById("pause").onclick = (e) => {
  paused = !paused;
  e.target.textContent = paused ? "Play" : "Pause";
  send({ type: "param", key: "paused", value: paused });
};
document.getElementById("clear").onclick = () => send({ type: "clear" });
document.getElementById("reset").onclick = () => send({ type: "reset" });

// ---- pointer -> field coordinates ----
function relCoords(ev) {
  const r = view.getBoundingClientRect();
  return { x: (ev.clientX - r.left) / r.width, y: (ev.clientY - r.top) / r.height };
}
view.addEventListener("click", (ev) => {
  const { x, y } = relCoords(ev);
  send({ type: ev.shiftKey ? "oscillator" : "drip", x, y });
});

document.getElementById("freqval").textContent = (+freq.value).toFixed(2);
document.getElementById("brightval").textContent = (+bright.value).toFixed(2);
connect();
