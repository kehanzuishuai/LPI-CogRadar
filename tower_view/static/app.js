"use strict";

const el = id => document.getElementById(id);
const canvas = el("towerCanvas");
const ctx = canvas.getContext("2d");
let replay = null;
let frameIndex = 0;
let selectedGlobalId = null;
let timer = null;
let bounds = { minX: -8000, maxX: 8000, minY: -8000, maxY: 8000 };

const colors = ["#32d6d0", "#ffb94e", "#dd7cf6", "#6bdd8a", "#ff716d", "#8aa9ff"];
const colorFor = id => colors[Math.abs([...id].reduce((a,c) => a + c.charCodeAt(0), 0)) % colors.length];
const fmt = (value, digits=1) => value === null || value === undefined ? "—" : Number(value).toFixed(digits);
const vec = values => values ? `[${values.map(v => fmt(v, 1)).join(", ")}]` : "—";

async function loadManifest() {
  const response = await fetch("/api/replays");
  if (!response.ok) throw new Error(`回放清单加载失败: ${response.status}`);
  const entries = await response.json();
  const select = el("scenarioSelect");
  select.innerHTML = entries.map(item => `<option value="${item.url}">${item.title}</option>`).join("");
  const requested = new URLSearchParams(location.search).get("replay");
  if (requested && entries.some(item => item.url === requested)) select.value = requested;
  select.addEventListener("change", () => loadReplay(select.value));
  if (!entries.length) throw new Error("没有可用回放，请先运行 --generate-only");
  await loadReplay(select.value || entries[0].url);
}

async function loadReplay(url) {
  stop();
  el("statusPill").textContent = "正在载入";
  const response = await fetch(url);
  if (!response.ok) throw new Error(`回放加载失败: ${response.status}`);
  const data = await response.json();
  if (data.schema_version !== "tower-view-v1") throw new Error(`不支持的 schema: ${data.schema_version}`);
  replay = data;
  frameIndex = 0;
  selectedGlobalId = null;
  el("scenarioTitle").textContent = data.scenario.title;
  el("timeline").max = Math.max(0, data.frames.length - 1);
  bounds = computeBounds(data.frames);
  const note = data.summary.frozen_evaluation_annotation;
  const banner = el("negativeBanner");
  if (note) {
    banner.textContent = `已冻结关联负结果：ID switch ${note.global_id_switches} · fragmentation ${note.fragmentation} · duplicate ${note.duplicate_global_tracks_mean}`;
    banner.classList.remove("hidden");
  } else banner.classList.add("hidden");
  const debug = Boolean(data.debug_overlay && data.debug_overlay.enabled);
  const debugBanner = el("debugBanner");
  debugBanner.classList.toggle("hidden", !debug);
  el("statusPill").textContent = debug
    ? `DEBUG / GROUND TRUTH · ${data.frames.length} 帧`
    : `只读回放 · ${data.frames.length} 帧`;
  render();
}

function computeBounds(frames) {
  const points = [];
  frames.forEach(frame => {
    frame.radar_nodes.forEach(row => points.push(row.position_m));
    frame.local_tracks.forEach(row => points.push(row.position_m));
    frame.global_tracks.forEach(row => points.push(row.position_m));
    (frame.debug_truth_tracks || []).forEach(row => points.push(row.position_m));
  });
  if (!points.length) return { minX:-8000,maxX:8000,minY:-8000,maxY:8000 };
  let minX = Math.min(...points.map(p => p[0])), maxX = Math.max(...points.map(p => p[0]));
  let minY = Math.min(...points.map(p => p[1])), maxY = Math.max(...points.map(p => p[1]));
  const padX = Math.max(1200, (maxX-minX)*.12), padY = Math.max(1200, (maxY-minY)*.12);
  return { minX:minX-padX, maxX:maxX+padX, minY:minY-padY, maxY:maxY+padY };
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return rect;
}

function project(position, rect) {
  const pad = 34;
  const x = pad + (position[0]-bounds.minX)/(bounds.maxX-bounds.minX)*(rect.width-pad*2);
  const y = rect.height-pad-(position[1]-bounds.minY)/(bounds.maxY-bounds.minY)*(rect.height-pad*2);
  return [x,y];
}

function drawGridAxes(rect) {
  const origin = project([0,0,0], rect);
  ctx.save(); ctx.strokeStyle = "rgba(88,126,145,.45)"; ctx.setLineDash([5,5]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, origin[1]); ctx.lineTo(rect.width, origin[1]); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(origin[0], 0); ctx.lineTo(origin[0], rect.height); ctx.stroke(); ctx.restore();
}

function drawTrail(globalId, rect, strong) {
  const points = [];
  for (let i=0; i<=frameIndex; i++) {
    const row = replay.frames[i].global_tracks.find(t => t.global_track_id === globalId);
    if (row) points.push(project(row.position_m, rect));
  }
  if (points.length < 2) return;
  ctx.save(); ctx.strokeStyle = colorFor(globalId); ctx.globalAlpha = strong ? .95 : .42; ctx.lineWidth = strong ? 3 : 1.5;
  ctx.beginPath(); points.forEach((p,i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.stroke(); ctx.restore();
}

function drawFrame(frame, rect) {
  drawGridAxes(rect);
  frame.global_tracks.forEach(track => drawTrail(track.global_track_id, rect, !selectedGlobalId || selectedGlobalId === track.global_track_id));
  const globalById = Object.fromEntries(frame.global_tracks.map(row => [row.global_track_id, row]));
  frame.local_tracks.forEach(local => {
    if (!local.mapped_global_track_id || !globalById[local.mapped_global_track_id]) return;
    const a = project(local.position_m, rect), b = project(globalById[local.mapped_global_track_id].position_m, rect);
    ctx.save(); ctx.strokeStyle = "rgba(117,148,165,.42)"; ctx.setLineDash([3,4]); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); ctx.restore();
  });
  frame.radar_nodes.forEach(node => {
    const [x,y] = project(node.position_m, rect); ctx.save(); ctx.translate(x,y); ctx.rotate(Math.PI/4);
    ctx.fillStyle = node.available ? "#ffb94e" : "#765736"; ctx.fillRect(-7,-7,14,14); ctx.restore();
    ctx.fillStyle="#c8d5dc"; ctx.font="11px Segoe UI"; ctx.fillText(node.node_id, x+12, y+4);
  });
  frame.local_tracks.forEach(local => {
    const [x,y] = project(local.position_m, rect); const isSelected = selectedGlobalId && local.mapped_global_track_id === selectedGlobalId;
    ctx.beginPath(); ctx.arc(x,y,isSelected?5:3.5,0,Math.PI*2); ctx.fillStyle = local.source_node_id === "NODE_A" ? "rgba(103,174,248,.9)" : "rgba(221,124,246,.82)"; ctx.fill();
    if (isSelected) { ctx.fillStyle="#bcd2df"; ctx.font="10px Segoe UI"; ctx.fillText(`${local.source_node_id}/${local.local_track_id}`,x+7,y-7); }
  });
  frame.global_tracks.forEach(track => {
    const [x,y] = project(track.position_m, rect); const selected = selectedGlobalId === track.global_track_id;
    ctx.save(); ctx.shadowColor=colorFor(track.global_track_id); ctx.shadowBlur=selected?18:8;
    ctx.beginPath(); ctx.arc(x,y,selected?8:6,0,Math.PI*2); ctx.fillStyle=colorFor(track.global_track_id); ctx.fill(); ctx.restore();
    ctx.fillStyle="#f0fbff"; ctx.font=`${selected?"600 ":""}11px Segoe UI`; ctx.fillText(track.global_track_id,x+10,y-10);
  });
  (frame.debug_truth_tracks || []).forEach(track => {
    const [x,y]=project(track.position_m,rect); ctx.save(); ctx.strokeStyle="#ff716d"; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(x-6,y-6); ctx.lineTo(x+6,y+6); ctx.moveTo(x+6,y-6); ctx.lineTo(x-6,y+6); ctx.stroke(); ctx.restore();
  });
}

function renderList(frame) {
  el("trackCount").textContent = frame.global_tracks.length;
  el("trackList").innerHTML = frame.global_tracks.map(track => `
    <button class="track-row ${track.global_track_id===selectedGlobalId?"selected":""}" data-id="${track.global_track_id}">
      <strong>${track.global_track_id}<span class="source-count">${track.active_source_count} active</span></strong>
      <small>${track.status} · age ${fmt(track.information_age_s)}s · ${track.fusion_method}</small>
    </button>`).join("") || `<div class="muted">本帧暂无 global track</div>`;
  document.querySelectorAll(".track-row").forEach(button => button.addEventListener("click", () => { selectedGlobalId=button.dataset.id; render(); }));
}

function renderDetail(frame) {
  const track = frame.global_tracks.find(row => row.global_track_id === selectedGlobalId);
  if (!track) { el("trackDetail").className="track-detail muted"; el("trackDetail").textContent="点击一条 global track 查看详情"; return; }
  const locals = frame.local_tracks.filter(row => row.mapped_global_track_id === selectedGlobalId);
  el("trackDetail").className="track-detail";
  el("trackDetail").innerHTML = `<div class="kv">
    <span>ID</span><span>${track.global_track_id}</span><span>状态</span><span>${track.status}</span>
    <span>来源雷达</span><span>${track.source_node_ids.join(", ")||"—"}</span><span>Active source</span><span>${track.active_source_count}</span>
    <span>Position m</span><span>${vec(track.position_m)}</span><span>Velocity m/s</span><span>${vec(track.velocity_mps)}</span>
    <span>Information age</span><span>${fmt(track.information_age_s)} s</span><span>Fusion</span><span>${track.fusion_method}</span>
    <span>最近 CI 权重</span><span>${Object.entries(track.fusion_weights).map(([k,v])=>`${k}:${fmt(v,2)}`).join(" · ")||"—"}</span>
    <span>Lifecycle</span><span>created ${fmt(track.created_at_s)}s · updates ${track.updates} · coasts ${track.coasts}</span>
    <span>Local tracks</span><span>${locals.map(row=>`<i class="local-chip">${row.source_node_id}/${row.local_track_id}</i>`).join("")||"—"}</span>
  </div>`;
}

function renderEvents(frame) {
  const rows = frame.recent_events.filter(row => !selectedGlobalId || !row.global_track_id || row.global_track_id===selectedGlobalId).slice(-8).reverse();
  el("eventList").className = rows.length ? "event-list" : "event-list muted";
  el("eventList").innerHTML = rows.map(row => `<div class="event-row ${row.event}"><b>${row.event}</b><br><small>${row.global_track_id||row.local_track_id||"—"} · ${row.reason||row.decision||""}</small></div>`).join("") || "本帧无新增事件";
}

function render() {
  if (!replay) return;
  const frame = replay.frames[frameIndex];
  if (selectedGlobalId && !frame.global_tracks.some(row => row.global_track_id===selectedGlobalId)) selectedGlobalId=null;
  const rect=resizeCanvas(); ctx.clearRect(0,0,rect.width,rect.height); drawFrame(frame,rect);
  renderList(frame); renderDetail(frame); renderEvents(frame);
  el("emptyState").style.display = frame.global_tracks.length ? "none" : "block";
  el("timeline").value=frameIndex; el("timeValue").textContent=`T+${fmt(frame.time_s)}s`; el("frameValue").textContent=`FRAME ${frameIndex+1} / ${replay.frames.length}`;
}

function step(delta) { frameIndex=Math.max(0,Math.min(replay.frames.length-1,frameIndex+delta)); render(); }
function stop() { if (timer) clearInterval(timer); timer=null; el("playButton").textContent="▶"; }
function togglePlay() { if (timer) return stop(); el("playButton").textContent="Ⅱ"; timer=setInterval(()=>{ if(frameIndex>=replay.frames.length-1){stop();return;} step(1); },700); }

el("prevButton").addEventListener("click",()=>{stop();step(-1);});
el("nextButton").addEventListener("click",()=>{stop();step(1);});
el("playButton").addEventListener("click",togglePlay);
el("timeline").addEventListener("input",event=>{stop();frameIndex=Number(event.target.value);render();});
window.addEventListener("resize",render);
loadManifest().catch(error=>{el("statusPill").textContent="载入失败";el("scenarioTitle").textContent=error.message;console.error(error);});
