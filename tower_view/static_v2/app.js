"use strict";

const el = id => document.getElementById(id);
const canvas = el("towerCanvas");
const ctx = canvas.getContext("2d");
let manifest = [];
let replay = null;
let replayUrl = "";
let frameIndex = 0;
let selectedGlobalId = null;
let selectedEvent = null;
let timer = null;
let bounds = {minX:-8000,maxX:8000,minY:-8000,maxY:8000};
const colors = ["#32d6d0","#ffb94e","#dd7cf6","#6bdd8a","#ff716d","#8aa9ff"];
const criticalEvents = new Set(["ID_SWITCH","FRAGMENTATION","STALE_REJECTED","DROPPED","GATE_REJECTED"]);
const fusionEvents = new Set(["CI_FUSED","ASSOCIATED","RECONNECTED","HANDOVER"]);
const colorFor = id => colors[Math.abs([...id].reduce((sum,char)=>sum+char.charCodeAt(0),0))%colors.length];
const fmt = (value,digits=1) => value===null||value===undefined ? "—" : Number(value).toFixed(digits);
const vec = values => values ? `[${values.map(value=>fmt(value,1)).join(", ")}]` : "—";
const safe = value => String(value??"").replace(/[&<>"']/g, char=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));

async function loadManifest(){
  const response=await fetch("/api/replays");
  if(!response.ok) throw new Error(`回放清单加载失败: ${response.status}`);
  manifest=await response.json();
  if(!manifest.length) throw new Error("没有 tower-view-v2 回放");
  const scenarios=[...new Map(manifest.map(item=>[item.scenario_id,item])).values()];
  el("scenarioSelect").innerHTML=scenarios.map(item=>`<option value="${safe(item.scenario_id)}">${safe(item.title)} · seed ${safe(item.seed)}</option>`).join("");
  el("scenarioSelect").addEventListener("change",()=>refreshModes(true));
  el("modeSelect").addEventListener("change",()=>loadSelection());
  refreshModes(false);
  await loadSelection();
}

function refreshModes(triggerLoad){
  const scenario=el("scenarioSelect").value||manifest[0].scenario_id;
  const available=manifest.filter(item=>item.scenario_id===scenario);
  const previous=el("modeSelect").value;
  el("modeSelect").innerHTML=available.map(item=>`<option value="${safe(item.sharing_mode)}">${safe(item.sharing_mode)}</option>`).join("");
  if(available.some(item=>item.sharing_mode===previous)) el("modeSelect").value=previous;
  else if(available.some(item=>item.sharing_mode==="track_share")) el("modeSelect").value="track_share";
  if(triggerLoad) loadSelection();
}

async function loadSelection(){
  const scenario=el("scenarioSelect").value;
  const mode=el("modeSelect").value;
  const entry=manifest.find(item=>item.scenario_id===scenario&&item.sharing_mode===mode);
  if(entry) await loadReplay(entry.url);
}

async function loadReplay(url){
  stop(); el("statusPill").textContent="正在载入";
  const response=await fetch(url); if(!response.ok) throw new Error(`回放加载失败: ${response.status}`);
  const data=await response.json(); if(data.schema_version!=="tower-view-v2") throw new Error(`不支持的 schema: ${data.schema_version}`);
  replay=data; replayUrl=url; frameIndex=0; selectedGlobalId=null; selectedEvent=null;
  el("scenarioTitle").textContent=`${data.scenario.title} · ${data.sharing_mode}`;
  el("timeline").max=Math.max(0,data.frames.length-1); bounds=computeBounds(data.frames);
  const debug=Boolean(data.debug_overlay&&data.debug_overlay.enabled);
  el("debugBanner").classList.toggle("hidden",!debug);
  el("statusPill").textContent=debug?`DEBUG / GROUND TRUTH · ${data.frames.length} 帧`:`只读诊断 · ${data.frames.length} 帧`;
  renderComparison(); renderEventMarkers(); render();
}

function computeBounds(frames){
  const points=[]; frames.forEach(frame=>{
    frame.radar_nodes.forEach(row=>points.push(row.position_m)); frame.local_tracks.forEach(row=>points.push(row.position_m));
    frame.global_tracks.forEach(row=>points.push(row.position_m)); (frame.debug_truth_tracks||[]).forEach(row=>points.push(row.position_m));
  });
  if(!points.length) return {minX:-8000,maxX:8000,minY:-8000,maxY:8000};
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]); let minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const padX=Math.max(1200,(maxX-minX)*.12),padY=Math.max(1200,(maxY-minY)*.12); return {minX:minX-padX,maxX:maxX+padX,minY:minY-padY,maxY:maxY+padY};
}
function resizeCanvas(){const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);return rect;}
function project(position,rect){const pad=32;return [pad+(position[0]-bounds.minX)/(bounds.maxX-bounds.minX)*(rect.width-pad*2),rect.height-pad-(position[1]-bounds.minY)/(bounds.maxY-bounds.minY)*(rect.height-pad*2)];}
function drawGrid(rect){const origin=project([0,0,0],rect);ctx.save();ctx.strokeStyle="rgba(88,126,145,.4)";ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(0,origin[1]);ctx.lineTo(rect.width,origin[1]);ctx.stroke();ctx.beginPath();ctx.moveTo(origin[0],0);ctx.lineTo(origin[0],rect.height);ctx.stroke();ctx.restore();}
function drawTrail(id,rect,strong){const points=[];for(let index=0;index<=frameIndex;index++){const row=replay.frames[index].global_tracks.find(track=>track.global_track_id===id);if(row)points.push(project(row.position_m,rect));}if(points.length<2)return;ctx.save();ctx.strokeStyle=colorFor(id);ctx.globalAlpha=strong?.95:.35;ctx.lineWidth=strong?3:1.3;ctx.beginPath();points.forEach((point,index)=>index?ctx.lineTo(...point):ctx.moveTo(...point));ctx.stroke();ctx.restore();}
function lifecycleAt(id,time,status){const transient=(replay.frames[frameIndex]?.events||[]).find(event=>event.global_track_id===id&&["HANDOVER","RECONNECTED","DROPPED"].includes(event.event_type));if(!transient)return status;return transient.event_type==="RECONNECTED"?"reconnect":transient.event_type.toLowerCase();}
function drawFrame(frame,rect){
  drawGrid(rect); frame.global_tracks.forEach(track=>drawTrail(track.global_track_id,rect,!selectedGlobalId||selectedGlobalId===track.global_track_id));
  const globals=Object.fromEntries(frame.global_tracks.map(row=>[row.global_track_id,row])); frame.local_tracks.forEach(local=>{if(!local.mapped_global_track_id||!globals[local.mapped_global_track_id])return;const a=project(local.position_m,rect),b=project(globals[local.mapped_global_track_id].position_m,rect);ctx.save();ctx.strokeStyle="rgba(117,148,165,.42)";ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.restore();});
  frame.radar_nodes.forEach(node=>{const [x,y]=project(node.position_m,rect);ctx.save();ctx.translate(x,y);ctx.rotate(Math.PI/4);ctx.fillStyle=node.available?"#ffb94e":"#65484a";ctx.fillRect(-7,-7,14,14);ctx.restore();ctx.fillStyle="#c8d5dc";ctx.font="10px Segoe UI";ctx.fillText(node.node_id,x+11,y+3);});
  frame.local_tracks.forEach(local=>{const [x,y]=project(local.position_m,rect),match=selectedGlobalId&&local.mapped_global_track_id===selectedGlobalId;ctx.beginPath();ctx.arc(x,y,match?5:3.5,0,Math.PI*2);ctx.fillStyle=local.source_node_id==="NODE_A"?"rgba(103,174,248,.9)":"rgba(221,124,246,.82)";ctx.fill();if(match){ctx.fillStyle="#bcd2df";ctx.font="9px Segoe UI";ctx.fillText(`${local.source_node_id}/${local.local_track_id}`,x+7,y-7);}});
  frame.global_tracks.forEach(track=>{const [x,y]=project(track.position_m,rect),selected=selectedGlobalId===track.global_track_id,status=lifecycleAt(track.global_track_id,frame.time_s,track.status);ctx.save();ctx.shadowColor=colorFor(track.global_track_id);ctx.shadowBlur=selected?18:7;ctx.beginPath();ctx.arc(x,y,selected?8:6,0,Math.PI*2);ctx.fillStyle=colorFor(track.global_track_id);ctx.fill();if(status==="coasting"||status==="stale_coasting"){ctx.strokeStyle=status==="stale_coasting"?"#ff716d":"#8aa9ff";ctx.lineWidth=2;ctx.setLineDash([3,2]);ctx.stroke();}ctx.restore();ctx.fillStyle="#f1fbff";ctx.font=`${selected?"600 ":""}10px Segoe UI`;ctx.fillText(track.global_track_id,x+9,y-9);});
  (frame.debug_truth_tracks||[]).forEach(track=>{const [x,y]=project(track.position_m,rect);ctx.save();ctx.strokeStyle="#ff4545";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x-6,y-6);ctx.lineTo(x+6,y+6);ctx.moveTo(x+6,y-6);ctx.lineTo(x-6,y+6);ctx.stroke();ctx.restore();});
}

function renderComparison(){const metrics=replay.comparison_metrics,items=[["Coverage",fmt(metrics.coverage,3)],["RMSE m",fmt(metrics.rmse_m,1)],["ID switch",metrics.id_switch],["Fragmentation",metrics.fragmentation],["Duplicate",fmt(metrics.duplicate,3)],["Info age s",fmt(metrics.information_age_s,2)],["Comm bytes",fmt(metrics.communication_bytes,0)],["Utilization",fmt(metrics.message_utilization,3)]];el("comparisonBar").innerHTML=items.map(([key,value])=>`<div class="metric"><small>${safe(key)}</small><strong>${safe(value)}</strong></div>`).join("");}
function renderTelemetry(frame){const comm=frame.communication,fusion=frame.fusion,items=[["sent",comm.sent_messages],["arrived",comm.arrived_messages],["used",comm.used_messages],["rejected",comm.rejected_messages],["bytes",fmt(comm.communication_bytes,0)],["util",fmt(comm.message_utilization,2)],["local",fusion.local_track_count],["global",fusion.global_track_count],["single",fusion.single_source_global_count],["multi",fusion.multi_source_global_count],["CI",fusion.ci_count],["radar/source",`${fusion.active_radar_count}/${fusion.active_source_count}`]];el("telemetryGrid").innerHTML=items.map(([key,value])=>`<div class="metric-tile"><span>${safe(key)}</span><b>${safe(value)}</b></div>`).join("");}
function renderList(frame){el("trackCount").textContent=frame.global_tracks.length;el("trackList").innerHTML=frame.global_tracks.map(track=>{const lifecycle=lifecycleAt(track.global_track_id,frame.time_s,track.status);return `<button class="track-row ${track.global_track_id===selectedGlobalId?"selected":""}" data-id="${safe(track.global_track_id)}"><strong>${safe(track.global_track_id)}<span class="lifecycle-badge ${safe(lifecycle)}">${safe(lifecycle)}</span></strong><small>${track.active_source_count} active / ${track.retained_source_node_ids.length} retained · age ${fmt(track.information_age_s)}s · ${safe(track.fusion_method)}</small></button>`;}).join("")||`<div class="muted">本帧暂无 global track</div>`;document.querySelectorAll(".track-row").forEach(button=>button.addEventListener("click",()=>{selectedGlobalId=button.dataset.id;render();}));}
function findTrackState(id){for(let index=frameIndex;index>=0;index--){const track=replay.frames[index].global_tracks.find(row=>row.global_track_id===id);if(track)return track;}return null;}
function renderDetail(frame){const track=selectedGlobalId?findTrackState(selectedGlobalId):null;if(!track){el("trackDetail").className="track-detail muted";el("trackDetail").textContent="点击一条 global track 查看详情";return;}const locals=frame.local_tracks.filter(row=>row.mapped_global_track_id===selectedGlobalId),lifecycle=replay.track_lifecycle.find(row=>row.global_track_id===selectedGlobalId),history=(lifecycle?.states||[]).filter(row=>Number(row.time_s)<=Number(frame.time_s)+1e-12).slice(-8).reverse();const status=lifecycleAt(selectedGlobalId,frame.time_s,track.status);el("trackDetail").className="track-detail";el("trackDetail").innerHTML=`<div class="kv"><span>ID</span><span>${safe(track.global_track_id)}</span><span>Lifecycle</span><span><i class="lifecycle-badge ${safe(status)}">${safe(status)}</i></span><span>Active sources</span><span>${safe(track.active_source_node_ids.join(", ")||"—")}</span><span>Retained sources</span><span>${safe(track.retained_source_node_ids.join(", ")||"—")}</span><span>Last update</span><span>T+${fmt(track.last_update_time_s)}s</span><span>Information age</span><span>${fmt(track.information_age_s)}s</span><span>Position / velocity</span><span>${safe(vec(track.position_m))}<br>${safe(vec(track.velocity_mps))}</span><span>Fusion method</span><span>${safe(track.fusion_method)}</span><span>CI sources/weights</span><span>${safe(Object.entries(track.fusion_weights).map(([key,value])=>`${key}:${fmt(value,2)}`).join(" · ")||"—")}</span><span>Local tracks</span><span>${locals.map(row=>`<i class="local-chip">${safe(row.source_node_id)}/${safe(row.local_track_id)}</i>`).join("")||"—"}</span></div><div class="history">${history.map(row=>`<div class="history-row">T+${fmt(row.time_s)} · <b>${safe(row.status)}</b> ${safe(row.reason||"")}</div>`).join("")}</div>`;}
function eventClass(event){if(criticalEvents.has(event.event_type))return"critical";if(fusionEvents.has(event.event_type))return"fusion";return"";}
function jumpToEvent(event){selectedEvent=event;if(event.global_track_id)selectedGlobalId=event.global_track_id;const target=replay.frames.findIndex(frame=>Number(frame.time_s)+1e-12>=Number(event.time_s||0));frameIndex=Math.max(0,target);stop();render();}
function renderEventMarkers(){const duration=Number(replay.summary.duration_s)||1;el("eventMarkers").innerHTML=replay.events.map((event,index)=>`<button class="event-marker ${eventClass(event)}" data-event-index="${index}" style="left:${Math.max(0,Math.min(100,Number(event.time_s||0)/duration*100))}%" title="${safe(event.event_type)} @ ${fmt(event.time_s)}s"></button>`).join("");document.querySelectorAll(".event-marker").forEach(button=>button.addEventListener("click",()=>jumpToEvent(replay.events[Number(button.dataset.eventIndex)])));}
function renderEvents(frame){const rows=frame.events;el("eventStrip").innerHTML=rows.map(event=>`<button class="event-chip ${eventClass(event)}" data-event-key="${safe(replay.events.indexOf(event))}"><b>${safe(event.event_type)}</b><small>T+${fmt(event.time_s)} · ${safe(event.global_track_id||event.local_track_id||event.message_id||"—")}</small></button>`).join("")||`<span class="muted">本帧无事件</span>`;document.querySelectorAll(".event-chip").forEach((button,index)=>button.addEventListener("click",()=>{selectedEvent=rows[index];if(selectedEvent.global_track_id)selectedGlobalId=selectedEvent.global_track_id;render();}));}
function renderEvidence(){if(!selectedEvent){el("evidenceChain").className="evidence-list muted";el("evidenceChain").textContent="点击底部事件查看证据链";return;}const chain=replay.message_evidence.find(row=>row.message_id===selectedEvent.message_id);const steps=chain?.steps||[selectedEvent];el("evidenceChain").className="evidence-list";el("evidenceChain").innerHTML=steps.map(step=>`<div class="evidence-step"><b>T+${fmt(step.time_s)} · ${safe(step.event_type)}</b><small>${safe(step.source_node_id||"")} ${safe(step.local_track_id||"")} → ${safe(step.global_track_id||"—")}<br>${safe(step.decision||"")} ${safe(step.reason||"")}</small></div>`).join("");}
function render(){if(!replay)return;const frame=replay.frames[frameIndex],rect=resizeCanvas();ctx.clearRect(0,0,rect.width,rect.height);drawFrame(frame,rect);renderTelemetry(frame);renderList(frame);renderDetail(frame);renderEvents(frame);renderEvidence();el("emptyState").style.display=frame.global_tracks.length?"none":"block";el("timeline").value=frameIndex;el("timeValue").textContent=`T+${fmt(frame.time_s)}s`;el("frameValue").textContent=`FRAME ${frameIndex+1} / ${replay.frames.length}`;}
function step(delta){frameIndex=Math.max(0,Math.min(replay.frames.length-1,frameIndex+delta));selectedEvent=null;render();}
function stop(){if(timer)clearInterval(timer);timer=null;el("playButton").textContent="▶";}
function togglePlay(){if(timer)return stop();el("playButton").textContent="Ⅱ";timer=setInterval(()=>{if(frameIndex>=replay.frames.length-1){stop();return;}step(1);},700);}
function downloadBlob(blob,name){const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);}
function exportPng(){canvas.toBlob(blob=>blob&&downloadBlob(blob,`${replay.scenario.scenario_id}.${replay.sharing_mode}.t${replay.frames[frameIndex].time_s}.png`),"image/png");}
function exportJson(){downloadBlob(new Blob([JSON.stringify(replay,null,2)+"\n"],{type:"application/json"}),`${replay.scenario.scenario_id}.${replay.sharing_mode}.json`);}
function exportHtml(){const metrics=Object.entries(replay.comparison_metrics).map(([key,value])=>`<tr><th>${safe(key)}</th><td>${safe(value)}</td></tr>`).join("");const events=replay.events.map(event=>`<tr><td>${fmt(event.time_s)}</td><td>${safe(event.event_type)}</td><td>${safe(event.global_track_id||"")}</td><td>${safe(event.message_id||"")}</td><td>${safe(event.reason||event.decision||"")}</td></tr>`).join("");const html=`<!doctype html><meta charset="utf-8"><title>Tower View v2 Diagnostic</title><style>body{font:14px sans-serif;margin:32px;color:#17232b}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #ccd6dc;padding:6px;text-align:left}</style><h1>Tower View v2 诊断报告</h1><p>${safe(replay.scenario.title)} · ${safe(replay.sharing_mode)} · seed ${safe(replay.scenario.seed)}</p><p>只读导出；运行时对象与底层结果未被修改。</p><h2>对照指标</h2><table>${metrics}</table><h2>事件</h2><table><tr><th>time</th><th>event</th><th>global</th><th>message</th><th>reason</th></tr>${events}</table>`;downloadBlob(new Blob([html],{type:"text/html"}),`${replay.scenario.scenario_id}.${replay.sharing_mode}.diagnostic.html`);}

el("prevButton").addEventListener("click",()=>{stop();step(-1);});el("nextButton").addEventListener("click",()=>{stop();step(1);});el("playButton").addEventListener("click",togglePlay);el("timeline").addEventListener("input",event=>{stop();frameIndex=Number(event.target.value);selectedEvent=null;render();});el("exportPng").addEventListener("click",exportPng);el("exportJson").addEventListener("click",exportJson);el("exportHtml").addEventListener("click",exportHtml);window.addEventListener("resize",render);
loadManifest().catch(error=>{el("statusPill").textContent="载入失败";el("scenarioTitle").textContent=error.message;console.error(error);});
