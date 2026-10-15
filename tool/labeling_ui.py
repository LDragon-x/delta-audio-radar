"""边听边标网页工具：本地起服务 + 自动开浏览器，纯鼠标点选打标，实时落盘。

用法：
  python tool/labeling_ui.py --session sessions/<时间戳>
  python tool/labeling_ui.py --session sessions/<时间戳> --port 8471

玩法（v3 模块化大类，无键盘快捷键——小键盘党出来再标）：
  点大类卡片（真人/人机/自己/环境/救援/技能/其他）= 标/取消这个大类；
  卡片点开会列出常用子类型（脚步/枪声/开门声…），点子类型 = 标
  「大类·细节」；每个子类型旁边 ✕ 可删，底部 ＋ 可自填新增——
  想到什么填什么，跨会话保留（labeling_subtypes.json）。
  同一段可挂多个大类、同大类下也可挂多个小类（脚步+枪声混着就两个都点）；
  点已亮的再点 = 取消。
  底部按钮：上一段 / 重播 / 下一段 / 撤销 / 听不出。

标注写入 sessions/<ts>/labeling/user_labels.csv，重开自动续上；
旧格式的标注（enemy_footstep、局内播报、enemy:脚步…）照样读得出来。
会话目录里有 video.mp4（采集器录屏）时，页面自动出现视频窗：
点哪段音频就同步播对应时刻 ±1.2s 的画面（静音，声音只听音频段）。
会话目录里有 keys.jsonl（按键记录）时，每条音频下方显示此刻按着的键
和 ±1.2s 内的按键边沿（如 +0.3s W↓），辅助判断声音是不是自己发出的。
"""
import argparse
import csv
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CUSTOM_PATH = ROOT / "labeling_tags.json"     # 旧版自定义标签文件，仅 /api/tag 兼容用
SUB_PATH = ROOT / "labeling_subtypes.json"    # 各大类的子类型清单，用户可增删
COLOR_POOL = ["#d65a9e", "#5ad6b0", "#b58fd6", "#d6c25a", "#6fa8d6",
              "#d67f5a", "#7fd65a", "#5a6fd6", "#d65a6f", "#8fa3b5"]

CATS = [  # (大类名(进数据集), 中文, 颜色)
    ("enemy", "真人", "#e85f5f"),
    ("bot", "人机", "#e8b13f"),
    ("self", "自己", "#5a8dd6"),
    ("env", "环境", "#4fd6d0"),
    ("rescue", "救援", "#d65a9e"),
    ("skill", "技能", "#b06fd6"),
    ("other", "其他", "#9aa7b5"),
]

DEFAULT_SUBTYPES = {  # 各大类自带的常用子类型；用户增删后写 labeling_subtypes.json
    "enemy": ["脚步", "枪声", "开门声", "换弹/打药", "喷气"],
    "bot": ["脚步", "枪声", "说话/鬼叫"],
    "self": ["脚步", "枪声", "技能", "打药/换弹"],
    "env": ["播报", "搜物资", "杂音", "水声"],
    "rescue": ["救队友", "请求救援"],
    "skill": ["炸弹", "烟", "火", "喷气"],
    "other": ["受伤", "杂音"],
}
CAT_NAMES = {c[0] for c in CATS}


def load_custom():
    """旧版用户自建标签 [{key,en,cn,color}]，损坏/不存在就空表。"""
    try:
        data = json.loads(CUSTOM_PATH.read_text(encoding="utf-8"))
        return [t for t in data if t.get("cn")]
    except Exception:
        return []


def save_custom(tags):
    CUSTOM_PATH.write_text(json.dumps(tags, ensure_ascii=False, indent=1),
                           encoding="utf-8")


def load_subtypes():
    """各大类子类型清单；首次运行用默认表落盘，之后以文件为准。"""
    if not SUB_PATH.exists():
        SUB_PATH.write_text(json.dumps(DEFAULT_SUBTYPES, ensure_ascii=False,
                                       indent=1), encoding="utf-8")
        return {k: list(v) for k, v in DEFAULT_SUBTYPES.items()}
    try:
        data = json.loads(SUB_PATH.read_text(encoding="utf-8"))
        return {cat: [str(x) for x in data.get(cat, [])] for cat in CAT_NAMES}
    except Exception:
        return {k: list(v) for k, v in DEFAULT_SUBTYPES.items()}


def save_subtypes(subs):
    SUB_PATH.write_text(json.dumps(subs, ensure_ascii=False, indent=1),
                        encoding="utf-8")

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>边听边标</title>
<style>
  body { background:#10141c; color:#cfd8e3; font-family:"Microsoft YaHei UI",sans-serif;
         margin:0; display:flex; height:100vh; overflow:hidden; }
  #main { flex:1; display:flex; flex-direction:column; align-items:center;
          justify-content:center; gap:14px; padding:16px; overflow-y:auto; }
  #clipname { font-size:15px; color:#8fa0b3; }
  #big { font-size:44px; font-weight:bold; font-family:Consolas,monospace; }
  #meta { font-size:13px; color:#5f7080; }
  #bar { width:420px; height:6px; background:#1c2430; border-radius:3px; overflow:hidden; }
  #bar i { display:block; height:100%; background:#3fd68f; width:0; transition:width .2s; }
  #cats { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; max-width:900px; }
  .cat { background:#131a26; border:1px solid #2a3546; border-radius:10px;
         min-width:150px; max-width:230px; }
  .cat.open { border-color:#3a4a60; }
  .cathdr { display:flex; align-items:center; gap:8px; padding:10px 12px;
            cursor:pointer; user-select:none; }
  .cathdr .dot { width:11px; height:11px; border-radius:3px; }
  .cathdr b { font-size:16px; }
  .cathdr .arr { margin-left:auto; color:#5f7080; font-size:12px; transition:transform .15s; }
  .cat.open .cathdr .arr { transform:rotate(90deg); }
  .cat.lit { box-shadow:0 0 10px rgba(63,214,143,.28); border-color:#3fd68f; }
  .cat.lit .cathdr b { color:#fff; }
  .chips { display:none; padding:0 10px 10px; flex-wrap:wrap; gap:6px; }
  .cat.open .chips { display:flex; }
  .chip { position:relative; background:#1a2230; border:1px solid #2a3546;
          border-radius:7px; padding:5px 9px; cursor:pointer; font-size:13px;
          display:inline-flex; align-items:center; gap:4px; }
  .chip:hover { border-color:#3fd68f; }
  .chip.lit { background:#263247; border-color:#3fd68f; color:#fff; }
  .chip .del { color:#5f7080; font-size:11px; padding:0 2px; }
  .chip .del:hover { color:#e85f5f; }
  .chipadd { border-style:dashed; color:#8fa0b3; }
  .chipin { background:#1a2230; border:1px solid #3fd68f; border-radius:7px;
            color:#cfd8e3; padding:4px 7px; font-size:13px; width:110px; }
  #nav { display:flex; gap:8px; align-items:center; justify-content:center;
         flex-wrap:wrap; font-size:13px; }
  #nav button { background:#1a2230; border:1px solid #2a3546; border-radius:7px;
                color:#cfd8e3; padding:7px 14px; cursor:pointer; font-size:13px; }
  #nav button:hover { border-color:#3fd68f; }
  #nav .go { background:#1e3a2c; border-color:#3fd68f; color:#3fd68f; font-weight:bold; }
  #hint { font-size:12px; color:#5f7080; }
  #side { width:250px; background:#0b0e14; overflow-y:auto; padding:8px;
          font-size:12px; font-family:Consolas,monospace; }
  .row { padding:3px 6px; border-radius:4px; white-space:nowrap; overflow:hidden;
         text-overflow:ellipsis; color:#5f7080; }
  .row.done { color:#cfd8e3; background:#141a24; }
  .row.cur { outline:1px solid #3fd68f; color:#fff; }
  #toast { position:fixed; top:14px; left:50%; transform:translateX(-50%);
           background:#1a2230; border:1px solid #2a3546; padding:6px 16px;
           border-radius:6px; font-size:13px; opacity:0; transition:opacity .3s; }
</style>
</head>
<body>
<div id="main">
  <div id="clipname">—</div>
  <div id="big">…</div>
  <div id="meta">—</div>
  <div id="keyline" style="font-size:13px;color:#c8d65a;font-family:Consolas,monospace;
                           min-height:18px"></div>
  <video id="vid" muted playsinline
         style="max-width:700px;width:90%;border-radius:8px;border:1px solid #2a3546;
                %VIDSTYLE%"></video>
  <div id="bar"><i></i></div>
  <div id="cats"></div>
  <div id="nav">
    <button onclick="show(cur-1,false)">← 上一段</button>
    <button onclick="replay()">↻ 重播</button>
    <button class="go" onclick="show(cur+1,true)">下一段 →</button>
    <button onclick="show(cur+1,true)">听不出，跳过</button>
    <button onclick="undo()">撤销</button>
    <span id="voffbox" style="display:none">| 视频偏移(s):
      <input id="voff" type="number" step="0.5" value="0" style="width:64px;
             background:#1a2230;border:1px solid #2a3546;border-radius:6px;
             color:#c8d65a;padding:2px 6px"></span>
  </div>
  <div id="hint">点大类/小类 = 标/取消（都允许多选）· 子类型 ✕ 删 ＋ 加 · 同段随便挂几类</div>
</div>
<div id="side"></div>
<div id="toast"></div>
<audio id="au" preload="auto"></audio>
<script>
const CATS = %CATS%;      // [大类名, 中文, 颜色]
const CN = {};            // 大类名 → 中文
CATS.forEach(c => { CN[c[1]] = c[2]; });
let SUBTYPES = %SUBTYPES%;  // {大类: [子类型…]}，服务端持久化
const CLIPS = %CLIPS%;
const KEYS = %KEYS%;      // {键名: [[t, down], ...]} 按键边沿，无 keys.jsonl 是 {}
const KEYCN = k => ({left:'左键', right:'右键', middle:'中键'}[k] || k.toUpperCase());
let cur = 0, labels = %SAVED%;
const au = document.getElementById('au');
const VIDEO = %HASVIDEO%;
const vid = document.getElementById('vid');
let segStart = 0, segEnd = 0;
let VMAP = null, VIDFPS = 15, VIDLEN = 0;   // 帧时间戳映射：会话秒 → mp4 秒
function labText(lab) {  // 显示用：enemy:脚步 → 真人·脚步；裸类/旧名 → 中文或原文
  const i = lab.indexOf(':');
  if (i > 0) return (CN[lab.slice(0, i)] || lab.slice(0, i)) + '·' + lab.slice(i + 1);
  return CN[lab] || lab;
}
function catOf(lab) { const i = lab.indexOf(':'); return i > 0 ? lab.slice(0, i) : lab; }
if (VIDEO) {
  vid.src = '/video.mp4';
  fetch('/video_frames.json').then(r => r.ok ? r.json() : null).then(m => {
    if (m && m.times && m.times.length) { VMAP = m.times; VIDFPS = m.fps || 15; }
    else document.getElementById('voffbox').style.display = '';  // 无帧表：给手动偏移
  }).catch(() => {});
  vid.onloadedmetadata = () => { VIDLEN = vid.duration; };
  vid.ontimeupdate = () => {
    if (vid.currentTime > segEnd || vid.currentTime < segStart - 0.5)
      vid.currentTime = segStart;
  };
}
function conv(t) {  // 会话时间(秒) → mp4 时间轴（优先逐帧时间戳，退化为按时长线性）
  t += voff();  // 手动微调（旧会话无帧表时校准用）
  if (VMAP && VMAP.length) {
    let lo = 0, hi = VMAP.length - 1, idx = 0;
    while (lo <= hi) { const m = (lo + hi) >> 1;
      if (VMAP[m] <= t) { idx = m; lo = m + 1; } else hi = m - 1; }
    return idx / VIDFPS;
  }
  if (VIDLEN) {
    const total = CLIPS.reduce((a, c) => Math.max(a, parseFloat(c.t)), 0);
    return t * (VIDLEN / Math.max(1, total));
  }
  return t;
}
function voff() {
  const e = document.getElementById('voff');
  return e ? (parseFloat(e.value) || 0) : 0;
}
const catUI = {};   // cat → {card, hdr, chipsBox, chipOf:{label→el}}
CATS.forEach(buildCat);
const side = document.getElementById('side');
const voffInp = document.getElementById('voff');
if (voffInp) voffInp.addEventListener('input', () => show(cur, false));
CLIPS.forEach((c, i) => {
  const r = document.createElement('div');
  r.className = 'row'; r.id = 'r' + i;
  r.textContent = c.clip;
  r.onclick = () => show(i, true);
  side.appendChild(r);
});
function keyInfo(t) {
  if (!Object.keys(KEYS).length) return '';
  const held = [], win = [];
  for (const k in KEYS) {
    const ev = KEYS[k];
    let lo = 0, hi = ev.length - 1, idx = -1;      // 二分找最后一条 t 之前的边沿
    while (lo <= hi) { const m = (lo + hi) >> 1;
      if (ev[m][0] <= t) { idx = m; lo = m + 1; } else hi = m - 1; }
    if (idx >= 0 && ev[idx][1]) held.push(k);
    for (const [et, dn] of ev)
      if (et > t - 1.2 && et <= t + 1.2)
        win.push([et, k, dn]);
  }
  win.sort((a, b) => a[0] - b[0]);
  const parts = [];
  if (held.length) parts.push('按着: ' + held.map(KEYCN).join(' '));
  if (win.length) parts.push('±1.2s: ' + win.map(([et, k, dn]) =>
    `${et - t >= 0 ? '+' : ''}${(et - t).toFixed(1)}s ${KEYCN(k)}${dn ? '↓' : '↑'}`).join('  '));
  return parts.join('　|　');
}
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.opacity = 1;
  setTimeout(() => t.style.opacity = 0, 1400);
}
function buildCat([cat, cn, color]) {
  const card = document.createElement('div');
  card.className = 'cat';
  const hdr = document.createElement('div');
  hdr.className = 'cathdr';
  hdr.innerHTML = `<span class="dot" style="background:${color}"></span><b>${cn}</b>` +
                  `<span class="arr">▸</span>`;
  hdr.onclick = () => {
    card.classList.add('open');   // 点大类 = 标/取消 + 展开选细节
    toggleLabel(cat);
  };
  const chipsBox = document.createElement('div');
  chipsBox.className = 'chips';
  card.appendChild(hdr);
  card.appendChild(chipsBox);
  document.getElementById('cats').appendChild(card);
  const ui = catUI[cat] = { card, hdr, chipsBox, chipOf: {} };
  // 点大类右上小箭头只展开/收起，不打标
  const arr = hdr.querySelector('.arr');
  arr.addEventListener('click', e => {
    e.stopPropagation();
    card.classList.toggle('open');
  });
  ui.bareChip = addChip(cat, null, '只标' + cn);   // 第一片：只标大类
  (SUBTYPES[cat] || []).forEach(det => addChip(cat, det));
  addAdder(cat);
}
function addChip(cat, det, textOverride) {  // det=null → 裸大类
  const ui = catUI[cat];
  const name = det ? cat + ':' + det : cat;
  const chip = document.createElement('span');
  chip.className = 'chip';
  chip.textContent = textOverride || det;
  chip.onclick = () => { ui.card.classList.add('open'); toggleLabel(name); };
  if (det) {  // 子类型带删除 ✕（删除不打标）
    const del = document.createElement('span');
    del.className = 'del';
    del.textContent = '✕';
    del.onclick = e => {
      e.stopPropagation();
      fetch('/api/subtype_del', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({cat: cat, name: det})})
        .then(() => {
          SUBTYPES[cat] = (SUBTYPES[cat] || []).filter(x => x !== det);
          chip.remove(); delete ui.chipOf[name];
          toast('已删除 ' + CN[cat] + '·' + det);
        });
    };
    chip.appendChild(del);
  }
  ui.chipsBox.insertBefore(chip, ui.adderChip || null);
  ui.chipOf[name] = chip;
  return chip;
}
function addAdder(cat) {  // 底部 ＋ 添加：内联输入，回车存服务端
  const ui = catUI[cat];
  const chip = document.createElement('span');
  chip.className = 'chip chipadd';
  chip.textContent = '＋ 添加';
  ui.adderChip = chip;
  ui.chipsBox.appendChild(chip);
  chip.onclick = () => {
    chip.style.display = 'none';
    const inp = document.createElement('input');
    inp.className = 'chipin';
    inp.placeholder = '如：撞门';
    inp.onclick = e => e.stopPropagation();
    inp.onkeydown = e => { e.stopPropagation(); };
    inp.onblur = () => { inp.remove(); chip.style.display = ''; };
    inp.onkeyup = e => {
      if (e.key !== 'Enter') return;
      const name = inp.value.trim().replace(/[:;,]/g, '').slice(0, 20);
      if (!name) { inp.remove(); chip.style.display = ''; return; }
      fetch('/api/subtype', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({cat: cat, name: name})})
        .then(r => r.json()).then(t => {
          if (t.existed) { toast('已有了：' + name); }
          else {
            SUBTYPES[cat] = SUBTYPES[cat] || [];
            SUBTYPES[cat].push(name);
            addChip(cat, name);
            toast('已加 ' + CN[cat] + '·' + name);
          }
          inp.remove(); chip.style.display = '';
        });
    };
    ui.chipsBox.appendChild(inp);
    inp.focus();
  };
}
function refreshBtns() {  // 当前段已挂的类别卡片/对应子类型片点亮
  const cur_labs = labels[CLIPS[cur].clip] || [];
  for (const cat in catUI) {
    const ui = catUI[cat];
    const on = cur_labs.some(l => catOf(l) === cat);
    ui.card.classList.toggle('lit', on);
    for (const name in ui.chipOf)
      ui.chipOf[name].classList.toggle('lit', cur_labs.includes(name));
    if (ui.bareChip) ui.bareChip.classList.toggle('lit', cur_labs.includes(cat));
  }
}
function saveLabels(clip, arr) {  // 更新内存 + 落盘
  if (arr.length) labels[clip] = arr; else delete labels[clip];
  fetch('/api/label', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({clip: clip, label: arr.join(';')})});
}
function toggleLabel(name) {  // 点一下挂上、再点取消；大类小类都允许多选
  const c = CLIPS[cur];
  const arr0 = labels[c.clip] || [];
  let arr, msg;
  if (arr0.includes(name)) {
    arr = arr0.filter(l => l !== name);
    msg = '移除 ' + labText(name);
  } else {
    arr = arr0.concat([name]);
    msg = '加上 ' + labText(name);
  }
  saveLabels(c.clip, arr);
  paint();
  const now = labels[c.clip] || [];
  toast(msg + (now.length ? '（现: ' + now.map(labText).join('+') + '）'
                          : '（已清空）'));
}
function replay() {
  au.currentTime = 0; au.play().catch(() => {});
  if (VIDEO) { vid.currentTime = segStart; vid.play().catch(() => {}); }
}
function show(i, autoplay) {
  cur = Math.max(0, Math.min(CLIPS.length - 1, i));
  const c = CLIPS[cur];
  document.getElementById('clipname').textContent = c.clip;
  document.getElementById('big').textContent = (cur + 1) + ' / ' + CLIPS.length;
  document.getElementById('meta').textContent =
    `引擎判断 self=${c.auto_self}  corr=${c.corr}  t=${c.t}s`;
  document.getElementById('keyline').textContent = keyInfo(parseFloat(c.t));
  refreshBtns();
  document.getElementById('bar').firstElementChild.style.width =
    (Object.keys(labels).length / CLIPS.length * 100) + '%';
  document.querySelectorAll('.row').forEach(r => r.classList.remove('cur'));
  const row = document.getElementById('r' + cur);
  row.classList.add('cur'); row.scrollIntoView({block:'nearest'});
  au.src = '/clips/' + encodeURIComponent(c.clip);
  if (autoplay) au.play().catch(() => {});
  if (VIDEO) {
    const ct = parseFloat(c.t);
    segStart = Math.max(0, conv(ct - 1.2));
    segEnd = Math.max(segStart + 0.5, conv(ct + 1.2));
    vid.currentTime = segStart;
    vid.play().catch(() => {});
  }
}
function paint() {
  CLIPS.forEach((c, i) => {
    const row = document.getElementById('r' + i);
    row.classList.toggle('done', !!labels[c.clip]);
    const lab = labels[c.clip];
    row.textContent = c.clip + (lab && lab.length
      ? '  ✓ ' + lab.map(labText).join('+') : '');
  });
  document.getElementById('bar').firstElementChild.style.width =
    (Object.keys(labels).length / CLIPS.length * 100) + '%';
  refreshBtns();
}
function undo() {
  const keys = Object.keys(labels);
  if (!keys.length) return;
  const last = keys[keys.length - 1];
  saveLabels(last, []);
  paint(); toast('已撤销 ' + last);
}
au.onended = () => { if (VIDEO) vid.pause(); };  // 听完停画面，↻ 重播
paint();
show(0, false);
</script>
</body>
</html>"""


TIMELINE_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>时间轴段落标注</title>
<style>
body{background:#12141c;color:#dde;font-family:Microsoft YaHei,sans-serif;margin:0;padding:14px}
h2{margin:4px 0 10px;font-size:17px}
a{color:#7ab}#bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
#cv{width:100%;height:170px;background:#0a0c12;border:1px solid #334;border-radius:6px;cursor:crosshair;display:block}
#selInfo{color:#fc6;margin-left:8px}
button{background:#232838;color:#dde;border:1px solid #445;border-radius:5px;padding:5px 12px;cursor:pointer;font-size:13px}
button:hover{background:#2e3850}
#labelInput{background:#0a0c12;border:1px solid #445;color:#dde;border-radius:5px;padding:5px 8px;width:180px}
#ranges{margin-top:12px;font-size:13px}
#ranges div{padding:4px 8px;border-bottom:1px solid #232838;display:flex;gap:10px;align-items:center}
video{width:100%;max-height:42vh;background:#000;margin-top:10px;border-radius:6px}
.t{color:#8ab;width:130px}.del{color:#f66;cursor:pointer;border:none;background:none;font-size:14px}
small{color:#778}
</style>
</head>
<body>
<h2>时间轴段落标注 <small>拉框选一段时间 → 给标签 → 保存。喂"场景级"模型用：交战/人机多/真人活动…</small>
<a href="/" style="float:right;font-size:13px">← 返回片段标注</a></h2>
<div id="bar">
<span>快速标签：</span>
<span id="presets"></span>
<input id="labelInput" placeholder="自定义标签，如：交战">
<span id="selInfo">在波形上拖选一段</span>
<button id="saveBtn" onclick="saveRange()" disabled>保存段落</button>
</div>
<canvas id="cv"></canvas>
<video id="vid" muted playsinline %VIDSTYLE% src="/video.mp4"></video>
<audio id="aud" controls src="/session.wav"></audio>
<div id="ranges"></div>
<script>
const WAVE=%WAVE%, RANGES=%RANGES%, DUR=%DUR%;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const vid=document.getElementById('vid'), aud=document.getElementById('aud');
let sel=null, anchor=null, FTIMES=null, FPS=15;
fetch('/video_frames.json').then(r=>r.ok?r.json():null).then(m=>{
  if(m){FTIMES=m.times; FPS=m.fps||15;}}).catch(()=>{});
const hasVid = %HASVIDEO%;
function syncVideo(){
 if(!FTIMES||!hasVid||!aud.duration) return;
 const t=aud.currentTime;
 let lo=0, hi=FTIMES.length-1;
 while(lo<hi){const mid=(lo+hi)>>1; if(FTIMES[mid]<t) lo=mid+1; else hi=mid}
 const vt=Math.min(lo/FPS, vid.duration||1e9);
 if(isFinite(vt)&&Math.abs((vid.currentTime||0)-vt)>0.25) vid.currentTime=vt;
}
const PRE=["交战","人机多","真人活动","搜物资","打药","安静","赶路","自己突进"];
PRE.forEach(p=>{const b=document.createElement('button');b.textContent=p;
b.onclick=()=>{document.getElementById('labelInput').value=p;updBtn()};presets.appendChild(b)});
function fmt(t){t=Math.max(0,t|0);return String(t/60|0).padStart(2,'0')+':'+String(t%60).padStart(2,'0')}
function color(l){let h=0;for(const c of l)h=(h*31+c.charCodeAt(0))>>>0;return `hsl(${h%360},60%,45%)`}
function t2x(t){return t/DUR*cv.width}
function x2t(x){return x/cv.width*DUR}
function draw(){
 cv.width=cv.clientWidth; cv.height=170;
 const mid=85, amp=80;
 ctx.strokeStyle='#3a4'; ctx.lineWidth=1; ctx.beginPath();
 for(let i=0;i<WAVE.length;i++){
  const x=i/WAVE.length*cv.width, h=Math.max(1,WAVE[i]*amp*3);
  ctx.moveTo(x,mid-h); ctx.lineTo(x,mid+h);
 }
 ctx.stroke();
 RANGES.forEach((r,i)=>{ctx.fillStyle=color(r.label)+'55';
  ctx.fillRect(t2x(r.t0),2,t2x(r.t1)-t2x(r.t0),166);
  ctx.fillStyle=color(r.label);ctx.font='12px sans-serif';
  ctx.fillText(r.label,t2x(r.t0)+3,14)});
 if(sel){ctx.fillStyle='#fc64';ctx.fillRect(t2x(sel[0]),2,t2x(sel[1])-t2x(sel[0]),166);
  ctx.strokeStyle='#fc6';ctx.strokeRect(t2x(sel[0]),2,t2x(sel[1])-t2x(sel[0]),166)}
 const ct=aud.currentTime||0; ctx.strokeStyle='#fff'; ctx.beginPath();
 ctx.moveTo(t2x(ct),0); ctx.lineTo(t2x(ct),170); ctx.stroke();
}
function updBtn(){document.getElementById('saveBtn').disabled=!(sel&&document.getElementById('labelInput').value.trim())}
cv.onmousedown=e=>{const r=cv.getBoundingClientRect();anchor=x2t(e.clientX-r.left);sel=null};
cv.onmousemove=e=>{if(anchor===null)return;const r=cv.getBoundingClientRect();
 const t=x2t(e.clientX-r.left);sel=[Math.min(anchor,t),Math.max(anchor,t)];
 document.getElementById('selInfo').textContent=`选中 ${fmt(sel[0])} ~ ${fmt(sel[1])}（${(sel[1]-sel[0]).toFixed(1)}秒）`;updBtn();draw()};
cv.onmouseup=e=>{if(anchor===null)return;
 if(sel&&sel[1]-sel[0]>1){updBtn()}else{aud.currentTime=anchor;syncVideo();sel=null;document.getElementById('selInfo').textContent='跳转 '+fmt(anchor)}
 anchor=null;draw()};
document.getElementById('labelInput').oninput=updBtn;
async function saveRange(){
 const label=document.getElementById('labelInput').value.trim();
 await fetch('/api/range_save',{method:'POST',body:JSON.stringify({t0:+sel[0].toFixed(2),t1:+sel[1].toFixed(2),label})});
 RANGES.push({t0:sel[0],t1:sel[1],label});sel=null;updBtn();renderRanges();draw()}
async function delRange(i){
 await fetch('/api/range_del',{method:'POST',body:JSON.stringify({idx:i})});
 RANGES.splice(i,1);renderRanges();draw()}
function jump(t){aud.currentTime=t;syncVideo()}
function renderRanges(){
 const el=document.getElementById('ranges');el.innerHTML='';
 RANGES.forEach((r,i)=>{const d=document.createElement('div');
  d.innerHTML=`<span class="t" style="cursor:pointer" onclick="jump(${r.t0})">${fmt(r.t0)} ~ ${fmt(r.t1)}</span>
   <span style="color:${color(r.label)}">■</span><span>${r.label}</span>
   <span style="color:#778">(${(r.t1-r.t0).toFixed(1)}s)</span>
   <button class="del" onclick="delRange(${i})">删除</button>`;
  el.appendChild(d)})}
aud.ontimeupdate=()=>{draw();syncVideo()};
aud.onplay=()=>{if(hasVid)vid.play().catch(()=>{})};
aud.onpause=()=>vid.pause();
setInterval(draw,120);
window.onresize=draw;
renderRanges();draw();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--port", type=int, default=8471)
    args = ap.parse_args()

    sdir = Path(args.session)
    ldir = sdir / "labeling"
    clips_dir = ldir / "clips"
    if not clips_dir.exists():
        raise SystemExit(f"找不到 {clips_dir}，先跑 tool/make_labeling_pack.py")
    csv_path = ldir / "clips.csv"
    out_path = ldir / "user_labels.csv"

    clips = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clips.append({"clip": row["clip"], "t": row["t"],
                          "auto_self": row["auto_self"], "corr": row["corr"]})
    saved = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("label"):
                    saved[row["clip"]] = [x for x in row["label"].split(";") if x]

    subtypes = load_subtypes()

    video_path = sdir / "video.mp4"
    has_video = video_path.exists()
    # 按键边沿按键分组注入：标注时显示该片段时刻按着哪些键
    keys_map = {}
    kjsonl = sdir / "keys.jsonl"
    if kjsonl.exists():
        try:
            for line in kjsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                keys_map.setdefault(e["key"], []).append([e["t"], e["down"]])
            for k in keys_map:
                keys_map[k].sort()
        except Exception:
            keys_map = {}
    html = (HTML.replace("%CATS%", json.dumps(CATS, ensure_ascii=False))
            .replace("%SUBTYPES%", json.dumps(subtypes, ensure_ascii=False))
            .replace("%CLIPS%", json.dumps(clips, ensure_ascii=False))
            .replace("%SAVED%", json.dumps(saved, ensure_ascii=False))
            .replace("%KEYS%", json.dumps(keys_map, ensure_ascii=False))
            .replace("%HASVIDEO%", "true" if has_video else "false")
            .replace("%VIDSTYLE%", "" if has_video else "display:none"))

    # ---- 时间轴段落标注：整局波形 + 拉框选段 + 自定义标签（ranges.csv） ----
    import wave as _wave
    import struct
    ranges_path = ldir / "ranges.csv"
    wf_cache = ldir / "waveform.json"

    def load_ranges():
        rows = []
        if ranges_path.exists():
            with open(ranges_path, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append({"t0": float(r["t0"]), "t1": float(r["t1"]),
                                     "label": r["label"]})
                    except (KeyError, ValueError):
                        pass
        return rows

    def save_ranges(rows):
        with open(ranges_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["t0", "t1", "label"])
            w.writeheader()
            w.writerows(rows)

    duration = 0.0
    if wf_cache.exists():
        wave_peaks = json.loads(wf_cache.read_text(encoding="utf-8"))
        duration = len(wave_peaks) * 0.25
    else:
        try:
            with _wave.open(str(sdir / "session.wav"), "rb") as w:
                sr, ch, sw, nf = (w.getframerate(), w.getnchannels(),
                                  w.getsampwidth(), w.getnframes())
                duration = nf / sr
                n_bucket = max(1, int(duration / 0.25))
                per = max(1, int(sr * 0.25))
                peaks = []
                for _ in range(n_bucket):
                    frames = w.readframes(per)
                    if not frames:
                        break
                    if sw == 2:
                        import array
                        a = array.array("h")
                        a.frombytes(frames)
                        if ch > 1:
                            a = a[::ch]
                        peaks.append(round(max(abs(x) for x in a) / 32768, 4)
                                     if a else 0)
                    else:
                        peaks.append(0)
            wave_peaks = peaks
            wf_cache.write_text(json.dumps(peaks), encoding="utf-8")
        except Exception as e:
            print(f"[!] 波形生成失败: {e}")
            wave_peaks = []

    timeline_html = (TIMELINE_HTML
                     .replace("%WAVE%", json.dumps(wave_peaks))
                     .replace("%RANGES%", json.dumps(load_ranges(),
                                                     ensure_ascii=False))
                     .replace("%DUR%", str(round(duration, 2)))
                     .replace("%HASVIDEO%", "true" if has_video else "false")
                     .replace("%VIDSTYLE%", "" if has_video
                              else 'style="display:none"')
                     .replace("%AUDIOSTYLE%", 'style="display:none"' if has_video
                              else ""))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, html)
            elif self.path == "/timeline":
                self._send(200, timeline_html)
            elif self.path == "/api/ranges":
                self._send(200, json.dumps(load_ranges(), ensure_ascii=False),
                           "application/json")
            elif self.path == "/session.wav":
                wav_path = sdir / "session.wav"
                if not wav_path.exists():
                    self._send(404, "not found", "text/plain")
                    return
                size = wav_path.stat().st_size
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    spec = rng[6:].split("-")
                    start = int(spec[0]) if spec[0] else 0
                    end = int(spec[1]) if len(spec) > 1 and spec[1] else size - 1
                    end = min(end, size - 1)
                    with open(wav_path, "rb") as f:
                        f.seek(start)
                        data = f.read(end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    with open(wav_path, "rb") as f:
                        self.wfile.write(f.read())
            elif self.path == "/video.mp4" and has_video:
                # HTML5 video 拖动进度条依赖 Range：bytes=start-end → 206
                size = video_path.stat().st_size
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    spec = rng[6:].split("-")
                    start = int(spec[0]) if spec[0] else 0
                    end = int(spec[1]) if len(spec) > 1 and spec[1] else size - 1
                    end = min(end, size - 1)
                    with open(video_path, "rb") as f:
                        f.seek(start)
                        data = f.read(end - start + 1)
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Content-Range",
                                     f"bytes {start}-{end}/{size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    with open(video_path, "rb") as f:
                        self.wfile.write(f.read())
            elif self.path == "/video_frames.json" and has_video:
                fm = video_path.with_name("video_frames.json")
                if fm.exists():
                    self._send(200, fm.read_text(encoding="utf-8"),
                               "application/json")
                else:
                    self._send(404, "{}", "application/json")
            elif self.path.startswith("/clips/"):
                name = self.path[len("/clips/"):]
                p = clips_dir / name
                if p.exists() and p.suffix == ".wav":
                    self._send(200, p.read_bytes(), "audio/wav")
                else:
                    self._send(404, "not found", "text/plain")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path == "/api/range_save":
                n = int(self.headers.get("Content-Length", 0))
                row = json.loads(self.rfile.read(n))
                t0, t1 = float(row.get("t0", 0)), float(row.get("t1", 0))
                label = str(row.get("label", "")).strip()[:30] \
                    .replace(";", "").replace(",", "")
                if not label or t1 - t0 < 0.5:
                    self._send(400, "{}", "application/json")
                    return
                rows = load_ranges()
                rows.append({"t0": round(t0, 2), "t1": round(t1, 2),
                             "label": label})
                rows.sort(key=lambda r: r["t0"])
                save_ranges(rows)
                print(f"[段落] {t0:.0f}-{t1:.0f}s = {label}")
                self._send(200, "{}", "application/json")
            elif self.path == "/api/range_del":
                n = int(self.headers.get("Content-Length", 0))
                row = json.loads(self.rfile.read(n))
                idx = int(row.get("idx", -1))
                rows = load_ranges()
                if 0 <= idx < len(rows):
                    print(f"[段落] 删除 {rows[idx]['t0']:.0f}s {rows[idx]['label']}")
                    rows.pop(idx)
                    save_ranges(rows)
                self._send(200, "{}", "application/json")
            elif self.path == "/api/subtype":  # 某大类新增子类型，跨会话保留
                n = int(self.headers.get("Content-Length", 0))
                row = json.loads(self.rfile.read(n))
                cat = str(row.get("cat", ""))
                name = str(row.get("name", "")).strip()[:20]
                name = name.replace(":", "").replace(";", "").replace(",", "")
                if cat not in CAT_NAMES or not name:
                    self._send(400, "{}", "application/json")
                    return
                subs = load_subtypes()
                if name in subs[cat]:
                    self._send(200, json.dumps({"existed": True},
                                               ensure_ascii=False),
                               "application/json")
                    return
                subs[cat].append(name)
                save_subtypes(subs)
                print(f"[子类型] {cat} + {name}")
                self._send(200, json.dumps({"existed": False},
                                           ensure_ascii=False),
                           "application/json")
            elif self.path == "/api/subtype_del":  # 删除某大类的子类型
                n = int(self.headers.get("Content-Length", 0))
                row = json.loads(self.rfile.read(n))
                cat = str(row.get("cat", ""))
                name = str(row.get("name", "")).strip()[:20]
                if cat not in CAT_NAMES:
                    self._send(400, "{}", "application/json")
                    return
                subs = load_subtypes()
                if name in subs.get(cat, []):
                    subs[cat] = [x for x in subs[cat] if x != name]
                    save_subtypes(subs)
                    print(f"[子类型] {cat} - {name}")
                self._send(200, "{}", "application/json")
            elif self.path == "/api/tag":
                # 旧版 F 键自定义标签接口：新界面不再调用，保留兼容
                n = int(self.headers.get("Content-Length", 0))
                row = json.loads(self.rfile.read(n))
                cn = str(row.get("cn", "")).strip()[:20].replace(";", "").replace(",", "")
                if not cn:
                    self._send(400, "{}", "application/json")
                    return
                tags = load_custom()
                for t in tags:  # 同名直接复用
                    if t["cn"] == cn or t["en"] == cn:
                        self._send(200, json.dumps({**t, "existed": True},
                                                   ensure_ascii=False),
                                   "application/json")
                        return
                used = {t["key"] for t in tags} | {"1", "2", "3", "4", "5",
                                                   "6", "7"}
                key = next((f"F{i}" for i in range(1, 13) if f"F{i}" not in used),
                           None)
                if key is None:
                    self._send(200, json.dumps({"error": "F1-F12 已满"},
                                               ensure_ascii=False),
                               "application/json")
                    return
                tag = {"key": key, "en": cn, "cn": cn,
                       "color": COLOR_POOL[len(tags) % len(COLOR_POOL)]}
                tags.append(tag)
                save_custom(tags)
                print(f"[自定义标签] {key} = {cn}")
                self._send(200, json.dumps(tag, ensure_ascii=False),
                           "application/json")
            elif self.path == "/api/label":
                n = int(self.headers.get("Content-Length", 0))
                row = json.loads(self.rfile.read(n))
                rows = {}
                if out_path.exists():
                    with open(out_path, encoding="utf-8-sig") as f:
                        for r in csv.DictReader(f):
                            rows[r["clip"]] = r.get("label", "")
                if row["label"]:
                    rows[row["clip"]] = row["label"]
                else:
                    rows.pop(row["clip"], None)
                with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(["clip", "label"])
                    for k, v in rows.items():
                        w.writerow([k, v])
                self._send(200, "{}")
            else:
                self._send(404, "{}", "application/json")

    port = args.port
    while True:
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            port += 1
    url = f"http://127.0.0.1:{port}/"
    print(f"标注页: {url}")
    print(f"clips {len(clips)} 个，已标 {len(saved)} 个；标注落盘 {out_path.name}")
    print("Ctrl+C 停止服务\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print("服务已停止，标注都在 user_labels.csv 里")


if __name__ == "__main__":
    main()
