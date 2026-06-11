"""Zeus web control panel — Phase 1.

Mobile web UI served from the Pi (reachable over Tailscale at
http://100.117.250.1:8080). Serves its own bandwidth-friendly MJPEG stream
(/stream: downscaled + recompressed from Vilib's frame buffer — vilib's own
:9000/mjpg is ~20 Mbit/s and unusable over Wi-Fi), preset action buttons
grouped into move/look/posture/tricks, and a text command box that feeds
handle_command() — same path as a spoken command.

Phase 2 (planned): phone mic -> upload -> on-Pi transcription -> editable text.
Phase 3 (planned): robot speech mirrored to the phone (audio stream/answer text).
"""

import logging
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request

# Only one command/action runs at a time; extra requests are rejected with 429
# instead of queueing, so a laggy phone can't stack up movements.
_busy = threading.Lock()

# Stream tuning: ~12 fps, 480px wide, JPEG q60 ≈ 1.5 Mbit/s — fits Wi-Fi uplink
# so frames don't queue in TCP buffers (the cause of the multi-second lag).
_STREAM_FPS = 12
_STREAM_WIDTH = 480
_STREAM_QUALITY = 60


def _run_locked(fn):
    try:
        fn()
    finally:
        _busy.release()


def start(handle_command, run_actions, action_names, get_frame=None, port=8080):
    """Start the web UI in a daemon thread. Never raises into the caller."""
    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.get("/")
    def index():
        return _PAGE

    @app.get("/api/actions")
    def actions():
        return jsonify(sorted(action_names))

    @app.get("/stream")
    def stream():
        if get_frame is None:
            return jsonify(error="no camera"), 503

        def gen():
            interval = 1.0 / _STREAM_FPS
            params = [cv2.IMWRITE_JPEG_QUALITY, _STREAM_QUALITY]
            while True:
                t0 = time.time()
                frame = get_frame()
                if frame is not None:
                    h, w = frame.shape[:2]
                    if w > _STREAM_WIDTH:
                        frame = cv2.resize(
                            frame, (_STREAM_WIDTH, int(h * _STREAM_WIDTH / w)))
                    ok, jpg = cv2.imencode(".jpg", frame, params)
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + jpg.tobytes() + b"\r\n")
                time.sleep(max(0.0, interval - (time.time() - t0)))

        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/action")
    def do_action():
        name = (request.get_json(silent=True) or {}).get("name", "")
        if name not in action_names:
            return jsonify(error="unknown action"), 400
        if not _busy.acquire(blocking=False):
            return jsonify(error="busy"), 429
        threading.Thread(target=_run_locked,
                         args=(lambda: run_actions([name]),), daemon=True).start()
        return jsonify(ok=True)

    @app.post("/api/command")
    def do_command():
        text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
        if not text:
            return jsonify(error="empty"), 400
        if not _busy.acquire(blocking=False):
            return jsonify(error="busy"), 429
        threading.Thread(target=_run_locked,
                         args=(lambda: handle_command(text),), daemon=True).start()
        return jsonify(ok=True)

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True,
                               use_reloader=False),
        daemon=True,
    ).start()


_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Zeus</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;user-select:none}
body{margin:0;font-family:-apple-system,system-ui,sans-serif;background:#0b0e14;color:#e6e9f0}
header{display:flex;align-items:center;justify-content:space-between;padding:10px 14px}
h1{font-size:18px;margin:0}
#status{font-size:12px;color:#8b93a7}
#cam{width:100%;aspect-ratio:4/3;background:#000;display:block;object-fit:contain}
section{padding:10px 14px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#8b93a7;margin:14px 0 8px}
#cmdrow{display:flex;gap:8px}
#cmd{flex:1;padding:12px;border-radius:10px;border:1px solid #2a3040;background:#141927;color:#e6e9f0;font-size:16px}
button{border:0;border-radius:12px;background:#1d2536;color:#e6e9f0;font-size:14px;padding:12px 8px;cursor:pointer}
button:active{background:#33405e;transform:scale(.96)}
#send{background:#3b62d6;font-weight:600;padding:12px 18px}
.pads{display:flex;justify-content:space-around;gap:12px;margin-top:4px}
.pad{display:grid;grid-template-columns:repeat(3,58px);grid-template-rows:repeat(3,58px);gap:6px}
.pad button{font-size:22px;border-radius:14px}
.pad .lbl{grid-column:1/4;text-align:center;font-size:11px;color:#8b93a7;text-transform:uppercase;letter-spacing:.1em;align-self:end}
.move-pad button{background:#23304a}
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1d2536;border:1px solid #2a3040;padding:8px 16px;border-radius:10px;font-size:13px;opacity:0;transition:opacity .2s;pointer-events:none;z-index:9}
</style></head><body>
<header><h1>Zeus</h1><span id="status">connecting…</span></header>
<img id="cam" alt="camera">
<section>
  <div id="cmdrow">
    <input id="cmd" placeholder="Tell Zeus something…" autocomplete="off">
    <button id="send">Send</button>
  </div>

  <div class="pads">
    <div class="pad move-pad">
      <div class="lbl" style="grid-row:1">Move</div>
      <div></div><button data-a="forward" style="grid-row:2">▲</button><div></div>
      <button data-a="turn_left" style="grid-row:3">◀</button>
      <button data-a="backward" style="grid-row:3">▼</button>
      <button data-a="turn_right" style="grid-row:3">▶</button>
    </div>
    <div class="pad">
      <div class="lbl" style="grid-row:1">Look</div>
      <div></div><button data-a="look_up" style="grid-row:2">▲</button><div></div>
      <button data-a="look_left" style="grid-row:3">◀</button>
      <button data-a="look_down" style="grid-row:3">▼</button>
      <button data-a="look_right" style="grid-row:3">▶</button>
    </div>
  </div>

  <h2>Gait</h2>
  <div class="row">
    <button data-a="spin">🌀 spin</button>
    <button data-a="creep">🐾 creep</button>
    <button data-a="swagger">😎 swagger</button>
  </div>

  <h2>Posture</h2>
  <div class="grid4">
    <button data-a="stand">🧍 stand</button>
    <button data-a="sit">🪑 sit</button>
    <button data-a="crouch_rise">⬇⬆ crouch</button>
    <button data-a="sway">〰 sway</button>
  </div>

  <h2>Tricks</h2>
  <div class="grid4">
    <button data-a="wave_hand">👋 wave</button>
    <button data-a="shake_hand">🤝 shake</button>
    <button data-a="nod">✅ nod</button>
    <button data-a="shake_head">❌ no</button>
    <button data-a="excited">🎉 excited</button>
    <button data-a="fighting">🥊 fight</button>
    <button data-a="play_dead">💀 dead</button>
    <button data-a="push_up">💪 pushup</button>
    <button data-a="warm_up">🤸 warmup</button>
    <button data-a="bob">🎶 bob</button>
    <button data-a="play_music">🎵 music</button>
  </div>
</section>
<div id="toast"></div>
<script>
const cam=document.getElementById("cam");
cam.src="/stream";
cam.onerror=()=>setTimeout(()=>{cam.src="/stream?t="+Date.now()},2000);
const toast=t=>{const e=document.getElementById("toast");e.textContent=t;e.style.opacity=1;
  clearTimeout(e._t);e._t=setTimeout(()=>e.style.opacity=0,1500)};
async function post(url,body){
  try{const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)});
    if(r.status===429){toast("Zeus is busy…");return false}
    if(!r.ok){toast("Error");return false}
    return true;
  }catch(e){toast("No connection");return false}}
for(const b of document.querySelectorAll("button[data-a]"))
  b.onclick=async()=>{if(await post("/api/action",{name:b.dataset.a}))
    toast(b.dataset.a.replaceAll("_"," "))};
const cmd=document.getElementById("cmd");
async function send(){const t=cmd.value.trim();if(!t)return;
  if(await post("/api/command",{text:t})){cmd.value="";toast("Sent")}}
document.getElementById("send").onclick=send;
cmd.addEventListener("keydown",e=>{if(e.key==="Enter")send()});
fetch("/api/actions").then(r=>r.ok&&(document.getElementById("status").textContent="online"))
  .catch(()=>document.getElementById("status").textContent="offline");
</script></body></html>
"""
