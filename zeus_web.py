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

import audioop
import logging
import queue
import subprocess
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from flask_sock import Sock

# Only one command/action runs at a time; extra requests are rejected with 429
# instead of queueing, so a laggy phone can't stack up movements.
_busy = threading.Lock()

# Stream tuning, overridable per-request (?w=&q=&fps=). Campus Wi-Fi gives
# ~200 kbit/s device-to-device, so the default is small frames: throughput-
# starved links get more fps and less TCP-buffer delay from smaller JPEGs.
_STREAM_FPS = 15
_STREAM_WIDTH = 320
_STREAM_QUALITY = 45


# Phase 3: everything Zeus speaks is mirrored here — text for the chat log,
# raw piper audio (22050 Hz S16 mono) wrapped as WAV for phone playback.
_speech_lock = threading.Lock()
_speech_events = []          # [{"id", "text", "ts"}], audio kept separately
_speech_audio = {}           # id -> wav bytes
_SPEECH_KEEP = 20


def note_speech_text(text):
    """Called from zeus.speak() as the utterance STARTS — text shows in the
    chat immediately; the audio attaches via note_speech_audio when done."""
    with _speech_lock:
        sid = (_speech_events[-1]["id"] + 1) if _speech_events else 1
        _speech_events.append({"id": sid, "text": text, "ts": time.time()})
        while len(_speech_events) > _SPEECH_KEEP:
            old = _speech_events.pop(0)
            _speech_audio.pop(old["id"], None)
    return sid


def note_speech_audio(sid, raw_pcm):
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(raw_pcm)
    with _speech_lock:
        _speech_audio[sid] = buf.getvalue()


# Live voice fan-out: speak() feeds PCM chunks here as piper synthesizes them,
# so a phone with 🔊 on hears Zeus in near-real-time over /ws/speech.
_live_lock = threading.Lock()
_live_clients = set()


def feed_live(chunk):
    with _live_lock:
        for q in _live_clients:
            try:
                q.put_nowait(chunk)
            except queue.Full:
                pass


# ── WebRTC call (aiortc): Opus, jitter buffer, real AEC on the phone side ───
_rtc_loop = None
_rtc_pc = None


def _get_rtc_loop():
    global _rtc_loop
    if _rtc_loop is None:
        import asyncio
        _rtc_loop = asyncio.new_event_loop()
        threading.Thread(target=_rtc_loop.run_forever, daemon=True).start()
    return _rtc_loop


async def _rtc_close_current():
    global _rtc_pc
    pc = _rtc_pc
    _rtc_pc = None
    if pc is not None:
        try:
            await pc.close()
        except Exception:
            pass


async def _rtc_answer(offer, call_mode):
    global _rtc_pc
    import asyncio
    import fractions

    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

    await _rtc_close_current()
    loop = asyncio.get_event_loop()
    call_mode.set()

    # Acquire the mic in the background (the wake-word loop takes a couple of
    # seconds to release it) so SDP signaling isn't blocked — MicTrack sends
    # silence until it's ready.
    rec_holder = [None]

    async def acquire_mic():
        for _ in range(10):
            await asyncio.sleep(0.5)
            cand = subprocess.Popen(
                ["arecord", "-D", "plughw:3,0", "-f", "S16_LE",
                 "-r", "16000", "-c", "1", "-q"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            head = await loop.run_in_executor(None, cand.stdout.read, 64)
            if head and cand.poll() is None:
                rec_holder[0] = cand
                return
            try:
                cand.terminate()
            except Exception:
                pass

    asyncio.ensure_future(acquire_mic())

    play = subprocess.Popen(
        ["aplay", "-D", "robothat", "-f", "S16_LE", "-r", "16000",
         "-c", "1", "-q", "--buffer-time=150000", "--period-time=30000"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    speaker_until = [0.0]
    closed = [False]

    class MicTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self.pts = 0

        async def recv(self):
            rec = rec_holder[0]
            if rec is None:                      # mic not released yet: silence
                await asyncio.sleep(0.04)
                data = b"\x00" * 1280
            else:
                data = await loop.run_in_executor(None, rec.stdout.read, 1280)  # 40 ms
            if not data:
                raise MediaStreamError
            # Duck (not mute) the robot mic while its speaker plays the phone's
            # voice — phone-side WebRTC AEC handles the rest of the echo.
            if time.time() < speaker_until[0]:
                data = audioop.mul(data, 2, 0.05)
            frame = av.AudioFrame(format="s16", layout="mono",
                                  samples=len(data) // 2)
            frame.planes[0].update(data)
            frame.sample_rate = 16000
            frame.pts = self.pts
            self.pts += len(data) // 2
            frame.time_base = fractions.Fraction(1, 16000)
            return frame

    pc = RTCPeerConnection()
    _rtc_pc = pc
    pc.addTrack(MicTrack())

    async def pump_to_speaker(track):
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        try:
            while True:
                frame = await track.recv()
                for f in resampler.resample(frame):
                    data = bytes(f.planes[0])
                    try:
                        if audioop.rms(data, 2) > 120:
                            speaker_until[0] = time.time() + 0.25
                    except Exception:
                        pass
                    await loop.run_in_executor(None, play.stdin.write, data)
        except Exception:
            pass

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.ensure_future(pump_to_speaker(track))

    async def cleanup():
        if closed[0]:
            return
        closed[0] = True
        for p in (rec_holder[0], play):
            if p is None:
                continue
            try:
                p.terminate()
            except Exception:
                pass
        call_mode.clear()

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await cleanup()
            try:
                await pc.close()
            except Exception:
                pass

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


def _run_locked(fn):
    try:
        fn()
    finally:
        _busy.release()


_NO_MOVE_FLAG = "/home/pi/.zeus_no_move"
_NO_TRACK_FLAG = "/home/pi/.zeus_no_track"
_SERVICES = ["zeus", "ollama", "tailscaled", "NetworkManager"]


def _sh(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception as e:
        return f"error: {e}"


def _wav_header(rate=16000):
    """Streaming WAV header with max data length (for the live intercom feed)."""
    import struct
    return (b"RIFF" + struct.pack("<I", 0x7fffffff) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", 0x7fffffff))


def start(handle_command, run_actions, action_names, get_frame=None,
          transcribe=None, zero_servos=None, call_mode=None,
          intercom_play=None, port=8080):
    """Start the web UI in a daemon thread. Never raises into the caller."""
    app = Flask(__name__)
    sock = Sock(app)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @sock.route("/ws/speech")
    def ws_speech(ws):
        """Zeus's voice, live (22050 Hz S16 mono PCM as he speaks)."""
        q = queue.Queue(maxsize=400)
        with _live_lock:
            _live_clients.add(q)
        try:
            while True:
                try:
                    ws.send(q.get(timeout=20))
                except queue.Empty:
                    ws.send(b"")   # keepalive
        except Exception:
            pass
        finally:
            with _live_lock:
                _live_clients.discard(q)

    @sock.route("/ws/call")
    def ws_call(ws):
        """Full-duplex intercom: 16 kHz S16 mono PCM both directions.
        Half-duplex auto-gate: while the phone is talking, the robot mic is
        muted briefly so the speaker doesn't feed back to the phone."""
        if call_mode is None:
            return
        call_mode.set()
        # The wake-word loop's arecord can take a couple of seconds to release
        # the device (single capture subdevice) — retry until we get audio.
        rec = None
        for _ in range(10):
            time.sleep(0.5)
            cand = subprocess.Popen(
                ["arecord", "-D", "plughw:3,0", "-f", "S16_LE",
                 "-r", "16000", "-c", "1", "-q"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            head = cand.stdout.read(64)
            if head and cand.poll() is None:
                rec = cand
                break
            try:
                cand.terminate()
            except Exception:
                pass
        if rec is None:
            call_mode.clear()
            return
        # Small ALSA buffer (150 ms): keeps robot-side playback near-live and
        # lets blocking writes act as backpressure instead of growing a queue.
        play = subprocess.Popen(
            ["aplay", "-D", "robothat", "-f", "S16_LE",
             "-r", "16000", "-c", "1", "-q",
             "--buffer-time=150000", "--period-time=30000"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        stop = threading.Event()
        phone_talking_until = [0.0]

        def mic_pump():
            while not stop.is_set():
                chunk = rec.stdout.read(1600)   # 50 ms
                if not chunk:
                    break
                if time.time() < phone_talking_until[0]:
                    continue
                try:
                    ws.send(chunk)
                except Exception:
                    break

        threading.Thread(target=mic_pump, daemon=True).start()
        try:
            while True:
                data = ws.receive()
                if data is None:
                    break
                if isinstance(data, str):
                    continue
                # Half-duplex echo gate: phone speech wins. Mute the robot mic
                # while the speaker will be playing the phone's voice (arrival
                # + ALSA buffer latency + hangover), so it can't loop back.
                try:
                    if audioop.rms(data, 2) > 150:
                        phone_talking_until[0] = time.time() + 0.9
                except Exception:
                    pass
                try:
                    play.stdin.write(data)
                except Exception:
                    break
        except Exception:
            pass
        finally:
            stop.set()
            for p in (rec, play):
                try:
                    p.terminate()
                except Exception:
                    pass
            call_mode.clear()

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
        width = max(160, min(640, request.args.get("w", _STREAM_WIDTH, type=int)))
        quality = max(20, min(90, request.args.get("q", _STREAM_QUALITY, type=int)))
        fps = max(1, min(30, request.args.get("fps", _STREAM_FPS, type=int)))

        def gen():
            interval = 1.0 / fps
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            while True:
                t0 = time.time()
                frame = get_frame()
                if frame is not None:
                    h, w = frame.shape[:2]
                    if w > width:
                        frame = cv2.resize(
                            frame, (width, int(h * width / w)))
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

    # ── System panel ─────────────────────────────────────────────────────────
    @app.get("/api/system")
    def system():
        import os
        throttled = _sh(["vcgencmd", "get_throttled"]).split("=")[-1]
        try:
            t = int(throttled, 16)
        except ValueError:
            t = 0
        mem = _sh(["free", "-m"]).splitlines()
        mem_line = mem[1].split() if len(mem) > 1 else []
        return jsonify(
            temp=_sh(["vcgencmd", "measure_temp"]).replace("temp=", ""),
            volts=_sh(["vcgencmd", "measure_volts", "core"]).replace("volt=", ""),
            throttled=throttled,
            under_voltage_now=bool(t & 0x1),
            throttled_now=bool(t & 0x2),
            under_voltage_past=bool(t & 0x10000),
            throttled_past=bool(t & 0x40000),
            mem_used_mb=int(mem_line[2]) if len(mem_line) > 2 else 0,
            mem_total_mb=int(mem_line[1]) if len(mem_line) > 1 else 0,
            uptime=_sh(["uptime", "-p"]),
            flags={"no_move": os.path.exists(_NO_MOVE_FLAG),
                   "no_track": os.path.exists(_NO_TRACK_FLAG)},
            services=[{"name": s, "active": _sh(["systemctl", "is-active", s])}
                      for s in _SERVICES],
        )

    @app.post("/api/system/flag")
    def set_flag():
        import os
        d = request.get_json(silent=True) or {}
        path = {"no_move": _NO_MOVE_FLAG, "no_track": _NO_TRACK_FLAG}.get(d.get("name"))
        if not path:
            return jsonify(error="unknown flag"), 400
        if d.get("on"):
            open(path, "w").close()
        elif os.path.exists(path):
            os.remove(path)
        return jsonify(ok=True, note="restart Zeus to apply")

    @app.post("/api/system/service")
    def service_action():
        d = request.get_json(silent=True) or {}
        name, action = d.get("name"), d.get("action")
        if name not in _SERVICES or action not in ("restart", "start", "stop"):
            return jsonify(error="not allowed"), 400
        # detached so restarting zeus itself doesn't kill the request mid-flight
        subprocess.Popen(["systemd-run", "--no-block", "systemctl", action, name])
        return jsonify(ok=True)

    @app.get("/api/system/processes")
    def processes():
        return Response(_sh(["sh", "-c",
                             "ps aux --sort=-%cpu | head -14"]), mimetype="text/plain")

    @app.get("/api/system/logs")
    def logs():
        out = _sh(["sh", "-c",
                   "tail -n 80 /home/pi/zeus.log | tr -cd '[:print:]\\n'"])
        return Response(out, mimetype="text/plain")

    @app.post("/api/system/power")
    def power():
        what = (request.get_json(silent=True) or {}).get("what")
        if what == "restart_zeus":
            subprocess.Popen(["systemd-run", "--no-block", "systemctl", "restart", "zeus"])
        elif what == "reboot":
            subprocess.Popen(["systemd-run", "--no-block", "reboot"])
        else:
            return jsonify(error="unknown"), 400
        return jsonify(ok=True)

    @app.post("/api/zero")
    def zero():
        if zero_servos is None:
            return jsonify(error="unavailable"), 503
        if not _busy.acquire(blocking=False):
            return jsonify(error="busy"), 429
        threading.Thread(target=_run_locked, args=(zero_servos,),
                         daemon=True).start()
        return jsonify(ok=True)

    # ── WebRTC signaling ─────────────────────────────────────────────────────
    @app.post("/api/rtc/offer")
    def rtc_offer():
        if call_mode is None:
            return jsonify(error="unavailable"), 503
        import asyncio
        offer = request.get_json(silent=True) or {}
        if not offer.get("sdp"):
            return jsonify(error="bad offer"), 400
        fut = asyncio.run_coroutine_threadsafe(
            _rtc_answer(offer, call_mode), _get_rtc_loop())
        try:
            return jsonify(fut.result(timeout=25))
        except Exception as e:
            call_mode.clear()
            return jsonify(error=str(e)), 500

    @app.post("/api/rtc/hangup")
    def rtc_hangup():
        import asyncio
        asyncio.run_coroutine_threadsafe(_rtc_close_current(), _get_rtc_loop())
        return jsonify(ok=True)

    # ── Intercom (call mode) ─────────────────────────────────────────────────
    @app.post("/api/intercom/say")
    def intercom_say():
        if intercom_play is None:
            return jsonify(error="unavailable"), 503
        blob = request.get_data()
        if not blob or len(blob) > 10 * 1024 * 1024:
            return jsonify(error="bad audio"), 400
        src, wav = "/tmp/zeus_intercom.bin", "/tmp/zeus_intercom.wav"
        with open(src, "wb") as f:
            f.write(blob)
        conv = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src,
                               "-ar", "22050", "-ac", "1", wav],
                              capture_output=True, timeout=30)
        if conv.returncode != 0:
            return jsonify(error="convert failed"), 422
        threading.Thread(target=intercom_play, args=(wav,), daemon=True).start()
        return jsonify(ok=True)

    @app.get("/api/intercom/listen")
    def intercom_listen():
        if call_mode is None:
            return jsonify(error="unavailable"), 503

        def gen():
            call_mode.set()
            time.sleep(0.8)  # let the voice loop release the mic
            proc = subprocess.Popen(
                ["arecord", "-D", "plughw:3,0", "-f", "S16_LE",
                 "-r", "16000", "-c", "1", "-q"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                yield _wav_header(16000)
                while True:
                    chunk = proc.stdout.read(3200)  # 100 ms
                    if not chunk:
                        break
                    yield chunk
            finally:
                proc.terminate()
                call_mode.clear()

        return Response(gen(), mimetype="audio/wav",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/speech")
    def speech():
        after = request.args.get("after", 0, type=int)
        with _speech_lock:
            return jsonify([e for e in _speech_events if e["id"] > after])

    @app.get("/api/speech/<int:sid>.wav")
    def speech_wav(sid):
        with _speech_lock:
            wav = _speech_audio.get(sid)
        if wav is None:
            return jsonify(error="gone"), 404
        return Response(wav, mimetype="audio/wav")

    @app.post("/api/transcribe")
    def do_transcribe():
        # Phone mic audio (webm/ogg/mp4 from MediaRecorder) -> 16k mono wav ->
        # whisper.cpp. Returns the text for the user to edit before sending.
        if transcribe is None:
            return jsonify(error="no transcriber"), 503
        blob = request.get_data()
        if not blob or len(blob) > 10 * 1024 * 1024:
            return jsonify(error="bad audio"), 400
        src = "/tmp/zeus_web_mic.bin"
        wav = "/tmp/zeus_web_mic.wav"
        with open(src, "wb") as f:
            f.write(blob)
        conv = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", src,
             "-ar", "16000", "-ac", "1", wav],
            capture_output=True, timeout=30)
        if conv.returncode != 0:
            return jsonify(error="convert failed"), 422
        return jsonify(text=transcribe(wav) or "")

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
#mic,#spk{font-size:18px;padding:12px 14px}
#spk.on{background:#1e7d4f}
#chat{display:flex;flex-direction:column;gap:6px;margin-bottom:10px;max-height:180px;overflow-y:auto}
#chat:empty{display:none}
.msg{padding:8px 12px;border-radius:12px;font-size:14px;max-width:85%}
.msg.zeus{background:#1d2536;align-self:flex-start}
.msg.me{background:#3b62d6;align-self:flex-end}
#mic.rec{background:#c0392b;animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.6}}
.pads{display:flex;justify-content:space-around;gap:12px;margin-top:4px}
.pad{display:grid;grid-template-columns:repeat(3,58px);grid-template-rows:auto 58px 58px;gap:6px}
.pad button{font-size:22px;border-radius:14px}
.pad .lbl{grid-column:1/4;text-align:center;font-size:11px;color:#8b93a7;text-transform:uppercase;letter-spacing:.1em;align-self:end}
.move-pad button{background:#23304a}
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1d2536;border:1px solid #2a3040;padding:8px 16px;border-radius:10px;font-size:13px;opacity:0;transition:opacity .2s;pointer-events:none;z-index:9}
#tabs{display:flex;gap:8px;padding:0 14px 8px}
#tabs button{flex:1;font-weight:600}
#tabs button.sel{background:#3b62d6}
.callrow{display:flex;gap:8px;margin-top:10px}
.callrow button{flex:1;padding:14px}
#call.on{background:#1e7d4f}
#ptt:active{background:#c0392b}
pre{background:#141927;border:1px solid #2a3040;border-radius:10px;padding:10px;font-size:10px;overflow-x:auto;white-space:pre;max-height:240px;overflow-y:auto}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.card{background:#141927;border:1px solid #2a3040;border-radius:12px;padding:10px 12px}
.card b{display:block;font-size:11px;color:#8b93a7;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;background:#1e7d4f}
.badge.bad{background:#c0392b}
.badge.warn{background:#a67708}
.svc{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #1d2536}
.svc span.nm{flex:1}
.dot{width:10px;height:10px;border-radius:50%;background:#c0392b}
.dot.up{background:#27ae60}
.svc button{padding:6px 12px;font-size:12px}
.flagrow{display:flex;align-items:center;gap:10px;padding:8px 0}
.flagrow span{flex:1}
.danger{background:#5c1f1f}
</style></head><body>
<header><h1>Zeus</h1>
<span>
  <select id="vq" style="background:#1d2536;color:#e6e9f0;border:1px solid #2a3040;border-radius:8px;padding:4px 6px;font-size:12px">
    <option value="w=240&q=35&fps=15">Low (slow Wi-Fi)</option>
    <option value="w=320&q=45&fps=15" selected>Medium</option>
    <option value="w=480&q=60&fps=15">High</option>
    <option value="w=640&q=75&fps=20">Max (good Wi-Fi)</option>
  </select>
  <span id="status">connecting…</span>
</span></header>
<img id="cam" alt="camera">
<div id="tabs">
  <button id="tab-control" class="sel">Control</button>
  <button id="tab-system">System</button>
</div>
<section id="pg-control">
  <div id="chat"></div>
  <div id="cmdrow">
    <button id="spk" title="Hear Zeus on this phone">🔇</button>
    <button id="mic">🎤</button>
    <input id="cmd" placeholder="Tell Zeus something…" autocomplete="off">
    <button id="send">Send</button>
  </div>

  <div class="pads">
    <div class="pad move-pad">
      <div class="lbl" style="grid-area:1/1/2/4">Move</div>
      <button data-a="forward" style="grid-area:2/2">▲</button>
      <button data-a="turn_left" style="grid-area:3/1">◀</button>
      <button data-a="backward" style="grid-area:3/2">▼</button>
      <button data-a="turn_right" style="grid-area:3/3">▶</button>
    </div>
    <div class="pad">
      <div class="lbl" style="grid-area:1/1/2/4">Look</div>
      <button data-a="look_up" style="grid-area:2/2">▲</button>
      <button data-a="look_left" style="grid-area:3/1">◀</button>
      <button data-a="look_down" style="grid-area:3/2">▼</button>
      <button data-a="look_right" style="grid-area:3/3">▶</button>
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
    <button data-a="dance">💃 dance</button>
  </div>

  <h2>Call</h2>
  <div class="callrow">
    <button id="call">📞 Start call</button>
  </div>
  <audio id="rtcaudio" autoplay playsinline style="display:none"></audio>
</section>

<section id="pg-system" style="display:none">
  <div class="kv">
    <div class="card"><b>Power</b><span id="sys-power">…</span></div>
    <div class="card"><b>Temp / Volts</b><span id="sys-temp">…</span></div>
    <div class="card"><b>Memory</b><span id="sys-mem">…</span></div>
    <div class="card"><b>Uptime</b><span id="sys-up">…</span></div>
  </div>

  <h2>Safety flags <small style="text-transform:none;color:#667">(restart Zeus to apply)</small></h2>
  <div class="flagrow"><span>🛡 Brownout guard (no movement)</span>
    <button id="flag-no_move">…</button></div>
  <div class="flagrow"><span>👁 Disable face tracking only</span>
    <button id="flag-no_track">…</button></div>

  <h2>Services</h2>
  <div id="svcs"></div>

  <h2>Processes</h2>
  <pre id="procs">…</pre>

  <h2>Logs</h2>
  <pre id="logs">…</pre>

  <h2>Power actions</h2>
  <div class="row">
    <button id="zero">🦿 Zero servos</button>
    <button class="danger" id="rzeus">♻ Restart Zeus</button>
    <button class="danger" id="reboot">🔌 Reboot Pi</button>
  </div>
</section>
<div id="toast"></div>
<script>
const cam=document.getElementById("cam"),vq=document.getElementById("vq");
const setStream=()=>cam.src="/stream?"+vq.value+"&t="+Date.now();
vq.onchange=setStream;
if(localStorage.vq){vq.value=localStorage.vq}
vq.addEventListener("change",()=>localStorage.vq=vq.value);
setStream();
cam.onerror=()=>setTimeout(setStream,2000);
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
const chat=document.getElementById("chat");
function addMsg(text,who){const d=document.createElement("div");
  d.className="msg "+who;d.textContent=text;chat.appendChild(d);
  while(chat.children.length>30)chat.firstChild.remove();
  chat.scrollTop=chat.scrollHeight}
async function send(){const t=cmd.value.trim();if(!t)return;
  if(await post("/api/command",{text:t})){cmd.value="";addMsg(t,"me")}}
// Shared low-latency PCM player (Int16 mono chunks -> Web Audio).
function pcmPlayer(rate){
  const ctx=new (window.AudioContext||window.webkitAudioContext)();
  let at=0;
  const resume=()=>{if(ctx.state==="suspended")ctx.resume()};
  document.addEventListener("visibilitychange",resume);
  document.addEventListener("touchstart",resume,{passive:true});
  return{ctx,
    push(buf){const i16=new Int16Array(buf);if(!i16.length)return;
      resume();
      const f32=Float32Array.from(i16,v=>v/32768);
      const b=ctx.createBuffer(1,f32.length,rate);b.copyToChannel(f32,0);
      const s=ctx.createBufferSource();s.buffer=b;s.connect(ctx.destination);
      // Clamp backlog: if we've fallen >350 ms behind live (network burst,
      // tab pause), skip ahead instead of accumulating delay forever.
      if(at-ctx.currentTime>0.35)at=ctx.currentTime+0.1;
      const t=Math.max(ctx.currentTime+0.08,at);s.start(t);at=t+b.duration;},
    close(){try{ctx.close()}catch(e){}}};
}
const wsBase=(location.protocol==="https:"?"wss://":"ws://")+location.host;

// Chat: poll fast — text appears the moment Zeus STARTS speaking.
// 🔊 = live voice over WebSocket (near-real-time, plays while he talks).
const spk=document.getElementById("spk");
let hearing=false,lastId=0,booted=false,liveWs=null,livePl=null;
spk.onclick=()=>{
  hearing=!hearing;spk.textContent=hearing?"🔊":"🔇";
  spk.classList.toggle("on",hearing);
  if(hearing){
    livePl=pcmPlayer(22050);
    liveWs=new WebSocket(wsBase+"/ws/speech");
    liveWs.binaryType="arraybuffer";
    liveWs.onmessage=e=>livePl.push(e.data);
    liveWs.onclose=()=>{if(hearing)toast("Voice link lost")};
  }else{
    if(liveWs)liveWs.close();liveWs=null;
    if(livePl)livePl.close();livePl=null;
  }};
async function pollSpeech(){
  try{
    const r=await fetch("/api/speech?after="+lastId);
    const evs=await r.json();
    for(const e of evs){
      lastId=e.id;
      if(!booted)continue;          // skip backlog from before page load
      addMsg(e.text,"zeus");
    }
    booted=true;
  }catch(e){}
  setTimeout(pollSpeech,800)}
pollSpeech();
document.getElementById("send").onclick=send;
cmd.addEventListener("keydown",e=>{if(e.key==="Enter")send()});
// Tap mic to record, tap again to stop; transcript lands in the box for editing.
const micBtn=document.getElementById("mic");
let mr=null,chunks=[];
micBtn.onclick=async()=>{
  if(mr&&mr.state==="recording"){mr.stop();return}
  if(!navigator.mediaDevices){toast("Mic needs HTTPS — use the ts.net link");return}
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    mr=new MediaRecorder(stream);chunks=[];
    mr.ondataavailable=e=>chunks.push(e.data);
    mr.onstop=async()=>{
      stream.getTracks().forEach(t=>t.stop());
      micBtn.classList.remove("rec");
      toast("Transcribing…");
      try{
        const r=await fetch("/api/transcribe",{method:"POST",
          body:new Blob(chunks,{type:mr.mimeType})});
        const j=await r.json();
        if(r.ok&&j.text){cmd.value=j.text;cmd.focus()}
        else toast(j.text===""?"Heard nothing":"Transcribe failed");
      }catch(e){toast("Transcribe failed")}
    };
    mr.start();micBtn.classList.add("rec");toast("Recording — tap to stop");
  }catch(e){toast("Mic permission denied")}
};
fetch("/api/actions").then(r=>r.ok&&(document.getElementById("status").textContent="online"))
  .catch(()=>document.getElementById("status").textContent="offline");

// ── Tabs ──
const pgC=document.getElementById("pg-control"),pgS=document.getElementById("pg-system");
const tC=document.getElementById("tab-control"),tS=document.getElementById("tab-system");
let sysTimer=null;
tC.onclick=()=>{pgC.style.display="";pgS.style.display="none";
  tC.classList.add("sel");tS.classList.remove("sel");clearInterval(sysTimer)};
tS.onclick=()=>{pgC.style.display="none";pgS.style.display="";
  tS.classList.add("sel");tC.classList.remove("sel");
  refreshSys();refreshProcs();refreshLogs();
  clearInterval(sysTimer);sysTimer=setInterval(refreshSys,5000)};

// ── System page ──
async function refreshSys(){
  try{
    const s=await(await fetch("/api/system")).json();
    const pow=document.getElementById("sys-power");
    if(s.under_voltage_now)pow.innerHTML='<span class="badge bad">UNDER-VOLTAGE NOW</span>';
    else if(s.throttled_now)pow.innerHTML='<span class="badge bad">THROTTLING NOW</span>';
    else if(s.under_voltage_past||s.throttled_past)
      pow.innerHTML='<span class="badge warn">OK now, browned out earlier</span>';
    else pow.innerHTML='<span class="badge">HEALTHY</span>';
    pow.innerHTML+=" <small>"+s.throttled+"</small>";
    document.getElementById("sys-temp").textContent=s.temp+" / "+s.volts;
    document.getElementById("sys-mem").textContent=s.mem_used_mb+" / "+s.mem_total_mb+" MB";
    document.getElementById("sys-up").textContent=s.uptime;
    for(const f of ["no_move","no_track"]){
      const b=document.getElementById("flag-"+f);
      b.textContent=s.flags[f]?"ON":"OFF";
      b.style.background=s.flags[f]?"#1e7d4f":"#1d2536";
      b.onclick=async()=>{await post("/api/system/flag",{name:f,on:!s.flags[f]});
        toast("Saved — restart Zeus to apply");refreshSys()};
    }
    const sv=document.getElementById("svcs");sv.innerHTML="";
    for(const x of s.services){
      const d=document.createElement("div");d.className="svc";
      d.innerHTML='<span class="dot'+(x.active==="active"?" up":"")+'"></span>'+
        '<span class="nm">'+x.name+' <small style="color:#667">'+x.active+'</small></span>';
      const rb=document.createElement("button");rb.textContent="restart";
      rb.onclick=async()=>{if(confirm("Restart "+x.name+"?")){
        await post("/api/system/service",{name:x.name,action:"restart"});
        toast("Restarting "+x.name)}};
      d.appendChild(rb);sv.appendChild(d);
    }
  }catch(e){}
}
async function refreshProcs(){try{
  document.getElementById("procs").textContent=await(await fetch("/api/system/processes")).text()}catch(e){}}
async function refreshLogs(){try{
  const el=document.getElementById("logs");
  el.textContent=await(await fetch("/api/system/logs")).text();
  el.scrollTop=el.scrollHeight}catch(e){}}
document.getElementById("procs").onclick=refreshProcs;
document.getElementById("logs").onclick=refreshLogs;
document.getElementById("zero").onclick=async()=>{
  if(await post("/api/zero",{}))toast("Zeroing servos")};
document.getElementById("rzeus").onclick=async()=>{
  if(confirm("Restart Zeus? ~60s downtime."))
    {await post("/api/system/power",{what:"restart_zeus"});toast("Restarting Zeus…")}};
document.getElementById("reboot").onclick=async()=>{
  if(confirm("Reboot the whole Pi? A few minutes downtime."))
    {await post("/api/system/power",{what:"reboot"});toast("Rebooting Pi…")}};

// ── Intercom call: WebRTC (Opus + jitter buffer + real echo cancellation) ──
const callBtn=document.getElementById("call"),rtcAudio=document.getElementById("rtcaudio");
let pc=null,callStream=null;
async function startCall(){
  try{
    callStream=await navigator.mediaDevices.getUserMedia(
      {audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
  }catch(e){toast("Mic permission denied");return}
  callBtn.textContent="⏳ Connecting…";
  pc=new RTCPeerConnection({iceServers:[]});
  callStream.getTracks().forEach(t=>pc.addTrack(t,callStream));
  pc.ontrack=e=>{rtcAudio.srcObject=e.streams[0];rtcAudio.play().catch(()=>{})};
  pc.onconnectionstatechange=()=>{
    if(!pc)return;
    if(pc.connectionState==="connected"){
      callBtn.textContent="📵 End call";callBtn.classList.add("on");
      toast("Call connected — talk freely");}
    else if(["failed","disconnected","closed"].includes(pc.connectionState))endCall(true);};
  try{
    await pc.setLocalDescription(await pc.createOffer());
    await new Promise(r=>{
      if(pc.iceGatheringState==="complete")return r();
      const c=()=>{if(pc.iceGatheringState==="complete"){
        pc.removeEventListener("icegatheringstatechange",c);r()}};
      pc.addEventListener("icegatheringstatechange",c);
      setTimeout(r,2000);});
    const resp=await fetch("/api/rtc/offer",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({sdp:pc.localDescription.sdp,type:pc.localDescription.type})});
    if(!resp.ok)throw new Error("offer rejected");
    await pc.setRemoteDescription(await resp.json());
  }catch(e){toast("Call failed");endCall(true)}
}
function endCall(dropped){
  if(callStream){callStream.getTracks().forEach(t=>t.stop());callStream=null}
  const p=pc;pc=null;
  if(p){try{p.close()}catch(e){}}
  fetch("/api/rtc/hangup",{method:"POST"}).catch(()=>{});
  rtcAudio.srcObject=null;
  callBtn.textContent="📞 Start call";callBtn.classList.remove("on");
  if(dropped!==false)toast(dropped?"Call dropped":"Call ended");
}
callBtn.onclick=()=>{pc?endCall(false):startCall()};
</script></body></html>
"""
