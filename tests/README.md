# Tests & self-checks

Hardware and pipeline diagnostics. Most need real hardware. **Always stop the
`zeus` service first** (`sudo systemctl stop zeus`) so the test is the sole owner
of the servos, camera, and mic.

> 🛡️ **Power safety:** the PiCrawler can brown out the Pi 5 when many servos move
> at once. `test_actions_safe.py` checks `vcgencmd` voltage/throttle before and
> after every move and **aborts** on a brownout. Run motion tests with the robot
> on a stand and a strong power supply.

## Movement / hardware

| File | Purpose |
|------|---------|
| [`test_actions_safe.py`](test_actions_safe.py) | Brownout-gated run of first-wake wave + every action-with-response. Requires `~/.zeus_no_move` removed. The safest motion test to start with. |
| [`test_movements.py`](test_movements.py) | Walks every gait + preset action with labels and pauses — quick check after a hardware change. |
| [`test_face_track.py`](test_face_track.py) | Camera + face-tracking servo loop. |
| [`test_yaw.py`](test_yaw.py) | Minimal yaw-servo sanity check. |

## Audio / STT / LLM pipeline (no human needed)

| File | Purpose |
|------|---------|
| [`audio_selftest.py`](audio_selftest.py) | Measures ambient noise floor, then acoustic loopback: plays a known phrase, records it, transcribes it. |
| [`cmd_capture_test.py`](cmd_capture_test.py) | Validates the VAD command-capture + transcription path without a wake word. |
| [`test_vosk_piper.py`](test_vosk_piper.py) | Exercises Vosk STT and Piper TTS together. |
| [`self_test_drive.py`](self_test_drive.py) | End-to-end: imports the live `zeus` module and feeds commands straight into `handle_command()` — full pipeline, no human. |
| [`test_introspect.py`](test_introspect.py) | Validates the model's self-aware answers against the augmented system prompt (off-robot). |
| [`test_ollama_text.py`](test_ollama_text.py) | One-shot Ollama text sanity check. |

## Typical flow

```bash
sudo systemctl stop zeus
touch ~/.zeus_no_move           # optional: disable motion while testing audio/LLM
python3 tests/audio_selftest.py
rm -f ~/.zeus_no_move           # re-enable motion
sudo python3 tests/test_actions_safe.py
```
