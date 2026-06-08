# Tests & self-checks

Hardware and pipeline diagnostics. Most of these need the real robot. Stop the
service first (`sudo systemctl stop zeus`) so the test is the only thing talking
to the servos, camera, and mic.

A warning that's worth repeating: the PiCrawler can brown out the Pi 5 when a
bunch of servos move at once. `test_actions_safe.py` checks the voltage and
throttle state before and after every move and stops if it sees a brownout. Run
motion tests with the robot on a stand and a power supply that means it.

## Movement / hardware

- **[`test_actions_safe.py`](test_actions_safe.py)** — brownout-gated run of the first-wake wave plus every action-with-response. Needs `~/.zeus_no_move` removed. Start here for motion.
- **[`test_movements.py`](test_movements.py)** — walks every gait and preset action with labels and pauses. Good after you've changed something physical.
- **[`test_face_track.py`](test_face_track.py)** — the camera + face-tracking head loop.
- **[`test_yaw.py`](test_yaw.py)** — minimal yaw-servo sanity check.

## Audio / STT / LLM (no human required)

- **[`audio_selftest.py`](audio_selftest.py)** — measures the ambient noise floor, then plays a known phrase, records it back, and transcribes it.
- **[`cmd_capture_test.py`](cmd_capture_test.py)** — checks the VAD command-capture and transcription path without needing a wake word.
- **[`test_vosk_piper.py`](test_vosk_piper.py)** — exercises Vosk STT and Piper TTS together.
- **[`self_test_drive.py`](self_test_drive.py)** — the end-to-end one. Imports the live `zeus` module and feeds commands straight into `handle_command()`. Full pipeline, nobody has to say anything.
- **[`test_introspect.py`](test_introspect.py)** — checks the model's self-aware answers against the augmented system prompt, off-robot.
- **[`test_ollama_text.py`](test_ollama_text.py)** — one-shot Ollama text check.

## A normal testing session

```bash
sudo systemctl stop zeus
touch ~/.zeus_no_move           # talk but don't move, while you work on audio/LLM
python3 tests/audio_selftest.py
rm -f ~/.zeus_no_move           # let it move again
sudo python3 tests/test_actions_safe.py
```
