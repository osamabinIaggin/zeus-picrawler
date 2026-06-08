#!/usr/bin/env python3
"""Brownout-gated action test: confirms first-wake wave+Hey and actions-with-responses.

Run with zeus.service STOPPED and ~/.zeus_no_move REMOVED (so MOVEMENT_ENABLED=True).
Checks vcgencmd throttle/volts before and after every single move and ABORTS at the
first sign of under-voltage. Uses only low-current moves (no gaits/spins/fighting).
Importing zeus does NOT start continuous face-tracking (that's only in main()), so
this is the gentlest possible way to exercise movement.
"""
import sys
import time
import subprocess
import zeus


def throttle():
    out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                         text=True).stdout.strip()
    val = int(out.split("=")[1], 16) if "=" in out else -1
    return val, out


def volts():
    return subprocess.run(["vcgencmd", "measure_volts"], capture_output=True,
                         text=True).stdout.strip().replace("volt=", "")


def guard(label):
    """Return True if power is still clean; print + return False if brownout seen."""
    val, raw = throttle()
    v = volts()
    under_now = bool(val & 0x1)
    under_ever = bool(val & 0x10000)
    flag = "OK" if val == 0 else ("UNDER-VOLT NOW!" if under_now else
                                  ("brownout occurred" if under_ever else f"{raw}"))
    print(f"   [{label}] {raw} volts={v}  -> {flag}", flush=True)
    return val == 0


print(f"movement enabled = {zeus.MOVEMENT_ENABLED}", flush=True)
if not zeus.MOVEMENT_ENABLED:
    print("ABORT: no_move flag still present — movement is suppressed.", flush=True)
    sys.exit(2)

print("waiting for subsystems...", flush=True)
zeus._vosk_ready.wait(timeout=40)
zeus._camera_ready.wait(timeout=40)

if not guard("baseline"):
    print("ABORT: not clean at baseline.", flush=True)
    try:
        zeus.Vilib.camera_close()
    except Exception:
        pass
    sys.exit(1)


def finish():
    print("\n=== returning to sit + releasing camera ===", flush=True)
    try:
        zeus.crawler.do_action("sit", speed=50)
    except Exception as e:
        print("sit failed:", e, flush=True)
    try:
        zeus.Vilib.camera_close()
        print("camera released", flush=True)
    except Exception as e:
        print("camera_close failed:", e, flush=True)
    print("=== action test complete ===", flush=True)


# 1) First-wake: the wave that should accompany "Hey!"
print("\n########## FIRST-WAKE: wave_hand + 'Hey!'", flush=True)
zeus.action_in_progress.set()
try:
    zeus.wave_hand(zeus.crawler)
finally:
    zeus.action_in_progress.clear()
zeus.speak("Hey!")
if not guard("after first-wake wave"):
    print("!!! brownout after first-wake wave — aborting", flush=True)
    finish()
    sys.exit(1)

# 2) Actions-with-responses (low current only), gated each step
ACTIONS = [
    ("wave",       ["wave_hand"],  "Hey! How's it going?"),
    ("nod",        ["nod"],        "Yep, got it."),
    ("look left",  ["look_left"],  "Looking to my left."),
    ("look right", ["look_right"], "And now to my right."),
    ("shake head", ["shake_head"], "Nope, not happening."),
]

for label, acts, answer in ACTIONS:
    if not guard(f"before {label}"):
        print(f"!!! brownout before {label} — stopping", flush=True)
        break
    print(f"\n########## ACTION: {label} -> {acts}  | says: {answer!r}", flush=True)
    zeus.handle_response(acts, answer)
    if not guard(f"after {label}"):
        print(f"!!! brownout after {label} — stopping", flush=True)
        break
    time.sleep(1.2)

finish()
