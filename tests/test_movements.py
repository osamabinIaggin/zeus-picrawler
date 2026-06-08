#!/usr/bin/env python3
"""
Quick movement test for PiCrawler after a hardware change.
Walks through every gait + preset action with labels and pauses so you can
watch each one and spot a miscalibrated / binding / dead servo.

Usage:
  python3 test_movements.py            # run the full sequence
  python3 test_movements.py forward    # run just one item by name
  python3 test_movements.py --list     # list available items
"""
import sys
import time

sys.path.append("/home/pi/picrawler/examples")
from picrawler import Picrawler
from preset_actions import (  # type: ignore
    wave_hand, shake_hand, fighting, excited, play_dead, nod, shake_head,
    look_left, look_right, look_up, look_down, warm_up, push_up,
)

SPEED = 55          # gentle-ish; raise if you want more vigour
PAUSE = 1.3         # seconds between moves so you can observe

crawler = Picrawler()


def gait(name, steps=2, speed=SPEED):
    crawler.do_action(name, steps, speed)


# (label, callable) — ordered from safe/static to dynamic
TESTS = [
    ("stand",            lambda: crawler.do_action("stand", speed=SPEED)),
    ("sit",              lambda: crawler.do_action("sit", speed=SPEED)),
    ("stand (again)",    lambda: crawler.do_action("stand", speed=SPEED)),
    ("look_up",          lambda: look_up(crawler)),
    ("look_down",        lambda: look_down(crawler)),
    ("look_left",        lambda: look_left(crawler)),
    ("look_right",       lambda: look_right(crawler)),
    ("nod",              lambda: nod(crawler)),
    ("shake_head",       lambda: shake_head(crawler)),
    ("wave_hand",        lambda: wave_hand(crawler)),
    ("shake_hand",       lambda: shake_hand(crawler)),
    ("warm_up",          lambda: warm_up(crawler)),
    ("push_up",          lambda: push_up(crawler)),
    ("forward",          lambda: gait("forward")),
    ("backward",         lambda: gait("backward")),
    ("turn left",        lambda: gait("turn left")),
    ("turn right",       lambda: gait("turn right")),
    ("turn left angle",  lambda: gait("turn left angle")),
    ("turn right angle", lambda: gait("turn right angle")),
    ("excited",          lambda: excited(crawler)),
    ("fighting",         lambda: fighting(crawler)),
    ("play_dead",        lambda: play_dead(crawler)),
]

BY_NAME = {label: fn for label, fn in TESTS}


def run_one(label, fn):
    print(f"\n▶  {label} ...", flush=True)
    t0 = time.time()
    try:
        fn()
        print(f"   ✅ {label}  ({time.time()-t0:.1f}s)", flush=True)
        return True
    except Exception as e:
        print(f"   ❌ {label}  FAILED: {e}", flush=True)
        return False


def main():
    args = sys.argv[1:]
    if args and args[0] in ("--list", "-l"):
        print("Available:", ", ".join(BY_NAME))
        return

    print("Initializing PiCrawler...")
    time.sleep(1.0)
    crawler.do_action("stand", speed=SPEED)
    time.sleep(1.0)

    if args:
        label = " ".join(args)
        if label not in BY_NAME:
            print(f"Unknown: {label}\nAvailable: {', '.join(BY_NAME)}")
            return
        run_one(label, BY_NAME[label])
    else:
        results = []
        for label, fn in TESTS:
            results.append((label, run_one(label, fn)))
            time.sleep(PAUSE)
        passed = sum(1 for _, ok in results if ok)
        print(f"\n=== Summary: {passed}/{len(results)} passed ===")
        for label, ok in results:
            if not ok:
                print(f"   ❌ {label}")

    print("\nReturning to sit.")
    crawler.do_action("sit", speed=SPEED)


if __name__ == "__main__":
    main()
