from picrawler import Picrawler
from vilib import Vilib
from time import sleep
import time

c = Picrawler()
c.do_action('stand', speed=60)
sleep(1)

Vilib.camera_start(vflip=False, hflip=False)
Vilib.display(local=False, web=True)
Vilib.face_detect_switch(True)
sleep(2)

print('Face tracking active!')
print('Press Ctrl+C to stop.')

last_face_time = time.time()
searching = False
search_dir_h = 1
search_h = 0
search_phase = 0
pitch_t = 0.0
pitch_target = None

# Current smooth values (for easing)
current_yaw = 0.0
current_pitch_t = 0.0

# Z positions per leg: [right_front, left_front, left_rear, right_rear]
STAND_Z =     [-50, -50, -50, -50]
LOOK_UP_Z =   [-76, -76, -38, -30]
LOOK_DOWN_Z = [-28, -40, -68, -76]

SMOOTHING = 0.2  # easing factor (0=no move, 1=instant)
SEARCH_SMOOTHING = 0.15

def lerp(a, b, t):
    return a + (b - a) * t

def get_pitch_z(target_z, t):
    return [lerp(STAND_Z[i], target_z[i], t) for i in range(4)]

def get_combined_pitch_z(up_t, down_t):
    """Blend stand toward up or down based on signed pitch."""
    z = STAND_Z[:]
    if up_t > 0:
        z = [lerp(STAND_Z[i], LOOK_UP_Z[i], up_t) for i in range(4)]
    elif down_t > 0:
        z = [lerp(STAND_Z[i], LOOK_DOWN_Z[i], down_t) for i in range(4)]
    return z

def apply_pose(yaw, pitch_z):
    step = [
        [45 - yaw*0.3, 45 + yaw, pitch_z[0]],
        [45 + yaw*0.3, 0 - yaw,  pitch_z[1]],
        [45 - yaw*0.3, 0 + yaw,  pitch_z[2]],
        [45 + yaw*0.3, 45 - yaw, pitch_z[3]],
    ]
    c.do_step(step, speed=100)

try:
    while True:
        fn = Vilib.detect_obj_parameter.get('human_n', 0)

        if fn > 0:
            fx = Vilib.detect_obj_parameter.get('human_x', 320)
            fy = Vilib.detect_obj_parameter.get('human_y', 240)

            # Target yaw from X: face left of frame -> rotate left
            target_yaw = (fx - 320) / 320 * 20

            # Target pitch from Y: face above center -> look up, below -> look down
            # Y=0 is top of frame, Y=480 is bottom
            # Center = 240, above center = look up (negative pitch_t), below = look down
            y_error = (fy - 240) / 240  # -1 (top) to +1 (bottom)
            # pitch_t: negative = up, positive = down, range roughly -1 to 1
            target_pitch = y_error * 0.8  # scale down a bit

            # Smooth easing
            current_yaw = lerp(current_yaw, target_yaw, SMOOTHING)
            current_pitch_t = lerp(current_pitch_t, target_pitch, SMOOTHING)

            # Convert pitch_t to Z coords
            if current_pitch_t < 0:
                # Looking up
                pitch_z = get_pitch_z(LOOK_UP_Z, min(abs(current_pitch_t), 1.0))
            else:
                # Looking down
                pitch_z = get_pitch_z(LOOK_DOWN_Z, min(current_pitch_t, 1.0))

            apply_pose(current_yaw, pitch_z)

            last_face_time = time.time()
            searching = False
            search_h = 0
            search_phase = 0
            print(f'Track: x={fx:.0f} y={fy:.0f} yaw={current_yaw:.1f} pitch={current_pitch_t:.2f}')

        else:
            elapsed = time.time() - last_face_time

            if elapsed > 3:
                if not searching:
                    print('No face for 3s... searching!')
                    searching = True
                    search_phase = 0
                    search_h = 0
                    search_dir_h = 1
                    pitch_t = 0.0
                    pitch_target = None

                # Smooth horizontal sweep
                if search_phase == 0:
                    target_h = search_h + search_dir_h * 2
                    if target_h > 20:
                        search_dir_h = -1
                    elif target_h < -20:
                        search_dir_h = 1
                        search_phase = 1
                    search_h = target_h
                    pitch_target = None
                    # Ease pitch back to neutral
                    pitch_t = lerp(pitch_t, 0, SEARCH_SMOOTHING)
                    pitch_z = STAND_Z[:] if abs(pitch_t) < 0.01 else get_pitch_z(
                        LOOK_UP_Z if pitch_t < 0 else LOOK_DOWN_Z, abs(pitch_t))

                elif search_phase == 1:
                    # Gradually tilt up while sweeping
                    target_h = search_h + search_dir_h * 2
                    if target_h > 20:
                        search_dir_h = -1
                    elif target_h < -20:
                        search_dir_h = 1
                        search_phase = 2
                    search_h = target_h
                    pitch_target = 'up'
                    pitch_t = lerp(pitch_t, -1.0, 0.03)
                    pitch_z = get_pitch_z(LOOK_UP_Z, min(abs(pitch_t), 1.0))

                elif search_phase == 2:
                    # Gradually tilt down while sweeping
                    target_h = search_h + search_dir_h * 2
                    if target_h > 20:
                        search_dir_h = -1
                    elif target_h < -20:
                        search_dir_h = 1
                        search_phase = 3
                    search_h = target_h
                    pitch_target = 'down'
                    pitch_t = lerp(pitch_t, 1.0, 0.03)
                    if pitch_t < 0:
                        pitch_z = get_pitch_z(LOOK_UP_Z, abs(pitch_t))
                    else:
                        pitch_z = get_pitch_z(LOOK_DOWN_Z, pitch_t)

                elif search_phase >= 3:
                    # Return to neutral
                    pitch_t = lerp(pitch_t, 0, SEARCH_SMOOTHING)
                    search_h = lerp(search_h, 0, SEARCH_SMOOTHING)
                    if abs(pitch_t) < 0.05 and abs(search_h) < 1:
                        pitch_t = 0
                        search_h = 0
                        search_phase = 0
                        pitch_z = STAND_Z[:]
                        print('Search cycle done, restarting...')
                    else:
                        if pitch_t < 0:
                            pitch_z = get_pitch_z(LOOK_UP_Z, abs(pitch_t))
                        else:
                            pitch_z = get_pitch_z(LOOK_DOWN_Z, pitch_t)

                # Smooth the horizontal search movement too
                current_yaw = lerp(current_yaw, search_h, SEARCH_SMOOTHING)
                current_pitch_t = pitch_t
                apply_pose(current_yaw, pitch_z)
                print(f'Search h={current_yaw:.1f} phase={search_phase} pitch={pitch_t:.2f}')

            else:
                # Waiting period - hold last pose
                print(f'No face ({elapsed:.1f}s)')

        sleep(0.08)

except KeyboardInterrupt:
    print('Stopping...')
    c.do_action('sit', speed=60)
    Vilib.camera_close()
