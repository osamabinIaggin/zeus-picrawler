from picrawler import Picrawler
from time import sleep

c = Picrawler()
c.do_action('stand', speed=60)
sleep(1)

print('Testing extended yaw...')
for angle in range(0, 21, 2):
    step = [
        [45 - angle*0.3, 45 + angle, -50],
        [45 + angle*0.3, 0 - angle, -50],
        [45 - angle*0.3, 0 + angle, -50],
        [45 + angle*0.3, 45 - angle, -50],
    ]
    c.do_step(step, speed=100)
    sleep(0.08)

sleep(0.5)

for angle in range(20, -21, -2):
    step = [
        [45 - angle*0.3, 45 + angle, -50],
        [45 + angle*0.3, 0 - angle, -50],
        [45 - angle*0.3, 0 + angle, -50],
        [45 + angle*0.3, 45 - angle, -50],
    ]
    c.do_step(step, speed=100)
    sleep(0.08)

sleep(0.5)
c.do_step('stand', 90)
sleep(0.5)
c.do_action('sit', speed=60)
print('Done! Was the rotation bigger?')
