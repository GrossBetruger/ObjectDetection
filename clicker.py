import json
import pyautogui as ag
import time
import sys

from random import randint


def move_around(point):
    for _ in range(7):
        point = (point[0]+randint(-5, 5), point[1]+randint(-5, 5))
        ag.moveTo(*point)
        time.sleep(.3)

def detection_parser(detection_output):
    parsed1 = detection_output.split("\n")
    parsed2 = [json.loads(x) for x in parsed1 if x]
    return parsed2


if __name__ == "__main__":
    detection_result = open(sys.argv[1]).read()
    for _ in detection_parser(detection_result):
        print move_around(_["centroid"])
