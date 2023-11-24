import cv2
import numpy as np

class LineTracker:
    def __init__(self):
        self._delta = 0.0
        self._delta2 = 0.0

    def process(self, img: np.ndarray) -> None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        h, w, d = img.shape
        search_top = int(2 * h / 4)
        search_bot = int(2 * h / 4 + 20)

        mask[0:h, 0:int(2 * w / 4)] = 0
        mask[0:h, int(2 * w / 4 + 25):w] = 0
        mask[int(h / 2 + 20):h, int(w / 2):int(w / 2 + 25)] = 0
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1)
            err = cy - h / 2
            self._delta = err
        cv2.imshow("window", img)
        cv2.imshow("mask", mask)
        cv2.waitKey(3)

    def process2(self, img: np.ndarray) -> None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white2 = np.array([0, 0, 200])
        upper_white2 = np.array([180, 30, 255])

        mask2 = cv2.inRange(hsv, lower_white2, upper_white2)

        h, w, d = img.shape
        search_top = int(2 * h / 4)
        search_bot = int(2 * h / 4 + 20)

        mask2[0:h, 0:int(w / 2):w] = 0
        mask2[0:h, int(3 * w / 4):w] = 0
        mask2[0:h, int(w / 2 + 25):int(3 * w / 4 - 25)] = 0
        mask2[int(h / 2 + 20):h, 0:w] = 0

        M = cv2.moments(mask2)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 20, (0, 0, 255), -1)
            err = cx - w / 2
            self._delta = err

        cv2.imshow("window2", img)
        cv2.imshow("mask2", mask2)
        cv2.waitKey(3)

    @property
    def delta(self):
        return self._delta


def main():
    tracker = LineTracker()
    import time
    for i in range(100):
        img = cv2.imread('/home/ros2/Ros2Projects/oom_ws/src/py_follower/worlds/sample.png')
        tracker.process(img)
        tracker.process2(img)
        time.sleep(0.1)


if __name__ == "__main__":
    main()