from imutils.video import VideoStream
from imutils import face_utils

from multiprocessing import Process
from playsound import playsound
from threading import Timer

import cv2
import dlib
import imutils
import logging
import numpy as np
import time


logging.basicConfig(format='%(asctime)s - %(message)s')
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

CLOSED_EYES_AR = 0.26
CLOSED_EYES_TIME = 3
CANCEL_DELAY = 3

faceEdgeColor = (0, 255, 0)
openEyesColor = (255, 255, 0)
closedEyesColor = (0, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontThickness = 1

alarm_proc = None


def start_alarm():
    ''' Start sound alarm player in a separate process '''
    global alarm_proc
    if alarm_proc:
        return
    log.info("Starting sound alarm process...")
    alarm_proc = Process(target=playsound, args=('alarm.wav',))
    alarm_proc.start()


def stop_alarm():
    ''' Delayed stop sound alarm player '''
    global alarm_proc
    if alarm_proc is None:
        return
    log.info("Terminating sound alarm process in %d seconds...", CANCEL_DELAY)

    timer = Timer(CANCEL_DELAY, alarm_proc.terminate)
    timer.start()
    alarm_proc = None


def eye_aspect_ratio(eye):
    ''' Calculate eye aspect ratio
        by dividing vertical to horizontal
        extension
    '''
    x = [item[1] for item in eye]
    y = [item[0] for item in eye]
    aspect_ratio = (max(x) - min(x))/(max(y) - min(y))
    return aspect_ratio


def draw_text(frame, leftAR, rightAR):
    ''' Draw eyes aspect ratio text '''
    leftColor = openEyesColor if leftAR > CLOSED_EYES_AR else closedEyesColor
    rightColor = openEyesColor if rightAR > CLOSED_EYES_AR else closedEyesColor

    frame = cv2.putText(frame, 'Left Eye AR: %4.2f' % leftAR, (10, 20),
                        font, fontScale, leftColor,
                        fontThickness, cv2.LINE_AA)

    frame = cv2.putText(frame, 'Right Eye AR: %4.2f' % rightAR, (10, 40),
                        font, fontScale, rightColor,
                        fontThickness, cv2.LINE_AA)

    return frame


def main(args):
    ''' Detect closed eyes using custom shape predictor
        and play a sound alarm if eyes were closed
        long enough
    '''
    log.info("Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()

    log.info("Loading eyes landmark predictor...")
    predictor = dlib.shape_predictor(args["shape_predictor"])

    log.info("Waiting for camera...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    start_time = None

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), faceEdgeColor, 2)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye, rightEye = shape[:6], shape[6:]

            leftEyeHull = cv2.convexHull(np.array(leftEye, dtype=np.int32))
            rightEyeHull = cv2.convexHull(np.array(rightEye, dtype=np.int32))

            cv2.drawContours(frame, [leftEyeHull], -1, openEyesColor, 1)
            cv2.drawContours(frame, [rightEyeHull], -1, openEyesColor, 1)

            leftAR = eye_aspect_ratio(leftEye)
            rightAR = eye_aspect_ratio(rightEye)

            if leftAR + rightAR < 2 * CLOSED_EYES_AR:
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                if elapsed_time > CLOSED_EYES_TIME:
                    start_alarm()
            else:
                stop_alarm()
                start_time = None

            frame = draw_text(frame, leftAR, rightAR)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--shape-predictor",
                        required=True,
                        help="path to landmark predictor")

    args = vars(parser.parse_args())
    main(args)
