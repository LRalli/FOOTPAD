import cv2
import sys

def func():
    videoFile = "Dataset/video.avi"
    videoCap = cv2.VideoCapture(videoFile)
    success, image = videoCap.read()
    length = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = 1  #<-- frame shot every x seconds
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    multiplier = fps * seconds
    animation = "|/-\\"
    j = 0
    print("")
    while success:
        for i in range(length):
            frameId = int(round(videoCap.get(1)))
            success, image = videoCap.read()

            if frameId % multiplier == 0:
                j += 1
                cv2.imwrite("Dataset/Frames/%d.jpg" % frameId, image)

            x = i % 4
            sys.stdout.write("{}analyzing {} [{} : {}] - {} images shotted{}"
                         .format("\r", animation[i % len(animation)], i, length, j, "." * x))
            sys.stdout.flush()

    videoCap.release()
    print(" End")

func()
