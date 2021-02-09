import cv2, os, sys

def makeDir(key):
    if not os.path.exists(key):
        os.makedirs(key)

root = "./data/affwild"
ids = [int(i[0:3]) for i in os.listdir("{}/{}".format(root, "annotations/train/arousal"))]
makeDir("{}/frames".format(root))

for idx in ids:
    print(idx, end="\r")
    if idx > 200:
        makeDir("{}/frames/{}".format(root, idx))
        vidcap = cv2.VideoCapture('{}/videos/train/{}.mp4'.format(root, idx))
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("{}/frames/{}/{}.png".format(root, idx, count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame: {} {} {}'.format(idx, count, success), end="\r")
            count += 1