import os
import cv2
from imutils.video import FPS
from imutils.video import FileVideoStream
from tqdm import tqdm

def main(dataRootPath, savePath):
    dataNameList = ['train', 'validation']
    for dataName in dataNameList:
        dataPath = dataRootPath + '/' + dataName
        SList = os.listdir(dataPath)
        for SName in SList:
            SPath = dataPath + '/' + SName
            CList = os.listdir(SPath)
            CList = [c for c in CList if c != 'map.png']
            for CName in CList:
                print("[INFO] Start to process {} {}".format(SName, CName))
                CPath = SPath + '/' + CName
                videoPathName = CPath + '/video.mp4'
                gtPathName = CPath + '/label.txt'
                cap = cv2.VideoCapture(videoPathName)
                fps = FPS().start()
                with open(gtPathName, 'r') as gtt:
                    for ind, line in enumerate(gtt):
                        words = line.split(',')
                        frameOrd = int(words[0]) - 1
                        IDName = int(words[1])
                        if frameOrd == 100:
                            break
                        print('\rFrame: {:010d} ID:{:04}'.format(frameOrd, IDName), end='')
                        BBox = [int(words[2]), int(words[3]), int(words[4]), int(words[5])]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frameOrd)
                        res, frame = cap.read()
                        savePathNew = savePath + '/' + dataName + '/{}_imgs/{}'.format(SName, IDName)
                        if not os.path.exists(savePathNew):
                            os.makedirs(savePathNew)
                        saveName = 'c0{}_{}_{}.jpg'.format(int(CName.replace('c', '')), IDName, frameOrd + 1)   # c[camID]_[ID]_[frame].jpg
                        cropImg = frame[BBox[1]:BBox[1] + BBox[3], BBox[0]:BBox[0] + BBox[2], :]
                        cv2.imwrite(savePathNew + '/' + saveName, cropImg)
                        fps.update()
                fps.stop()
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

                cap.release()

                print('\n{} {} done!'.format(SName, CName))
                return

if __name__ == '__main__':
    dataRootPath = '../data/AIC23_Track1_MTMC_Tracking'
    savePath = '../data/AIC23_REID_DATA'

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    main(dataRootPath, savePath)