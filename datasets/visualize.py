import cv2
import os

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255,  0), (0, 128,  0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139,  0,
                                                               139), (100, 149, 237), (138, 43, 226), (238, 130, 238),
             (255,  0, 255), (0, 100,  0), (127, 255,  0), (255,  0,
                                                            255), (0,  0, 205), (255, 140,  0), (255, 239, 213),
             (199, 21, 133), (124, 252,  0), (147, 112, 219), (106, 90,
                                                               205), (176, 196, 222), (65, 105, 225), (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199,
                                                               21, 133), (148,  0, 211), (255, 99, 71), (144, 238, 144),
             (255, 255,  0), (230, 230, 250), (0,  0, 255), (128, 128,
                                                             0), (189, 183, 107), (255, 255, 224), (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128,
                                                               128), (72, 209, 204), (139, 69, 19), (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135,
                                                               206, 235), (0, 191, 255), (176, 224, 230), (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139,
                                                                 139), (143, 188, 143), (255,  0,  0), (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42,
                                                              42), (178, 34, 34), (175, 238, 238), (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]

def draw_bboxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def visualize(vidPath, gtPath):
    gt = open(os.path.join(gtPath))
    gt_dict = dict()

    for line in gt.readlines():
        fid, pid, x, y, w, h, flag = [int(float(a)) for a in line.strip().split(',')][:7]
        if fid not in gt_dict:
            gt_dict[fid] = [(pid, x, y, w, h)]
        else:
            gt_dict[fid].append((pid, x, y, w, h))


    cap = cv2.VideoCapture(vidPath)
    frame_id = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            bbox_xyxy = []
            pids = []
            if frame_id in gt_dict:
                for pid, x, y, w, h in gt_dict[frame_id]:
                    bbox_xyxy.append([x, y, x+w, y+h])
                    pids.append(pid)

                frame = draw_bboxes(frame, bbox_xyxy, pids)
            
            cv2.putText(frame, "%d" % (frame_id), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 2)
            cv2.imshow('image',frame)

            frame_id += 1
            key = cv2.waitKey(500)
            if key & 255 == 27:  # ESC
                break
            if key & 255 == 32:  # SPACE
                cv2.waitKey(-1) #wait until any key is pressed
        else:
            break

if __name__ == '__main__':
    videoPath = "../../data/AIC23_Track1_MTMC_Tracking/train/S002/c009/video.mp4"
    gtPath = "../../data/AIC23_Track1_MTMC_Tracking/train/S002/c009/label.txt"
    visualize(videoPath, gtPath)