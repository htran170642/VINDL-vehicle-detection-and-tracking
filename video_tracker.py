import argparse
import numpy as np
import cv2
import warnings
from collections import deque
import time
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
import get_output_yolo

from core.config import cfg
from core.utils import read_class_names


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = r"city_street.mp4")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
# COLORS = np.random.randint(0, 255, size=(200, 3),
# 	dtype="uint8")
COLORS = {'car': (255,0,0), 'motorbike': (0,255,0), 'truck': (0,0,255), 'bus':(0,225,225)}
INPUT_SIZE = 608

## load model yolo
yolov4 = get_output_yolo.load_model_yolov4()

## read classes name
class_names = read_class_names(cfg.YOLO.CLASSES)

def main():
    video_capture = cv2.VideoCapture(args["input"])
    writeVideo_flag = True
    name_video = args["input"].split("/")[-1].split(".")[0] + '_1cos1'
    model_filename = 'data/market1501.pb'
    max_cosine_distance = 1
    nn_budget = None
    nms_max_overlap = 1
    counter = []
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    ## ghi ra video
    if writeVideo_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        out = cv2.VideoWriter(r'./output/{}.avi'.format(name_video), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))
        list_file = open('./output/{}_rslt.txt'.format(name_video), 'w')
        frame_index = -1

    fps = 0.0
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        t1 = time.time()
        boxs, confidence, class_idx = get_output_yolo.get_boxes(yolov4, frame, input_size=INPUT_SIZE)
        if len(boxs) == 0 or len(confidence) == 0 or len(class_idx) == 0:
            continue
        boxs_xywh = []
        for idx, (x1, y1, x2, y2) in enumerate(boxs):
            boxs_xywh.append((x1, y1, x2 - x1, y2 - y1))

        features = encoder(frame, boxs_xywh)
        # break
        detections = [Detection(bbox, conf, feature) for bbox, conf, feature in zip(boxs_xywh, confidence, features)]
        # print("========", detections)
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        class_idx = [class_idx[i] for i in indices]
        # print(class_idx)


        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        
        for det,n in zip(detections,class_idx):
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[class_names[n]], 2)
            cv2.putText(frame, class_names[n], (int(bbox[0]), int(bbox[3] + 20)), 0, 5e-3 * 150, (255, 255, 255), 2)
            
            # print(class_names)
            # print(class_names[p])

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # print("=====",track.track_id)
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            # color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            # print(frame_index)
            list_file.write(str(frame_index) + ',')
            list_file.write(str(track.track_id) + ',')
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            b0 = str(bbox[0])  # .split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1])  # .split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2] - bbox[0])  # .split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3] - bbox[1])

            list_file.write(str(b0) + ',' + str(b1) + ',' + str(b2) + ',' + str(b3))
            list_file.write('\n')
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (255,255,255), 2)

            i += 1
            # bbox_center_point(x,y)
            # center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # track_id[center]

            # pts[track.track_id].append(center)

            # thickness = 5
            # center point
            # cv2.circle(frame, (center), 1, color, thickness)

            # draw motion path
            # for j in range(1, len(pts[track.track_id])):
            #     if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
            #         continue
            #     thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            #     cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        # count = len(set(counter))

        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.namedWindow("YOLO4_Deep_SORT", 0)
        cv2.resizeWindow('YOLO4_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO4_Deep_SORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1. / (time.time() - t1))) / 2
        # out.write(frame)
        # frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    # if len(pts[track.track_id]) != None:
        # print(args["input"] + ": " + str(count) + ' vehicles' + ' Found')

    # else:
    #     print("[No Found]")
    # print("[INFO]: model_image_size = (960, 960)")
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()





if __name__ =="__main__":
    main()