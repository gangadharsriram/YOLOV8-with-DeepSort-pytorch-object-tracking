
import os
import cv2
import time


import torch
from ultralytics import YOLO

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


main_path=os.getcwd()
all_files=os.path.join(main_path,"input_video")
path=os.listdir(path=all_files)[0]
save_path=os.path.join(main_path,"output")
out=os.path.splitext(path)[0]+"_output"+os.path.splitext(path)[1]
output_file=os.path.join(save_path,out)
paths=os.path.join(all_files,path)


cap = cv2.VideoCapture(paths)
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

#Importing Deepsort Config files #
cfg_deep = get_config()
cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")


deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)


video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


#YOLO V8 Nan "Model Will Be Downloader Automatically If Not Available" #

model = YOLO("yolov8n.pt")
detection_threshold = 0.5

def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    #draw the bounding box
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        obj_name = names[object_id[i]]
        id = int(identities[i]) if identities is not None else 0
        
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)
        cv2.putText(img, str(label), (x1, y1-5),fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 0.5,
                    color = (255, 0, 0),
                    thickness = 1,lineType=cv2.LINE_AA )
        cv2.rectangle(img,(x1,y1), (x2,y2),(0,0,255), 5,lineType=cv2.LINE_AA)
        
def xyxy_to_xywh(x1, y1, x2, y2):
    """ Calculates the relative bounding box from absolute pixel values """
    bbox_left = min([x1, x2])
    bbox_top = min([y1,y2])
    bbox_w = abs(x1 -x2)
    bbox_h = abs(y1 - y2)
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


while True:
    ret, frame = cap.read()
    start=time.time()
    results = model(frame)
    
    xyxy=[]
    scr=[]
    ids=[]
    for result in results:        
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)            
            y2 = int(y2)
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(x1, y1, x2, y2)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    
            class_id = int(class_id)
            if score > detection_threshold: 
                xyxy.append(xywh_obj)
                scr.append(score)
                ids.append(class_id)
            
    xyxy_box=torch.tensor(xyxy)
    scrs=torch.tensor(scr)
    
    outputs = deepsort.update(xyxy_box, scrs, ids,frame)
    
    if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(frame, bbox_xyxy, model.names, object_id,identities)
            end=time.time()        
            fp=end-start
            fps=str(round(1/fp))
            fps="FPS : "+fps
            cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0,255 ,255), 2)

            
    cv2.imshow("Output",frame)
    
    video_writer.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
       
cv2.destroyAllWindows()
