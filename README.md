# Machine-Learning-For-Sunda-Pangolin-Behavioural-Detection

# train

from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data="data_custom.yaml", imgsz = 640, batch = 8, epochs = 100, workers = 0, device = 0)

# predict video

from ultralytics import YOLO

model = YOLO("yolo.pt")

model.predict(source = "video.mp4", show=True, save=True, conf=0.7,
              line_width = 2, save_crop = False, save_txt = False, show_labels = True,
              show_conf = True, classes=[0,1,2,3,4,5,6,7,8])

# predict image

from ultralytics import YOLO

model = YOLO("yolo.pt")

model.predict(source = "1.png", show=True, save=True)
