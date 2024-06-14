from ultralytics import YOLO
import cv2

model = YOLO('yolov9c')

# results = model.predict("RocaWall1.jpg")

# image = results.render()
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
ws
# print(results)

results = model.train(data='data.yaml', epochs=10, imgsz=640)