from roboflow import Roboflow
import supervision as sv
import cv2
import Rockhold, route, privateVars
import matplotlib.pyplot as plt


holdList = []

rf = Roboflow(api_key=privateVars.api_key)
project = rf.workspace().project("rock-climbing")
model = project.version(52).model

result = model.predict("RocaWall1.jpg", confidence=40, overlap=30).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_inference(result)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoundingBoxAnnotator()

image = cv2.imread("RocaWall1.jpg")

for i in range(len(detections)):
    type = detections[i].data["class_name"]
    startX = detections[i].xyxy[0][0]
    startY = detections[i].xyxy[0][1]
    endX = detections[i].xyxy[0][2]
    endY = detections[i].xyxy[0][3]

    hold = Rockhold.Rockhold(type, startX, startY, endX, endY)
    holdList.append(hold)
    #  image =  hold.draw(image)

print(image.shape[0], image.shape[1])

route = route.Route(holdList, "v9", image.shape)
route.generateRoute()
image = route.draw(image)
# print(detections.data["class_name"])
# print(detections[0].xyxy[0])
# print("start", detections[0].xyxy[0][0])
# print("end", detections[0].xyxy[0][2])

# annotated_image = bounding_box_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections, labels=labels)

# sv.plot_image(image=annotated_image)

# cv2.imshow('image', image)
# cv2.waitKey(0)
cv2.imwrite("annotated_image.jpg", image)