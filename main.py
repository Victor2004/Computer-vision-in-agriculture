from imageai.Detection import ObjectDetection
import os

filename = 5

# execution_path = os.getcwd()
detector = ObjectDetection()
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath(os.path.join(execution_path , "models/yolov3.pt"))
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(os.path.join(execution_path , "models/tiny-yolov3.pt"))
detector.setModelTypeAsRetinaNet()  # Самая крупная модель
detector.setModelPath("models/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=rf"images/{filename}.jpg", output_image_path=r"images/imagenew.jpg", minimum_percentage_probability=10)
# detections = detector.detectObjectsFromImage(input_image=r"image.jpg", output_image_path=r"imagenew.jpg", minimum_percentage_probability=50)
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=30)

animals = {}
for eachObject in detections:
    if eachObject["name"] in animals:
        animals[eachObject["name"]] += 1
    else:
        animals[eachObject["name"]] = 1
    print(eachObject["name"] + " - " + str(eachObject["percentage_probability"]), eachObject["box_points"])
print(animals)
