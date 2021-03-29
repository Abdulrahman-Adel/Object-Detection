# Object-Detection
Implementing Yolov1 and Yolov3 research papers on the PASCALVOC dataset

## PASCAL VOC Dataet

The [PASCAL Visual Object Classes](https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2) (VOC)  dataset contains 20 object categories including vehicles, household, animals, and other: aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, TV/monitor, bird, cat, cow, dog, horse, sheep, and person. Each image in this dataset has pixel-level segmentation annotations, bounding box annotations, and object class annotations. This dataset has been widely used as a benchmark for object detection, semantic segmentation, and classification tasks.

<p align="center">
  <img src="https://user-images.githubusercontent.com/57441828/112892764-0c62bd80-90da-11eb-92a7-44d8e03b41bc.PNG" />
</p>


## YOLOv1 

[YOLOv1](https://arxiv.org/abs/1506.02640v5) is a single-stage object detection model. Object detection is framed as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

The architechture of the model
<p align="center">
  <img src="https://user-images.githubusercontent.com/57441828/112893809-70d24c80-90db-11eb-98ec-544a240ed95d.PNG" />
</p>
The Loss function
<p align="center">
  <img src="https://user-images.githubusercontent.com/57441828/112893890-86e00d00-90db-11eb-84c0-e2867fddaaeb.PNG" />
</p>
 Results
