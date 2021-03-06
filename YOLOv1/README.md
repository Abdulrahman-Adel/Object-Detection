## YOLOv1 

[YOLOv1](https://arxiv.org/abs/1506.02640v5) is a single-stage object detection model. Object detection is framed as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

<b>The architecture of the model</b>:
<p align="center">
  <img src="https://user-images.githubusercontent.com/57441828/112893809-70d24c80-90db-11eb-98ec-544a240ed95d.PNG" />
</p>
<b>The Loss function</b>:
<p align="center">
  <img src="https://user-images.githubusercontent.com/57441828/112893890-86e00d00-90db-11eb-84c0-e2867fddaaeb.PNG" />
</p>
<b>Results</b>
After train the model on 100 examples (due to limitations in processing power) for 100 epochs:


|![Figure 2021-03-30 015159](https://user-images.githubusercontent.com/57441828/112914252-ce769100-90fb-11eb-8cb6-502ff3718065.png)|![Figure 2021-03-30 015206](https://user-images.githubusercontent.com/57441828/112914291-e64e1500-90fb-11eb-8fdc-2a9172cb35bc.png)|![Figure 2021-03-30 015212](https://user-images.githubusercontent.com/57441828/112914306-f960e500-90fb-11eb-90c1-c55bcf8f1dfa.png)|![Figure 2021-03-30 015232](https://user-images.githubusercontent.com/57441828/112914323-0382e380-90fc-11eb-9584-6f4e8788c5ae.png)|![image5](https://user-images.githubusercontent.com/57441828/112916822-9d00c400-9101-11eb-869b-5a67e5df37d5.jpg)|![image2](https://user-images.githubusercontent.com/57441828/112916832-a38f3b80-9101-11eb-8eca-a8313e33ca36.jpg)

