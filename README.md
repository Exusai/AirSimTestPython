# Visual Navigation For Drones
The simulator used is AirSim from Microsoft, window identification was implmented on the Unreal version of AirSim and precision landing wa implemented in the Unity version.

## Precision Landing
This is achieved by a proportional visual controller that identifies the center of the marker and moves 
the drone in a parallel plane to the marker until it is centered enough, then it descends and this process is repeated until the dron is at a safa landing distance.

https://user-images.githubusercontent.com/47704357/159741744-82feedfa-6210-4662-b68b-878478764728.mp4

## Window Identification and Segmentation
The idea was to implment a neural network able to identify windows and then fly trough them, however due to time constraints only the identification of windows 
was implemented.

Data was collected using AirSim´s API, as shown bellow. The data and images provided by the simulaton were written to a directory in pascal_voc format.

https://user-images.githubusercontent.com/47704357/159741963-3b32b3c2-6f21-4ab3-abef-77314825307c.mp4

Thanks to the simulation, a lot of information was gathered with different climatic and illumination conditions. 
Using Google Colab, a version of [EfficientDet](https://arxiv.org/abs/1911.09070) was trained.

Simulated environment:
![image](https://user-images.githubusercontent.com/47704357/159744851-db05cf19-8470-4343-b929-a673b3eaafdc.png)

Data collected:
![image](https://user-images.githubusercontent.com/47704357/159744908-63239ef0-038b-48c6-ab7f-ed6c927e2eec.png)

Network test with dust storm:
![image](https://user-images.githubusercontent.com/47704357/159745289-a284dbf7-e376-456f-8656-552cbb95a5c4.png)

Network test in simulation:
![image](https://user-images.githubusercontent.com/47704357/159745254-0ad858dc-46ae-4ec5-991e-0bfdc5250152.png)

## Video

https://user-images.githubusercontent.com/47704357/159742056-8f13d125-e0a6-4c1d-b70c-002addbdfad8.mp4
