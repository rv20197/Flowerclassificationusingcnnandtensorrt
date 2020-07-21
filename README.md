# Flowerclassificationusingcnnandtensorrt

What is TensorRT?
TensorRT is an optimization tool provided by NVIDIA that applies graph optimization and layer fusion, and finds the fastest implementation of a deep learning model. In other words, TensorRT will optimize our deep learning model so that we expect a faster inference time than the original model (before optimization), such as 5x faster or 2x faster. The bigger model we have, the bigger space for TensorRT to optimize the model. Furthermore, this TensorRT supports all NVIDIA GPU devices, such as 1080Ti, Titan XP for Desktop, and Jetson TX1, TX2 for embedded device.

LayMan’s Term
TensorRt helps us to execute and run our Tensorflow at high speed by compreesing the excessive nodes resulting in faster execution, results, lower waiting time. Helpful in situation where the speed and precision is important. Examples: (Automotive Vehicles, Embedded System,etc)

We have built a tensorflow CNN model which will predict the type of flower and it's an example how we can reduce the execution time of a tensorflow model usind TensorRt which 
results in an almost 3x to 5x times of improvement in the over execution time of the model which can be proved to be major improvement where the execution time of any model is 
highly important.
