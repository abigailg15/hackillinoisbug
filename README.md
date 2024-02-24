- the robot will move forward for 1 minute
- every time it detects a bug, it will stop
- if the bug is a pest/bad : LED yellow
- the car will move straight over the bug
- if the bug is good : LED green
- the car will move to the right and then to the left (move around the bug)

- it will take constant pictures and constantly compare to the dataset
- as soon as it detects something it will stop

- push dataset on github
- clone repo to rasp pi
- train dataset using opencv

- we will have a numerical value corresponding to good/bad of a bug
- use tensorflow to train the model

frozen_inference_graph.pb: This file typically contains the frozen graph of a trained TensorFlow model. A frozen graph is a TensorFlow graph where the variables are converted into constants, resulting in a single file that encapsulates the model's architecture and trained weights. It's optimized for inference and can be used directly for deployment without requiring the original model definition or training code.

graph.pbtxt: This file is a text representation of the TensorFlow graph defined in the "frozen_inference_graph.pb" file. It provides a human-readable representation of the graph's structure, including information about the operations and their connections. While the binary "frozen_inference_graph.pb" file is used for efficient inference, the "graph.pbtxt" file can be helpful for understanding the model's architecture and debugging.

These files are typically generated during the training process of an object detection model using TensorFlow's Object Detection API or similar frameworks. Once the training is complete, the frozen graph is exported along with its text representation for later use during inference. When performing object detection inference, you load the frozen graph using TensorFlow's API, and it allows you to perform predictions on new images or video frames.





