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

Collect and Annotate Your Dataset:
        
          Gather a dataset of images that contain the objects you want to detect.
          Annotate the images with bounding boxes indicating the location of each object of interest. Tools like LabelImg can help with this task.
          Prepare Your Dataset:
          
          Organize your dataset into a directory structure suitable for training.
          Convert the annotations into a format compatible with TensorFlow's Object Detection API, such as TFRecord files.
          Choose a Model Architecture:
          
          Select an object detection architecture that suits your needs. You may start with a simple architecture like Single Shot Multibox Detector (SSD) or You Only Look Once (YOLO) and then experiment with more complex models if needed.
          Configure the Training Pipeline:
          
          Define a configuration file specifying the model architecture, training parameters, and input data configuration.
          Adjust parameters such as learning rate, batch size, and number of training steps based on your dataset size and computational resources.
          Initialize the Model:
          
          Initialize the model with random weights.
          Training:
          
          Train the model on your annotated dataset using the TensorFlow Object Detection API.
          Monitor the training process and adjust hyperparameters as needed.
          Evaluation:
          
          Evaluate the trained model on a separate validation dataset to assess its performance.
          Compute evaluation metrics such as mean Average Precision (mAP) to quantify the model's accuracy.
          Export the Trained Model:
          
          Export the trained model checkpoint to a format suitable for inference, such as a frozen graph (.pb file) or a SavedModel.
          Inference:
          
          Use the exported trained model for inference on new images or video streams.

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

                
                import tensorflow as tf
                from tensorflow.keras import layers, models
                import numpy as np

                # Define a simple CNN model
                model = models.Sequential([
                    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.Flatten(),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(num_classes, activation='softmax')
                ])
                
                # Compile the model
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                
                # Train the model
                model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
                
                # Evaluate the model
                test_loss, test_acc = model.evaluate(test_images, test_labels)
                print('Test accuracy:', test_acc)
                
                Replace img_height, img_width, train_images, train_labels, val_images, val_labels, test_images, and test_labels with your actual data. Additionally, adjust the model architecture, hyperparameters, and training parameters as needed for your specific task and dataset.

 !wget https://cainvas-static.s3.amazonaws.com/media/user_data/cainvas-admin/archive_2.zip

 import tensorflow as tf
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

main_dir = 'archive/insects'
num_fldrs = 5

# dictionary of labels
insect_names = {'1':"Butterfly",'2':"Dragonfly",
               '3':"Grasshopper",'4':"Ladybird",
               '5':"Mosquito"}

def getdata(folder_path,num_subfolders):
    global insect_names
    insects = pd.DataFrame(columns=['image_abs_path','image_labels'])
    for label in range(1,num_subfolders+1):
        #print("processing for label: {}".format(label))
        label_i = folder_path+"/"+insect_names[str(label)]
        #read directory
        dirs_label_i =  os.listdir(label_i)
        idx = 0
        for image in dirs_label_i:
            #create a absolute image path
            insect_i = os.path.join(label_i,image)
            #print('Absolute path for image no. {} and label {}: {}'\
                  #.format(idx,label,flower_i))

            #fill the dataframe with path and label
            insects = insects.append({'image_abs_path':insect_i,
                            'image_labels':insect_names[str(label)]},
                           ignore_index=True)
            idx += 1
    return insects

                
                
