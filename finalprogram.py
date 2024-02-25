import time
import tensorflow as tf
import numpy as np
from PIL import Image
from src import motor as motor_module
from src import vehicle as vehicle_module
from src import led as led_module
from src import camera as camera_module
import cv2

from src import led as led_module
def preprocess_image(image):
    # Resize the image to match the model's input shape
    resized_image = image.resize((150, 150))
    # Convert the image to a numpy array
    resized_image = np.array(resized_image)
    # Check if the image has 4 channels (RGBA) and convert it to RGB if needed
    if resized_image.shape[-1] == 4:
        resized_image = resized_image[:,:,:3]  # Keep only the first 3 channels (RGB)
    # Normalize pixel values
    resized_image = resized_image / 255.0
    # Add batch dimension
    input_image = np.expand_dims(resized_image, axis=0)
    return input_image

if __name__ == '__main__':
    total_seconds = 15
    sample_hz = 10
    cycle_time = 3
    vehicle = vehicle_module.Vehicle(
        {
            "motors": {
                "left": {
                    "pins": {
                        "speed": 13,
                        "control1": 5,
                        "control2": 6
                    }
                },
                "right": {
                    "pins": {
                        "speed": 12,
                        "control1": 7,
                        "control2": 8
                    }
                }
            }
        }
    )
    led1 = led_module.LED({
        "pin": 20
    })

    led2 = led_module.LED({
        "pin": 21
    })
    # Initialize camera module
    camera = camera_module.Camera({
        "show_preview": False
    })
    start_time = time.time()
    
    # Load the TensorFlow model
    model = tf.keras.models.load_model('/home/pi/HackIllinois2024/code/hackillinoisbug/Dataset/saved_models/Model01.tf')
    model.summary()
    insect_names =  {'1':"Butterfly",'2':"Dragonfly",
               '3':"Grasshopper",'4':"Ladybird",
               '5':"Mosquito"}
    o = 0
    while time.time() - start_time < total_seconds:
        vehicle.drive_forward()
        # Capture an image
        camera.capture()
        # Preprocess the captured image
        input_image = preprocess_image(Image.fromarray(camera.image_array))
        
        # Make prediction
        predictions = model.predict(input_image)
        #print(insect_names[str(predictions[0]+1)])
        print(predictions)
        # Adjust the delay to maintain the desired sampling frequency
        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1/sample_hz - elapsed_time))
        m=0
        
        for x in predictions:
            for i in range(len(x)):
             if (x[i]>m):
                m = x[i]
                o = i
        if (o == 2) :
            print("bad")
            vehicle.drive_forward(1)
            time.sleep(0.5)
            vehicle.stop()
            led2.on()
            time.sleep(1)
            led2.off()
            vehicle.drive(0.5, True, 1, True)
            time.sleep(1)
            vehicle.drive_forward(1)
            time.sleep(0.5)
            vehicle.drive(1, True, 0.5, True)
            time.sleep(1.5)
            vehicle.drive_forward(1)
            time.sleep(0.5)
            vehicle.drive(0.5, True, 1, True)
            time.sleep(0.4)
            
    vehicle.stop()    
    print(o)
                                 
