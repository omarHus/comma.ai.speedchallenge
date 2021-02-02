import numpy as np
import cv2
import os
from train_model import background_sub
from train_model import calc_optical_flow
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def main():

    # Load data
    filename = os.path.join('data', 'test.mp4')
    data_for_predictions = load_data(filename)

    # Load model
    model_filepath = os.path.join('saved_models', 'optical_flow')
    model = load_model(model_filepath, compile=True)

    # Make Predictions
    predictions = model.predict(np.array(data_for_predictions))

    # Plot results
    # classes = np.array2string(predictions)
    # output_file = open("predictions.txt","w")
    # output_file.write(classes)
    # output_file.close()
    # Plot history for accuracy
    plt.plot(predictions)
    plt.title('Model Speed Predictions')
    plt.ylabel('Speed')
    plt.xlabel('frame')
    plt.show()


def load_data(video_filename):
    
    # Start streaming video file
    cap = cv2.VideoCapture(video_filename)


    images = []
    print("Loading frames")

    # loop over frames from the video file stream
    while (True):

        # Read two frames from the file
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        if frame1 is None:
            break

        # Background subtraction
        # mask = background_sub(frame1, frame2)
        img = calc_optical_flow(frame1, frame2)

        # Combine all data for the model
        images.append(img)
    
    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("Loading Frames Done!")

    return images


if __name__ == "__main__":
    main()