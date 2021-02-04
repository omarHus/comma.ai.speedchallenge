import numpy as np
import cv2
import os
from train_model import background_sub
from train_model import calc_optical_flow
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train_model import IMG_HEIGHT, IMG_WIDTH


def main():

    # Load data
    filename = os.path.join('data', 'test.mp4')
    data_for_predictions = load_data(filename)

    # Load model
    model_filepath = os.path.join('saved_models', 'optical_flow')
    model = load_model(model_filepath, compile=True)

    # Make Predictions
    predictions = model.predict(np.array(data_for_predictions))

    # Separate frames
    frame_speeds = [elem[0] for elem in predictions for i in range(2)]

    # Plot results
    plt.plot(frame_speeds)
    plt.title('Model Speed Predictions')
    plt.ylabel('Speed')
    plt.xlabel('frames')
    plt.show()

    # Create output txt file with predictions
    output_file = open("predictions.txt", "w")
    for frame_speed in frame_speeds:
        output_file.write("%s\n" % str(frame_speed))
    output_file.close()



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

        # Resize img
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)

        # Combine all data for the model
        images.append(img)
    
    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("Loading Frames Done!")

    return images


if __name__ == "__main__":
    main()