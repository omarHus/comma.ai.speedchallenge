import cv2
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split

TEST_SIZE  = 0.2
EPOCHS     = 10
IMG_WIDTH  = 200
IMG_HEIGHT = 66

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python train_model.py model_directory")

    # Training file paths
    saved_model_dir = sys.argv[1]
    filename        = os.path.join('data', 'train.mp4')
    label_filename  = os.path.join('data', 'train.txt')

    # Load data for training
    images, labels = load_data(filename, label_filename)

    # Split training and testing data and shuffle
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, shuffle=True
    )

    # Train model and make predictions
    print("Training Model")
    
    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS, steps_per_epoch=400)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    model_file = os.path.join('saved_models', saved_model_dir)
    model.save(model_file)
    print(f"Model saved to {model_file}.")

    # Plot history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

def load_data(video_filename, label_filename):
    """
    Load frames from the video file and load labels from the label file.
    
    Combine successive frames into frame-pairs using either Background Sub
    or Optical Flow

    Returns a list of the processed frame-pairs and a list of the average 
    label values for each pair.
    """

    # Start streaming video file
    cap = cv2.VideoCapture(video_filename)

    # Generator for labels list
    speeds = load_labels(label_filename)

    images = []
    labels   = []

    print("Loading frames")
    # loop over frames from the video file stream
    while (True):

        # Read two frames from the file
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        if frame1 is None:
            break

        # # Background subtraction
        # img = background_sub(frame1, frame2)

        # Optical Flow
        img = calc_optical_flow(frame1, frame2)

        # Resize img for neural network
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)

        # Get car speed from labels
        v1 = next(speeds)
        v2 = next(speeds)

        # Calculate Mean speed
        mean_speed = 0.5*(v1 + v2)

        # Combine all data for the model
        images.append(img)
        labels.append(mean_speed)

        # show the frame and update the FPS counter
        # cv2.imshow("optical flow frame", img)
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     break

    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("Loading Frames Done!")

    return (images, labels)


def background_sub(frame1, frame2):
    """
    Perform background subtraction on two successive frames.
    Reshape the foreground mask img and return it
    """

    # Background Subtraction object
    backSub = cv2.createBackgroundSubtractorMOG2()

    #Create foreground mask using both frames
    mask = backSub.apply(frame1)
    mask = backSub.apply(frame2)

    # reshape mask
    img = np.zeros_like(frame1)
    img[:,:,0] = mask
    img[:,:,1] = mask
    img[:,:,2] = mask

    return img

def calc_optical_flow(frame1, frame2):
    """
    Calculate the Dense Optical Flow for successive frames
    Return the color-coded vector field
    """
    
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    # Grayscale images
    gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Calculate dense optical Flow
    flow = cv2.calcOpticalFlowFarneback(gray1,gray2, None, 0.5, 1, 15, 2, 5, 1.3, 0)

    # Change to polar coordinates
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # Color code pixels
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return bgr



def load_labels(filename):
    """
    Return Generator for labels so that we can get successive labels
    And calculate the mean speed of each frame-pair
    """

    for row in open(filename, 'r'):
        yield float(row)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    Base the CNN off of the NVIDIA architecture from 2017
    The output layer should have 1 unit - the speed of the car in the frame
    """
    # Define input shape of image
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    # Need a sequential model
    model = tf.keras.models.Sequential([

        # Add Normalization layer - avoids weights exceeding certain values
        tf.keras.layers.BatchNormalization(input_shape=input_shape),

        # Add convolution layer 1
        tf.keras.layers.Conv2D(24, (5, 5), activation='elu', strides=(2, 2)),

        # Add convolution layer 2
        tf.keras.layers.Conv2D(36, (5, 5), activation='elu', strides=(2, 2)),

        # Add convolution layer 3
        tf.keras.layers.Conv2D(48, (5, 5), activation='elu', strides=(2, 2)),

        # Add convolution layer 4 - non strided
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', strides=(1, 1)),

        # Add final convolution layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', strides=(1, 1)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Three fully connected layers
        tf.keras.layers.Dense(100, activation='elu'),
        tf.keras.layers.Dense(50, activation='elu'),
        tf.keras.layers.Dense(10, activation='elu'),

        # Final Output layer - 1 speed value
        tf.keras.layers.Dense(1, name='output')

    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss="mse",
    )

    return model

if __name__ == "__main__":
    main()