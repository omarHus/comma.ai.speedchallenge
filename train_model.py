import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Machine Learning libraries
from sklearn.model_selection import train_test_split
import tensorflow as tf

TEST_SIZE = 0.2
EPOCHS = 10
IMG_WIDTH = 480
IMG_HEIGHT = 640

def main():

    # Training file paths
    filename       = os.path.join('data', 'train.mp4')
    label_filename = os.path.join('data', 'train.txt')

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
    history = model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    model_file = os.path.join('saved_models', 'optical_flow')
    model.save(model_file)
    print(f"Model saved to {model_file}.")

    # Plot history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # Plot history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

def load_data(video_filename, label_filename):

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

        # Get car speed from labels
        v1 = next(speeds)
        v2 = next(speeds)

        # Calculate Mean speed
        mean_speed = 0.5*(v1 + v2)

        # Combine all data for the model
        images.append(img)
        labels.append(mean_speed)

        # show the frame and update the FPS counter
        # cv2.imshow("background sub frame", img)
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     break
    
    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("Loading Frames Done!")

    return (images, labels)



def background_sub(frame1, frame2):
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
    
    
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    # Grayscale images
    gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Calculate dense optical Flow
    flow = cv2.calcOpticalFlowFarneback(gray1,gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Change to polar coordinates
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # Color code pixels
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return bgr



def load_labels(filename):
    for row in open(filename, 'r'):
        yield float(row)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Need a sequential model
    model = tf.keras.models.Sequential([

        # Input layer with 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Max pooling layer 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # 2nd convolution layer with 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

        # 2nd Max pooling layer 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add hidden layer with dropout (avoids overfitting)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # do not put activation at the end because we want to have exact output, not a class identifier
        tf.keras.layers.Dense(1, name = 'output', kernel_initializer = 'he_normal'),

    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss="mse",
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    main()