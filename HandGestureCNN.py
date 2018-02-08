from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from matplotlib import pyplot as plt


''' Label Output is 4. It's meaning in this case, be able to  predict
    4 types of Hand Gesture.
    Ex: FREINDLINESS | PUNCH | HASMONEY | KNIFE
    After change this, note that you must also adjust Labels again.
'''
im_predict_classes = 4
# Epochs
im_epoches = 15  # 25
# Convolutional filters will be used
im_filters = 32
# Max pooling
im_max_pooling = 2
# Size of convolution kernel
im_kernel_convo = 3
# Number of Image channel. Here channel of GrayScale is "1".
im_channels = 1
# Batch size for Training.
batch_size = 32

# Image Dimension
im_row, im_col = 200, 200


# Directory what contain collected image and will be fed to training model.
data_set_path = "./data_set"

# Weight File Name
weight_file_name = ["hand_crime_prevention.hdf5", "hi_punch_weights.hdf5"]

# Label Output
output = ["FRIENDLINESS", "HASMONEY", "PUNCH", "KNIFE"]

# %%
def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        # This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist


# Load CNN model
def load_cnn(wf_index):
    global get_output
    model = Sequential()

    model.add(Conv2D(im_filters, (im_kernel_convo, im_kernel_convo),
                     padding='valid',
                     input_shape=(im_channels, im_row, im_col)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(im_filters, (im_kernel_convo, im_kernel_convo)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(im_max_pooling, im_max_pooling)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(im_predict_classes))
    model.add(Activation('softmax'))
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Model summary
    model.summary()
    # Model conig details
    model.get_config()


    if wf_index >= 0:
        # Load pretrained weights
        fname = weight_file_name[int(wf_index)]
        print("Loading file ->>>", fname)
        model.load_weights(fname)

    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model


# This function does the guessing work based on input images
def predict(model, img):
    global output, get_output
    # Load image and flatten it
    image = np.array(img).flatten()

    # reshape it
    image = image.reshape(im_channels, im_row, im_col)

    # float32
    image = image.astype('float32')

    # normalize it
    image = image / 255

    # reshape for NN
    rimage = image.reshape(1, im_channels, im_row, im_col)

    # Now feed it to the NN, to fetch the predictions
    # index = model.predict_classes(rimage)
    # prob_array = model.predict_proba(rimage)

    prob_array = get_output([rimage, 0])[0]

    # print prob_array

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    # Get the output with maximum probability
    import operator

    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob = d[guess]

    if prob > 70.0:
        print("Prediction: " + guess + "  Probability: ", prob)

        return guess

    else:
        return "SOMETHING"


# %%
def initializers():
    imlist = modlistdir(data_set_path)

    image1 = np.array(Image.open(data_set_path + '/' + imlist[0]))  # open one image to get size
    # plt.imshow(im1)

    m, n = image1.shape[0:2]  # get the size of the images
    total_images = len(imlist)  # get the 'total' number of images

    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(data_set_path + '/' + images).convert('L')).flatten()
                         for images in imlist], dtype='f')

    print(immatrix.shape)

    input("Press any key")

    #########################################################
    ## Label the set of images per respective gesture type.
    ##
    label = np.ones((total_images,), dtype=int)

    samples_per_class = total_images / im_predict_classes
    print("samples_per_class - ", samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(im_predict_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class

    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''

    data, Label = shuffle(immatrix, label, random_state=2)
    train_data = [data, Label]

    (X, y) = (train_data[0], train_data[1])

    # Split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    X_train = X_train.reshape(X_train.shape[0], im_channels, im_row, im_col)
    X_test = X_test.reshape(X_test.shape[0], im_channels, im_row, im_col)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, im_predict_classes)
    Y_test = np_utils.to_categorical(y_test, im_predict_classes)
    return X_train, X_test, Y_train, Y_test


def trainModel(model):
    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=im_epoches,
                     verbose=1, validation_split=0.2)

    visualizeHis(hist)

    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = input("Enter file name - ")
        fname = "./" + str(filename) + ".hdf5"
        model.save_weights(fname, overwrite=True)
    else:
        model.save_weights("newWeight.hdf5", overwrite=True)

    # Save model as well
    # model.save("newModel.hdf5")


# %%

def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(im_epoches)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    # plt.style.use(['classic'])

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)

    plt.show()


# %%
def visualizeLayers(model, img, layerIndex):
    imlist = modlistdir('./imgs')
    if img <= len(imlist):

        image = np.array(Image.open('./imgs/' + imlist[img - 1]).convert('L')).flatten()

        ## Predict
        predict(model, image)

        # reshape it
        image = image.reshape(im_channels, im_row, im_col)

        # float32
        image = image.astype('float32')

        # normalize it
        image = image / 255

        # reshape for NN
        input_image = image.reshape(1, im_channels, im_row, im_col)
    else:
        X_train, X_test, Y_train, Y_test = initializers()

        # the input image
        input_image = X_test[:img + 1]

    # visualizing intermediate layers
    # output_layer = model.layers[layerIndex].output
    # output_fn = theano.function([model.layers[0].input], output_layer)
    # output_image = output_fn(input_image)

    if layerIndex >= 1:
        visualizeLayer(model, img, input_image, layerIndex)
    else:
        tlayers = len(model.layers[:])
        print("Total layers - {}".format(tlayers))
        for i in range(1, tlayers):
            visualizeLayer(model, img, input_image, i)


# %%
def visualizeLayer(model, img, input_image, layerIndex):
    layer = model.layers[layerIndex]

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
    activations = get_activations([input_image, 0])[0]
    output_image = activations

    ## If 4 dimensional then take the last dimension value as it would be no of filters
    if output_image.ndim == 4:
        # Rearrange dimension so we can plot the result
        o1 = np.rollaxis(output_image, 3, 1)
        output_image = np.rollaxis(o1, 3, 1)

        print("Dumping filter data of layer{} - {}".format(layerIndex, layer.__class__.__name__))
        filters = len(output_image[0, 0, 0, :])

        fig = plt.figure(figsize=(8, 8))
        # This loop will plot the 32 filter data for the input image
        for i in range(filters):
            ax = fig.add_subplot(6, 6, i + 1)
            # ax.imshow(output_image[img,:,:,i],interpolation='none' ) #to see the first filter
            ax.imshow(output_image[0, :, :, i], 'gray')
            # ax.set_title("Feature map of layer#{} \ncalled '{}' \nof type {} ".format(layerIndex,
            #                layer.name,layer.__class__.__name__))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()
        # plt.show()
        fig.savefig("img_" + str(img) + "_layer" + str(layerIndex) + "_" + layer.__class__.__name__ + ".png")
        # plt.close(fig)
    else:
        print("Can't dump data of this layer{}- {}".format(layerIndex, layer.__class__.__name__))


