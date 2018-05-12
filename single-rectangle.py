# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import os


NB_EPOCH = 1


def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U


def vectorize_img(img_path, img_xsize, img_ysize):
    img = cv2.imread(img_path)
    hot_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_norm = cv2.resize(hot_grey, (img_xsize, img_ysize))
    return img_norm


if __name__ == "__main__":
    # quit()
    # Create images with random rectangles and bounding boxes.

    target_bbox = (34, 32, 15, 7)
    num_imgs = 500

    img_size = 10**2
    min_object_size = 1
    max_object_size = 4
    num_objects = 1

    bboxes = np.zeros((num_imgs, num_objects, 4))
    # for b in bboxes:
    #     print b[2]
    # quit()
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

    training_set_path = "hot_starts"
    paths = [os.path.join(training_set_path, img_path) for img_path in os.listdir(training_set_path)]
    hots = [vectorize_img(p, img_size, img_size) for p in paths]
    hot = vectorize_img("hot_starts/9731914136151-1173170842.jpg", img_size, img_size)
    imgs = np.array([hot] * num_imgs)

    for i_img in range(num_imgs):
        for i_object in range(num_objects):
            x, y, w, h = target_bbox
            # imgs[i_img, x:x + w, y:y + h] = 1.  # set rectangle to 1
            bboxes[i_img, i_object] = [x, y, w, h]

    imgs.shape, bboxes.shape

    # In[5]:

    i = 0
    plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    for bbox in bboxes[i]:
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))

    # Reshape and normalize the image data to mean 0 and std 1.
    X = (imgs.reshape(num_imgs, -1) - np.mean(imgs)) / np.std(imgs)
    X.shape, np.mean(X), np.std(X)

    # Normalize x, y, w, h by img_size, so that all values are between 0 and 1.
    # Important: Do not shift to negative values (e.g. by setting to mean 0), because the IOU calculation needs positive w and h.
    y = bboxes.reshape(num_imgs, -1) / img_size
    y.shape, np.mean(y), np.std(y)

    # Split training and test.
    i = int(0.8 * num_imgs)
    train_X = X[:i]
    test_X = X[i:]
    train_y = y[:i]
    test_y = y[i:]
    test_imgs = imgs[i:]
    test_bboxes = bboxes[i:]

    # Build the model.
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.optimizers import SGD

    print "building model..."
    model = Sequential([
        Dense(200, input_dim=X.shape[-1]),
        Activation('relu'),
        Dropout(0.2),
        Dense(y.shape[-1])
    ])
    model.compile('adadelta', 'mse')
    # Train.
    print "training..."
    model.fit(train_X, train_y, nb_epoch=NB_EPOCH, validation_data=(test_X, test_y), verbose=2)

    # Predict bounding boxes on the test images.
    pred_y = model.predict(test_X)
    pred_bboxes = pred_y * img_size
    pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
    pred_bboxes.shape

    # Show a few images and predicted bounding boxes from the test dataset.
    plt.figure(figsize=(12, 3))
    for i_subplot in range(1, 5):
        plt.subplot(1, 4, i_subplot)
        i = np.random.randint(len(test_imgs))
        plt.imshow(test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower',
                   extent=[0, img_size, 0, img_size])
        for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
            print pred_bbox
            plt.gca().add_patch(
                matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r',
                                             fc='none'))
            plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox, exp_bbox)),
                         (pred_bbox[0], pred_bbox[1] + pred_bbox[3] + 0.2), color='r')

    # Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset.
    summed_IOU = 0.
    for pred_bbox, test_bbox in zip(pred_bboxes.reshape(-1, 4), test_bboxes.reshape(-1, 4)):
        summed_IOU += IOU(pred_bbox, test_bbox)
    mean_IOU = summed_IOU / len(pred_bboxes)
    mean_IOU
    print test_X[0]
    hot = cv2.imread("hot_starts/9731914136151-1173170842.jpg")
    hot_grey = cv2.cvtColor(hot, cv2.COLOR_RGB2GRAY )
    # print hot_grey.shape
    # quit()
    hot_norm = cv2.resize(hot_grey, (img_size, img_size))
    hots = np.array(np.array([hot_norm]))
    inpt = (hots.reshape(len(hots), -1) - np.mean(hots)) / np.std(hots)

    pred_bbox = model.predict([inpt])[0]
    print "pred", pred_bbox
    plt.close()
    plt.imshow(hots[0])

    plt.gca().add_patch(
        matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r',
                                     fc='none'))

    # plt.gca().add_patch(
    #     matplotlib.patches.Rectangle((34, 32), 15, 7, ec='r',
    #                                  fc='none'))

    plt.show()


