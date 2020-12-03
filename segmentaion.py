from lib import *


def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (352, 480, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def segment(filename):

    image = load(filename)

    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 16
    CLASSES = ['passport']
    LR = 0.0001
    EPOCHS = 10

    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

    model.load_weights('./models/segmentation.h5')

    file_size = (cv2.imread(filename).shape[1],cv2.imread(filename).shape[0])

    predict = model.predict(image)


    pred = predict[0,:,:,0]
    pred = np.dstack([pred, pred, pred])
    pred = (pred * 255).astype(np.uint8)
    img = Image.fromarray(pred, 'RGB')
    img = img.resize(file_size)
    img.save('./images/pred_org.png')
    return img


if __name__ == "__main__":

    filename = 'images/image.jpg'
    img = segment(filename)
    plt.imshow(img)
    plt.show()