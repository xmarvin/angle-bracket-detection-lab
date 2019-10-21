import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import params

input_size = params.input_size
epochs = params.max_epochs
batch_size = params.batch_size
model = params.model_factory()

class MaskGenerator:
  def __init__(self, batch_size):
    self.image = cv2.imread('data/sample.jpg', cv2.IMREAD_GRAYSCALE)
    self.mask  = cv2.imread('data/mask.jpg', cv2.IMREAD_GRAYSCALE)
    self.image = cv2.resize(self.image, (input_size, input_size))
    self.mask  = cv2.resize(self.mask, (input_size, input_size))
    self.brackets = np.asarray(params.get_boundaries(self.mask))
    self.batch_size = batch_size

  def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.7, 0.7),
                           scale_limit=(-0.5, 3),
                           rotate_limit=(-3, 3), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT):
    height, width = image.shape

    angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
    scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
    aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
    sx = scale * aspect / (aspect ** 0.5)
    sy = scale / (aspect ** 0.5)
    dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
    dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

    cc = np.math.cos(angle / 180 * np.math.pi) * sx
    ss = np.math.sin(angle / 180 * np.math.pi) * sy
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                borderValue=(
                                    0, 0,
                                    0,))
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                               borderValue=(
                                   0, 0,
                                   0,))

    return image, mask

  def generator(self):
    while True:
      x_batch = []
      y_batch = []
      for i in range(self.batch_size):
        img = self.image
        msk = self.mask
        if np.random.rand()>0.3:
          img, msk = params.disable_bracket(img, msk, self.brackets)

        img, msk = MaskGenerator.randomShiftScaleRotate(img, msk)
        img =  params.simplify_image(img)
        img = np.expand_dims(img, axis=2)
        msk = np.expand_dims(msk, axis=2)
        x_batch.append(img)
        y_batch.append(msk)

      x_batch = np.asarray(x_batch)
      y_batch = np.asarray(y_batch)
      x_batch = np.array(x_batch, np.float32) / 255.0
      y_batch = np.array(y_batch, np.float32) / 255.0

      yield x_batch, y_batch

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=4,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights.h5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]

gen = MaskGenerator(batch_size)
print(model.summary())
model.fit_generator(generator=gen.generator(),
                    steps_per_epoch=10,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=gen.generator(),
                    validation_steps=2
                    )
