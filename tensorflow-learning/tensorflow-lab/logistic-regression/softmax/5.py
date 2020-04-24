from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import os

checkpoint_path = "savecheck/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=5)
IO = Dense(units=3, input_shape=[4], activation='softmax')
model = Sequential([IO])
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.01), metrics=['accuracy'])
history = model.fit(x, y, epochs=1000, callbacks=[cp_callback])
