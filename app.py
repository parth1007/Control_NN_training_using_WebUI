import tensorflow as tf
from tensorflow import keras
import numpy as np

import time
from datetime import datetime as dt
import os
from threading import Thread

from flask import Flask, render_template



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(time.time_ns() % ((1 << 32) - 1))
tf.random.set_seed(time.time_ns())


N = 6400
a, b = 0.3, 5
x = np.linspace(0, 32, N)
y = a * x + b + np.random.uniform(-1, 1, N)


m,c = 50,90
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 500
total_epochs = None
training_result = None
model = None
weights=None
biases = None



class TrainingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global total_epochs, TRAINING_EPOCHS
        print(f"epoch {total_epochs + epoch}/{TRAINING_EPOCHS}", end="\r", flush=True)


def perform_training():
    global training, total_epochs, training_start_time

    training_start_time = time.time()

    while training and total_epochs < TRAINING_EPOCHS:
        model.fit(
            x,
            y,
            epochs=TRAINING_EPOCHS // 100,
            callbacks=[TrainingCallback()],
            verbose=1
        )

        total_epochs += TRAINING_EPOCHS // 100
        print(f"epoch {total_epochs}/{TRAINING_EPOCHS}")

        weights, biases = model.get_weights()
        cost = model.evaluate(x, y, verbose=0)

    training = False
    finalize_training()



def finalize_training():
    global training_end_time, training_result, weights, biases

    training_end_time = time.time()

    print(f"[{dt.time(dt.now())}]: training completed!")

    weights, biases = model.get_weights()
    cost = model.evaluate(x, y, verbose=0)

    training_result = '\n'.join([
        f"It took {training_end_time - training_start_time: .3f} seconds",
        f"We used {m = }, {c = } and some noise",
        f"After training, we have m = {weights[0][0]: .3f}, c = {biases[0]: .3f}"
    ])

    print(training_result)


def init_model():
    model = keras.Sequential([
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model


app = Flask(__name__)

@app.route("/", endpoint="inner", methods=['GET','POST'])
def index():
    return render_template('index.html')


total_epochs = 0

@app.route("/start", endpoint="start_training", methods=['GET'])
def start_training():
    global model, training, training_end_time, training_thread

    if model is None:
        model = init_model()
        training = True
        
        training_thread = Thread(target=perform_training)

        print(f"[{dt.time(dt.now())}]: Model training has started!")

        training_start_time = time.time()
        training_thread.start()

        return f"""
            [{dt.time(dt.now())}]: training successfully started!
        """
    elif training:
        return """
            Training is already in progress!
            See <a href="/status">/status</a>
        """


    return """
        Model is already trained!
    """


@app.route("/pause", endpoint="stop_training", methods=['GET','POST'])
def pause_training():
    global model, training, training_end_time, training_thread

    if model is not None:
        if training:
            training = False
            training_thread.join()

            return f"Training paused! After training, we have m = {weights[0][0]: .3f}, c = {biases[0]: .3f}"
        else:
            return f"Training Already completed! After training, we have m = {weights[0][0]: .3f}, c = {biases[0]: .3f}"

    return """
        Model is not initiliazed!
        See <a href="/start">/start</a>
    """


@app.route("/status", endpoint="status_training", methods=['GET','POST'])
def status_training():
    global model, training, training_end_time, training_thread

    if model is not None:
        if training:
            return f"""
                Training is in progress...
                Epochs completed: {total_epochs}
                Time spent: {time.time() - training_start_time: .3f} seconds
            """

        return f"""
            Training completed!
            Total epochs completed: {total_epochs}
            Time taken: {training_end_time - training_start_time: .3f} seconds
            After training, we have m = {weights[0][0]: .3f}, c = {biases[0]: .3f}
        """

    return """
        Model is not initiliazed!
        Click <a href="/start">/start</a>
    """



@app.route("/resume", endpoint="resume_training", methods=['GET'])
def resume_training():
    global model, training, training_end_time, training_thread

    if model is None:
        return "Model is not initiliazed. Please start training first"

    elif training is False:

        training = True
        m = weights[0][0]
        c = biases[0]
        training_thread = Thread(target=perform_training)
        print(f"[{dt.time(dt.now())}]: Model training is resumed!")
        training_thread.start()

        return """
            Training is Resumed!
            See <a href="/status">/status</a>
        """

    return "Training is already ongoing!"


if __name__ == "__main__":
    app.run()
