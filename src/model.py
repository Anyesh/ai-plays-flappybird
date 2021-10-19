import tensorflow as tf


class NeuralNet:
    def __init__(self, model=None):
        self.model: tf.keras.models.Sequential = model

    @classmethod
    def build_model(cls, input_shape, output_shape):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(
                512,
                input_shape=input_shape,
                activation="relu",
                kernel_initializer="he_uniform",
            )
        )
        model.add(
            tf.keras.layers.Dense(
                256, activation="relu", kernel_initializer="he_uniform"
            )
        )
        model.add(
            tf.keras.layers.Dense(
                64, activation="relu", kernel_initializer="he_uniform"
            )
        )
        model.add(
            tf.keras.layers.Dense(
                output_shape, activation="linear", kernel_initializer="he_uniform"
            )
        )
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.RMSprop(
                learning_rate=0.0001, rho=0.95, epsilon=0.01
            ),
            metrics=["accuracy"],
        )
        return cls(model).model


if __name__ == "__main__":
    model = NeuralNet.build_model(input_shape=(1,), output_shape=1)
    model.summary()
