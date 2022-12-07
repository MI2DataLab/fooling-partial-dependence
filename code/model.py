from tensorflow import keras

# basic model consisting of 3 Dense layers
class BasicModel(keras.Model):
    def __init__(
        self,
        size: int,
        normalizer
    ):
        super().__init__()
        self.normalizer = normalizer
        self.dense1 = keras.layers.Dense(size, activation="relu")
        self.dense2 = keras.layers.Dense(size, activation="relu")
        self.dense3 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, input):
        x = self.normalizer(input)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)