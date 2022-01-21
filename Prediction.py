from keras.models import load_model
import numpy as np

class Prediction:
        def __init__(self):
                self.model = load_model("90_90_model.h5")
                #self.model = load_model("90_65_cv_model.h5") ##
        def predict(self, arr):
                arr = [arr]
                #arr = np.array(arr) ##
                prediction = self.model.predict(arr)
                observation = round(prediction[0][0])
                if observation == 0:
                        observation = "Negative Breast Mass Characteristics"
                else:
                        observation = "Positive Breast Mass Characteristics"
                confidence = abs(prediction[0][0] - 0.5) * 2 * 100
                return [observation, str(confidence)]


