import argparse
import os
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from pathlib import Path

def predict(series_path, csv_path):
    loaded_model = tf.keras.models.load_model("model.best.h5")

    series_list = os.listdir(series_path)
    probabilities = {"SeriesInstanceUID": [], "prediction": []}

    for s in series_list:
        slices = [pydicom.dcmread(os.path.join(series_path, s, dcm)) for dcm in os.listdir(os.path.join(series_path, s))]
        slices.sort(key = lambda x: int(x.InstanceNumber))

        image = np.stack([slice.pixel_array for slice in slices])
        image = image.astype(np.int16)

        width=256
        height=256
        if (image.shape[1] < height) or (image.shape[2] < width):
            arr = np.squeeze(tf.image.resize_with_pad(np.expand_dims(image, axis=-1), height, width))
        else:
            arr = image

        y = arr.shape[1]
        x = arr.shape[2]
        startx = x//2 - (width//2)
        starty = y//2 - (height//2)
        cropped_image = arr[:, starty:starty+height, startx:startx+width]

        normalized_array = (cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min())
        normalized_array = normalized_array.astype("float32")


        if len(normalized_array) == 16:
             image = normalized_array

        elif len(normalized_array) > 16:
            remainder = len(normalized_array) - 16
            image = normalized_array[int(remainder / 2): 16 + int(remainder / 2)]


        pred = loaded_model.predict(np.expand_dims(image, axis=(0, -1)))
        #pred = loaded_model.predict(np.expand_dims(image, axis=0))
        probability = pred[0][0]
        probabilities["SeriesInstanceUID"].append(s)
        probabilities["prediction"].append(probability)

    df = pd.DataFrame(probabilities)
    df.to_csv(csv_path, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, help="path to the data directory")
    parser.add_argument("--predictions-file-path", type=Path, help="path to save the predictions CSV")
    args = parser.parse_args()
    predict(args.data_dir, args.predictions_file_path)

