import numpy as np
import pickle as pkl

# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials


def my_predict(df):
    with open("model_dump_o3.pkl", "rb") as file:
        model1 = pkl.load(file)
    with open("model_dump_no2.pkl", "rb") as file:
        model2 = pkl.load(file)

    # Load your model file

    # Make two sets of predictions, one for O3 and another for NO2
    pred_o3 = model1.predict(
        df[["temp", "humidity", "no2op1", "no2op2", "o3op1", "o3op2"]])
    pred_no2 = model2.predict(
        df[["temp", "humidity", "no2op1", "no2op2", "o3op1", "o3op2"]])

    # Return both sets of predictions
    return (pred_o3, pred_no2)
