import pandas as pd
from ludwig.api import LudwigModel


def main():
    data = pd.read_csv('./winequality-red.csv')
    for dummy_i in range(len(data["quality"])):
        if data["quality"][dummy_i] < 6.5:
            data.at[dummy_i, "quality"] = 0
        else:
            data.at[dummy_i, "quality"] = 1
    model_definition = {
        "input_features": [
                           {'name': 'fixed_acidity', 'type': 'numerical'},
                           {'name': 'volatile_acidity', 'type': 'numerical'},
                           {'name': 'citric_acid', 'type': 'numerical'},
                           {'name': 'residual_sugar', 'type': 'numerical'},
                           {'name': 'chlorides', 'type': 'numerical'},
                           {'name': 'free_sulfur_dioxide', 'type': 'numerical'},
                           {'name': 'total_sulfur_dioxide', 'type': 'numerical'},
                           {'name': 'density', 'type': 'numerical'},
                           {'name': 'pH', 'type': 'numerical'},
                           {'name': 'sulphates', 'type': 'numerical'},
                           {'name': 'alcohol', 'type': 'numerical'}
                           ],
        "output_features": [
                            {'name': 'quality', 'type': 'binary'}
                           ]
    }
    model = LudwigModel(model_definition)
    trained_model = model.train(data)
    predictions, test_stats = model.test(data_df=data)
    model.save("wine")
    model.close()


if __name__ == "__main__":
    main()
