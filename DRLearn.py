from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from statistics import mean

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
    import pandas as pd
    import seaborn as sns
    import shap
    import warnings
    from eli5 import show_weights


class DRLearn:
    def __init__(self):
        self.explainer = None
        self.shap_values = None

    @staticmethod
    def plot_passenger_gender(df):
        fig = plt.figure(figsize=(14, 6), dpi=100)
        sns.set(font_scale=2)
        sns.set_style("whitegrid")
        plt.grid(False)
        f = sns.barplot(
            x="Sex", y="Survived", data=df, palette="Greys", ci=0, edgecolor="dimgrey"
        )
        f.set(
            title="Survival rate by gender",
            ylabel="Survival rate (%)",
            xlabel="",
        )

        ylabels = ["{:.0%}".format(x) for x in f.get_yticks()]
        _ = f.set_yticklabels(ylabels)

    @staticmethod
    def plot_passenger_class(df):
        fig = plt.figure(figsize=(14, 6), dpi=100)
        class_map = {1: "1st class", 2: "2nd class", 3: "3rd class"}
        sns.set(font_scale=2)
        sns.set_style("whitegrid")

        df["pclass_label"] = df["Pclass"].apply(lambda x: class_map.get(x))
        plt.grid(False)
        f = sns.barplot(
            x="pclass_label",
            order=["1st class", "2nd class", "3rd class"],
            y="Survived",
            data=df,
            palette="Greys",
            ci=0,
            edgecolor="dimgrey",
        )
        f.set(title="Survival rate by class", xlabel="", ylabel="Survival rate")

        ylabels = ["{:.0%}".format(x) for x in f.get_yticks()]
        _ = f.set_yticklabels(ylabels)

    @staticmethod
    def extract_features(df):
        sex_encoder = LabelEncoder()
        df["Gender"] = sex_encoder.fit_transform(df["Sex"])
        df["Family_Size"] = df.SibSp + df.Parch + 1

        classes = pd.get_dummies(df.Pclass, prefix="Class")

        X = pd.concat([df.Gender, df.Family_Size, classes], axis=1)
        y = df["Survived"]

        print("Feature extraction complete.")

        return X, y

    @staticmethod
    def train_model(X_train, y_train):
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        print("The model has been trained!")
        return clf

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        preds = model.predict(X_test)
        print("Evaluating model...")
        score = accuracy_score(y_test, preds) * 100
        print(f"The model achieved {round(score, 2)}% accuracy on the test dataset")

    @staticmethod
    def split_dataset(X, y, split=0.5):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=42
        )
        pctg_size = round(X_train.shape[0] / X.shape[0] * 100)
        print(f"Training set is {pctg_size}% the size of the original dataset")
        return X_train, X_test, y_train, y_test

    @staticmethod
    def visualise_training_progress(model, X_train, y_train, X_test, y_test):
        sizes = [2, 8, 10, 12, 16, 20, 24, 32, 40, 52, 64, 128, 256, 512, 720]
        train_scores = []
        test_scores = []

        for size in sizes:
            x_tr = X_train[:size]
            y_tr = y_train[:size]

            model.fit(x_tr, y_tr)
            train_scores.append(accuracy_score(y_tr, model.predict(x_tr)))
            test_scores.append(accuracy_score(y_test, model.predict(X_test)))

        fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
        plt.grid(False)

        ax.plot(sizes, test_scores, color="black", label="Test Score", lw=5)[0]
        ax.plot(sizes, train_scores, color="grey", label="Train Score", lw=3)
        plt.xlabel("Dataset Size")
        plt.ylabel("Accuracy")
        ylabels = ["{:.0%}".format(x) for x in ax.get_yticks()]
        ax.set_yticklabels(ylabels)

        plt.legend()

    @staticmethod
    def interpret_model(model, X, y):
        expected_value = 99
        limit = 1000
        while (expected_value > 1 or expected_value < 0) and limit > 0:
            limit -= 1
            explainer = shap.TreeExplainer(model, data=X, model_output="probability")
            expected_value = explainer.expected_value[1]

        shap_values = explainer.shap_values(X, y)[1]

        return [explainer, shap_values]

    @staticmethod
    def analyze_passenger_prediction(model_interpretation, X, index):
        shap.initjs()
        gender = X.Gender.map({0: "Female", 1: "Male"})
        plot = shap.force_plot(
            model_interpretation[0].expected_value[1],
            model_interpretation[1][index, :],
            X.assign(Gender=gender).iloc[index, :],
        )

        return plot

    @staticmethod
    def explain_model(model, X_train):
        return show_weights(
            model,
            feature_names=X_train.columns.tolist(),
            show=(
                "method",
                "description",
                "transition_features",
                "targets",
                "feature_importances",
            ),
        )
