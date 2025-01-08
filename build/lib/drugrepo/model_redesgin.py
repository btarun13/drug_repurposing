import sqlite3
import time
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

alt.data_transformers.disable_max_rows()


class Model:
    def __init__(self, layers):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.model = Model.get_model(layers).to(self.device)

    def add_data(self, data, features, test_size=0.2):
        self.features = features
        self.scalar = StandardScaler()
        X = self.scalar.fit_transform(data[features].to_numpy())
        y = data[["target"]].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_train = (
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        )
        self.y_test = (
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)
        )

    @staticmethod
    def get_model(layers):
        network = []
        network.append(torch.nn.Linear(layers[0], layers[1]))
        network.append(torch.nn.ReLU())

        for i in range(1, len(layers) - 1):
            network.append(torch.nn.Linear(layers[i], layers[i + 1]))
            network.append(torch.nn.ReLU())

        network.append(torch.nn.Linear(layers[-1], 1))
        return torch.nn.Sequential(*network)

    def train(self, epochs, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        start = time.time()

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            train_loss = self.criterion(outputs, self.y_train)
            train_loss.backward()
            optimizer.step()

            # Evaluate on the test set
            self.model.eval()
            with torch.no_grad():
                self.y_pred = self.model(self.X_test)

            if (epoch + 1) % 500 == 0:
                with torch.no_grad():
                    train_pred = self.model(self.X_train)
                    test_pred = self.model(self.X_test)
                r2_train = r2_score(
                    self.y_train.cpu().numpy(), train_pred.cpu().numpy()
                )
                r2_test = r2_score(self.y_test.cpu().numpy(), test_pred.cpu().numpy())
                print(
                    f"Epoch [{epoch+1}/{epochs}], Train: {r2_train:.4f}, "
                    f"Test: {r2_test:.4f}, Time: {time.time()-start:.2f}"
                )

    def compare(self, data, query, x="label"):
        train = data.eval("case='train'")
        test = data.eval("case='test'")
        test["target"] = self.predict(data[self.features].to_numpy())
        df = pd.concat([train, test], axis=0, ignore_index=True)
        return (
            alt.Chart(df.query(query))
            .mark_line(point=True)
            .encode(
                alt.X("label", scale=alt.Scale(zero=False)),
                alt.Y("target", scale=alt.Scale(zero=False)),
                alt.Color("case"),
            )
            .properties(height=400, width=500)
        )

    def predict(self, X):
        X = self.scalar.transform(X)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.model(X).detach().cpu().numpy().flatten()

    def evaluate(self):
        with torch.no_grad():
            y_pred = self.model(self.X_test).cpu().numpy().flatten()

        y_test = self.y_test.cpu().numpy().flatten()
        error = pd.DataFrame({"true": y_test, "pred": y_pred, "error": y_pred - y_test})
        scatter = (
            alt.Chart(error)
            .mark_circle(fillOpacity=0.3)
            .encode(
                x=alt.X("true", title="True Values", scale=alt.Scale(zero=False)),
                y=alt.Y("pred", title="Predicted Values", scale=alt.Scale(zero=False)),
                tooltip=[
                    "true",
                    "pred",
                ],  # Optional: Add tooltips to see exact values on hover
            )
        )

        # Create the diagonal line where predicted equals true
        line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [error["true"].min(), error["true"].max()],
                        "y": [error["true"].min(), error["true"].max()],
                    }
                )
            )
            .mark_line(color="red")
            .encode(x="x:Q", y="y:Q")
        )

        # Combine the scatter plot and diagonal line
        diagonal_error_chart = scatter + line

        # Display the chart
        return error, diagonal_error_chart


