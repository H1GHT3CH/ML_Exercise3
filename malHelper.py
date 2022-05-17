import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score

def draw_estimation_scatterplot(model, X, y):
    predictions = model.predict(X)
    mae, r2 = mean_absolute_error(y, predictions), r2_score(y, predictions)
    plot_data = pd.DataFrame({'MEDV': y, 'predictions': predictions})
    return px.scatter(plot_data, x='MEDV', y='predictions', title=f"MAE:{mae:.2f}  R2:{r2:.2f}")


    