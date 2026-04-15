import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("data/train_data.csv")
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    # correlation matrix thats easy to look at 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    corr_features = ['Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp']
    corr = df[corr_features].corr()
    plt.figure(figsize=(120,100))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
