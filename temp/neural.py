import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    from statsmodels.tsa.stattools import adfuller

    return adfuller, mo, pd, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Les 3 Neural Prophet
    In het huiswerk zagen we dat je met het SARIMAX model best heel goed tijdsreeksen kunt voorspellen. Nu heeft SARIMAX alleen als nadeel dat zodra verbanden niet meer lineair zijn, of tijdsreeksen onderbrekingen kennen SARIMAX hier meer moeite mee krijgt. In deze les zullen we daarom een alternatief uitwerken in de vorm van NeurlaPropet.

    ## Business Understanding

    We hebben de verkoopdata van een Nederlandse winkel. Deze winkel, kent net als de meeste winkels een soort wekelijks patroon. Verder is deze winkel gesloten op de Nederlandse zon en feestdagen en de 1e 3 weken van juli i.v.m. vakantie. De winkelier wil graag de verkopen voor de komende maand weten, zodat zij kan inschatten hoeveel personeel er nodig is.

    ## data understanding
    In het bijgevoegde `winkel_verkoopdata.csv` vind je alle verkoopgegevens, plus nog wat extra variabelen:
    - date: Datum (dagelijks, 1 jan 2022 t/m 31 dec 2023)
    - sales: Verkoopbedrag in euro’s
    - promo: Dummy (1 = promotie actief, 0 = geen promotie)
    - temperature: Dagelijkse gemiddelde temperatuur (als voorbeeld van exogene invloed)

    Verder weten we dat de winkel is gesloten op zon- en feestdagen en de 1e 3 weken van juli. Verder verwachten we komende maand geen promo's en een temperatuur die dicht bij het jaargemiddelde van 3 graden ligt.

    **Werkt dit mb.v.NeuralProphet uit. Neem hier ook de zon- en feestdagen, vakanties, promoties, en temperatuur in mee.**
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("temp/winkel_verkoopdata.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ik zie geen missende waardes wel nul , promoties lijken redelijk zeldzaam aangezien het nog steeds niet 1.00 is bij 75%. Tempertuur en sales lijkt redelijk verdeeld
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laten we nu een paar kolomen over tijd plotten
    """)
    return


@app.cell
def _(df, sns):
    sns.lineplot(data=df,x="date",y="sales")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    De sales lijken redelijk sporadisch, ook kan je duidelijk feestdagen zien aangezien ze sales missen. Ik ga het plotten per maand om te zien of er een duidelijker patroon is
    """)
    return


@app.cell
def _(df, pd, sns):
    df["month"] = pd.to_datetime(df["date"]).dt.month
    sns.lineplot(data=df,x="month",y="sales")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hier zij jee duidelijker dat er patronen zijn over tijd. Wat opvallend is dat rond maand 7 de sales hard naar benden gaan
    """)
    return


@app.cell
def _(df, sns):
    sns.lineplot(data=df,x="date",y="temperature")
    return


@app.cell
def _(df):
    # drop na values
    df_preped = df.dropna()
    return (df_preped,)


@app.cell
def _(adfuller, df_preped, pd):

    df_preped["sales"] = pd.to_datetime(df_preped["date"])
    data_diff = df_preped.copy().diff()
    result = adfuller(df_preped['sales'])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return (data_diff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### de data is nog niet gestationeerd dus ik ga hem nog een keer differentieren over 12 maanden
    """)
    return


@app.cell
def _(data_diff):
    df_diff_seizoen = data_diff.copy().diff(31)
    return (df_diff_seizoen,)


@app.cell
def _(adfuller, df_diff_seizoen):
    result_seizoen = adfuller(df_diff_seizoen['sales'])
    print('ADF Statistic: %f' % result_seizoen[0])
    print('p-value: %f' % result_seizoen[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nice hij is seizoenaal
    """)
    return


@app.cell
def _(df):
    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    from matplotlib import pyplot as plt
    # PACF-plot voor AR-component (p)
    plot_pacf(df['sales'], lags=30)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    oke misschien is differentiëren niet nodig
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural trainen
    """)
    return


@app.cell
def _(df, np, pd):
    from neuralprophet import NeuralProphet
    converted_ds = pd.to_datetime(df, utc=True).view(dtype=np.int64)
    # Daily aggregation
    prophetdata = (
        converted_ds.groupby("date", as_index=False)["sales"]
          .sum()
          .rename(columns={"date": "ds", "sales": "y"})
          .dropna()
    )

    # Make sure ds is datetime
    prophetdata["ds"] = pd.to_datetime(prophetdata["ds"])

    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )

    metrics = model.fit(prophetdata, freq="D")
    print(metrics)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
