import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

df = pd.read_csv("nvda_final_2023-01-03_to_2023-12-15.csv")
df["Date"] = pd.to_datetime(df["Date"])

sentiment_labels = ["Optimism", "Uncertainty", "Surprise", "Immediacy", "Relief"]

app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("NVDA Stock Price with Sentiment Radar", style={'textAlign': 'center'}),
    dcc.Graph(id="price-graph", style={"width": "70%", "display": "inline-block"}),
    dcc.Graph(id="radar-graph", style={"width": "28%", "display": "inline-block"})
])

# Callback to update radar graph on hover
@app.callback(
    Output("radar-graph", "figure"),
    Input("price-graph", "hoverData")
)
def update_radar(hoverData):
    if hoverData:
        idx = hoverData["points"][0]["pointIndex"]
        r = df.loc[idx, sentiment_labels].values
        date_str = df.loc[idx, "Date"].strftime("%Y-%m-%d")
    else:
        r = [0] * 5
        date_str = "Hover over a point"

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=sentiment_labels,
        fill='toself',
        name='Sentiment',
        line_color='crimson'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=f"Sentiment on {date_str}"
    )
    return fig

# Render price graph
@app.callback(
    Output("price-graph", "figure"),
    Input("price-graph", "id")  # Dummy trigger
)
def render_price_graph(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines+markers",
        name="Close Price",
        marker=dict(color="royalblue")
    ))
    fig.update_layout(title="NVDA Close Price", hovermode="x unified")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
