import threading
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table


df = pd.read_csv("nvda_final_2022-01-03_to_2023-12-15.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

sentiment_labels = ["Optimism", "Uncertainty", "Surprise", "Immediacy", "Relief"]

def build_price_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines+markers",
        name="Close",
        customdata=df[sentiment_labels].values,  # used by hover to update radar
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="NVDA Close Price",
        hovermode="x unified",
        margin=dict(l=40, r=10, t=50, b=40),
    )
    return fig

price_fig = build_price_fig()
last_close = float(df["Close"].iloc[-1])
date_min = df["Date"].min().date()
date_max = df["Date"].max().date()

# Last dominant sentiment on the last available date
_last_sent_series = df[sentiment_labels].iloc[-1]
last_dom_label = _last_sent_series.idxmax()
last_dom_value = float(_last_sent_series.max())

results_cache = {"ready": False}

def train_once():
    try:
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Supervised target: next-day close
        df_model = df.copy()
        df_model["Next_Close"] = df_model["Close"].shift(-1)
        df_model = df_model.dropna(subset=["Next_Close"])

        X = df_model[["Close"] + sentiment_labels]
        y = df_model["Next_Close"]

        # Time-based split (80/20)
        split_index = int(len(df_model) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        dates_test = df_model["Date"].iloc[split_index:]

        # Train model
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        # Chart: Actual vs Predicted (test)
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=dates_test, y=y_test, mode="lines+markers", name="Actual Next Close"
        ))
        forecast_fig.add_trace(go.Scatter(
            x=dates_test, y=y_pred, mode="lines+markers", name="Predicted Next Close"
        ))
        forecast_fig.update_layout(
            title="Actual vs Predicted (Next-Day Close) — Test Split",
            margin=dict(l=40, r=10, t=50, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # ---- Last 10 forecasts (Date + Predicted), newest first ----
        last_10 = pd.DataFrame({
            "Date": dates_test.values,
            "Predicted_Next_Close": y_pred
        }).tail(10).copy()

        last_10 = last_10.sort_values("Date", ascending=False)
        last_10 = last_10.assign(
            Date=last_10["Date"].dt.strftime("%Y-%m-%d"),
            Predicted_Next_Close=last_10["Predicted_Next_Close"].round(4)
        )

        results_cache.update(dict(
            ready=True,
            rmse=rmse, mae=mae, r2=r2,
            forecast_fig=forecast_fig,
            table_columns=[{"name": c, "id": c} for c in ["Date", "Predicted_Next_Close"]],
            table_data=last_10[["Date", "Predicted_Next_Close"]].to_dict("records"),
        ))

    except Exception as e:
        results_cache.update(dict(ready=True, error=str(e)))

threading.Thread(target=train_once, daemon=True).start()


app = Dash(__name__)

def kpi_card(label, value):
    return html.Div(
        className="kpi-card",
        children=[html.Div(label, className="kpi-label"),
                  html.Div(value, className="kpi-value")],
        style={
            "padding": "12px 16px",
            "border": "1px solid #eaeaea",
            "borderRadius": "12px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
            "background": "white",
            "minWidth": "160px"
        }
    )

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, Arial, sans-serif", "padding": "16px", "background": "#fafafa"},
    children=[
        html.H2("MultiRadar — NVDA Sentiment & Forecasting Dashboard", style={"marginBottom": "12px"}),

        # State + Poller
        dcc.Store(id="results-store"),
        dcc.Interval(id="poller", interval=800, n_intervals=0, disabled=False),

        # TOP: KPIs (order requested)
        html.Div(
            id="kpi-panel",
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"},
            children=[
                kpi_card("Data Range", f"{date_min} → {date_max}"),
                kpi_card("Test RMSE", "…"),
                kpi_card("Test MAE", "…"),
                kpi_card("R² (Test)", "…"),
                kpi_card("Last Close", f"{last_close:.2f}"),
                kpi_card("Last Dominant Sentiment", f"{last_dom_label}: {last_dom_value:.2f}"),
            ]
        ),

        # MIDDLE: Price + Radar
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "12px",
                   "alignItems": "stretch", "marginBottom": "16px"},
            children=[
                html.Div(
                    style={"background": "white", "borderRadius": "12px", "padding": "8px","border": "1px solid #eaeaea"},
                    children=[dcc.Graph(id="price-graph", figure=price_fig, style={"height": "430px"})]
                ),
                html.Div(
                    style={"background": "white", "borderRadius": "12px", "padding": "8px","border": "1px solid #eaeaea"},
                    children=[dcc.Graph(id="radar-graph", style={"height": "430px"})]
                ),
            ]
        ),

        # BOTTOM: Left = Actual vs Predicted; Right = Last 10 forecasts (Date + Predicted), newest first
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "12px", "alignItems": "stretch"},
            children=[
                html.Div(
                    style={"background": "white", "borderRadius": "12px", "padding": "8px","border": "1px solid #eaeaea"},
                    children=[dcc.Graph(id="forecast-graph", style={"height": "420px"})]
                ),
                html.Div(
                    style={"background": "white", "borderRadius": "12px", "padding": "8px","border": "1px solid #eaeaea"},
                    children=[
                        html.Div("Last 10 Forecasts (Newest First)", style={"fontWeight":"600", "marginBottom":"6px"}),
                        dash_table.DataTable(
                            id="forecast-table",
                            columns=[{"name": "Date", "id": "Date"},
                                     {"name": "Predicted_Next_Close", "id": "Predicted_Next_Close"}],
                            data=[],
                            style_table={"height": "380px", "overflowY": "auto"},
                            style_cell={"padding":"6px", "fontFamily":"Inter, system-ui, Arial, sans-serif"},
                            style_header={"fontWeight": "600", "backgroundColor": "#f7f7f7"},
                        )
                    ]
                ),
            ]
        ),
    ]
)


@app.callback(
    Output("results-store", "data"),
    Output("poller", "disabled"),
    Input("poller", "n_intervals"),
    prevent_initial_call=False
)
def poll_results(_):
    if results_cache.get("ready"):
        return results_cache, True
    return no_update, False

@app.callback(
    Output("kpi-panel", "children"),
    Output("forecast-graph", "figure"),
    Output("forecast-table", "columns"),
    Output("forecast-table", "data"),
    Input("results-store", "data"),
    State("kpi-panel", "children")
)
def update_when_ready(data, current_kpis):
    if not data:
        return current_kpis, no_update, no_update, no_update

    if "error" in data:
        kpis = [
            kpi_card("Data Range", f"{date_min} → {date_max}"),
            kpi_card("Test RMSE", "—"),
            kpi_card("Test MAE", "—"),
            kpi_card("R² (Test)", "—"),
            kpi_card("Last Close", f"{last_close:.2f}"),
            kpi_card("Last Dominant Sentiment", f"{last_dom_label}: {last_dom_value:.2f}"),
        ]
        return kpis, no_update, no_update, no_update

    # Rebuild KPI panel in the requested order with filled metrics
    kpis = [
        kpi_card("Data Range", f"{date_min} → {date_max}"),
        kpi_card("Test RMSE", f"{data['rmse']:.4f}"),
        kpi_card("Test MAE", f"{data['mae']:.4f}"),
        kpi_card("R² (Test)", f"{data['r2']:.3f}"),
        kpi_card("Last Close", f"{last_close:.2f}"),
        kpi_card("Last Dominant Sentiment", f"{last_dom_label}: {last_dom_value:.2f}"),
    ]
    return kpis, data["forecast_fig"], data["table_columns"], data["table_data"]

@app.callback(
    Output("radar-graph", "figure"),
    Input("price-graph", "hoverData")
)
def update_radar(hoverData):
    if hoverData and "points" in hoverData and len(hoverData["points"]) > 0:
        p = hoverData["points"][0]
        idx = p["pointIndex"]
        r = df.loc[idx, sentiment_labels].values
        date_str = df.loc[idx, "Date"].strftime("%Y-%m-%d")
    else:
        r = [0] * 5
        date_str = "Hover over a point"

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r, theta=sentiment_labels, fill='toself',
        mode='lines+markers',
        line=dict(shape='spline', smoothing=1.3),
        marker=dict(size=6),
        name='Sentiment'
    ))
    fig.update_layout(
        title=f"Sentiment Radar — {date_str}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=40, r=10, t=50, b=40),
        transition=dict(duration=500, easing="cubic-in-out")
    )
    return fig

# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8051, debug=False)
