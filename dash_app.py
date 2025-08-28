import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, dash_table

CSV_PATH = "nvda_final_2022-01-03_to_2023-12-15.csv"


load_error = None
try:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at: {os.path.abspath(CSV_PATH)}")

    df = pd.read_csv(CSV_PATH)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("CSV must contain at least 'Date' and 'Close' columns.")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None)
    if df["Date"].isna().any():
        raise ValueError("Some Date values could not be parsed. Check your CSV format.")

    df = df.sort_values("Date").reset_index(drop=True)
except Exception as e:
    df = pd.DataFrame({"Date": pd.to_datetime(["1970-01-01"]), "Close": [0.0]})
    load_error = str(e)

sentiment_labels = ["Optimism", "Uncertainty", "Surprise", "Immediacy", "Relief"]
for col in sentiment_labels:
    if col not in df.columns:
        df[col] = 0.0

# Dominant label per day
df["Dominant"] = df[sentiment_labels].idxmax(axis=1)

# Colors & emojis
SENTIMENT_COLORS = {
    "Optimism":   "#2ecc71",
    "Uncertainty":"#e67e22",
    "Surprise":   "#8e44ad",
    "Immediacy":  "#3498db",
    "Relief":     "#f1c40f",
}
SENTIMENT_EMOJI = {
    "Optimism": "🟢",
    "Uncertainty": "🟠",
    "Surprise": "🟣",
    "Immediacy": "🔵",
    "Relief": "🟡",
}

date_min = df["Date"].min().date()
date_max = df["Date"].max().date()
last_close = float(df["Close"].iloc[-1]) if not df.empty else 0.0
_last_sent_series = df[sentiment_labels].iloc[-1]
last_dom_label = _last_sent_series.idxmax()
last_dom_value = float(_last_sent_series.max())

forecast_fig = go.Figure()
forecast_table_cols = [{"name": "Date", "id": "Date"}, {"name": "Predicted_Next_Close", "id": "Predicted_Next_Close"}]
forecast_table_data = []
rmse = mae = r2 = None
training_note = None
try:
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    if len(df) > 10:
        df_model = df.copy()
        df_model["Next_Close"] = df_model["Close"].shift(-1)
        df_model = df_model.dropna(subset=["Next_Close"]).reset_index(drop=True)

        X = df_model[["Close"] + sentiment_labels]
        y = df_model["Next_Close"]

        split_index = int(len(df_model) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        dates_test = df_model["Date"].iloc[split_index:]

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        forecast_fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode="lines+markers", name="Actual Next Close"))
        forecast_fig.add_trace(go.Scatter(x=dates_test, y=y_pred, mode="lines+markers", name="Predicted Next Close"))
        forecast_fig.update_layout(
            title="Actual vs Predicted (Next-Day Close) — Test Split",
            margin=dict(l=40, r=10, t=50, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        last_10 = pd.DataFrame({"Date": dates_test.values, "Predicted_Next_Close": y_pred}).tail(10).copy()
        last_10 = last_10.sort_values("Date", ascending=False).assign(
            Date=lambda x: pd.to_datetime(x["Date"]).dt.strftime("%Y-%m-%d"),
            Predicted_Next_Close=lambda x: x["Predicted_Next_Close"].round(4),
        )
        forecast_table_data = last_10[["Date", "Predicted_Next_Close"]].to_dict("records")
except Exception as e:
    training_note = f"Note: training skipped ({e})."


def _naive(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        return ts.tz_localize(None)
    except TypeError:
        return ts

def chips_legend():
    chips = []
    for s in sentiment_labels:
        chips.append(
            html.Div(
                style={
                    "display": "inline-flex","alignItems": "center","gap": "6px",
                    "padding": "4px 8px","border": "1px solid #e5e5e5",
                    "borderRadius": "999px","background": "#fff","fontSize": "12px",
                    "marginRight": "6px"
                },
                children=[
                    html.Span(style={
                        "display":"inline-block","width":"10px","height":"10px",
                        "borderRadius":"50%","background":SENTIMENT_COLORS[s],
                        "border":"1px solid rgba(0,0,0,0.15)"
                    }),
                    html.Span(f"{s}")
                ]
            )
        )
    return html.Div(chips, style={"marginTop": "4px"})

def kpi(label, value):
    return html.Div(
        style={"padding":"12px 16px","border":"1px solid #eaeaea","borderRadius":"12px",
               "boxShadow":"0 1px 3px rgba(0,0,0,0.06)","background":"white","minWidth":"160px"},
        children=[html.Div(label, style={"fontWeight":600,"marginBottom":"4px"}), html.Div(value)]
    )

# Page 1
def build_price_fig(df_subset, sentiments_selected):
    fig = go.Figure()
    # base blue line only
    fig.add_trace(go.Scatter(
        x=df_subset["Date"], y=df_subset["Close"], mode="lines",
        line=dict(color="#1f77b4", width=2),
        name="Close",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
    ))
    # colored markers only for selected sentiments
    if sentiments_selected:
        show_set = set(sentiments_selected)
        marker_colors = [
            SENTIMENT_COLORS.get(dom, "#999") if dom in show_set else "rgba(0,0,0,0)"
            for dom in df_subset["Dominant"]
        ]
        fig.add_trace(go.Scatter(
            x=df_subset["Date"], y=df_subset["Close"], mode="markers",
            marker=dict(size=7, color=marker_colors, line=dict(width=0)),
            name="Selected sentiments", hoverinfo="skip"
        ))
    fig.update_layout(
        title="NVDA Close Price",
        hovermode="x unified", margin=dict(l=40, r=10, t=50, b=40), showlegend=False
    )
    return fig

def filter_by_date_only(start_date, end_date):
    start = _naive(start_date) if start_date is not None else df["Date"].min()
    end = _naive(end_date) if end_date is not None else df["Date"].max()
    if pd.isna(start) or pd.isna(end): start, end = df["Date"].min(), df["Date"].max()
    if end < start: start, end = end, start
    out = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    return out if not out.empty else df.tail(1).copy()

def layout_page1():
    return html.Div(style={"fontFamily":"Inter, Arial, sans-serif","padding":"16px","background":"#fafafa"}, children=[
        html.H2("MultiRadar — NVDA Sentiment & Forecasting Dashboard", style={"marginBottom":"12px"}),

        # Environment notes
        html.Div([
            html.Div(f"CSV load error: {load_error}", style={"color":"white","background":"#d9534f",
                     "padding":"8px","borderRadius":"8px"}) if load_error else None,
            html.Div(training_note, style={"color":"#856404","background":"#fff3cd","padding":"8px",
                     "borderRadius":"8px","marginTop":"6px"}) if training_note else None,
        ]),

        # Controls
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap","margin":"10px 0"}, children=[
            html.Div([
                html.Div("Date Range", style={"fontWeight":600,"marginBottom":"4px"}),
                dcc.DatePickerRange(
                    id="p1-date-range",
                    min_date_allowed=date_min, max_date_allowed=date_max,
                    start_date=date_min, end_date=date_max, display_format="YYYY-MM-DD"
                )
            ]),
            html.Div([
                html.Div("Dominant Sentiment (markers)", style={"fontWeight":600,"marginBottom":"4px"}),
                dcc.Dropdown(
                    id="p1-sentiment-filter",
                    options=[{"label": f"{SENTIMENT_EMOJI[s]} {s}", "value": s} for s in sentiment_labels],
                    value=[], multi=True, placeholder="Select sentiments to show markers"
                ),
                chips_legend()
            ], style={"minWidth":"280px"}),
        ]),

        # KPIs
        html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"16px"}, children=[
            kpi("Data Range", f"{date_min} → {date_max}"),
            kpi("Test RMSE", f"{rmse:.4f}" if rmse is not None else "—"),
            kpi("Test MAE", f"{mae:.4f}" if mae is not None else "—"),
            kpi("R² (Test)", f"{r2:.3f}" if r2 is not None else "—"),
            kpi("Last Close", f"{last_close:.2f}"),
            kpi("Last Dominant Sentiment", f"{last_dom_label}: {last_dom_value:.2f}"),
        ]),

        # Middle: Price + Radar
        html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr","gap":"12px","alignItems":"stretch",
                        "marginBottom":"16px"}, children=[
            html.Div(style={"background":"white","borderRadius":"12px","padding":"8px","border":"1px solid #eaeaea"},
                     children=[dcc.Graph(id="p1-price-graph", figure=build_price_fig(df, []), style={"height":"430px"})]),
            html.Div(style={"background":"white","borderRadius":"12px","padding":"8px","border":"1px solid #eaeaea"},
                     children=[dcc.Graph(id="p1-radar-graph", style={"height":"430px"})]),
        ]),

        # Bottom: Forecast + Table
        html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr","gap":"12px","alignItems":"stretch"}, children=[
            html.Div(style={"background":"white","borderRadius":"12px","padding":"8px","border":"1px solid #eaeaea"},
                     children=[dcc.Graph(id="p1-forecast-graph", figure=forecast_fig, style={"height":"420px"})]),
            html.Div(style={"background":"white","borderRadius":"12px","padding":"8px","border":"1px solid #eaeaea"},
                     children=[
                        html.Div("Last 10 Forecasts (Newest First)", style={"fontWeight":"600","marginBottom":"6px"}),
                        dash_table.DataTable(
                            id="p1-forecast-table",
                            columns=forecast_table_cols, data=forecast_table_data,
                            style_table={"height":"380px","overflowY":"auto"},
                            style_cell={"padding":"6px","fontFamily":"Inter, Arial, sans-serif"},
                            style_header={"fontWeight":"600","backgroundColor":"#f7f7f7"},
                        )
                     ]),
        ]),

        # Simple nav to Page 2
        html.Div(style={"marginTop":"18px"}, children=[
            dcc.Link("→ Go to Sentiment Overview", href="/sentiment", style={"fontWeight":600})
        ])
    ])

# Page 2
def layout_page2():
    return html.Div(style={"fontFamily":"Inter, Arial, sans-serif","padding":"16px","background":"#fafafa"}, children=[
        html.H2("Sentiment Overview", style={"marginBottom":"12px"}),

        # Back link
        html.Div([dcc.Link("← Back to Price & Radar", href="/", style={"fontWeight":600})],
                 style={"marginBottom":"12px"}),

        # Controls: date range + which sentiment series to plot
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap","margin":"10px 0"}, children=[
            html.Div([
                html.Div("Date Range", style={"fontWeight":600,"marginBottom":"4px"}),
                dcc.DatePickerRange(
                    id="p2-date-range",
                    min_date_allowed=date_min, max_date_allowed=date_max,
                    start_date=date_min, end_date=date_max, display_format="YYYY-MM-DD"
                )
            ]),
            html.Div([
                html.Div("Sentiment Series (top chart)", style={"fontWeight":600,"marginBottom":"4px"}),
                dcc.Dropdown(
                    id="p2-sentiment-series",
                    options=[{"label": f"{SENTIMENT_EMOJI[s]} {s}", "value": s} for s in sentiment_labels],
                    value=sentiment_labels, multi=True
                ),
                chips_legend()
            ], style={"minWidth":"320px"}),
        ]),

        # Top: sentiment values over time (selected series)
        html.Div(style={"background":"white","borderRadius":"12px","padding":"8px","border":"1px solid #eaeaea",
                        "marginBottom":"12px"},
                 children=[dcc.Graph(id="p2-sentiment-lines", style={"height":"420px"})]),

        # Bottom: histogram of Dominant frequency (respects date filter & series selection)
        html.Div(style={"background":"white","borderRadius":"12px","padding":"8px","border":"1px solid #eaeaea"},
                 children=[dcc.Graph(id="p2-dominant-hist", style={"height":"380px"})]),
    ])


app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-content")
])

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/sentiment":
        return layout_page2()
    return layout_page1()

# Page 1 callbacks
@app.callback(
    Output("p1-price-graph", "figure"),
    Input("p1-date-range", "start_date"),
    Input("p1-date-range", "end_date"),
    Input("p1-sentiment-filter", "value"),
)
def p1_update_price(start_date, end_date, sentiments_selected):
    window = filter_by_date_only(start_date, end_date)
    return build_price_fig(window, sentiments_selected or [])

@app.callback(
    Output("p1-radar-graph", "figure"),
    Input("p1-price-graph", "hoverData")
)
def p1_update_radar(hoverData):
    vals = [0,0,0,0,0]
    date_str = "Hover over a point"
    if not load_error and hoverData and "points" in hoverData and hoverData["points"]:
        hover_dt = _naive(hoverData["points"][0]["x"])
        row = df.loc[df["Date"] == hover_dt]
        if not row.empty:
            vals = row[sentiment_labels].iloc[0].values
            date_str = hover_dt.strftime("%Y-%m-%d")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=sentiment_labels, fill='toself',
        mode='lines+markers', line=dict(shape='spline', smoothing=1.3),
        marker=dict(size=6), name='Sentiment'
    ))
    fig.update_layout(
        title=f"Sentiment Radar — {date_str}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, margin=dict(l=40, r=10, t=50, b=40),
        transition=dict(duration=500, easing="cubic-in-out")
    )
    return fig

# Page 2 callbacks
def _filter_for_p2(start_date, end_date):
    start = _naive(start_date) if start_date is not None else df["Date"].min()
    end = _naive(end_date) if end_date is not None else df["Date"].max()
    if pd.isna(start) or pd.isna(end): start, end = df["Date"].min(), df["Date"].max()
    if end < start: start, end = end, start
    out = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    return out if not out.empty else df.tail(1).copy()

@app.callback(
    Output("p2-sentiment-lines", "figure"),
    Output("p2-dominant-hist", "figure"),
    Input("p2-date-range", "start_date"),
    Input("p2-date-range", "end_date"),
    Input("p2-sentiment-series", "value"),
)
def p2_update_figs(start_date, end_date, series_selected):
    series_selected = series_selected or []  # allow empty
    window = _filter_for_p2(start_date, end_date)

    # Top chart: sentiment lines (one trace per selected series)
    top_fig = go.Figure()
    if series_selected:
        for s in series_selected:
            top_fig.add_trace(go.Scatter(
                x=window["Date"], y=window[s], mode="lines",
                line=dict(width=2, color=SENTIMENT_COLORS.get(s, "#888")),
                name=s, hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{s}: %{{y:.3f}}<extra></extra>"
            ))
    else:
        # If nothing selected, show a faint hint line = 0 to indicate "select series"
        top_fig.add_trace(go.Scatter(
            x=window["Date"], y=[0]*len(window), mode="lines",
            line=dict(width=1, color="#cccccc", dash="dot"),
            name="Select a series", hoverinfo="skip", showlegend=False
        ))
    top_fig.update_layout(
        title="Sentiment Values Over Time",
        hovermode="x unified", margin=dict(l=40, r=10, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 1])
    )

    # Bottom chart: histogram (Dominant counts) respecting date filter and (if any) series selection
    # If series_selected provided, count only days whose Dominant is in that set (intuitive subset)
    dom_counts = window["Dominant"]
    if series_selected:
        dom_counts = dom_counts[dom_counts.isin(series_selected)]

    # Build a bar chart in a fixed sentiment order so colors match
    counts = dom_counts.value_counts()
    cats = [s for s in sentiment_labels if s in counts.index]
    dom_fig = go.Figure()
    if len(cats) > 0:
        dom_fig.add_trace(go.Bar(
            x=cats, y=[counts[c] for c in cats],
            marker=dict(color=[SENTIMENT_COLORS[c] for c in cats]),
            name="Frequency"
        ))
    else:
        dom_fig.add_trace(go.Bar(x=[], y=[]))

    dom_fig.update_layout(
        title="Dominant Sentiment Frequency (Filtered Range)",
        xaxis_title="Sentiment", yaxis_title="Count",
        margin=dict(l=40, r=10, t=50, b=40),
        showlegend=False
    )

    return top_fig, dom_fig

if __name__ == "__main__":
    # http://127.0.0.1:8060/
    app.run(host="127.0.0.1", port=8060, debug=False, use_reloader=False, threaded=False)
