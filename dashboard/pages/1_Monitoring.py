"""Monitoring dashboard: data drift, prediction drift, fairness drift and performance.

Reads the aggregated tables written by `python -m src.monitoring.monitor` (reports/monitoring/).
No customer-level data is loaded.
"""
import json

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import logic as L

NAVY, GOLD, GREEN, RED, GREY, AMBER = "#1F3864", "#E8A33D", "#2E7D32", "#C0392B", "#8C96A8", "#B7791F"
st.set_page_config(page_title="ABC Bank model monitoring", page_icon="📈", layout="wide")


def style(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13),
                      plot_bgcolor="white", legend_title_text="")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF1F6")
    return fig


st.title("Model monitoring")
summary = json.loads((L.MONITORING_DIR / "summary.json").read_text(encoding="utf-8"))
st.caption(f"Last run {summary['run_at']} · reference: training cohort {summary['reference_cohort']} · "
           "each month's active customers scored with the deployed models")

model_key = st.radio("Model", ["main", "inactive_only"], horizontal=True,
                     format_func=lambda k: {"main": "Main model (inactive or dropping)",
                                            "inactive_only": "Comparison model (inactive only)"}[k])
alerts = L.load_monitoring("alerts")
alerts = alerts[alerts["model"] == model_key]
pred = L.load_monitoring("prediction_drift").query("model == @model_key")
drift = L.load_monitoring("data_drift").query("model == @model_key")
fair = L.load_monitoring("fairness_drift").query("model == @model_key")
perf = L.load_monitoring("performance").query("model == @model_key")

real_months = [m for m in pred["month"] if m != "Simulated stress"]
latest = real_months[-1]
st.subheader(f"Status for the latest month ({latest})")
icons = {"ok": "🟢 OK", "alert": "🔴 Alert"}
cols = st.columns(4)
for col, (check, state) in zip(cols, L.monitoring_status(alerts, latest).items()):
    col.metric(check.capitalize(), icons[state])
col = cols[3]
col.metric("Performance", "⏳ Labels due")
col.caption("Outcomes for a month's cohort arrive 3 months later")

st.markdown("---")
a, b = st.columns(2)
with a:
    fig = px.line(pred, x="month", y="flagged_rate", markers=True, title="Prediction drift: share of customers flagged")
    fig.update_traces(line_color=NAVY)
    ref = pred.iloc[0]["flagged_rate"]
    fig.add_hrect(y0=ref - 0.05, y1=ref + 0.05, fillcolor=GREEN, opacity=0.08, line_width=0,
                  annotation_text="±5 points of training", annotation_position="top left")
    fig.update_yaxes(tickformat=".0%", title="", range=[0, max(0.7, pred["flagged_rate"].max() + 0.05)])
    fig.update_xaxes(title="")
    st.plotly_chart(style(fig))
with b:
    heat = drift.pivot_table(index="feature", columns="month", values="psi")
    heat = heat.loc[heat.max(axis=1).sort_values(ascending=False).index[:8]]
    fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=[L.FEATURE_NAMES.get(f, f) for f in heat.index],
                               colorscale=[[0, "#EEF2F8"], [0.4, GOLD], [1, RED]], zmin=0, zmax=0.5,
                               text=heat.round(2).values, texttemplate="%{text}",
                               colorbar=dict(title="PSI")))
    fig.update_layout(title="Data drift: PSI by feature (watch 0.10, alert 0.25)")
    st.plotly_chart(style(fig))

st.subheader("Fairness drift: offer rate by age band, compared with the overall rate")
age = fair[(fair["attribute"] == "age_band") & (fair["judged"])]
fig = px.line(age, x="month", y="ratio_to_overall", color="group", markers=True)
fig.add_hrect(y0=summary["fairness_band"][0], y1=summary["fairness_band"][1], fillcolor=GREEN, opacity=0.08,
              line_width=0, annotation_text="Agreed band 0.8–1.25", annotation_position="top left")
fig.update_yaxes(title="Offer rate ÷ overall rate", rangemode="tozero")
fig.update_xaxes(title="")
st.plotly_chart(style(fig, 380))
st.caption("Groups with fewer than 500 customers are reported in the table below but not judged.")

st.subheader("Alerts")
if alerts.empty:
    st.success("No alerts.")
else:
    st.dataframe(alerts[["month", "type", "detail"]], hide_index=True)
st.caption("'Simulated stress' is a labelled test cohort (income -20%, one fewer active month for 30% of "
           "customers) that shows the alerts working. It is not real data.")

st.subheader("Performance once outcomes are known")
st.dataframe(perf.drop(columns=["model"]), hide_index=True)

with st.expander("What happens when an alert fires"):
    st.markdown("- **Data or prediction drift:** the monthly flow retrains a candidate model; it is promoted only "
                "if it passes the recall, AUC and fairness gates *and* a named approver signs off.\n"
                "- **Fairness:** an incident is opened for Digital Governance and Risk within 2 working days; "
                "the affected tier can be paused while it is reviewed.\n"
                "- Full plan: docs/model_governance.md in the GitHub repository.")
with st.expander("All fairness figures"):
    st.dataframe(fair.drop(columns=["model"]), hide_index=True)
