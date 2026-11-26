"""ABC Bank customer drift: stakeholder dashboard (Module 5).

Run locally from the repository root:   streamlit run dashboard/app.py
Deployed on Streamlit Community Cloud with dashboard/requirements.txt.

The dashboard shows aggregated results and scores fictional example profiles.
It loads no customer-level data and never uses age, sex or nationality to score anyone.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import logic as L

NAVY, GOLD, GREEN, RED, GREY = "#1F3864", "#E8A33D", "#2E7D32", "#C0392B", "#8C96A8"

st.set_page_config(page_title="ABC Bank customer drift dashboard", page_icon="📊", layout="wide")


@st.cache_resource
def get_model(key):
    return L.load_model(key)


@st.cache_data
def get_meta(key):
    return L.load_meta(key)


@st.cache_data
def get_table(key, name):
    return L.load_table(key, name)


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13),
                      plot_bgcolor="white", legend_title_text="")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF1F6")
    return fig


# ---------- Sidebar ----------
st.sidebar.title("ABC Bank")
st.sidebar.caption("Customer drift: early warning for the reactivation offer")
key = st.sidebar.radio("Which definition of drift?", list(L.MODELS), format_func=lambda k: L.MODELS[k]["label"])
meta, model = get_meta(key), get_model(key)
H = L.headline(meta)
if H["accepted"]:
    st.sidebar.success("Meets the agreed accuracy bars (recall ≥ 0.70, AUC ≥ 0.75). Fails the fairness rules.")
else:
    st.sidebar.warning("Narrowly misses the agreed accuracy bar (AUC 0.745 < 0.75). Not approved for use.")
st.sidebar.markdown("---")
st.sidebar.caption("Audience: Head of Retail Banking and Head of Marketing.  \n"
                   "Data: Santander bank data (Kaggle, 2016) used as a stand-in for ABC Bank.  \n"
                   "BAN6800 capstone · Irene Osei Fordjour")
st.sidebar.markdown("[Code on GitHub](https://github.com/oseifordjourirene-bit/abc-bank-segmentation)")

st.title("Which active customers are about to go quiet?")
st.caption(f"{L.MODELS[key]['label']} · {H['champion']} · drift measured over 3 months")

tabs = st.tabs(["Overview", "What drives the score", "What-if explorer", "Example customers",
                "Fairness & ethics", "Limits & transparency", "Targeting & precision"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Monitoring:** open the *Monitoring* page in the menu above for monthly drift and "
                    "fairness checks.")

# ---------- 1. Overview ----------
with tabs[0]:
    st.markdown("**In plain language:** the model looks at how a customer used the bank over the last three "
                "months (products held, products added or dropped, months active, time with the bank) and gives "
                "a score from 0 to 1. Customers above the threshold would receive the reactivation offer "
                "before they drift. The score ranks customers; it is not a percentage chance.")
    c = st.columns(4)
    for col, (label, value, note) in zip(c, [
            ("Customers in the study", f"{H['customers']:,}", f"{H['drift_rate']:.1%} of them drifted"),
            ("Drifters caught", f"{H['recall']:.0%}", "agreed target: at least 70%"),
            ("Offers reaching a drifter", f"{H['precision']:.0%}", f"{H['flagged']:.0%} of customers flagged"),
            ("Caught in riskiest 20%", f"{H['top20']:.0%}", "random targeting would catch 20%")]):
        col.metric(label, value)
        col.caption(note)

    comp = get_table(key, "model_comparison").copy()
    comp["Model"] = comp["model"].map(L.PLAIN_NAMES)
    comp = comp[comp["model"] != "baseline_prior"]
    # Never round a value up onto the bar: 0.7498 is shown as 0.7498, not 0.75
    labels = comp["roc_auc"].map(lambda v: f"{v:.4f}" if L.AUC_TARGET - 0.001 < v < L.AUC_TARGET else f"{v:.3f}")
    fig = px.bar(comp, x="Model", y="roc_auc", text=labels,
                 color=comp["model"].isin(["logistic_regression", "random_forest", "lightgbm"]).map(
                     {True: "Machine-learning model", False: "Simple comparison"}),
                 color_discrete_map={"Machine-learning model": NAVY, "Simple comparison": GREY},
                 title="How well each approach ranks drifters above non-drifters (AUC, 0.5 = guessing)")
    fig.add_hline(y=L.AUC_TARGET, line_dash="dot", line_color=RED,
                  annotation_text="Agreed bar 0.75", annotation_position="top left")
    fig.update_yaxes(range=[0, 1], title="")
    fig.update_xaxes(title="")
    st.plotly_chart(style(fig))

    st.markdown("**The two definitions side by side**")
    st.dataframe(pd.DataFrame({
        "": ["What counts as drift", "AUC (bar 0.75)", "Drifters caught", "Riskiest 20% catches",
             "Do product offers lower the score?", "Fairness rules"],
        L.MODELS["main"]["short"]: [L.headline(get_meta("main"))["definition"], "0.745 (misses)", "70%", "50%",
                                    "No: every offer raises it", "Fails on age"],
        L.MODELS["inactive"]["short"]: [L.headline(get_meta("inactive"))["definition"], "0.812 (passes)", "75%",
                                        "65%", "Yes: e.g. direct debit −0.18", "Fails on age and sex"],
    }), hide_index=True)

# ---------- 2. Drivers ----------
with tabs[1]:
    st.markdown("**What the model pays most attention to** (SHAP values from Module 4: a longer bar means "
                "the factor moves the score more, up or down).")
    imp = get_table(key, "shap_importance").head(8).iloc[::-1]
    fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h", color_discrete_sequence=[NAVY],
                 title="Top factors behind the score")
    fig.update_xaxes(title="Average influence on the score")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 380))
    st.markdown("**What each top factor means for a customer**")
    for feat, note in L.DRIVER_NOTES[key].items():
        st.markdown(f"- **{feat}:** {note}")
    if key == "main":
        st.info("Because this definition counts any dropped product as drift, customers who hold or have just "
                "added products look riskier. The model partly learns the shape of the definition, not real "
                "disengagement.")

# ---------- 3. What-if ----------
with tabs[2]:
    st.markdown("Change a customer's behaviour and see how the score moves. These are fictional profiles; "
                "age, sex and nationality are not inputs.")
    personas = L.load_personas()
    start = st.selectbox("Start from", list(personas))
    p = dict(personas[start])
    a, b, c = st.columns(3)
    with a:
        p["product_count"] = st.slider("Products held", 0, 10, int(p["product_count"]))
        p["products_added_2m"] = st.slider("Products added in the last 3 months", 0, 5, int(p["products_added_2m"]))
        p["products_dropped_2m"] = st.slider("Products dropped in the last 3 months", 0, 5, int(p["products_dropped_2m"]))
    with b:
        p["months_active_3m"] = st.slider("Months active (of the last 3)", 1, 3, int(p["months_active_3m"]))
        p["tenure_months"] = st.slider("Months with the bank", 0, 300, int(p["tenure_months"]), step=6)
        p["household_income"] = st.number_input("Household income", 0, 1_000_000, int(p["household_income"]),
                                                step=5_000)
    with c:
        for label, col in [("Direct debit", "has_direct_debit"), ("Salary paid in", "has_salary"),
                           ("Card", "has_card"), ("E-account (digital banking)", "has_e_account"),
                           ("Credit product", "has_credit")]:
            p[col] = int(st.toggle(label, bool(p[col]), key=f"{start}-{col}"))
    s = L.score(model, p, meta["features"])
    g = go.Figure(go.Indicator(mode="gauge+number", value=s, number={"valueformat": ".2f"},
                               gauge={"axis": {"range": [0, 1]}, "bar": {"color": NAVY},
                                      "threshold": {"line": {"color": RED, "width": 4}, "value": meta["threshold"]},
                                      "steps": [{"range": [meta["threshold"], 1], "color": "#FBE9E7"}]},
                               title={"text": "Drift score (red line = offer threshold)"}))
    st.plotly_chart(style(g, 300))
    if s >= meta["threshold"]:
        st.warning("This customer would be flagged for the reactivation offer.")
    else:
        st.success("This customer would not be flagged.")
    t = L.what_if_table(model, p, meta["features"], meta["threshold"])
    fig = px.bar(t, x="change_in_score", y="change", orientation="h",
                 color=t["change_in_score"].map(lambda v: "Lowers the score" if v < 0 else "Raises the score"),
                 color_discrete_map={"Lowers the score": GREEN, "Raises the score": RED},
                 title="What one change would do to this customer's score")
    fig.update_xaxes(title="Change in score")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 330))
    if key == "main":
        st.info("Read with care: under the main definition, dropping a product or skipping a month can *lower* the "
                "score, and new products raise it. That is because any dropped product counts as drift, so holding "
                "more products looks risky. Switch to the comparison model in the sidebar to see behaviour that "
                "matches common sense (more activity and more products lower the score).")

# ---------- 4. Examples ----------
with tabs[3]:
    st.markdown("Three fictional customers, scored by the model, with a plain-language reading.")
    for name, prof in L.load_personas().items():
        sc = L.score(model, prof, meta["features"])
        with st.container(border=True):
            st.markdown(f"**{name}**  \nProducts {prof['product_count']} · added {prof['products_added_2m']} · "
                        f"active {prof['months_active_3m']} of 3 months · {prof['tenure_months']} months with the bank")
            st.progress(min(max(sc, 0.0), 1.0), text=f"Score {sc:.2f}")
            st.write(L.plain_explanation(model, prof, meta["features"], meta["threshold"]))
            loc = L.local_explanation(model, prof, meta["features"], L.load_reference(key))
            if len(loc):
                loc = loc.iloc[::-1]
                fig = px.bar(loc, x="contribution", y="factor", orientation="h",
                             color=loc["contribution"].gt(0).map({True: "Raises the score", False: "Lowers the score"}),
                             color_discrete_map={"Raises the score": RED, "Lowers the score": GREEN},
                             title="Why this score: top factors compared with a typical customer")
                fig.update_xaxes(title="Change in score", zeroline=True, zerolinecolor=GREY)
                fig.update_yaxes(title="")
                st.plotly_chart(style(fig, 260))
    st.caption("Local explanation: each factor is set to the typical customer's value and the change in score is "
               "shown. The live API stores the same explanation with every prediction, for audit. "
               "SHAP and LIME (Module 4 notebook) give the same leading drivers.")

# ---------- 5. Fairness ----------
with tabs[4]:
    st.markdown("**Ethical compliance check** (Fairlearn metrics from Module 4). The offer is a benefit, so "
                "the question is whether each group is offered it fairly and whether each group's drifters are "
                "caught equally often. Groups under 500 customers are shown but not judged.")
    mitigated = st.toggle("Show the result after bias mitigation (Fairlearn ExponentiatedGradient)", False)
    groups = get_table(key, "fairness_by_group_mitigated" if mitigated else "fairness_by_group")
    st.dataframe(L.fairness_summary(groups), hide_index=True)
    attr = st.selectbox("Look at", groups["attribute"].unique(),
                        format_func=lambda a: a.replace(" (H/V as coded in source)", " (codes as in source data)"))
    g = groups[groups["attribute"] == attr].copy()
    g["Judged"] = g["customers"].map(lambda n: "Judged (500+ customers)" if n >= L.MIN_GROUP else "Too small to judge")
    left, right = st.columns(2)
    with left:
        fig = px.bar(g, x="group", y="disparate_impact", color="Judged",
                     color_discrete_map={"Judged (500+ customers)": NAVY, "Too small to judge": "#B8C2D6"},
                     title="Offer rate compared with the average (1.0 = same)")
        fig.add_hrect(y0=L.DI_BAND[0], y1=L.DI_BAND[1], fillcolor=GREEN, opacity=0.08, line_width=0,
                      annotation_text="Agreed fair band 0.8–1.25", annotation_position="top left")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        st.plotly_chart(style(fig))
    with right:
        fig = px.bar(g, x="group", y="recall_tpr", color="Judged",
                     color_discrete_map={"Judged (500+ customers)": NAVY, "Too small to judge": "#B8C2D6"},
                     title="Share of each group's drifters who are caught")
        fig.update_yaxes(range=[0, 1], title="")
        fig.update_xaxes(title="")
        st.plotly_chart(style(fig))
    st.markdown("**What was found and what was done**")
    if key == "main":
        st.markdown("- Customers aged 18–24 are offered less often (0.78 of the average), partly because fewer of "
                    "them drift.\n- Mitigation brought every age group inside the band but cut the share of drifters "
                    "caught from 70% to 46%, so it was **not adopted**.\n- Only 44% of drifters aged 65+ are caught "
                    "(a small group, 359 customers): flagged for monitoring.")
    else:
        st.markdown("- Young customers really do go quiet more often (6% of 18–24s against about 2% of 35–54s), so "
                    "this model offers 18–24s 2.6 times as often as average.\n- Recall for sex code H and V differs by "
                    "0.16, outside the agreed 0.10.\n- Mitigation evened out offer rates but halved the drifters caught.")
    st.caption("Base rates: share of each group that actually drifted in the data.")
    st.dataframe(get_table(key, "data_bias").rename(columns={"base_drift_rate": "drifted", "share_pct": "% of customers"}),
                 hide_index=True)

# ---------- 6. Limits ----------
with tabs[5]:
    st.markdown("### What this model can and cannot do")
    l, r = st.columns(2)
    with l:
        st.markdown("**It can**\n- Rank active customers by how likely they are to drift in the next 3 months\n"
                    "- Catch about 7 in 10 drifters at the current threshold\n"
                    "- Show which behaviours raise or lower a customer's score")
    with r:
        st.markdown("**It cannot**\n- Give a true probability (scores are rankings)\n"
                    "- Prove that an offer will change behaviour (it shows associations, not causes)\n"
                    "- Be used for credit, pricing, account closure or any decision against a customer\n"
                    "- Speak for ABC Bank customers yet: it was trained on Spanish bank data from 2015–16")
    st.markdown("### Transparency statement")
    st.markdown("This dashboard is a decision-support prototype. Scores support, not replace, staff judgement. "
                "Age, sex, nationality and province are never used to score a customer; they are used only to "
                "check fairness. No customer records are stored or shown here: the examples are fictional and "
                "every chart uses totals. Any live use would keep customer data in Ghana, in line with the Bank of "
                "Ghana Cyber and Information Security Directive (2026) and the Data Protection Act, 2012 (Act 843).")
    st.markdown("### Points for stakeholders to weigh")
    st.markdown("- Which behaviour matters more to the business: customers going quiet, or customers dropping products?\n"
                "- How many offers per real drifter is acceptable for the campaign budget?\n"
                "- What level of difference in offer rates between age groups is acceptable, and who signs it off?")

# ---------- 7. Targeting & precision (final project) ----------
with tabs[6]:
    st.markdown("**How many customers should we contact?** Precision is low at the agreed line because only a "
                "small share of customers drift. Contacting fewer, riskier customers raises precision but misses "
                "more drifters. Test-set results (customers the model never saw):")
    pts = get_table(key, "precision_operating_points")
    summ = L.load_json(key, "precision_summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Drifters among test customers", f"{summ['base_rate']:.1%}")
    c1.caption("Precision of contacting at random")
    c2.metric("PR-AUC", f"{summ['pr_auc']:.3f}")
    c2.caption(f"{summ['pr_auc_lift']:.1f} times the random baseline")
    c3.metric("Riskiest 10%: precision", f"{pts.iloc[1]['precision']:.0%}")
    c3.caption(f"vs {pts.iloc[-1]['precision']:.0%} at the agreed threshold")
    long = pts.melt(id_vars="operating_point", value_vars=["precision", "recall"], var_name="measure")
    long["measure"] = long["measure"].map({"precision": "Precision (offers reaching a drifter)",
                                           "recall": "Recall (drifters caught)"})
    fig = px.bar(long, x="operating_point", y="value", color="measure", barmode="group", text_auto=".0%",
                 color_discrete_sequence=[GOLD, NAVY], title="Precision against recall at each contact level")
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="")
    fig.update_xaxes(title="")
    st.plotly_chart(style(fig, 380))
    show = pts.rename(columns={"operating_point": "Contact level", "share_flagged": "Share contacted",
                               "drifters_caught": "Drifters caught", "recall": "Recall", "precision": "Precision",
                               "lift_vs_random": "Lift vs random", "offers_per_drifter": "Offers per drifter"})
    show = show.drop(columns=["customers_flagged"])
    for c, f in {"Share contacted": "{:.0%}", "Recall": "{:.1%}", "Precision": "{:.1%}",
                 "Lift vs random": "{:.2f}", "Offers per drifter": "{:.1f}"}.items():
        show[c] = show[c].map(f.format)
    st.dataframe(show, hide_index=True)

    st.markdown("**Two-tier contact policy** (keeps 70% recall, spends the costly channel where precision is high)")
    tiers = get_table(key, "precision_tiers")
    tshow = tiers.rename(columns={"operating_point": "Tier", "customers_flagged": "Customers",
                                  "drifters_caught": "Drifters", "recall": "Share of all drifters",
                                  "precision": "Precision", "offers_per_drifter": "Offers per drifter"})
    tshow = tshow[["Tier", "Customers", "Drifters", "Share of all drifters", "Precision", "Offers per drifter"]]
    for c, f in {"Share of all drifters": "{:.0%}", "Precision": "{:.1%}", "Offers per drifter": "{:.1f}"}.items():
        tshow[c] = tshow[c].map(f.format)
    st.dataframe(tshow, hide_index=True)

    st.markdown("**Break-even check:** enter your own campaign figures. Contacting pays when precision × "
                "save rate × value of a kept customer is greater than the cost of one contact.")
    b1, b2, b3 = st.columns(3)
    cost = b1.number_input("Cost per contact (GHS)", min_value=0.0, value=10.0, step=5.0)
    value = b2.number_input("Value of a kept customer (GHS)", min_value=1.0, value=500.0, step=50.0)
    save = b3.slider("Share of contacted drifters who stay", 0.01, 1.0, 0.20, 0.01)
    be = L.break_even_precision(cost, value, save)
    ok = tiers.assign(pays=tiers["precision"] >= be)
    st.info(f"Break-even precision: **{be:.1%}**. " + " ".join(
        f"{r.operating_point}: {'pays' if r.pays else 'does not pay'} ({r.precision:.1%})." for r in ok.itertuples()))
    st.caption("Default figures are placeholders for illustration, not ABC Bank data: replace them with "
               "Marketing's own costs and values.")
