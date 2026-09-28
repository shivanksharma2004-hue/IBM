"""
app.py
Employee Retention & Pay Check: a Streamlit app for non-technical managers.

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy on share.streamlit.io
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    EDUCATION, LABELS, LINEAR_CAT, LINEAR_NUM, LOGIT_CAT, LOGIT_NUM, SCALE_1_4, WLB_1_4,
    coef_table, explain_logit, linear_metrics, load_data, logit_metrics, risk_band,
    train_models, what_if,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Employee Retention & Pay Check", page_icon="📋", layout="wide")

GREEN, AMBER, RED, INK, GREY = "#2F6F5E", "#C98A1B", "#B23A3A", "#1C2430", "#8A94A3"
BAND_COLOR = {"Low": GREEN, "Medium": AMBER, "High": RED}

# Paste your Google Form link here before sharing the app
FEEDBACK_FORM_URL = "https://forms.gle/REPLACE_WITH_YOUR_FORM"

st.markdown(
    f"""
    <style>
    .block-container {{max-width: 1150px; padding-top: 2rem;}}
    .risk-number {{font-size: 3.2rem; font-weight: 700; line-height: 1;}}
    .risk-sub {{color: {GREY}; font-size: 0.95rem; margin-top: 0.3rem;}}
    .note {{color: {GREY}; font-size: 0.88rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_models():
    df = load_data()
    return df, train_models(df)


df, M = get_models()
logit, linear = M["logit"], M["linear"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this tool")
    st.write(
        "Two models built on 1,470 employee records. One estimates how likely an "
        "employee is to leave. The other estimates what their monthly pay should be "
        "for their role and experience."
    )
    threshold = st.slider(
        "Flag an employee as 'likely to leave' when risk is at least",
        min_value=0.20, max_value=0.80, value=0.50, step=0.05,
        help="Lower it to catch more leavers (more false alarms). Raise it for fewer, surer flags.",
    )
    st.divider()
    st.write("**Used the tool?** Share your views in a 5-minute form.")
    st.link_button("Give feedback", FEEDBACK_FORM_URL, width="stretch")
    st.divider()
    st.caption(
        "Data: IBM HR Analytics Employee Attrition dataset (Kaggle). "
        "Built for Assignment 2, Data Mining and Machine Learning, IRMA."
    )

st.title("Employee Retention & Pay Check")
st.write(
    "Enter an employee's details to see how likely they are to leave, what is driving that "
    "risk, and whether their pay is in line with similar roles."
)

tab_risk, tab_pay, tab_team, tab_model = st.tabs(
    ["Attrition risk", "Pay benchmark", "Score a whole team", "How the models work"]
)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def money(x: float) -> str:
    return f"{x:,.0f}"


# ---------------------------------------------------------------------------
# Tab 1: Attrition risk (logistic regression)
# ---------------------------------------------------------------------------
with tab_risk:
    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Profile")
            age = st.number_input("Age", 18, 60, 32)
            dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
            distance = st.number_input("Distance from home (km)", 1, 29, 8)
            total_yrs = st.number_input("Total work experience (years)", 0, 40, 8)
            n_comp = st.number_input("Number of past employers", 0, 9, 2)
        with c2:
            st.subheader("Job")
            income = st.number_input("Monthly income", 1000, 20000, 5000, step=250)
            overtime = st.radio("Works overtime regularly?", ["No", "Yes"], horizontal=True)
            travel = st.selectbox(
                "Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                format_func=lambda s: {"Non-Travel": "None", "Travel_Rarely": "Occasional",
                                       "Travel_Frequently": "Frequent"}[s],
            )
            stock = st.select_slider("Stock option level", [0, 1, 2, 3], value=1)
            yrs_mgr = st.number_input("Years with current manager", 0, 17, 3)
            yrs_promo = st.number_input("Years since last promotion", 0, 15, 1)
            training = st.number_input("Trainings attended last year", 0, 6, 3)
        with c3:
            st.subheader("Engagement (from survey)")
            env = st.select_slider("Work environment satisfaction", [1, 2, 3, 4], 3, format_func=SCALE_1_4.get)
            jobsat = st.select_slider("Job satisfaction", [1, 2, 3, 4], 3, format_func=SCALE_1_4.get)
            relsat = st.select_slider("Relationship with colleagues", [1, 2, 3, 4], 3, format_func=SCALE_1_4.get)
            wlb = st.select_slider("Work-life balance", [1, 2, 3, 4], 3, format_func=WLB_1_4.get)
            involve = st.select_slider("Job involvement", [1, 2, 3, 4], 3, format_func=SCALE_1_4.get)
        submitted = st.form_submit_button("Predict attrition risk", type="primary")

    row = pd.DataFrame([{
        "Age": age, "DistanceFromHome": distance, "MonthlyIncome": income,
        "NumCompaniesWorked": n_comp, "TotalWorkingYears": total_yrs,
        "YearsWithCurrManager": yrs_mgr, "YearsSinceLastPromotion": yrs_promo,
        "TrainingTimesLastYear": training, "StockOptionLevel": stock,
        "EnvironmentSatisfaction": env, "JobSatisfaction": jobsat,
        "RelationshipSatisfaction": relsat, "WorkLifeBalance": wlb, "JobInvolvement": involve,
        "OverTime": overtime, "BusinessTravel": travel, "MaritalStatus": marital, "Department": dept,
    }])[LOGIT_NUM + LOGIT_CAT]

    p = float(logit.predict_proba(row)[:, 1][0])
    band = risk_band(p)
    flagged = p >= threshold

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"<div class='risk-number' style='color:{BAND_COLOR[band]}'>{pct(p)}</div>"
            f"<div class='risk-sub'>{band} risk of leaving. "
            f"{'Flagged' if flagged else 'Not flagged'} at your {pct(threshold)} cut-off.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"<div class='note'>In the data, {pct(M['base_rate'])} of employees left. The model "
            "gives extra weight to leavers so it does not miss them, which pushes all scores up. "
            "Read the score as a ranking: higher means more likely to leave than a colleague with "
            "a lower score.</div>",
            unsafe_allow_html=True,
        )

    with right:
        contrib = explain_logit(logit, row).head(8).iloc[::-1]
        colors = [RED if v > 0 else GREEN for v in contrib["contribution"]]
        fig = go.Figure(go.Bar(
            x=contrib["contribution"], y=contrib["label"], orientation="h", marker_color=colors,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
        fig.update_layout(
            title="What is pushing this score up (red) or down (green)",
            height=340, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Effect on risk compared with an average employee",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("What you could change")
    wi = what_if(logit, row)
    wi_show = wi.assign(**{
        "Risk now": wi["Risk now"].map(pct),
        "Risk after": wi["Risk after"].map(pct),
        "Change": (wi["Change"] * 100).round().map(lambda v: f"{v + 0:+.0f} points"),
    })
    st.dataframe(wi_show, hide_index=True, width="stretch")
    st.caption(
        "Each row changes one thing and keeps everything else the same. These are statistical "
        "associations from past data, not guaranteed effects."
    )

# ---------------------------------------------------------------------------
# Tab 2: Pay benchmark (linear regression)
# ---------------------------------------------------------------------------
with tab_pay:
    roles = sorted(df["JobRole"].unique())
    level_range = df.groupby("JobRole")["JobLevel"].agg(["min", "max"])

    with st.form("pay_form"):
        c1, c2 = st.columns(2)
        with c1:
            role = st.selectbox("Job role", roles, index=roles.index("Sales Executive"))
            lo, hi = int(level_range.loc[role, "min"]), int(level_range.loc[role, "max"])
            level = st.number_input(f"Job level (seen in data for this role: {lo} to {hi})", 1, 5, max(lo, 2))
            p_total = st.number_input("Total work experience (years)", 0, 40, 8, key="p_total")
            p_years = st.number_input("Years at this company", 0, 40, 4)
        with c2:
            edu = st.select_slider("Education", [1, 2, 3, 4, 5], 3, format_func=EDUCATION.get)
            perf = st.radio("Performance rating", [3, 4], horizontal=True,
                            format_func=lambda v: {3: "3 - Excellent", 4: "4 - Outstanding"}[v])
            actual = st.number_input("Actual monthly income (optional, 0 to skip)", 0, 25000, 0, step=250)
        pay_go = st.form_submit_button("Estimate fair pay", type="primary")

    prow = pd.DataFrame([{
        "JobLevel": level, "TotalWorkingYears": p_total, "YearsAtCompany": p_years,
        "Education": edu, "PerformanceRating": perf, "JobRole": role,
    }])[LINEAR_NUM + LINEAR_CAT]
    expected = float(linear.predict(prow)[0])
    band_lo, band_hi = expected - 1.28 * M["resid_sd"], expected + 1.28 * M["resid_sd"]

    if level < lo or level > hi:
        st.warning(
            f"No {role} in the data sits at job level {level}. The estimate goes beyond "
            "what the model has seen, so treat it with caution."
        )

    a, b, c = st.columns(3)
    a.metric("Expected monthly income", money(expected))
    b.metric("Typical range (80% of similar staff)", f"{money(max(band_lo, 1000))} to {money(band_hi)}")
    if actual > 0:
        gap = (actual - expected) / expected
        c.metric("Actual vs expected", money(actual), f"{gap * 100:+.0f}%")
        if gap <= -0.15:
            st.error(
                f"This person earns {abs(gap) * 100:.0f}% below the benchmark. Low pay relative to "
                "peers raises attrition risk in the other model. Check it in the Attrition risk tab."
            )
        elif gap >= 0.15:
            st.info(f"This person earns {gap * 100:.0f}% above the benchmark.")
        else:
            st.success("Pay is within 15% of the benchmark.")

    st.subheader("How pay is built up in this model")
    ct = coef_table(linear)
    lvl = ct.loc[ct.feature == "JobLevel", "coefficient"].item()
    exp_c = ct.loc[ct.feature == "TotalWorkingYears", "coefficient"].item()
    st.write(
        f"Each step up in job level adds about **{money(lvl)}** a month. Each extra year of "
        f"total experience adds about **{money(exp_c)}**. Role matters too: Managers and "
        "Research Directors earn about 4,000 a month more than a Healthcare Representative at the "
        "same level. Education and performance rating barely move pay in this data."
    )

# ---------------------------------------------------------------------------
# Tab 3: Score a team from a CSV
# ---------------------------------------------------------------------------
with tab_team:
    st.write(
        "Upload a CSV of your team to rank everyone by attrition risk and check each person's "
        "pay against the benchmark. Download the template to see the columns needed."
    )
    template_cols = ["EmployeeNumber"] + LOGIT_NUM + LOGIT_CAT + ["JobLevel", "YearsAtCompany",
                                                                  "Education", "PerformanceRating", "JobRole"]
    template = df.sample(15, random_state=7)[list(dict.fromkeys(template_cols))]
    st.download_button("Download template (15 sample employees)", template.to_csv(index=False),
                       "team_template.csv", "text/csv")

    up = st.file_uploader("Upload your team CSV", type="csv")
    team = pd.read_csv(up) if up else None
    if team is not None:
        need = set(LOGIT_NUM + LOGIT_CAT + LINEAR_NUM + LINEAR_CAT)
        missing = sorted(need - set(team.columns))
        if missing:
            st.error("The file is missing these columns: " + ", ".join(missing))
        else:
            out = team.copy()
            out["Attrition risk"] = logit.predict_proba(team[LOGIT_NUM + LOGIT_CAT])[:, 1]
            out["Risk band"] = out["Attrition risk"].map(risk_band)
            out["Expected income"] = linear.predict(team[LINEAR_NUM + LINEAR_CAT]).round(0)
            out["Pay gap %"] = ((out["MonthlyIncome"] - out["Expected income"]) / out["Expected income"] * 100).round(0)
            out = out.sort_values("Attrition risk", ascending=False)
            n_flag = int((out["Attrition risk"] >= threshold).sum())
            st.write(f"**{n_flag} of {len(out)}** employees are flagged at your {pct(threshold)} cut-off.")
            show_cols = [c for c in ["EmployeeNumber", "JobRole", "Department"] if c in out.columns]
            show = out[show_cols + ["Attrition risk", "Risk band", "OverTime", "YearsSinceLastPromotion",
                                    "MonthlyIncome", "Expected income", "Pay gap %"]]
            st.dataframe(show.style.format({"Attrition risk": "{:.0%}", "MonthlyIncome": "{:,.0f}",
                                            "Expected income": "{:,.0f}", "Pay gap %": "{:+.0f}"}),
                         hide_index=True, width="stretch")
            st.download_button("Download results", out.to_csv(index=False), "team_scored.csv", "text/csv")

# ---------------------------------------------------------------------------
# Tab 4: Model performance and limits
# ---------------------------------------------------------------------------
with tab_model:
    Xte, yte, p_test = M["logit_test"]
    lm = logit_metrics(yte, p_test, threshold)
    tn, fp, fn, tp = lm["cm"]

    st.subheader("Attrition model: logistic regression")
    st.write(
        f"Tested on {len(yte)} employees the model never saw during training "
        f"({int(yte.sum())} of them left)."
    )
    k = st.columns(5)
    for col, name in zip(k, ["Accuracy", "ROC AUC", "Recall (leavers caught)",
                             "Precision (flags that were right)", "F1 score"]):
        col.metric(name, f"{lm[name]:.2f}")
    st.write(
        f"At your {pct(threshold)} cut-off the model caught **{tp} of {tp + fn}** leavers "
        f"and raised **{fp}** false alarms among {tn + fp} people who stayed."
    )

    c1, c2 = st.columns(2)
    with c1:
        cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]], x=["Predicted: stays", "Predicted: leaves"],
            y=["Actually stayed", "Actually left"], colorscale=[[0, "#EEF3F1"], [1, GREEN]],
            text=[[tn, fp], [fn, tp]], texttemplate="%{text}", showscale=False,
        ))
        cm.update_layout(title="Confusion matrix", height=320, margin=dict(l=10, r=10, t=40, b=10),
                         yaxis_autorange="reversed")
        st.plotly_chart(cm, width="stretch")
    with c2:
        fpr, tpr = lm["roc"]
        roc = go.Figure()
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=GREEN, width=3),
                                 name=f"Model (AUC {lm['ROC AUC']:.2f})"))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"),
                                 name="Random guess"))
        roc.update_layout(title="ROC curve", height=320, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="False alarm rate", yaxis_title="Leavers caught",
                          legend=dict(x=0.45, y=0.1))
        st.plotly_chart(roc, width="stretch")

    lct = coef_table(logit)
    lct["Odds ratio"] = np.exp(lct["coefficient"])
    lct = lct.sort_values("coefficient", ascending=False)
    st.write("**Coefficients.** Numeric inputs are standardised, so an odds ratio of 1.5 means one "
             "standard deviation more of that input multiplies the odds of leaving by 1.5. "
             "For yes/no inputs it compares with the baseline group (no overtime, no travel, "
             "divorced, HR department).")
    st.dataframe(lct[["label", "coefficient", "Odds ratio"]].rename(columns={"label": "Input"})
                 .style.format({"coefficient": "{:+.2f}", "Odds ratio": "{:.2f}"}),
                 hide_index=True, width="stretch")

    st.divider()
    st.subheader("Pay model: linear regression")
    Xlte, ylte, yl_pred = M["linear_test"]
    lin = linear_metrics(ylte.values, yl_pred)
    k = st.columns(4)
    k[0].metric("R squared", f"{lin['R squared']:.2f}")
    k[1].metric("RMSE", money(lin["RMSE"]))
    k[2].metric("MAE", money(lin["MAE"]))
    k[3].metric("MAPE", f"{lin['MAPE'] * 100:.0f}%")
    st.write(
        f"The model explains {lin['R squared'] * 100:.0f}% of the variation in monthly income. "
        f"On average its estimate is off by {money(lin['MAE'])} a month."
    )
    sc = go.Figure(go.Scatter(x=ylte, y=yl_pred, mode="markers",
                              marker=dict(color=GREEN, opacity=0.55, size=6)))
    mx = float(max(ylte.max(), yl_pred.max()))
    sc.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=GREY, dash="dash")))
    sc.update_layout(title="Predicted vs actual monthly income (test set)", height=380, showlegend=False,
                     xaxis_title="Actual", yaxis_title="Predicted", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(sc, width="stretch")

    st.divider()
    st.subheader("Limits you should know")
    st.markdown(
        "- The data is a sample dataset published by IBM, not records from one real company. "
        "Patterns may differ in your organisation.\n"
        "- Both models show associations, not causes. Removing overtime is linked to lower "
        "attrition in the data; that does not prove removing it will keep a specific person.\n"
        "- The attrition model misses some leavers and flags some stayers. Use the score to "
        "decide whom to talk to first, not to make decisions about a person on its own.\n"
        "- Satisfaction scores come from employee surveys. If your surveys are out of date, "
        "the score will be too."
    )
