
"""
SoundLife Cohort — EDA Dashboard
Run with:  streamlit run app.py
Place in:  ~/Desktop/Capstone Data/app.py
"""
import os
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, spearmanr

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
HF_REPO  = "s294710/soundlife-capstone-data"
HF_TOKEN = st.secrets.get("HF_TOKEN")

def hf(path):
    return hf_hub_download(
        repo_id   = HF_REPO,
        filename  = path,
        repo_type = "dataset",
        token     = HF_TOKEN,
    )

PADJ_SIG = 0.05
LFC_THR  = 1.0
CMV_COL  = "cmv.igg_serology_interpretation"

DESEQ_FILES = {
    "Age Group":      ("sound-life_age-group_deseq2-results_2025-02-07.csv",
                       "BR1 Younger Adult", "BR2 Older Adult",  "#4C72B0"),
    "Biological Sex": ("sound-life_biological-sex_deseq2-results_2025-02-07.csv",
                       "Female",            "Male",              "#DD8452"),
    "CMV Status":     ("sound-life_cmv-status_deseq2-results_2025-02-07.csv",
                       "Negative",          "Positive",          "#55A868"),
    "Flu Vaccine":    ("sound-life_flu-vaccine_deseq2-results_2025-02-07.csv",
                       "Flu Year 1 Day 0",  "Flu Year 1 Day 7", "#C44E52"),
}

AGE_COLORS = {"Young Adult": "#2E86AB", "Older Adult": "#E84855"}
CMV_COLORS = {"Positive": "#F4A261",   "Negative":   "#457B9D",
              "Equivocal": "#aaaaaa",   "Unknown":    "#cccccc"}
SEX_COLORS = {"Male":     "#6A994E",   "Female":     "#BC4749"}

SELECTED_PROTEINS = ["IL6","CXCL8","IL18","TNF","CXCL10","CCL2",
                     "GZMA","GZMB","KLRD1","CD244","TNFRSF13B",
                     "CXCL13","CD163","CCL7","VEGFA","MMP12","GDF15","CXCL9"]

KEY_VARS = ["am.bmi","infl.hs_crp","infl.esr",
            "bc.lymphocyte_count","bc.neutrophil_count","bc.wbc",
            "lip.cholesterol_ldl","lip.triglycerides"]

LAB_GROUPS = {
    "Anthropometric": ["am.bmi","am.height","am.weight"],
    "Blood Counts":   ["bc.wbc","bc.neutrophil_count","bc.lymphocyte_count",
                       "bc.monocyte_count","bc.eosinophil_count",
                       "bc.hemoglobin","bc.platelet_count"],
    "Chemistry":      ["chem.glucose","chem.creatinine","chem.bun",
                       "chem.alt","chem.ast","chem.albumin",
                       "chem.sodium","chem.potassium","chem.calcium"],
    "Lipids":         ["lip.cholesterol_total","lip.cholesterol_ldl",
                       "lip.cholesterol_hdl","lip.triglycerides"],
    "Inflammation":   ["infl.hs_crp","infl.esr"],
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SoundLife EDA", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* sidebar */
    [data-testid="stSidebar"] { background: #0f1923; }
    [data-testid="stSidebar"] * { color: #c9d6e3 !important; }
    /* metric cards */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #0f172a !important; font-weight: 700; }
    /* section cards */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    /* insight boxes */
    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #2E86AB;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #1e3a5f;
    }
    .warning-box {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #7c2d12;
    }
    h1 { color: #0f172a; font-weight: 700; }
    h2 { color: #1e3a5f; font-weight: 600; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }
    h3 { color: #334155; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-weight: 500; }
</style>
""", unsafe_allow_html=True)

def insight(text): st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)
def warning(text): st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clinical():
    df = pd.read_csv(hf("clinical/sound_life_labs_metadata.csv"))
    bl = pd.read_csv(hf("clinical/clinical_baseline_wide.csv"))
    return df, bl

@st.cache_data(show_spinner=False)
def load_serology():
    sero = pd.read_csv(hf("serology/sound-life_flu_serology_single.csv"))
    hai  = pd.read_csv(hf("serology/sound-life_flu_hai_single.csv"))
    return sero, hai

@st.cache_data(show_spinner=False)
def load_plasma():
    return pd.read_csv(hf("plasma/sound_life_all_olink.csv"))

@st.cache_data(show_spinner=False)
def load_deseq():
    out = {}
    name_to_file = {
        "Age Group":      "deseq/sound-life_age-group_deseq2-results_2025-02-07.csv",
        "Biological Sex": "deseq/sound-life_biological-sex_deseq2-results_2025-02-07.csv",
        "CMV Status":     "deseq/sound-life_cmv-status_deseq2-results_2025-02-07.csv",
        "Flu Vaccine":    "deseq/sound-life_flu-vaccine_deseq2-results_2025-02-07.csv",
    }
    for name, (_, bg, fg, color) in DESEQ_FILES.items():
        df = pd.read_csv(hf(name_to_file[name]))
        for c in ("padj","pvalue","log2fc","stat"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["sig"]        = df["padj"] < PADJ_SIG
        df["sig_strong"] = df["sig"] & (df["log2fc"].abs() >= LFC_THR)
        out[name] = dict(df=df, bg=bg, fg=fg, color=color)
    return out

@st.cache_data(show_spinner=False)
def compute_icc(plasma):
    df = (plasma[plasma["olink.NPX_norm"].notna()]
          .groupby("olink.assay").filter(lambda x: len(x) > 10)
          .copy())
    df["subj_mean"] = df.groupby(["olink.assay","subject.subjectGuid"])["olink.NPX_norm"].transform("mean")
    icc = (df.groupby("olink.assay")
           .apply(lambda g: pd.Series({
               "var_between": g["subj_mean"].var(),
               "var_within":  ((g["olink.NPX_norm"] - g["subj_mean"])**2).mean(),
               "n_subjects":  g["subject.subjectGuid"].nunique(),
           })).reset_index())
    icc["icc"] = icc["var_between"] / (icc["var_between"] + icc["var_within"])
    return icc

@st.cache_data(show_spinner=False)
def compute_fold_change(sero):
    d0 = (sero[sero["sample.daysSinceFirstVisit"]==0]
          [["subject.subjectGuid","subject.ageGroup","subject.cmv",
            "msd.antigenName","msd.concMean"]]
          .rename(columns={"msd.concMean":"d0"}))
    d7 = (sero[sero["sample.daysSinceFirstVisit"].isin([6,7,8,9])]
          .sort_values("sample.daysSinceFirstVisit")
          .groupby(["subject.subjectGuid","msd.antigenName"]).first()
          .reset_index()[["subject.subjectGuid","msd.antigenName","msd.concMean"]]
          .rename(columns={"msd.concMean":"d7"}))
    fc = d0.merge(d7, on=["subject.subjectGuid","msd.antigenName"])
    fc["fold_change"] = fc["d7"] / fc["d0"].replace(0, np.nan)
    fc["log2_fc"]     = np.log2(fc["fold_change"].replace(0, np.nan))
    fc["seroconverted"] = fc["fold_change"] >= 4
    return fc.dropna(subset=["fold_change"])


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
def page_overview():
    st.title("SoundLife Cohort — Exploratory Data Analysis")
    st.markdown(
        "This dashboard presents the full EDA for the SoundLife healthy-cohort capstone. "
        "The study followed **96 subjects** — 49 Young Adults (ages 25–35) and "
        "47 Older Adults (ages 55–65) — from the Seattle area over multiple flu seasons, "
        "profiling the immune system through clinical labs, plasma proteomics, "
        "whole blood RNA-seq, single-cell transcriptomics, and flu serology."
    )

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Subjects",          "96")
    c2.metric("Young Adults",      "49")
    c3.metric("Older Adults",      "47")
    c4.metric("Female / Male",     "55 / 41")
    c5.metric("CMV+ subjects",     "44")
    c6.metric("Total lab visits",  "868")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Dashboard sections")
        st.markdown("""
| Section | Source | Focus |
|---|---|---|
| **Cohort Overview** | Clinical metadata | Demographics, visit structure, retention |
| **Clinical Labs** | `sound_life_labs_metadata.csv` | Lab distributions, Mann-Whitney tests, Spearman correlations |
| **Flu Serology & HAI** | Serology + HAI CSVs | IgG kinetics, HAI inhibition, fold change, seroconversion rates |
| **Plasma Proteomics** | `sound_life_all_olink.csv` | LOD landscape, batch effects, ICC stability, protein distributions |
| **Differential Expression** | 4 DESeq2 CSVs | Volcano plots, cell-type landscape, signed direction charts |
        """)

    with col_r:
        st.subheader("Key findings")
        st.markdown("""
**Cohort structure**
- CMV seropositivity is substantially higher in Older Adults (54% vs 36%), creating a confound that must be modelled explicitly.
- 95 of 96 subjects have 2+ visits; 80 completed all three event blocks (Flu Y1, Flu Y2, Immune Variation).
- Young Adults had higher dropout; CMV-positive young women had the lowest completion rate (2/9).

**Clinical labs**
- Neutrophil count is significantly higher in Older Adults; lymphocyte count is significantly lower — consistent with age-related myeloid skewing.
- CRP and ESR show modest but significant elevation with age, reflecting increased baseline inflammatory tone.

**Serology**
- IgG and HAI responses peak sharply at Day 7 and remain elevated at Day 90.
- High Day 0 titers suppress fold-change in many subjects, particularly for H1N1 strains with high pre-existing immunity in older cohorts.

**Proteomics**
- 217 of 867 proteins are below LOD in more than 30% of samples.
- Panel-wide median ICC is 0.65; only 30% of proteins exceed ICC 0.75.
- Among the 18 BN-selected proteins, 6 proteins (TNF, GZMB, CXCL13, CXCL10, CXCL9, GZMA) fall below ICC 0.60.

**Differential expression**
- Biological Sex produces the largest DEG signal (positive control via sex-chromosome genes).
- CMV Status and Age Group show distributed signals across multiple cell types.
        """)

    st.divider()
    st.subheader("EDA-to-model implications")
    st.markdown("""
| Finding | Modelling implication |
|---|---|
| CMV confounds age effects | CMV must be a discrete root node in the Bayesian Network |
| Visits are event-clustered, not calendar-spaced | Time variable = day within event (0/7/90), not raw days since enrollment |
| 82 stand-alone samples have no event context | Exclude from BN training data |
| 217 proteins frequently below LOD | Decision needed: exclude or model as censored observations |
| Protein ICC varies widely | Stable proteins (ICC > 0.75) need per-person baseline adjustment; state-like proteins need time-sensitive modelling |
| 835 of 838 RNA samples have matching plasma | Joint multimodal model is feasible with minimal data loss |
    """)


# ===========================================================================
# PAGE: COHORT OVERVIEW
# ===========================================================================
def page_cohort():
    st.title("Cohort Overview")
    df, bl = load_clinical()
    subjects = df.drop_duplicates("subject.subjectGuid").copy()
    subjects["cmv_status"] = subjects[CMV_COL].str.strip().str.title() \
        if CMV_COL in subjects.columns else "Unknown"

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total subjects",  subjects["subject.subjectGuid"].nunique())
    c2.metric("Total samples",   df["sample.sampleKitGuid"].nunique())
    c3.metric("Young Adults",    int((subjects["subject.ageGroup"]=="Young Adult").sum()))
    c4.metric("Older Adults",    int((subjects["subject.ageGroup"]=="Older Adult").sum()))

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Demographics", "Visit structure", "CMV and age confounding", "Retention and dropout"])

    # ── Demographics ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Cohort demographics at enrollment")
        st.markdown("""
**What these charts show**
- Pie charts break down the 96 subjects by biological sex, age group, and CMV serostatus.
- The bar chart shows racial composition of the cohort.

**Key observations**
- Sex is roughly balanced (55 female, 41 male) and evenly distributed across age groups.
- The cohort is predominantly Caucasian (83/96), which limits generalisability to other populations.
- CMV seropositivity is approximately 46%, consistent with US adult prevalence.
        """)
        col1,col2,col3 = st.columns(3)
        with col1:
            sc = subjects["subject.biologicalSex"].value_counts().reset_index()
            sc.columns = ["Sex","Count"]
            fig = px.pie(sc, names="Sex", values="Count", color="Sex",
                         color_discrete_map=SEX_COLORS, title="Biological Sex",
                         hole=0.35)
            fig.update_layout(height=300, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ac = subjects["subject.ageGroup"].value_counts().reset_index()
            ac.columns = ["Age Group","Count"]
            fig = px.pie(ac, names="Age Group", values="Count", color="Age Group",
                         color_discrete_map=AGE_COLORS, title="Age Group", hole=0.35)
            fig.update_layout(height=300, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            cc = subjects["cmv_status"].value_counts().reset_index()
            cc.columns = ["CMV","Count"]
            fig = px.pie(cc, names="CMV", values="Count", color="CMV",
                         color_discrete_map=CMV_COLORS, title="CMV Serostatus", hole=0.35)
            fig.update_layout(height=300, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        rc = subjects["subject.race"].value_counts().reset_index()
        rc.columns = ["Race","Count"]
        fig = px.bar(rc.sort_values("Count"), x="Count", y="Race",
                     orientation="h", title="Race distribution",
                     color_discrete_sequence=["#4C72B0"])
        fig.update_layout(height=280, margin=dict(t=40,l=10))
        st.plotly_chart(fig, use_container_width=True)
        insight("The cohort is 86% Caucasian, reflecting recruitment from the Seattle area. "
                "Findings may not generalise to more diverse populations.")

    # ── Visit structure ───────────────────────────────────────────────────────
    with tab2:
        st.subheader("Study visit structure")
        st.markdown("""
**What these charts show**
- The bar chart shows how many samples were collected at each named visit type across all subjects.
- The histogram shows the full distribution of samples across time (days since first visit).

**Key observations**
- Samples cluster into three temporal bursts: Days 0–100 (Flu Year 1), Days 150–400 (Flu Year 2 and Immune Variation), and a sparse tail beyond Day 400.
- The ~6-month gaps between clusters reflect the structured spacing between study events.
- 82 stand-alone samples fall outside the Day 0 / 7 / 90 event structure and cannot be placed in a consistent timeline. These should be excluded from BN training.
        """)
        vc = df["sample.visitName"].value_counts().reset_index()
        vc.columns = ["Visit","Count"]
        fig = px.bar(vc.sort_values("Count", ascending=True),
                     x="Count", y="Visit", orientation="h",
                     title="Samples per visit type",
                     color_discrete_sequence=["#4C72B0"])
        fig.update_layout(height=480, margin=dict(t=40,l=10))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.histogram(df, x="sample.daysSinceFirstVisit", nbins=45,
                            color="subject.ageGroup", color_discrete_map=AGE_COLORS,
                            barmode="overlay", opacity=0.7,
                            title="Days since first visit — all samples by age group",
                            labels={"sample.daysSinceFirstVisit":"Days since first visit",
                                    "subject.ageGroup":"Age Group"})
        fig2.update_layout(height=340)
        st.plotly_chart(fig2, use_container_width=True)
        insight("Visit timing is event-based, not calendar-based. The correct time variable "
                "for modelling is day-within-event (0, 7, or 90), nested inside which event block "
                "(Flu Y1, Flu Y2, Immune Variation) — not raw days since enrollment.")

    # ── CMV x Age confounding ─────────────────────────────────────────────────
    with tab3:
        st.subheader("CMV seropositivity confounds age-group comparisons")
        st.markdown("""
**What these charts show**
- The stacked bar shows CMV serostatus proportions within each age group.
- The grouped bar shows sex balance within each age group.

**Key observations**
- 54% of Older Adults are CMV-positive vs 36% of Young Adults. Age and CMV are correlated.
- If CMV is omitted from the model, effects attributed to age may partly reflect CMV biology: T-cell compartment remodelling, elevated inflammatory tone, NK cell differentiation, and accelerated immune aging.
- Sex is balanced within both age groups, so it does not confound age-group comparisons.
        """)
        warning("CMV infection is more prevalent in Older Adults. Without CMV as a covariate, "
                "apparent age effects may be CMV effects in disguise. CMV must be a discrete root node in the BN.")

        cmv_age = (subjects.groupby(["subject.ageGroup","cmv_status"])
                   .size().reset_index(name="n"))
        tot = cmv_age.groupby("subject.ageGroup")["n"].transform("sum")
        cmv_age["pct"] = cmv_age["n"] / tot * 100
        fig = px.bar(cmv_age, x="subject.ageGroup", y="pct",
                     color="cmv_status", color_discrete_map=CMV_COLORS,
                     barmode="stack",
                     labels={"subject.ageGroup":"Age Group","pct":"% of subjects",
                             "cmv_status":"CMV Status"},
                     title="CMV serostatus by age group (%)",
                     text=cmv_age["pct"].round(1).astype(str)+"%")
        fig.update_traces(textposition="inside")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        sex_age = (subjects.groupby(["subject.ageGroup","subject.biologicalSex"])
                   .size().reset_index(name="n"))
        fig2 = px.bar(sex_age, x="subject.ageGroup", y="n",
                      color="subject.biologicalSex", color_discrete_map=SEX_COLORS,
                      barmode="group",
                      labels={"subject.ageGroup":"Age Group","n":"Subjects",
                              "subject.biologicalSex":"Sex"},
                      title="Sex distribution within age groups")
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Dropout ───────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Study retention and dropout patterns")
        st.markdown("""
**What this chart shows**
- The grouped histogram shows how many visits each subject completed, split by age group.

**Key observations**
- Older Adults were better retained (average 9.4 visits; 38/47 completed all 10).
- Young Adults had more dropout (average 8.1 visits; only 26/49 completed all 10).
- Among Young Adults, CMV-positive females had the lowest completion rate: an average of 7.2 visits and only 2 of 9 finishing the full study.
- Dropout is not fully random — it is related to observable characteristics (CMV status, sex, age group). The model should not assume missing later visits are a random subset.
        """)
        vps = (df.groupby(["subject.subjectGuid","subject.ageGroup"])
               ["sample.sampleKitGuid"].nunique().reset_index(name="n_visits"))
        fig = px.histogram(vps, x="n_visits", color="subject.ageGroup",
                           color_discrete_map=AGE_COLORS,
                           barmode="group", nbins=12,
                           title="Visits completed per subject by age group",
                           labels={"n_visits":"Number of visits",
                                   "subject.ageGroup":"Age Group"})
        fig.update_layout(height=360, bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

        summary = vps.groupby("subject.ageGroup")["n_visits"].agg(
            Mean="mean", Min="min", Max="max",
            Complete=lambda x: (x==x.max()).sum()
        ).reset_index().round(2)
        st.dataframe(summary, hide_index=True, use_container_width=True)
        warning("CMV-positive young women had fewer study visits (mean 7.2) "
                "and the lowest completion rate.")


# ===========================================================================
# PAGE: CLINICAL LABS
# ===========================================================================
def page_clinical():
    st.title("Clinical Labs and Metadata")
    df, bl = load_clinical()
    baseline = df[df["sample.daysSinceFirstVisit"]==0].drop_duplicates("subject.subjectGuid").copy()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Baseline subjects",   len(baseline))
    c2.metric("Total samples",       df["sample.sampleKitGuid"].nunique())
    c3.metric("Lab variable groups", len(LAB_GROUPS))
    c4.metric("CMV Positive",        int((baseline[CMV_COL].str.strip().str.title()=="Positive").sum())
              if CMV_COL in baseline.columns else "—")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Key variables by age", "Lab distributions", "Missingness", "Spearman correlations"])

    # ── Key vars + Mann-Whitney ───────────────────────────────────────────────
    with tab1:
        st.subheader("Key clinical variables by age group — Mann-Whitney U tests")
        st.markdown("""
**What this section shows**
- Box plots for 8 clinically important variables, comparing Young Adults and Older Adults at baseline.
- Each panel shows the Mann-Whitney U p-value and significance level. Mann-Whitney is used instead of a t-test because many clinical variables are not normally distributed.
- Significance thresholds: * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant.

**Key observations**
- Neutrophil count is significantly higher and lymphocyte count significantly lower in Older Adults, consistent with age-related myeloid skewing (a hallmark of inflammaging).
- CRP and ESR show modest but significant elevation in Older Adults, reflecting increased baseline inflammatory tone.
- BMI and LDL show no significant age-group difference in this cohort.
        """)

        existing_key = [v for v in KEY_VARS if v in baseline.columns]
        n_cols = 4
        rows = [existing_key[i:i+n_cols] for i in range(0, len(existing_key), n_cols)]
        for row in rows:
            cols = st.columns(len(row))
            for col, var in zip(cols, row):
                young = baseline[baseline["subject.ageGroup"]=="Young Adult"][var].dropna()
                older = baseline[baseline["subject.ageGroup"]=="Older Adult"][var].dropna()
                if len(young) > 0 and len(older) > 0:
                    stat, p = mannwhitneyu(young, older, alternative="two-sided")
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    label = var.split(".")[-1]
                    sub = baseline[["subject.ageGroup", var]].dropna()
                    fig = px.box(sub, x="subject.ageGroup", y=var,
                                 color="subject.ageGroup",
                                 color_discrete_map=AGE_COLORS,
                                 points="all",
                                 title=f"{label}<br><sup>p={p:.3f} {sig}</sup>",
                                 labels={"subject.ageGroup":"", var:label})
                    fig.update_layout(height=300, showlegend=False,
                                      margin=dict(t=50,b=10,l=10,r=10),
                                      title_font_size=13)
                    col.plotly_chart(fig, use_container_width=True)

        insight("Neutrophil count up, lymphocyte count down in Older Adults — "
                "a classic immune aging signature reflecting myeloid skewing. "
                "CRP and ESR elevation confirms increased baseline inflammatory tone.")

    # ── Lab distributions ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("Lab variable distributions at baseline")
        st.markdown("""
**What these charts show**
- Overlaid histograms and side-by-side box plots for a user-selected lab variable, split by age group and optionally by CMV status.
- Use this to explore individual variables in depth — look for shifts in median, spread, or outlier patterns between groups.
        """)
        col_g, col_v, col_grp = st.columns([1,1,1])
        with col_g:
            group = st.selectbox("Lab group", list(LAB_GROUPS.keys()))
        avail = [c for c in LAB_GROUPS[group] if c in baseline.columns]
        with col_v:
            var = st.selectbox("Variable", avail) if avail else None
        with col_grp:
            split = st.radio("Split by", ["Age Group","CMV Status"], horizontal=True)

        if var:
            split_col = "subject.ageGroup" if split=="Age Group" else CMV_COL
            cmap = AGE_COLORS if split=="Age Group" else CMV_COLORS
            sub = baseline[[split_col, var]].dropna()
            if CMV_COL in sub.columns:
                sub[CMV_COL] = sub[CMV_COL].str.strip().str.title()
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(sub, x=var, color=split_col,
                                   color_discrete_map=cmap,
                                   barmode="overlay", opacity=0.7, nbins=30,
                                   title=f"{var} — distribution",
                                   labels={var: var.split(".")[-1],
                                           split_col: split})
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.box(sub, x=split_col, y=var,
                              color=split_col, color_discrete_map=cmap,
                              points="all", title=f"{var} — boxplot",
                              labels={split_col:"", var: var.split(".")[-1]})
                fig2.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    # ── Missingness ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Missingness across lab variables at baseline")
        st.markdown("""
**What this chart shows**
- Percent of baseline subjects missing a value for each lab variable.
- The red dashed line marks the 20% threshold used in preprocessing to flag exclude candidates.

**Key observations**
- Most variables are missing in approximately 2% of subjects (2 individuals), likely reflecting failed blood draws.
- Inflammation markers (infl.anti_ccp3, infl.rf_iga_result, infl.rf_igm_result) are missing in approximately 75% of subjects — these were only collected for a clinical subset and are excluded from the BN node list.
- chem.ldh, chem.magnesium, and chem.phosphate are 100% missing — these tests were not run for this cohort.
        """)
        all_lab = [c for g in LAB_GROUPS.values() for c in g if c in baseline.columns]
        miss = (baseline[all_lab].isnull().mean()*100).reset_index()
        miss.columns = ["variable","pct_missing"]
        miss = miss.sort_values("pct_missing", ascending=False)
        miss["group"] = miss["variable"].apply(
            lambda v: next((g for g, cols in LAB_GROUPS.items() if v in cols), "Other"))

        fig = px.bar(miss, x="variable", y="pct_missing", color="group",
                     title="Percent missing at baseline by lab variable",
                     labels={"pct_missing":"% Missing","variable":"","group":"Group"})
        fig.add_hline(y=20, line_dash="dash", line_color="red",
                      annotation_text="20% threshold")
        fig.update_layout(height=440, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # ── Spearman correlations ─────────────────────────────────────────────────
    with tab4:
        st.subheader("Spearman correlation between clinical variables at baseline")
        st.markdown("""
**What this chart shows**
- A clustered heatmap of pairwise Spearman correlations between all clinical variables with at least 80% non-missing data at baseline.
- Spearman (rather than Pearson) correlation is used because clinical data is often not normally distributed.
- Hierarchical clustering groups variables that are correlated with each other.
- Red = strong positive correlation, blue = strong negative correlation, white = no correlation.

**Key observations**
- Blood count percentage columns (bc.perc_*) are tightly correlated with each other as expected — they sum to 100%.
- Neutrophil and lymphocyte counts are strongly negatively correlated, reflecting their inverse relationship in the differential.
- Lipid panel variables (total cholesterol, LDL, non-HDL) cluster together.
- Chemistry variables show expected clusters (liver enzymes, electrolytes).
        """)
        valid = [c for c in all_lab
                 if c in baseline.columns and baseline[c].notna().mean() >= 0.8]
        if len(valid) > 2:
            corr = baseline[valid].corr(method="spearman")
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            dist = (1 - corr.abs()).to_numpy().copy()
            np.fill_diagonal(dist, 0)
            link = linkage(squareform(dist), method="average")
            order = leaves_list(link)
            corr_ordered = corr.iloc[order, order]
            fig = px.imshow(corr_ordered, color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title="Spearman correlation — clinical variables (baseline, clustered)",
                            labels={"color":"Spearman r"})
            fig.update_layout(height=600, margin=dict(t=50))
            fig.update_xaxes(tickfont_size=9)
            fig.update_yaxes(tickfont_size=9)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough valid columns for correlation analysis.")


# ===========================================================================
# PAGE: FLU SEROLOGY & HAI
# ===========================================================================
def page_serology():
    st.title("Flu Serology and HAI")
    sero, hai = load_serology()
    fc = compute_fold_change(sero)

    sero_cmv = "subject.cmv" if "subject.cmv" in sero.columns else None
    col_map = {"Age Group": ("subject.ageGroup", AGE_COLORS)}
    if sero_cmv: col_map["CMV Status"] = (sero_cmv, CMV_COLORS)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("IgG subjects",  sero["subject.subjectGuid"].nunique())
    c2.metric("IgG antigens",  sero["msd.antigenName"].nunique())
    c3.metric("HAI subjects",  hai["subject.subjectGuid"].nunique())
    c4.metric("HAI antigens",  hai[hai["msd.antigenName"]!="BSA"]["msd.antigenName"].nunique())

    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "IgG kinetics", "HAI inhibition", "Fold change and log2FC",
        "Seroconversion rates", "Measurement quality"])

    # ── IgG kinetics ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("IgG antibody concentration over time")
        st.markdown("""
**What this chart shows**
- Box plots of IgG concentration (AU/mL) at Day 0 (pre-vaccine), Day 7 (one week post), and Day 90 (three months post) for a selected antigen.
- Groups can be split by age group or CMV status.

**Key observations**
- IgG rises sharply at Day 7 for all antigens, reflecting the acute antibody response to vaccination.
- Concentrations remain elevated at Day 90, indicating durable antibody production.
- Some subjects show high Day 0 titers, particularly for H1N1 strains — reflecting pre-existing immunity from prior seasons that can suppress the apparent fold-change.
        """)
        sero2 = sero.copy()
        sero2["timepoint"] = sero2["sample.daysSinceFirstVisit"].map(
            {0:"Day 0",6:"Day 7",7:"Day 7",8:"Day 7",9:"Day 7",90:"Day 90"})
        sero2 = sero2[sero2["timepoint"].notna()]

        col_a, col_g = st.columns([2,1])
        with col_a: antigen = st.selectbox("Antigen", sorted(sero2["msd.antigenName"].unique()), key="igg_antigen")
        with col_g: grp = st.radio("Split by", list(col_map.keys()), horizontal=True, key="igg_grp")
        col, cmap = col_map[grp]
        sub = sero2[sero2["msd.antigenName"]==antigen].dropna(subset=[col])
        if CMV_COL in sub.columns: sub[CMV_COL] = sub[CMV_COL].str.strip().str.title()
        fig = px.box(sub, x="timepoint", y="msd.concMean", color=col,
                     color_discrete_map=cmap, points="outliers",
                     category_orders={"timepoint":["Day 0","Day 7","Day 90"]},
                     title=f"IgG concentration — {antigen}",
                     labels={"msd.concMean":"IgG concentration (AU/mL)",
                             "timepoint":"Timepoint"})
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)

        # Pre-vaccination IgG across all antigens
        st.markdown("#### Pre-vaccination IgG (Day 0) across all antigens")
        st.markdown("Median baseline IgG level per antigen — higher Day 0 titer "
                    "indicates greater pre-existing immunity from prior exposure.")
        d0_all = sero[sero["sample.daysSinceFirstVisit"]==0].copy()
        fig2 = px.box(d0_all, x="msd.antigenName", y="msd.concMean",
                      color="subject.ageGroup", color_discrete_map=AGE_COLORS,
                      points=False,
                      title="Pre-vaccination IgG (Day 0) by antigen and age group",
                      labels={"msd.concMean":"IgG (AU/mL)","msd.antigenName":"Antigen",
                              "subject.ageGroup":"Age Group"})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        insight("B/Phuket consistently shows the highest pre-vaccination IgG levels, "
                "reflecting widespread prior exposure to B/Yamagata-lineage viruses.")

    # ── HAI inhibition ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("HAI percent neutralisation inhibition")
        st.markdown("""
**What this chart shows**
- Percent normalised inhibition measures how effectively antibodies in the sample block viral haemagglutination. Higher values indicate stronger neutralisation.
- Box plots compare inhibition at Day 0 and Day 7 across all antigens simultaneously (bottom chart) or for a single selected antigen over all timepoints (top chart).

**Key observations**
- Peak inhibition occurs at Day 7 across all antigens.
- There is substantial subject-to-subject variability, partly driven by pre-existing titer levels.
- BSA (bovine serum albumin) is the negative control antigen and is excluded from analysis.
        """)
        hai2 = hai[hai["msd.antigenName"]!="BSA"].copy()
        hai2["timepoint"] = hai2["sample.daysSinceFirstVisit"].map(
            {0:"Day 0",7:"Day 7",8:"Day 7",9:"Day 7",90:"Day 90"})
        hai2 = hai2[hai2["timepoint"].notna()]

        ag_h = st.selectbox("Antigen (HAI)", sorted(hai2["msd.antigenName"].unique()), key="hai_antigen")
        sub_h = hai2[hai2["msd.antigenName"]==ag_h]
        fig = px.box(sub_h, x="timepoint", y="msd.percentNormInhibition",
                     color="subject.ageGroup", color_discrete_map=AGE_COLORS,
                     category_orders={"timepoint":["Day 0","Day 7","Day 90"]},
                     points="outliers",
                     title=f"HAI % inhibition over time — {ag_h}",
                     labels={"msd.percentNormInhibition":"% Inhibition",
                             "timepoint":"Timepoint","subject.ageGroup":"Age Group"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Day 7 inhibition across all antigens
        st.markdown("#### Day 7 HAI inhibition across all antigens")
        d7_hai = hai2[hai2["sample.daysSinceFirstVisit"].isin([7,8,9])]
        fig2 = px.box(d7_hai, x="msd.antigenName", y="msd.percentNormInhibition",
                      color="subject.ageGroup", color_discrete_map=AGE_COLORS,
                      points=False,
                      title="HAI % inhibition at Day 7 — all antigens",
                      labels={"msd.percentNormInhibition":"% Inhibition",
                              "msd.antigenName":"Antigen","subject.ageGroup":"Age Group"})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Fold change ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader("IgG fold change and log2 fold change (Day 7 / Day 0)")
        st.markdown("""
**What these charts show**
- The violin/strip plot (left) shows the distribution of fold change (FC = Day 7 / Day 0) for a selected antigen, split by group.
- The box plot (right) shows log2 fold change across all antigens simultaneously.
- The dashed line at FC = 4 marks the seroconversion threshold.
- Log2 FC is used for cross-antigen comparison because it is symmetric around zero and compresses extreme values.

**Key observations**
- A/Victoria typically shows the most variable fold change, driven by high inter-subject variability in pre-existing immunity.
- Fold change is suppressed in subjects with high Day 0 titers, even when their absolute Day 7 levels are protective. This is a ceiling effect, not a vaccine failure.
        """)
        col_ag, col_grp2 = st.columns([2,1])
        with col_ag: ag_fc = st.selectbox("Antigen", sorted(fc["msd.antigenName"].unique()), key="fc_antigen")
        with col_grp2: grp2 = st.radio("Split by", list(col_map.keys()), horizontal=True, key="fc_grp")
        col_fc, cmap_fc = col_map[grp2]

        col1, col2 = st.columns(2)
        sub_fc = fc[fc["msd.antigenName"]==ag_fc].copy()
        with col1:
            fig = px.strip(sub_fc, x=col_fc, y="fold_change",
                           color=col_fc, color_discrete_map=cmap_fc,
                           title=f"Fold change — {ag_fc}",
                           labels={"fold_change":"Fold change (D7/D0)", col_fc:""})
            fig.add_hline(y=4, line_dash="dash", line_color="red",
                          annotation_text="Seroconversion (FC=4)")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(fc, x="msd.antigenName", y="log2_fc",
                          color="subject.ageGroup", color_discrete_map=AGE_COLORS,
                          points=False,
                          title="log2 fold change across all antigens",
                          labels={"log2_fc":"log2 FC (D7/D0)","msd.antigenName":"Antigen",
                                  "subject.ageGroup":"Age Group"})
            fig2.add_hline(y=np.log2(4), line_dash="dash", line_color="red",
                           annotation_text="log2(4) = seroconversion")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Seroconversion rates ──────────────────────────────────────────────────
    with tab4:
        st.subheader("Seroconversion rates by antigen and group")
        st.markdown("""
**What this chart shows**
- Seroconversion is defined as a ≥4-fold rise in IgG from Day 0 to Day 7. This is the established clinical benchmark for flu vaccine immunogenicity.
- The bar chart shows the percentage of subjects achieving seroconversion for each antigen, split by age group or CMV status.
- A population-level rate above approximately 40% per antigen is considered evidence of effective vaccine immunogenicity.

**Key observations**
- A/Victoria typically shows the highest seroconversion rates, consistent with lower pre-existing immunity to this strain.
- B/Phuket tends to show the lowest rates, as most subjects already have high Day 0 titers that suppress fold-change.
- Seroconversion rates per antigen will be the primary validation outcome for the Bayesian Network: predicting which baseline molecular profile leads to seroconversion vs. failure.
        """)
        grp3 = st.radio("Split by", list(col_map.keys()), horizontal=True, key="sc_grp")
        col_sc, cmap_sc = col_map[grp3]
        sc_rate = (fc.groupby(["msd.antigenName", col_sc])
                   .apply(lambda x: (x["seroconverted"]==True).mean()*100)
                   .reset_index(name="seroconversion_rate"))
        fig = px.bar(sc_rate, x="msd.antigenName", y="seroconversion_rate",
                     color=col_sc, color_discrete_map=cmap_sc,
                     barmode="group",
                     title="Seroconversion rate (FC >= 4) by antigen",
                     labels={"seroconversion_rate":"Seroconversion rate (%)",
                             "msd.antigenName":"Antigen", col_sc: grp3})
        fig.add_hline(y=40, line_dash="dash", line_color="black",
                      annotation_text="40% immunogenicity benchmark")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        tbl = sc_rate.pivot(index="msd.antigenName", columns=col_sc,
                            values="seroconversion_rate").round(1).reset_index()
        tbl.columns.name = None
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    # ── CV quality ────────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Measurement quality — coefficient of variation")
        st.markdown("""
**What these charts show**
- The histogram shows the distribution of coefficient of variation (CV%) across all IgG measurements.
- CV% measures replicate consistency within each well. Higher CV indicates less consistent measurement.
- The flag threshold is CV > 20%.

**Key observations**
- The vast majority of measurements have CV below 20%, indicating good assay reproducibility.
- Measurements flagged as CV > 20% are retained in the data with a paired flag column and are not silently removed.
        """)
        cv_pct = (sero["msd.concPercentCV"] > 20).mean() * 100
        col1, col2 = st.columns(2)
        col1.metric("IgG measurements with CV > 20%", f"{cv_pct:.1f}%")
        col2.metric("IgG measurements with CV <= 20%", f"{100-cv_pct:.1f}%")
        fig = px.histogram(sero, x="msd.concPercentCV", nbins=60,
                           color="msd.antigenName",
                           title="IgG coefficient of variation — by antigen",
                           labels={"msd.concPercentCV":"CV (%)",
                                   "msd.antigenName":"Antigen"})
        fig.add_vline(x=20, line_dash="dash", line_color="red",
                      annotation_text="20% flag threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: PLASMA PROTEOMICS
# ===========================================================================
def page_plasma():
    st.title("Plasma Proteomics (Olink)")
    plasma = load_plasma()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total rows",      f"{len(plasma):,}")
    c2.metric("Unique samples",  plasma["sample.sampleKitGuid"].nunique())
    c3.metric("Unique proteins", plasma["olink.assay"].nunique())
    c4.metric("Panels",          plasma["olink.panel"].nunique())

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "LOD landscape", "Protein stability (ICC)",
        "NPX distributions", "Batch effects"])

    # ── LOD ──────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Limit of detection (LOD) landscape")
        st.markdown("""
**What these charts show**
- The histogram shows how the fraction-below-LOD is distributed across all 867 proteins, coloured by Olink panel.
- The bar chart shows the 15 proteins most frequently below the detection limit.
- LOD is the minimum protein concentration the assay can reliably distinguish from background noise.

**Key observations**
- Approximately 217 proteins are below LOD in more than 30% of samples. These require a decision before BN inclusion: exclude them, or model them as censored observations where the value is known only to be below a threshold.
- LOD rates are comparable between age groups — missingness is not concentrated in one subgroup.
- The Oncology panel has more proteins with high LOD rates, as many tumour markers are undetectable in healthy subjects.
        """)
        plasma["below_lod"] = plasma["olink.NPX_norm"] < plasma["olink.LOD_norm"]
        lod = (plasma.groupby(["olink.assay","olink.panel"])
               .agg(pct_below_lod=("below_lod","mean")).reset_index())
        lod["pct_below_lod"] *= 100

        c1,c2,c3 = st.columns(3)
        c1.metric("Proteins > 50% below LOD", int((lod["pct_below_lod"]>50).sum()))
        c2.metric("Proteins > 30% below LOD", int((lod["pct_below_lod"]>30).sum()))
        c3.metric("Proteins <= 10% below LOD", int((lod["pct_below_lod"]<=10).sum()))

        fig = px.histogram(lod, x="pct_below_lod", nbins=50, color="olink.panel",
                           title="Distribution of % below LOD per protein (by panel)",
                           labels={"pct_below_lod":"% samples below LOD",
                                   "olink.panel":"Panel"})
        fig.add_vline(x=30, line_dash="dash", line_color="red",
                      annotation_text="30% flag threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        top_lod = lod.nlargest(15, "pct_below_lod")
        fig2 = px.bar(top_lod.sort_values("pct_below_lod"), x="pct_below_lod",
                      y="olink.assay", color="olink.panel", orientation="h",
                      title="Top 15 proteins most often below LOD",
                      labels={"pct_below_lod":"% below LOD","olink.assay":"Protein"})
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)
        warning("217 proteins are below LOD in >30% of samples.")

    # ── ICC ──────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Protein stability — Intraclass Correlation Coefficient (ICC)")
        st.markdown("""
**What these charts show**
- The histogram shows the distribution of ICC values across all 867 proteins.
- ICC measures what fraction of a protein's total variability is due to stable between-person differences (trait-like, ICC near 1) vs within-person fluctuation over time (state-like, ICC near 0).
- The bar chart shows ICC values for the 18 BN-selected proteins, colour-coded by reliability tier.

**Key observations**
- Panel-wide median ICC is approximately 0.65. Only 30% of proteins exceed ICC 0.75.
- Among the 18 selected proteins: 6 are reliable (ICC >= 0.75), 6 are borderline (0.60–0.75), and 6 fall below 0.60.
- Note: proteins such as CXCL10, CXCL9, and CXCL13 genuinely change post-vaccination — their lower ICC may reflect real biological variation over time rather than measurement noise.
        """)
        with st.spinner("Computing ICC across all proteins..."):
            icc_df = compute_icc(plasma)

        c1,c2,c3 = st.columns(3)
        c1.metric("Median ICC",  f"{icc_df['icc'].median():.2f}")
        c2.metric("ICC > 0.75",  int((icc_df["icc"]>0.75).sum()))
        c3.metric("ICC < 0.50",  int((icc_df["icc"]<0.50).sum()))

        fig = px.histogram(icc_df, x="icc", nbins=50,
                           title="ICC distribution across all 867 proteins",
                           labels={"icc":"ICC"},
                           color_discrete_sequence=["#4C72B0"])
        fig.add_vline(x=0.75, line_dash="dash", line_color="green",
                      annotation_text="0.75 threshold")
        fig.add_vline(x=0.50, line_dash="dot",  line_color="orange",
                      annotation_text="0.50")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        sel = icc_df[icc_df["olink.assay"].isin(SELECTED_PROTEINS)].sort_values("icc").copy()
        sel["reliability"] = sel["icc"].apply(
            lambda x: "ICC >= 0.75" if x>=0.75 else ("ICC 0.60-0.75" if x>=0.60 else "ICC < 0.60"))
        cmap_r = {"ICC >= 0.75":"#55A868","ICC 0.60-0.75":"#DD8452","ICC < 0.60":"#C44E52"}
        fig2 = px.bar(sel, x="icc", y="olink.assay", orientation="h",
                      color="reliability", color_discrete_map=cmap_r,
                      title="ICC for 18 BN-selected proteins",
                      labels={"icc":"ICC","olink.assay":"Protein"})
        fig2.add_vline(x=0.75, line_dash="dash", line_color="green")
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**ICC values for selected proteins**")
        st.dataframe(sel[["olink.assay","var_between","var_within","icc","reliability"]]
                     .rename(columns={"olink.assay":"Protein","var_between":"Var Between",
                                      "var_within":"Var Within","icc":"ICC",
                                      "reliability":"Reliability"})
                     .sort_values("ICC").round(3),
                     hide_index=True, use_container_width=True)

    # ── NPX distributions ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("NPX distributions by protein at baseline")
        st.markdown("""
**What these charts show**
- NPX (Normalized Protein eXpression) is Olink's log2-based unit for protein abundance.
- The histogram and box plot show the NPX distribution at baseline (Flu Year 1 Day 0) for a selected protein, split by age group or CMV status.
- The scatter plot below shows NPX vs BMI to illustrate potential metabolic correlations.

**How to use**
- Select the BN-selected proteins tab to quickly compare the 18 key proteins side by side across age groups.
        """)
        baseline_p = plasma[plasma["sample.visitName"].str.contains("Flu Year 1 Day 0", na=False)]

        view_mode = st.radio("View", ["Single protein explorer","BN-selected proteins overview"],
                             horizontal=True)
        if view_mode == "Single protein explorer":
            col_p, col_sg = st.columns([2,1])
            with col_p: protein = st.selectbox("Protein", sorted(baseline_p["olink.assay"].unique()))
            with col_sg: sg2 = st.radio("Split by", ["Age Group","CMV Status"], horizontal=True, key="npx_sg")
            sc2 = "subject.ageGroup" if sg2=="Age Group" else "subject.cmv"
            cm2 = AGE_COLORS if sg2=="Age Group" else CMV_COLORS
            sub_p = baseline_p[baseline_p["olink.assay"]==protein].copy()
            if "subject.cmv" in sub_p.columns:
                sub_p["subject.cmv"] = sub_p["subject.cmv"].str.strip().str.title()

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(sub_p, x="olink.NPX_norm", color=sc2,
                                   color_discrete_map=cm2, barmode="overlay",
                                   opacity=0.7, nbins=30,
                                   title=f"{protein} NPX distribution",
                                   labels={"olink.NPX_norm":"NPX"})
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.box(sub_p, x=sc2, y="olink.NPX_norm", color=sc2,
                              color_discrete_map=cm2, points="all",
                              title=f"{protein} NPX boxplot",
                              labels={"olink.NPX_norm":"NPX", sc2:""})
                fig2.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        else:
            sel_base = baseline_p[baseline_p["olink.assay"].isin(SELECTED_PROTEINS)]
            fig = px.box(sel_base, x="olink.assay", y="olink.NPX_norm",
                         color="subject.ageGroup", color_discrete_map=AGE_COLORS,
                         points=False,
                         title="NPX at baseline — all 18 BN-selected proteins by age group",
                         labels={"olink.NPX_norm":"NPX","olink.assay":"Protein",
                                 "subject.ageGroup":"Age Group"},
                         category_orders={"olink.assay": sorted(SELECTED_PROTEINS)})
            fig.update_layout(height=480, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            insight("GDF15 and IL18 show the clearest separation between age groups "
                    "at baseline, consistent with their roles as established inflammaging markers.")

    # ── Batch effects ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Batch effects across Olink panels")
        st.markdown("""
**What this chart shows**
- Box plots of NPX_norm across the 6 processing batches, split by panel.
- NPX_norm is already batch-corrected by Olink's pipeline. This chart verifies that the correction was effective.
- Residual batch effects would appear as systematic shifts in the median NPX across batches within the same panel.

**Key observations**
- After batch normalisation, NPX distributions are broadly consistent across batches within each panel.
- Any residual batch structure should be noted as a potential covariate for downstream modelling.
        """)
        fig = px.box(plasma, x="olink.batch_id", y="olink.NPX_norm",
                     color="olink.panel",
                     facet_col="olink.panel", facet_col_wrap=2,
                     title="NPX_norm distribution by batch and panel",
                     labels={"olink.NPX_norm":"NPX_norm","olink.batch_id":"Batch"})
        fig.update_layout(height=600, showlegend=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_xaxes(tickangle=-45, tickfont_size=8)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: DIFFERENTIAL EXPRESSION
# ===========================================================================
def page_deseq():
    st.title("Differential Expression (DESeq2)")
    deseq = load_deseq()
    if not deseq:
        st.error(f"No DESeq2 CSVs found at: {DESEQ_DIR}")
        return

    ov_rows = []
    for name, d in deseq.items():
        df = d["df"]
        n  = len(df)
        ov_rows.append(dict(
            Contrast=name, Significant=int(df["sig"].sum()),
            Strong=int(df["sig_strong"].sum()),
            Up=int((df["sig"] & (df["log2fc"]>0)).sum()),
            Down=int((df["sig"] & (df["log2fc"]<0)).sum()),
            Pct_sig=round(100*df["sig"].sum()/n,1),
            color=d["color"]))
    ov = pd.DataFrame(ov_rows)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Contrasts", len(deseq))
    c2.metric("Total DEG tests", f"{sum(len(d['df']) for d in deseq.values()):,}")
    c3.metric("Significant (padj<0.05)", f"{ov['Significant'].sum():,}")
    c4.metric("Strong (|log2FC|>=1)", f"{ov['Strong'].sum():,}")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Overview and direction", "Volcano explorer", "Cell-type landscape"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Significant results by contrast")
        st.markdown("""
**What these charts show**
- Left: grouped bar chart of total significant (padj < 0.05) and strong (padj < 0.05 AND |log2FC| >= 1) DEG tests per contrast.
- Right: signed bar chart showing how many significant genes are upregulated in the foreground vs the background group. Positive bars = higher in foreground; negative bars = higher in background.
- A single test = one gene in one cell type (AIFI_L3) for one contrast.

**Key observations**
- Biological Sex produces by far the most significant genes, driven by sex-chromosome genes (RPS4Y1, XIST, DDX3Y). This is expected and confirms the pipeline is working correctly.
- CMV Status and Age Group show real but more distributed signals across multiple cell types.
- Flu Vaccine contrast captures the acute transcriptional response to vaccination (Day 0 vs Day 7).
- P-value histograms with a spike near zero confirm genuine biological signal in all four contrasts.
        """)
        col1, col2 = st.columns(2)
        with col1:
            long = ov.melt(id_vars="Contrast", value_vars=["Significant","Strong"],
                           var_name="Type", value_name="Count")
            fig = px.bar(long, x="Contrast", y="Count", color="Type", barmode="group",
                         color_discrete_map={"Significant":"#9aa7b3","Strong":"#1f3a5f"},
                         text="Count", title="Significant DEG tests by contrast")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=400, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            for _, row in ov.iterrows():
                fig2.add_trace(go.Bar(name=row["Contrast"], x=[row["Contrast"]],
                                     y=[row["Up"]], marker_color="#C44E52",
                                     showlegend=False))
                fig2.add_trace(go.Bar(name=row["Contrast"], x=[row["Contrast"]],
                                     y=[-row["Down"]], marker_color="#4C72B0",
                                     showlegend=False))
            fig2.add_hline(y=0, line_color="black", line_width=1)
            fig2.update_layout(height=400, barmode="relative",
                               title="Direction of significant changes",
                               yaxis_title="# significant tests (signed)",
                               annotations=[
                                   dict(x=1.01, y=0.75, xref="paper", yref="paper",
                                        text="Red = up in foreground", showarrow=False,
                                        font=dict(color="#C44E52", size=11)),
                                   dict(x=1.01, y=0.65, xref="paper", yref="paper",
                                        text="Blue = up in background", showarrow=False,
                                        font=dict(color="#4C72B0", size=11))])
            st.plotly_chart(fig2, use_container_width=True)

        # p-value histograms
        st.markdown("#### Raw p-value distributions")
        st.markdown("A spike near p=0 indicates genuine biological signal. "
                    "A flat uniform distribution indicates weak or absent effects.")
        fig3 = make_subplots(rows=1, cols=4,
                             subplot_titles=list(deseq.keys()))
        for i, (name, d) in enumerate(deseq.items(), 1):
            pv = d["df"]["pvalue"].dropna()
            fig3.add_trace(go.Histogram(x=pv, nbinsx=40,
                                        marker_color=DESEQ_FILES[name][3],
                                        showlegend=False), row=1, col=i)
        fig3.update_layout(height=320, margin=dict(t=50))
        fig3.update_xaxes(title_text="p-value")
        fig3.update_yaxes(title_text="Count", col=1)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Volcano ───────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Interactive volcano plot")
        st.markdown("""
**How to read this plot**
- Each point represents one gene tested in one cell type (AIFI_L3). X-axis = log2 fold change; Y-axis = -log10(adjusted p-value).
- Red points are significantly higher in the foreground group (positive log2FC). Blue points are significantly higher in the background group.
- The dashed horizontal line marks padj = 0.05. Dotted vertical lines mark the ±1 log2FC threshold.
- Points above and beyond both thresholds are "strong" DEGs.
- Hover over any point to see gene name, cell type, fold change, and adjusted p-value.
- Use the cell type filter to zoom in on specific immune populations.
        """)
        name = st.selectbox("Contrast", list(deseq.keys()))
        d    = deseq[name]
        df   = d["df"]

        col_ct, col_gene = st.columns([2,1])
        with col_ct:
            if "AIFI_L3" in df.columns:
                cts = sorted(df["AIFI_L3"].dropna().unique())
                chosen_ct = st.multiselect("Filter by cell type (empty = all)", cts)
            else:
                chosen_ct = []
        with col_gene:
            gene_q = st.text_input("Search gene", "").strip().upper()

        sub = df if not chosen_ct else df[df["AIFI_L3"].isin(chosen_ct)]
        st.caption(f"Foreground: {d['fg']}  |  Background: {d['bg']}  |  "
                   f"{int(sub['sig'].sum()):,} significant  |  "
                   f"{int(sub['sig_strong'].sum()):,} strong")

        vdf = sub.dropna(subset=["padj","log2fc"]).copy()
        vdf = vdf[vdf["padj"]>0]
        vdf["neglog10padj"] = -np.log10(vdf["padj"])
        vdf["category"] = np.where(~vdf["sig"], "Not significant",
            np.where(vdf["log2fc"]>0, "Up in foreground", "Up in background"))
        ns = vdf[vdf["category"]=="Not significant"]
        if len(ns) > 8000: ns = ns.sample(8000, random_state=0)
        vdf_plot = pd.concat([ns, vdf[vdf["category"]!="Not significant"]], ignore_index=True)

        hover = {"log2fc":":.2f","padj":":.2e","neglog10padj":False}
        for c in ("gene","AIFI_L3"):
            if c in vdf_plot.columns: hover[c] = True

        fig_v = px.scatter(vdf_plot, x="log2fc", y="neglog10padj", color="category",
                           color_discrete_map={"Not significant":"#d3d8de",
                                               "Up in foreground":"#C44E52",
                                               "Up in background":"#4C72B0"},
                           hover_data=hover, opacity=0.65,
                           title=f"Volcano plot — {name}")
        fig_v.add_hline(y=-np.log10(PADJ_SIG), line_dash="dash", line_color="#555")
        fig_v.add_vline(x= LFC_THR, line_dash="dot", line_color="#999")
        fig_v.add_vline(x=-LFC_THR, line_dash="dot", line_color="#999")
        fig_v.update_traces(marker=dict(size=4))
        fig_v.update_layout(height=520, xaxis_title="log2(Fold Change)",
                             yaxis_title="-log10(padj)", legend_title="",
                             margin=dict(t=40))
        st.plotly_chart(fig_v, use_container_width=True)

        col_l, col_r = st.columns([3,2])
        with col_l:
            st.markdown("**Top significant hits**")
            top = sub[sub["sig"]].dropna(subset=["padj"]).copy()
            if gene_q and "gene" in top.columns:
                top = top[top["gene"].str.upper().str.contains(gene_q, na=False)]
            top = top.sort_values(["padj","pvalue"]).head(200)
            show_cols = [c for c in ["AIFI_L3","gene","log2fc","padj"] if c in top.columns]
            st.dataframe(
                top[show_cols].style.format(
                    {k:v for k,v in {"log2fc":"{:.2f}","padj":"{:.2e}"}.items()
                     if k in show_cols}),
                hide_index=True, use_container_width=True, height=380)
        with col_r:
            st.markdown("**Most differentially active cell types**")
            if "AIFI_L3" in sub.columns:
                ct_ct = (sub[sub["sig_strong"]].groupby("AIFI_L3").size()
                         .sort_values(ascending=False).head(15)
                         .rename("Strong DEGs").reset_index()
                         .rename(columns={"AIFI_L3":"Cell type"}))
                if len(ct_ct):
                    fig_ct = px.bar(ct_ct.iloc[::-1], x="Strong DEGs", y="Cell type",
                                    orientation="h",
                                    color_discrete_sequence=[d["color"]])
                    fig_ct.update_layout(height=420, margin=dict(t=10))
                    st.plotly_chart(fig_ct, use_container_width=True)
                else:
                    st.info("No strong DEGs for this selection.")

    # ── Cell-type landscape ───────────────────────────────────────────────────
    with tab3:
        st.subheader("Cell-type signal landscape — strong DEGs across all contrasts")
        st.markdown("""
**What this chart shows**
- A heatmap showing how many strong DEGs (padj < 0.05 AND |log2FC| >= 1) each cell type has in each contrast.
- Colour intensity = log1p(count). Brighter = more strong DEGs in that cell type for that contrast.
- Cell types are sorted by total signal across all contrasts.

**Key observations**
- Some cell types carry signal in multiple contrasts simultaneously, suggesting they are broadly responsive to demographic differences.
- Cell types with high signal in the Age Group and CMV Status contrasts are particularly relevant to the Bayesian Network, as these contrasts directly map to root nodes.
        """)
        if all("AIFI_L3" in d["df"].columns for d in deseq.values()):
            all_cts = sorted(set().union(*[d["df"]["AIFI_L3"].dropna().unique()
                                           for d in deseq.values()]))
            contrasts = list(deseq.keys())
            mat = np.zeros((len(all_cts), len(contrasts)))
            for j, (cname, d) in enumerate(deseq.items()):
                cnt = d["df"][d["df"]["sig_strong"]].groupby("AIFI_L3").size()
                for i, ct in enumerate(all_cts):
                    mat[i,j] = cnt.get(ct, 0)
            order = np.argsort(mat.sum(1))[::-1]
            mat   = mat[order]
            cts_o = [all_cts[i] for i in order]

            fig = px.imshow(np.log1p(mat), x=contrasts, y=cts_o,
                            color_continuous_scale="magma",
                            title="Strong DEGs per cell type and contrast (log1p scale)",
                            labels={"color":"log1p(count)","x":"Contrast","y":"Cell type"})
            fig.update_layout(height=max(500, len(cts_o)*14),
                              margin=dict(t=50,l=10), yaxis_tickfont_size=8)
            st.plotly_chart(fig, use_container_width=True)

            top_ct = cts_o[0]
            insight(f"'{top_ct}' carries the most strong DEGs summed across all four contrasts, "
                    "making it the most broadly responsive cell type in this dataset.")
        else:
            st.info("AIFI_L3 cell type column not found in all contrast files.")

# ===========================================================================
# PAGE: WHOLE BLOOD RNA-SEQ
# ===========================================================================
def page_wholeblood():
    st.title("Whole Blood RNA-seq")
    st.markdown(
        "Key findings from the unstimulated and ex vivo stimulated whole blood "
        "RNA-seq datasets. These files are too large to load at runtime so results "
        "from the EDA are presented here as a static summary."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Genes measured",       "58,302")
    c2.metric("Unstimulated samples", "838")
    c3.metric("Ex vivo samples",      "985")
    c4.metric("Plasma-RNA overlap",   "835 / 838")

    st.divider()
    tab1, tab2, tab3 = st.tabs([
        "Sequencing quality", "Ex vivo stimulation", "Variance partition"])

    # ── Sequencing quality ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Sequencing quality — unstimulated dataset")
        st.markdown("""
**What was measured**
- Total read counts per sample (nCount_RNA): the number of genetic messages captured from each blood sample
- Genes detected per sample (nFeature_RNA): how many of the 58,302 genes had measurable activity in that sample

**Key observations**
- Median total read count is approximately 3.5 million reads per sample
- The distribution of read counts is smooth and bell-shaped with no extreme outliers, meaning every sample was processed to a similar sequencing depth
- On average approximately 17,450 of 58,302 genes are detectable per sample
- The tight distribution of genes detected across samples indicates consistent RNA quality throughout the dataset
- Sequencing depth in the ex vivo stimulated dataset is comparable — the stimulation process did not degrade sample quality
        """)

        col1, col2 = st.columns(2)
        with col1:
            rc_data = pd.DataFrame({
                "Statistic": ["Min","25th pct","Median","Mean","75th pct","Max"],
                "nCount_RNA (millions)": [1.2, 2.8, 3.5, 3.6, 4.3, 7.1],
            })
            st.markdown("**Total read counts per sample**")
            st.dataframe(rc_data, hide_index=True, use_container_width=True)
        with col2:
            feat_data = pd.DataFrame({
                "Statistic": ["Min","25th pct","Median","Mean","75th pct","Max"],
                "nFeature_RNA (genes)": [11200, 15800, 17450, 17200, 18900, 23400],
            })
            st.markdown("**Genes detected per sample**")
            st.dataframe(feat_data, hide_index=True, use_container_width=True)

        st.markdown("""
**Dataset overlap with plasma proteomics**
- 835 of 838 unstimulated RNA-seq samples have a matching plasma sample
- The 32 unmatched plasma samples are scattered across visit types and age groups with no obvious pattern, consistent with random processing failures rather than systematic loss
- A joint multimodal model using both gene expression and protein levels is feasible with minimal data loss
        """)

    # ── Ex vivo stimulation ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Ex vivo stimulation conditions")
        st.markdown("""
**What the stimulations are**
- Each blood draw was split into four tubes, one per condition, producing 985 total samples
- **Null:** No stimulus — the control condition
- **LPS:** Bacterial cell wall component that triggers a strong innate immune response
- **SEB:** Staph toxin that activates T cells
- **Poly I:C:** Mimics a viral infection and triggers antiviral immune pathways
        """)

        stim_data = pd.DataFrame({
            "Condition":   ["Null", "LPS",  "SEB",  "Poly I:C"],
            "Samples":     [249,    248,    245,    243],
            "Description": [
                "Control — no immune activation",
                "Bacterial innate immune trigger",
                "T-cell activator",
                "Viral infection mimic",
            ]
        })
        fig = px.bar(stim_data, x="Condition", y="Samples",
                     color="Condition",
                     title="Sample counts per stimulation condition",
                     labels={"Samples": "Number of samples"},
                     color_discrete_sequence=["#457B9D","#E84855","#F4A261","#55A868"])
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
**Key findings from PCA analysis**
- The four stimulation conditions separate cleanly in gene expression space
- LPS and SEB, which both cause strong immune activation, cluster together on one side
- Unstimulated Null samples sit apart; Poly I:C falls in between
- This confirms the stimulations are working as expected

**Important modelling implication**
- Stimulation condition dominates gene expression variation in the ex vivo dataset
- The stimulated dataset must model stimulation condition as a primary factor before any age or CMV effects can be detected
        """)

    # ── Variance partition ────────────────────────────────────────────────────
    with tab3:
        st.subheader("What is driving gene expression differences?")
        st.markdown("""
**What variance partition analysis measures**
- For the 2,000 most variable genes in the unstimulated dataset, variance partition analysis estimates what fraction of each gene's total variability is explained by each biological factor
- Factors tested: individual identity (subject.subjectGuid), age group, CMV status, biological sex

**Key observations**
- Individual identity explains the largest fraction of gene expression variance — most gene-to-gene variation is simply people being different from each other
- Age group, CMV status, and biological sex each explain smaller but meaningful fractions
- A substantial portion of variance is unexplained by these four factors, likely driven by visit-to-visit fluctuations, vaccination timing, or biology not captured by these variables alone

**Modelling implications**
- Per-person random effects are essential in any mixed model applied to this data
- The large individual-identity component supports the Bayesian Network design: learning each person's immune baseline separately from group-level patterns
        """)

        vp_data = pd.DataFrame({
            "Factor":              ["Individual identity", "Age group", "CMV status",
                                   "Biological sex", "Residual"],
            "Approx % variance":   [35, 8, 6, 12, 39],
        })
        fig2 = px.bar(vp_data.sort_values("Approx % variance", ascending=True),
                      x="Approx % variance", y="Factor", orientation="h",
                      title="Approximate variance explained per factor (top 2,000 genes)",
                      color="Approx % variance",
                      color_continuous_scale="Blues",
                      labels={"Approx % variance": "% variance explained", "Factor": ""})
        fig2.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Values are approximate, derived from the variance partition analysis "
                   "in the EDA Rmd. Exact values depend on the gene subset and model specification.")

# ===========================================================================
# Navigation
# ===========================================================================
pages = [
    st.Page(page_overview, title="Overview",                default=True),
    st.Page(page_cohort,   title="Cohort Overview"),
    st.Page(page_clinical, title="Clinical Labs"),
    st.Page(page_serology, title="Flu Serology and HAI"),
    st.Page(page_plasma,   title="Plasma Proteomics"),
    st.Page(page_wholeblood, title="Whole Blood RNA-seq"),
    st.Page(page_deseq,    title="Differential Expression"),
]
st.navigation(pages).run()