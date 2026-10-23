"""
 S-DAC PLANT OPTIMIZER — Machine Learning & Techno-Economic Analysis
 Project: Chemical Technology - Round 3
 Description: 
    This application evaluates real experimental Solid Direct Air Capture (S-DAC) 
    data using a Histogram Gradient Boosting Machine (GBM). It predicts optimal 
    operating conditions (Temperature, Time, Humidity) and calculates Levelized 
    Cost of Capture (LCOC) using chemical engineering scaling laws.

 Authors: Chemical Technology Project : Group 5
 Date: 23 October 2026

"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# 1. PAGE CONFIGURATION & CONSTANTS
st.set_page_config(page_title="S-DAC Plant Optimizer", page_icon="🏭", layout="wide")

USD_INR   = 83.0           # Conversion rate: 1 USD to INR
M_CO2     = 0.044          # Molar mass of CO2 (kg/mol)
MOL_PER_T = 1e6 / 44.0     # Moles of CO2 in one metric tonne
RHO_AIR   = 1.2            # Density of air at standard conditions (kg/m³)
TDES_OPTS = [80, 90, 100, 110, 120] # Candidate desorption temperatures (°C)

SITES = {
    "🇮🇸 Iceland — Hellisheiði (Climeworks)":    dict(T=5,  RH=75, note="Geothermal heat · 4 kt (Orca) + 36 kt (Mammoth) operating"),
    "🇺🇸 USA — Texas (Occidental Stratos)":      dict(T=22, RH=55, note="Largest DAC under construction: 500 kt/yr by 2025"),
    "🇨🇦 Canada — British Columbia (Carbon Eng.)": dict(T=10, RH=70, note="Hydropower · 1 Mt/yr liquid DAC design"),
    "🇴🇲 Oman — Jebel Hafit (44.01)":             dict(T=28, RH=45, note="Mineralization in peridotite rock"),
    "🇰🇪 Kenya — Rift Valley (Octavia Carbon)":   dict(T=22, RH=55, note="First African DAC · geothermal powered"),
    "🇳🇴 Norway — Northern grid":                 dict(T=6,  RH=70, note="Hydropower · integration with CCS hubs"),
    "🇦🇪 UAE — Masdar City":                      dict(T=30, RH=55, note="Solar-powered DAC pilot"),
    "🇮🇳 India — Jaisalmer (hot desert)":         dict(T=35, RH=25, note="Dry heat · solar potential"),
    "🇮🇳 India — Delhi":                          dict(T=25, RH=55, note="Industrial corridor"),
    "🇮🇳 India — Mumbai (coastal)":               dict(T=28, RH=75, note="Humid coastal · high RH"),
    "🇮🇳 India — Ladakh (cold desert)":           dict(T=8,  RH=30, note="High altitude · cold · solar-rich"),
}

# 2. UI STYLING & HELPER FUNCTIONS
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#0b1020 0%,#111a3a 55%,#0d2230 100%);}
section[data-testid="stSidebar"]{background:#0a0f1f;border-right:1px solid rgba(56,189,248,.15);}
h1,h2,h3,h4{color:#f1f5f9!important;}
div[data-testid="stMetric"]{background:rgba(255,255,255,.04);border:1px solid rgba(56,189,248,.18);
  border-radius:14px;padding:14px 18px;}
div[data-testid="stMetricValue"]{color:#38bdf8!important;font-size:1.5rem!important;}
.stTabs [data-baseweb="tab-list"]{gap:8px;}
.stTabs [data-baseweb="tab"]{background:rgba(255,255,255,.04);border-radius:10px 10px 0 0;
  padding:10px 22px;font-weight:700;color:#94a3b8;}
.stTabs [aria-selected="true"]{background:rgba(56,189,248,.15)!important;color:#38bdf8!important;}
.card{background:linear-gradient(135deg,rgba(56,189,248,.09),rgba(52,211,153,.05));
  border:1px solid rgba(56,189,248,.22);border-radius:16px;padding:16px 18px;min-height:175px;}
.card h5{color:#38bdf8;margin:0 0 6px 0;font-size:.85rem;letter-spacing:.6px;text-transform:uppercase;}
.card .big{color:#f8fafc;font-size:1.25rem;font-weight:800;line-height:1.3;}
.card .sub{color:#94a3b8;font-size:.85rem;line-height:1.55;margin-top:6px;}
.chem{background:rgba(52,211,153,.06);border-left:4px solid #34d399;border-radius:0 12px 12px 0;
  padding:12px 16px;color:#cbd5e1;margin:6px 0;font-size:.92rem;min-height:160px;}
.chem b{color:#34d399;} .chem code{color:#fbbf24;background:transparent;}
.hero{background:linear-gradient(90deg,rgba(56,189,248,.16),rgba(167,139,250,.10));
  border:1px solid rgba(56,189,248,.25);border-radius:18px;padding:16px 24px;margin-bottom:12px;}
.hero h2{margin:0;} .hero p{color:#94a3b8;margin:4px 0 0 0;}
.site-banner{background:linear-gradient(90deg,rgba(52,211,153,.15),rgba(56,189,248,.08));
  border:1px solid rgba(52,211,153,.3);border-radius:14px;padding:12px 20px;margin:8px 0 16px 0;
  color:#e2e8f0;display:flex;justify-content:space-between;align-items:center;}
.recipe-box{background:rgba(56,189,248,.08);border:2px solid rgba(56,189,248,.4);
  border-radius:16px;padding:18px 22px;margin:14px 0;}
.recipe-box h3{color:#38bdf8!important;margin:0 0 10px 0;}
.recipe-row{display:flex;justify-content:space-between;padding:8px 0;
  border-bottom:1px dashed rgba(148,163,184,.2);}
.recipe-row:last-child{border-bottom:none;}
.recipe-label{color:#94a3b8;font-size:.9rem;}
.recipe-value{color:#f8fafc;font-weight:700;font-size:1rem;}
.recipe-tag{background:rgba(52,211,153,.15);color:#34d399;font-size:.7rem;padding:2px 8px;
  border-radius:6px;margin-left:8px;font-weight:600;text-transform:uppercase;}
.recipe-tag.input{background:rgba(251,191,36,.15);color:#fbbf24;}
.recipe-tag.calc{background:rgba(167,139,250,.15);color:#a78bfa;}
.plain{background:rgba(251,191,36,.07);border-left:3px solid #fbbf24;padding:8px 14px;
  border-radius:0 10px 10px 0;color:#cbd5e1;font-size:.88rem;margin-top:6px;}
footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

def style(fig, h=380, **kw):
    """Applies a uniform dark theme to all Plotly charts."""
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10), **kw)
    return fig

def recipe_item(label, value, tag="ml"):
    """Generates formatted HTML rows for the final optimized recipe."""
    tag_label = {"ml": "ML-PREDICTED", "input": "YOUR INPUT", "calc": "CALCULATED"}[tag]
    tag_class = {"ml": "", "input": "input", "calc": "calc"}[tag]
    return (f"<div class='recipe-row'><span class='recipe-label'>{label}</span>"
            f"<span class='recipe-value'>{value}"
            f"<span class='recipe-tag {tag_class}'>{tag_label}</span></span></div>")


# 3. DATA LOADING & PREPROCESSING
NA = ["-", " -", "- ", "", " ", "—", "–", "NA", "N/A", "na", "nan"]

@st.cache_data
def load_data():
    """Loads and sanitizes the main dataset from directory relative to this script."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(BASE_DIR, "dataset.xlsx")
    csv_path = os.path.join(BASE_DIR, "dataset.csv")

    df = None
    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path, na_values=NA)
        except Exception:
            df = None
    if df is None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, na_values=NA, encoding="latin1")
    if df is None:
        return None

    df.columns = (df.columns.astype(str)
                  .str.replace("\n", " ")
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True))
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].dropna(how="all")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: x.strftime("%H:%M:%S") if hasattr(x, "strftime")
                else ("" if x is None else str(x))
            )
            df[col] = df[col].replace(
                {"nan": np.nan, "None": np.nan, "NaT": np.nan, "": np.nan, "NaN": np.nan}
            )

    return df

df = load_data()
if df is None:
    st.error("❌ Critical Error: Please place `dataset.xlsx` in the same folder as app.py")
    st.stop()
df = df.copy()

def find_col(*keys, exclude=()):
    """Helper to dynamically locate columns regardless of exact naming."""
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keys) and not any(e in cl for e in exclude):
            return c
    return None

C = {
    "support": next((c for c in df.columns if c.lower() == "support"), None) or find_col("support", exclude=("bet", "pore")),
    "amine": find_col("amine 1 or"), "mw": find_col("mw"), "org": find_col("organic content"),
    "n": find_col("n content"), "bet": find_col("bet"), "pv": find_col("pore volume"),
    "pdia": find_col("pore diam"), "rh": find_col("humidity"), "conc": find_col("concentration"),
    "flow": find_col("flow rate"), "time": find_col("adsorption time"),
    "temp": find_col("adsorption temp", exclude=("desorption",)),
    "q": find_col("capacity (mmol"), "dH": find_col("heat of adsorption"),
}

missing = [k for k in ["support", "amine", "q", "temp", "time", "rh"] if C[k] is None]
if missing:
    st.error(f"❌ Missing critical columns in dataset: {missing}")
    st.stop()

SUP, AMN, Q = C["support"], C["amine"], C["q"]
TEMP, TIME, RHC = C["temp"], C["time"], C["rh"]
CONC, FLOW, ORG, NC, BET = C["conc"], C["flow"], C["org"], C["n"], C["bet"]

for k in ["mw", "org", "n", "bet", "pv", "pdia", "rh", "conc", "flow", "time", "temp", "q", "dH"]:
    if C[k]: df[C[k]] = pd.to_numeric(df[C[k]], errors="coerce")
for c in [SUP, AMN]:
    df[c] = df[c].astype(str).str.strip().replace({"nan": "None", "": "None"})

dH_default = 70
if C["dH"] and df[C["dH"]].notna().sum() > 3:
    med = df[C["dH"]].dropna().abs().median()
    if 0 < med < 1: med *= 1000
    if 30 < med < 120: dH_default = int(round(med))

CAT   = [SUP, AMN]
MAT   = CAT + [C[k] for k in ["mw", "org", "n", "bet", "pv", "pdia"] if C[k]]
OP    = [c for c in [RHC, CONC, FLOW, TIME, TEMP] if c]
FEATS = MAT + OP

ENC = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in CAT}

SHORT = {SUP: "Support", AMN: "Amine type", C["mw"]: "Amine MW", C["org"]: "Amine loading (wt%)",
         C["n"]: "N content (mmol/g)", C["bet"]: "BET area", C["pv"]: "Pore volume",
         C["pdia"]: "Pore diameter", RHC: "Relative humidity", CONC: "CO₂ concentration",
         FLOW: "Flow rate", TIME: "Adsorption time", TEMP: "Adsorption temperature"}

def encode(X):
    """Encodes categorical strings into numeric indices for the ML model."""
    X = X[FEATS].copy()
    for c in CAT: 
        X[c] = X[c].map(ENC[c])
    return X.astype(float)


# 4. MACHINE LEARNING MODEL 
@st.cache_resource(show_spinner="🧠 Training ML model on experimental data…")
def build_model():

   
    d = df[df[Q].notna()]
    X, y = encode(d), d[Q].values
    
    cat_mask = [c in CAT and len(ENC[c]) < 250 for c in FEATS]
    
    def create_model():
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=10, l2_regularization=0.1,
            categorical_features=cat_mask, random_state=42
        )
        
    cv = cross_val_score(create_model(), X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m_eval = create_model().fit(Xtr, ytr)
    yp = m_eval.predict(Xte)
    
    pi = permutation_importance(m_eval, Xte, yte, n_repeats=5, random_state=42)
    imp = pd.Series(pi.importances_mean, index=[SHORT.get(c, c) for c in FEATS]).sort_values()
    
    final_model = create_model().fit(X, y)
    
    metrics = dict(cv=cv.mean(), cv_sd=cv.std(), r2=r2_score(yte, yp),
                   mae=mean_absolute_error(yte, yp), n=len(d))
                   
    return final_model, metrics, yte, yp, imp

model, MET, y_test, y_pred, IMP = build_model()


@st.cache_data
def build_materials(min_exp):
    """Filters database to find statistically reliable sorbents."""
    d = df[df[Q].notna()]
    agg = {"n_exp": (Q, "size"), "q_obs": (Q, "max")}
    if FLOW: agg["flow_med"] = (FLOW, "median")
    
    mats = d.groupby(MAT, dropna=False).agg(**agg).reset_index()
    if FLOW: mats = mats.rename(columns={"flow_med": FLOW})
    
    mats = mats[mats["n_exp"] >= min_exp].reset_index(drop=True)
    mats["mat_id"] = np.arange(len(mats))
    
    def create_label(r):
        org = f" · {r[ORG]:.0f} wt%" if ORG and pd.notna(r[ORG]) else ""
        return f"{r[AMN]} / {r[SUP]}{org}"
        
    mats["label"] = mats.apply(create_label, axis=1)
    dup = mats["label"].duplicated(keep=False)
    mats.loc[dup, "label"] += " #" + mats.loc[dup, "mat_id"].astype(str)
    return mats


@st.cache_data(show_spinner="🔎 Searching optimal operating window…")
def predict_grid(T_opts, RH_opts, t_opts, sup_sel, am_sel, min_exp):
    """Generates multi-dimensional matrix of combinations and predicts CO2 capacity for all."""
    mats = build_materials(min_exp)
    mats = mats[mats[SUP].isin(sup_sel) & mats[AMN].isin(am_sel)]
    if mats.empty: return pd.DataFrame()
    
    # Cross-join materials with operating conditions
    cond = pd.DataFrame([(T, r, t) for T in T_opts for r in RH_opts for t in t_opts],
                        columns=[TEMP, RHC, TIME])
    g = mats.merge(cond, how="cross")
    if CONC: g[CONC] = 0.04
    
    # Predict capacity (clip to prevent impossible negative values)
    g["q"] = np.clip(model.predict(encode(g)), 0.01, None)
    return g

def predict_for(mrow, T, RH, t):
    """Targeted prediction wrapper for charting dynamic variables."""
    T, RH, t = np.broadcast_arrays(*[np.atleast_1d(np.asarray(a, float)) for a in (T, RH, t)])
    g = pd.concat([mrow] * len(T), ignore_index=True)
    g[TEMP], g[RHC], g[TIME] = T, RH, t
    if CONC: g[CONC] = 0.04
    return np.clip(model.predict(encode(g)), 0.01, None)


# 5. TECHNO-ECONOMIC ANALYSIS (TEA) ENGINE
def techno_econ(q, t_ads, T_ads, RH, T_des, P):
    """
    Core Chemical Engineering constraints and Economics block.
    Uses Standard Lang Factor approaches and Six-Tenths rule for CAPEX scaling.
    """
    q, t_ads, T_ads, RH, T_des = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(a, float)) for a in (q, t_ads, T_ads, RH, T_des)])

    # Physics & Cycle Modeling
    regen  = 1 - np.exp(-(T_des - 40) / 25)  # Empirical desorption kinetics
    dq     = np.maximum(q * regen, 1e-3)     # Working capacity (delta-Q)
    t_cyc  = t_ads + P["t_des"] + P["t_cool"]
    cycles = 525600 * P["avail"] / t_cyc
    prod   = dq * M_CO2 * cycles
    m_sorb = P["cap"] / prod                 # Mass of sorbent required

    # Energy Modeling (Thermodynamics)
    E_sens = P["cp"] * np.maximum(T_des - T_ads, 1) / dq   # Sensible heat
    E_rxn  = np.full_like(dq, P["dH"])                     # Reaction enthalpy
    E_h2o  = P["kw"] * RH / 100 * 44.0 / dq                # Latent heat of water
    E_th   = (E_sens + E_rxn + E_h2o) / P["eta_th"]
    GJ_th  = E_th / 44.0
    
    # Fluid Dynamics (Air Flow)
    V_air   = MOL_PER_T / (400e-6 * P["phi"]) * 0.0245
    air_t   = V_air * RHO_AIR / 1000                       # tonnes air per tonne CO2
    kWh_fan = P["dP"] * V_air / 0.6 / 3.6e6                # Fan energy
    kWh_el  = kWh_fan + P["aux"]
    GJ_tot  = GJ_th + kWh_el * 0.0036

    # CAPEX Scaling (Six-Tenths Rule)
    hrs   = 8760 * P["avail"]
    Q_MW  = GJ_th * P["cap"] * 1e3 / (hrs * 3600)
    V_s   = V_air * P["cap"] / (hrs * 3600)
    co2_h = P["cap"] / hrs
    
    pec_cont = 25.0 * (m_sorb / 500) ** 0.6
    pec_heat = 1.5 * (Q_MW / 10) ** 0.6
    pec_fan  = 2.0 * (V_s / 1000) ** 0.6
    pec_comp = 5.0 * (co2_h / 10) ** 0.6
    
    PEC  = pec_cont + pec_heat + pec_fan + pec_comp
    FCI  = PEC * P["lang"]  # Fixed Capital Investment
    sorb_fill = m_sorb * P["sorb_price"] / 1e3
    CAPEX = FCI + sorb_fill
    
    # Capital Recovery Factor (CRF)
    i, n = P["i"], P["life"]
    CRF = i * (1 + i) ** n / ((1 + i) ** n - 1)
    cap_charge = CAPEX * CRF

    # OPEX Estimation
    op_heat = GJ_th * P["cap"] * P["heat_price"] / 1e6
    op_elec = kWh_el * P["cap"] * P["elec_price"] / 1e6
    op_sorb = sorb_fill / P["sorb_life"]
    op_om   = P["om"] * FCI
    op_lab  = 1.0 * (P["cap"] / 1e5) ** 0.3
    OPEX = op_heat + op_elec + op_sorb + op_om + op_lab
    
    # Primary Output: Levelized Cost of Capture ($/tCO2)
    LCOC = (cap_charge + OPEX) * 1e6 / P["cap"]

    out = dict(dq=dq, regen=regen, t_cyc=t_cyc, cycles=cycles, prod=prod, m_sorb=m_sorb,
               E_sens=E_sens, E_rxn=E_rxn, E_h2o=E_h2o, GJ_th=GJ_th, kWh_el=kWh_el,
               kWh_fan=kWh_fan, GJ_tot=GJ_tot, Q_MW=Q_MW, V_air=V_air, air_t=air_t,
               pec_cont=pec_cont, pec_heat=pec_heat, pec_fan=pec_fan, pec_comp=pec_comp,
               PEC=PEC, FCI=FCI, sorb_fill=sorb_fill, CAPEX=CAPEX, cap_charge=cap_charge,
               op_heat=op_heat, op_elec=op_elec, op_sorb=op_sorb, op_om=op_om, op_lab=op_lab,
               OPEX=OPEX, LCOC=LCOC)
               
    return {k: np.broadcast_to(np.asarray(v, float), q.shape).astype(float) for k, v in out.items()}

def optimize_Tdes(g, P, obj_key):
    """Finds the theoretically ideal desorption temperature for a given scenario."""
    best, bm = None, None
    for Td in TDES_OPTS:
        e = techno_econ(g["q"].values, g[TIME].values, g[TEMP].values, g[RHC].values, Td, P)
        metric = e["GJ_tot"] if obj_key == "GJ_tot" else e["LCOC"]
        if best is None:
            best, bm = e, metric
            best["Tdes"] = np.full(len(g), Td, float)
        else:
            better = metric < bm
            for k in e: best[k] = np.where(better, e[k], best[k])
            best["Tdes"] = np.where(better, Td, best["Tdes"])
            bm = np.where(better, metric, bm)
    out = g.copy()
    for k, v in best.items(): out[k] = v
    return out


# 6. SIDEBAR CONTROLS & USER INPUTS
sups, ams = sorted(df[SUP].unique()), sorted(df[AMN].unique())
OBJ = {"💰 Minimum cost (LCOC)": "LCOC",
       "🫧 Maximum CO₂ capacity": "q",
       "⚡ Minimum energy": "GJ_tot"}

with st.sidebar:
    st.markdown("## 🏭 S-DAC Plant Optimizer")
    st.caption(f"📊 {len(df):,} experiments · {len(sups)} supports · {len(ams)} amines")
    st.markdown("---")

    st.markdown("### 🎯 Design Target")
    cap = st.select_slider(
        "Annual CO₂ capture target (t CO₂/yr)",
        options=[4000, 10000, 36000, 100000, 250000, 500000, 1000000],
        value=100000, format_func=lambda x: f"{x:,} t/yr",
        help="How much CO₂ do you want the plant to remove from air per year? "
             "This is YOUR engineering design goal. Example: Climeworks Orca = 4,000 t/yr."
    )

    st.markdown("### 📍 Plant Location")
    site_choice = st.selectbox("Choose site", ["🔎 Let optimizer choose best"] + list(SITES) + ["✏️ Custom"])
    if site_choice == "✏️ Custom":
        Tc = st.slider("Ambient T (°C)", -10, 45, 20)
        RHc = st.slider("Ambient RH (%)", 10, 90, 50)
        
    st.markdown("### 🎯 Optimization Objective")
    obj_name = st.radio("Optimize for:", list(OBJ))
    obj_key = OBJ[obj_name]

    with st.expander("🧪 Sorbent search space"):
        sup_sel = st.multiselect("Supports to consider", sups, default=sups)
        am_sel = st.multiselect("Amines to consider", ams, default=ams)
        min_exp = st.slider("Minimum experiments per sorbent", 1, 10, 3,
                            help="Reliability filter: only consider sorbents tested at least this many times.")
        st.caption("💡 These filter which materials the ML optimizer searches over. "
                   "Narrower = faster + more trustworthy; Wider = more candidates.")

    with st.expander("⚙️ Process assumptions"):
        t_des = st.slider("Desorption step (min)", 10, 90, 30)
        t_cool = st.slider("Cooling step (min)", 5, 60, 20)
        avail = st.slider("Plant availability", 0.70, 0.98, 0.90, 0.01)
        phi = st.slider("Air-side capture fraction φ", 0.50, 0.95, 0.75, 0.05)
        dP = st.slider("Contactor pressure drop ΔP (Pa)", 50, 500, 150, 10)
        cp = st.slider("Heat capacity (kJ/kg·K)", 0.8, 3.0, 1.5, 0.1)
        dH = st.slider("−ΔH adsorption (kJ/mol)", 40, 100, dH_default)
        kw = st.slider("H₂O uptake at 100% RH (mol/kg)", 0.0, 10.0, 5.0, 0.5)
        eta_th = st.slider("Heat efficiency η", 0.60, 0.98, 0.85, 0.01)
        aux = st.slider("Vacuum + compression (kWh/t)", 50, 500, 200, 10)

    with st.expander("💰 Economic assumptions"):
        sorb_price = st.slider("Sorbent cost ($/kg)", 5, 100, 30)
        sorb_life = st.slider("Sorbent lifetime (yr)", 0.5, 5.0, 2.0, 0.5)
        heat_price = st.slider("Heat ($/GJ)", 2, 30, 8)
        elec_price = st.slider("Electricity ($/kWh)", 0.02, 0.20, 0.06, 0.01)
        i_rate = st.slider("Discount rate", 0.03, 0.15, 0.08, 0.01)
        life = st.slider("Plant life (yr)", 10, 30, 20)
        lang = st.slider("Lang factor", 2.5, 5.0, 3.6, 0.1)
        om = st.slider("O&M (fraction FCI/yr)", 0.01, 0.08, 0.04, 0.005)

# Group parameters into dict for passing to calculations
P = dict(cap=cap, t_des=t_des, t_cool=t_cool, avail=avail, phi=phi, dP=dP, cp=cp, dH=dH,
         kw=kw, eta_th=eta_th, aux=aux, sorb_price=sorb_price, sorb_life=sorb_life,
         heat_price=heat_price, elec_price=elec_price, i=i_rate, life=life, lang=lang, om=om)

# 7. EXECUTE OPTIMIZER
# Define boundary conditions based on dataset distributions
t_lo = max(10.0, float(df[TIME].quantile(0.02)))
t_hi = float(df[TIME].quantile(0.98))
T_lo = max(-10.0, float(np.floor(df[TEMP].min())))
T_hi = min(45.0, float(df[TEMP].max()))
RH_hi = float(min(90, df[RHC].max())) if df[RHC].notna().any() else 0.0

if site_choice == "🔎 Let optimizer choose best":
    T_opts = tuple(np.arange(T_lo, T_hi + 0.1, 5.0))
    RH_opts = tuple(r for r in [20.0, 40.0, 60.0, 80.0] if r <= max(RH_hi, 20)) or (20.0,)
    site_info = None
elif site_choice == "✏️ Custom":
    T_opts, RH_opts = (float(Tc),), (float(RHc),)
    site_info = dict(T=Tc, RH=RHc, note="Custom climate")
else:
    s = SITES[site_choice]
    T_opts, RH_opts = (float(s["T"]),), (float(s["RH"]),)
    site_info = s

t_opts = tuple(np.unique(np.clip([30, 60, 120, 180, 240, 360, 480, 720], t_lo, t_hi)).round())

grid = predict_grid(T_opts, RH_opts, t_opts, tuple(sup_sel), tuple(am_sel), min_exp)

if grid.empty:
    st.warning("⚠️ No sorbents match the filters. Widen the search space in the sidebar.")
    st.stop()

# Run the techno-economic optimization
res = optimize_Tdes(grid, P, obj_key)
if obj_key == "q":
    res = res.sort_values(["q", "LCOC"], ascending=[False, True])
else:
    res = res.sort_values(obj_key)
    
res = res.reset_index(drop=True)
best = res.iloc[0]
top10 = res.drop_duplicates("mat_id").head(10).reset_index(drop=True)
mat_cols = MAT + ([FLOW] if FLOW else [])
mrow = res.iloc[[0]][mat_cols].reset_index(drop=True)


# 8. MAIN DASHBOARD UI
tab1, tab2, tab3 = st.tabs(["🎯 Optimal Operating Parameters", "💰 CAPEX · OPEX · LCOC",
                            "🧠 ML Model & Methodology"])

# --- TAB 1: OPTIMAL RECIPE & KEY INSIGHTS ---
with tab1:
    # Site environment banner
    if site_info:
        st.markdown(f"""<div class='site-banner'>
        <div><b style='color:#34d399;font-size:1.1rem'>📍 {site_choice}</b><br>
        <span style='color:#94a3b8;font-size:.9rem'>{site_info['note']}</span></div>
        <div style='text-align:right;color:#f8fafc;font-size:1.1rem;font-weight:700'>
        🌡️ {site_info['T']}°C &nbsp;·&nbsp; 💧 {site_info['RH']}% RH</div>
        </div>""", unsafe_allow_html=True)

    n_eval = len(res) * len(TDES_OPTS)
    st.markdown(f"""<div class='hero'><h2>🎯 Finding the Optimal Plant Recipe</h2>
    <p>Design target: <b>{cap:,} t CO₂/yr</b> · Searched <b>{res['mat_id'].nunique()}</b> real sorbents × 
    operating conditions = <b>{n_eval:,}</b> ML-predicted scenarios · Objective: <b>{obj_name}</b></p></div>""",
    unsafe_allow_html=True)

    st.markdown("### ✅ Optimizer Result")
    st.markdown(f"""<div class='recipe-box'>
    <h3>After scanning {n_eval:,} combinations, the ML model recommends:</h3>
    {recipe_item("🧱 Sorbent material", best["label"], "ml")}
    {recipe_item("🌡️ Adsorption temperature (T_ads)", f"{best[TEMP]:.0f} °C", "input" if site_info else "ml")}
    {recipe_item("💧 Relative humidity (RH)", f"{best[RHC]:.0f} %", "input" if site_info else "ml")}
    {recipe_item("⏱️ Adsorption time (t_ads)", f"{best[TIME]:.0f} min", "ml")}
    {recipe_item("🔥 Desorption temperature (T_des)", f"{best['Tdes']:.0f} °C", "ml")}
    {recipe_item("🫧 Predicted CO₂ capacity (q)", f"{best['q']:.2f} mmol CO₂ / g sorbent", "ml")}
    {recipe_item("♻️ Working capacity (Δq)", f"{best['dq']:.2f} mmol CO₂ / g sorbent / cycle", "calc")}
    {recipe_item("⚙️ Sorbent productivity", f"{best['prod']:.1f} t CO₂ / t sorbent / yr", "calc")}
    {recipe_item("📦 Sorbent inventory needed", f"{best['m_sorb']:,.0f} tonnes", "calc")}
    {recipe_item("🌬️ Air to process per tonne CO₂", f"{best['air_t']:,.0f} t air (≈ {best['V_air']/1e6:.2f} million m³)", "calc")}
    {recipe_item("💰 Levelized Cost (LCOC)", f"${best['LCOC']:,.0f} / t CO₂ (₹{best['LCOC']*USD_INR:,.0f})", "calc")}
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='plain'>
    <b>Legend:</b> 
    <span style='color:#34d399'>■ ML-PREDICTED</span> = output of the ML model &nbsp;·&nbsp; 
    <span style='color:#fbbf24'>■ YOUR INPUT</span> = from your site/slider choice &nbsp;·&nbsp; 
    <span style='color:#a78bfa'>■ CALCULATED</span> = from engineering equations
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    a, b = st.columns(2)
    with a:
        st.markdown("#### ⏱️ How does CO₂ capacity change with adsorption time?")
        t_line = np.linspace(t_lo, t_hi, 40)
        q_line = predict_for(mrow, best[TEMP], best[RHC], t_line)
        e_t = techno_econ(q_line, t_line, best[TEMP], best[RHC], best["Tdes"], P)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=t_line, y=q_line, name="CO₂ capacity (ML)",
                                 line=dict(color="#38bdf8", width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=t_line, y=e_t["LCOC"], name="LCOC ($/t)",
                                 line=dict(color="#f97316", width=3, dash="dot")), secondary_y=True)
        j = int(np.argmin(e_t["LCOC"]))
        fig.add_vline(x=t_line[j], line_dash="dash", line_color="#34d399",
                      annotation_text=f"min cost at {t_line[j]:.0f} min")
        fig.update_yaxes(title_text="q (mmol/g)", secondary_y=False)
        fig.update_yaxes(title_text="LCOC ($/t)", secondary_y=True)
        style(fig, 380, xaxis_title="Adsorption time (min)", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""<div class='plain'>
        Longer adsorption lets more CO₂ reach equilibrium, but after a point extra time just enlarges
        the plant without extra capture. The green line marks the <b>economic sweet spot</b>.</div>""",
        unsafe_allow_html=True)

    with b:
        st.markdown("#### 🔥 How does desorption temperature affect energy?")
        Td_line = np.arange(70, 131, 5)
        e_d = techno_econ(best["q"], best[TIME], best[TEMP], best[RHC], Td_line, P)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for k, nm, col in [("E_sens", "Sensible heat", "#f97316"),
                           ("E_rxn", "Reaction heat (−ΔH)", "#a78bfa"),
                           ("E_h2o", "Water desorption", "#38bdf8")]:
            fig.add_trace(go.Bar(x=Td_line, y=e_d[k] / eta_th, name=nm, marker_color=col),
                          secondary_y=False)
        fig.add_trace(go.Scatter(x=Td_line, y=e_d["LCOC"], name="LCOC", mode="lines+markers",
                                 line=dict(color="#34d399", width=3)), secondary_y=True)
                                 
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title_text="Thermal energy (kJ/mol CO₂)", secondary_y=False)
        fig.update_yaxes(title_text="LCOC ($/t)", secondary_y=True)
        style(fig, 380, xaxis_title="Desorption temperature (°C)", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""<div class='plain'>
        Higher desorption T releases CO₂ more completely but costs more sensible + water-evap heat
        and degrades amines faster. The optimizer balances these three contributions.</div>""",
        unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("#### 🏆 Top-10 sorbents for this plant")
        fig = px.bar(top10, x="LCOC", y="label", orientation="h", color="q",
                     color_continuous_scale="Teal",
                     labels={"LCOC": "LCOC ($/t CO₂)", "label": "", "q": "q (mmol/g)"})
        fig.update_yaxes(categoryorder="total descending")
        style(fig, 400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""<div class='plain'>
        Each bar is a real sorbent from the dataset. Shorter bar = cheaper plant. 
        Color shows CO₂ capacity. The <b>#1 bar</b> is the optimal recipe above.</div>""",
        unsafe_allow_html=True)

    with b:
        st.markdown("#### 🏆 What controls CO₂ capacity most? (feature importance)")
        fig = px.bar(x=IMP.values, y=IMP.index, orientation="h", color=IMP.values,
                     color_continuous_scale="Blues",
                     labels={"x": "Impact on prediction accuracy", "y": ""})
        style(fig, 400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""<div class='plain'>
        The ML model learned on its own that <b>these parameters matter most</b> for CO₂ capacity. 
        Longer bar = bigger impact. This guides where to focus R&D effort.</div>""",
        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 Chemistry behind the optimum")
    k1, k2, k3 = st.columns(3)
    k1.markdown("""<div class='chem'><b>🌡️ Thermodynamics vs kinetics</b><br>
    Amine–CO₂ binding is exothermic (−ΔH ≈ 60–90 kJ/mol), so equilibrium favours low T
    (Le Chatelier / van't Hoff). But in polymeric amines (PEI) many N-sites are buried
    in pores and become accessible only when chains are mobile → capacity often
    <b>peaks at 40–75 °C</b>.</div>""", unsafe_allow_html=True)
    
    k2.markdown("""<div class='chem'><b>💧 Humidity: help and harm</b><br>
    Dry: carbamate, 2 N per CO₂ &nbsp;<code>CO₂ + 2RNH₂ ⇌ RNHCOO⁻ + RNH₃⁺</code><br>
    Humid: bicarbonate, 1 N per CO₂ &nbsp;<code>CO₂ + RNH₂ + H₂O ⇌ RNH₃⁺ + HCO₃⁻</code><br>
    → higher amine efficiency, <b>but</b> co-adsorbed water must be evaporated in regeneration.</div>""",
    unsafe_allow_html=True)
    
    k3.markdown("""<div class='chem'><b>🔄 TSA cycle design</b><br>
    Longer adsorption → more CO₂/cycle but fewer cycles/yr (bigger plant).
    Higher T<sub>des</sub> → more complete regeneration but more heat + faster degradation.
    The optimum is where <b>marginal capture gain = marginal cost</b>.</div>""", unsafe_allow_html=True)

# --- TAB 2: TECHNO-ECONOMIC ANALYSIS ---
with tab2:
    st.markdown(f"""<div class='hero'><h2>💰 Techno-Economic Analysis</h2>
    <p>🎯 Your design target: <b>{cap:,} t CO₂/yr</b> (set in sidebar) · 
    Costs scaled using six-tenths rule + Lang factor · Class 4/5 estimate (±30–50%)</p></div>""",
    unsafe_allow_html=True)

    st.markdown("""<div class='plain' style='font-size:.95rem'>
    📘 <b>LCOC = Levelized Cost of CO₂ Capture</b> — the <b>all-inclusive cost to remove 1 tonne of CO₂</b> 
    from air. It bundles: capital cost spread over the 20-year plant life + annual energy + sorbent + maintenance. 
    <b>Formula:</b> &nbsp;<code>LCOC = (CAPEX × CRF + Annual OPEX) / Annual CO₂ captured  [$/tCO₂]</code><br>
    <b>Benchmarks:</b> Today ~$600–1,000/t · DOE 2030 target: $100/t
    </div>""", unsafe_allow_html=True)

    cands = top10.head(5).reset_index(drop=True)
    pick = st.selectbox(
        "📌 Choose which top-5 design to analyze:", range(len(cands)),
        format_func=lambda k: (f"#{k+1} · {cands.loc[k, 'label']} · {cands.loc[k, TEMP]:.0f}°C, "
                               f"{cands.loc[k, RHC]:.0f}%RH, t_ads {cands.loc[k, TIME]:.0f}min, "
                               f"T_des {cands.loc[k, 'Tdes']:.0f}°C"))
                               
    r = cands.loc[pick]
    E = {k: float(v[0]) for k, v in
         techno_econ(r["q"], r[TIME], r[TEMP], r[RHC], r["Tdes"], P).items()}

    st.markdown("### 💵 Cost Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 Total CAPEX", f"${E['CAPEX']:,.0f} M",
              help="One-time cost to build the plant. Includes contactors, heat system, "
                   "fans, CO₂ compression, piping, civil work, and initial sorbent fill.")
    m2.metric("📐 Specific CAPEX", f"${E['CAPEX']*1e6/cap:,.0f} per (t/yr)",
              help="Capital cost per unit of annual capture capacity.")
    m3.metric("🔄 Annual OPEX", f"${E['OPEX']:,.1f} M/yr",
              help="Yearly running cost: heat, electricity, sorbent replacement, labor, maintenance.")
    m4.metric("🎯 LCOC", f"${E['LCOC']:,.0f} /t CO₂",
              help="Levelized Cost of Capture — the all-in cost per tonne of CO₂ removed.")

    st.markdown("### 📊 Physical Metrics")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("📦 Sorbent inventory", f"{E['m_sorb']:,.0f} t",
              help="Total mass of sorbent needed in the plant (adsorbing + regenerating beds).")
    m6.metric("♨️ Thermal energy", f"{E['GJ_th']:.2f} GJ/t",
              help="Heat required to regenerate the sorbent per tonne of CO₂ captured.")
    m7.metric("⚡ Electrical energy", f"{E['kWh_el']:,.0f} kWh/t",
              help="Electricity for fans, vacuum, and CO₂ compression.")
    m8.metric("🌬️ Air processed", f"{E['air_t']:,.0f} t air/t CO₂",
              help="At 400 ppm, you must move ~2,300 t of air to extract 1 t of CO₂.")

    st.markdown("---")
    a, b = st.columns(2)

    with a:
        st.markdown("#### 🏗️ Where does the CAPEX go?")
        labels = ["Air contactors (sorbent beds)", "Heating / steam system", "Fans & blowers",
                  "CO₂ vacuum & compression", "Installation (Lang factor)", "Initial sorbent fill"]
        vals = [E["pec_cont"], E["pec_heat"], E["pec_fan"], E["pec_comp"],
                E["FCI"] - E["PEC"], E["sorb_fill"]]
                
        fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.5, textinfo="percent",
                               marker=dict(colors=["#38bdf8", "#f97316", "#a78bfa",
                                                   "#34d399", "#64748b", "#fbbf24"])))
        fig.add_annotation(text=f"${E['CAPEX']:,.0f} M", showarrow=False,
                           font=dict(size=20, color="white"))
        style(fig, 400, legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        st.markdown("#### 🧾 Cost of capture build-up ($/t CO₂)")
        comp = {"Capital charge": E["cap_charge"], "Heat": E["op_heat"],
                "Electricity": E["op_elec"], "Sorbent replacement": E["op_sorb"],
                "Maintenance & insurance": E["op_om"], "Labour": E["op_lab"]}
        per_t = [v * 1e6 / cap for v in comp.values()]
        
        fig = go.Figure(go.Waterfall(
            x=list(comp) + ["LCOC"], measure=["relative"] * len(comp) + ["total"],
            y=per_t + [0], text=[f"${v:,.0f}" for v in per_t] + [f"${E['LCOC']:,.0f}"],
            textposition="outside", connector=dict(line=dict(color="#475569")),
            increasing=dict(marker=dict(color="#38bdf8")),
            totals=dict(marker=dict(color="#f97316"))))
            
        fig.add_hline(y=100, line_dash="dash", line_color="#34d399",
                      annotation_text="DOE target: $100/t")
        style(fig, 400, yaxis_title="$/t CO₂")
        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.markdown("#### 🌪️ Sensitivity (±25%) — what moves the cost most?")
        base = E["LCOC"]
        items = [("Sorbent price", "sorb_price"), ("Sorbent lifetime", "sorb_life"),
                 ("Heat price", "heat_price"), ("Electricity price", "elec_price"),
                 ("Lang (installation) factor", "lang"), ("Discount rate", "i"),
                 ("Working capacity Δq (sorbent R&D)", "__q")]
                 
        rows = []
        for lab, key in items:
            vals = []
            for f in (0.75, 1.25):
                P2, qq = dict(P), r["q"]
                if key == "__q": qq = r["q"] * f
                else: P2[key] = P[key] * f
                vals.append(techno_econ(qq, r[TIME], r[TEMP], r[RHC], r["Tdes"], P2)["LCOC"][0])
            rows.append((lab, vals[0] - base, vals[1] - base))
            
        tor = pd.DataFrame(rows, columns=["p", "lo", "hi"])
        tor["span"] = (tor["hi"] - tor["lo"]).abs()
        tor = tor.sort_values("span")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(y=tor["p"], x=tor["lo"], base=base, orientation="h",
                             name="−25%", marker_color="#34d399"))
        fig.add_trace(go.Bar(y=tor["p"], x=tor["hi"], base=base, orientation="h",
                             name="+25%", marker_color="#f97316"))
        fig.add_vline(x=base, line_color="white", line_width=1)
        style(fig, 400, barmode="overlay", xaxis_title="LCOC ($/t CO₂)",
              legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        st.markdown("#### 📈 Economies of scale")
        caps = np.logspace(3, 6, 30)
        lc = [techno_econ(r["q"], r[TIME], r[TEMP], r[RHC], r["Tdes"], dict(P, cap=c))["LCOC"][0]
              for c in caps]
              
        fig = go.Figure(go.Scatter(x=caps, y=lc, mode="lines", line=dict(color="#38bdf8", width=3),
                                   name="This design"))
        for nm, c in [("Orca (4 kt)", 4000), ("Mammoth (36 kt)", 36000),
                       ("Stratos (500 kt)", 500000), ("Your plant", cap)]:
            yv = techno_econ(r["q"], r[TIME], r[TEMP], r[RHC], r["Tdes"], dict(P, cap=c))["LCOC"][0]
            fig.add_trace(go.Scatter(x=[c], y=[yv], mode="markers+text", text=[nm],
                                     textposition="top center", marker=dict(size=12), name=nm))
                                     
        style(fig, 400, xaxis_type="log", xaxis_title="Plant capacity (t CO₂/yr)",
              yaxis_title="LCOC ($/t)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    k1, k2, k3 = st.columns(3)
    k1.markdown("""<div class='chem'><b>🏗️ Why contactors dominate CAPEX</b><br>
    At 400 ppm, ~2,300 t of air must pass the sorbent per tonne of CO₂ captured. The plant is
    essentially a giant gas–solid contactor; its size scales with sorbent inventory = 
    plant capacity ÷ productivity.</div>""", unsafe_allow_html=True)
    k2.markdown("""<div class='chem'><b>🧱 Sorbent = the hidden OPEX</b><br>
    Amines degrade by oxidation (urea / imine formation) and evaporation. A 2-year life means the 
    entire inventory is replaced every 2 years — often the #1 or #2 cost item. 
    <b>Stability matters as much as capacity.</b></div>""", unsafe_allow_html=True)
    k3.markdown("""<div class='chem'><b>♨️ Low-grade heat advantage</b><br>
    S-DAC regenerates at 80–120 °C → can use <b>waste heat or geothermal</b> (why Climeworks is in 
    Iceland), unlike liquid DAC which needs ~900 °C calcination.</div>""", unsafe_allow_html=True)

# --- TAB 3: MACHINE LEARNING & METHODOLOGY ---
with tab3:
    st.markdown("<div class='hero'><h2>🧠 ML Model & Methodology</h2>"
                "<p>Histogram Gradient Boosting Regressor · handles missing values & categorical "
                "sorbents natively · 5-fold cross-validation</p></div>", unsafe_allow_html=True)

    st.markdown("### 📏 Model Accuracy Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("5-fold CV R²", f"{MET['cv']:.3f}", f"± {MET['cv_sd']:.3f}", delta_color="off")
    m2.metric("Hold-out R²", f"{MET['r2']:.3f}")
    m3.metric("MAE", f"{MET['mae']:.3f} mmol/g")
    m4.metric("Training experiments", f"{MET['n']:,}")

    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(f"""<div class='plain'><b>5-fold CV R²: {MET['cv']:.2f}</b><br>
    We split the data into 5 chunks, trained on 4 and tested on the 5th — repeated 5 times. 
    The model <b>consistently explains {MET['cv']*100:.0f}%</b> of the variation, 
    proving it's robust and not a lucky split.</div>""", unsafe_allow_html=True)
    e2.markdown(f"""<div class='plain'><b>Hold-out R²: {MET['r2']:.2f}</b><br>
    On a completely unseen 20% of the data, the model explains <b>{MET['r2']*100:.0f}%</b> 
    of what we observe — this is its real-world accuracy on new experiments.</div>""",
    unsafe_allow_html=True)
    e3.markdown(f"""<div class='plain'><b>MAE: {MET['mae']:.2f} mmol/g</b><br>
    On average, the prediction is off by ±{MET['mae']:.2f} mmol/g. If true capacity is 
    2.0, the model typically predicts between {2.0-MET['mae']:.2f} and {2.0+MET['mae']:.2f}.</div>""",
    unsafe_allow_html=True)
    e4.markdown(f"""<div class='plain'><b>{MET['n']:,} experiments</b><br>
    The model learned from this many real lab measurements across many sorbents, 
    temperatures, humidities, and contact times.</div>""", unsafe_allow_html=True)

    st.markdown("---")
    a, b = st.columns(2)
    with a:
        st.markdown("#### 🎯 Parity plot: Predicted vs Measured")
        fig = go.Figure(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                   marker=dict(color="#38bdf8", size=7, opacity=0.6,
                                               line=dict(width=0.5, color="white")),
                                   name="Test points"))
        lim = [0, float(max(y_test.max(), y_pred.max())) * 1.05]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction (y=x)",
                                 line=dict(color="#f97316", dash="dash")))
        style(fig, 420, xaxis_title="Measured CO₂ capacity (mmol/g)",
              yaxis_title="Predicted CO₂ capacity (mmol/g)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""<div class='plain'>Points close to the orange line = good predictions. 
        A tight cluster around the line proves the model generalizes well.</div>""",
        unsafe_allow_html=True)

    with b:
        st.markdown("#### 🏆 Feature importance (what controls CO₂ capacity)")
        fig = px.bar(x=IMP.values, y=IMP.index, orientation="h", color=IMP.values,
                     color_continuous_scale="Teal", labels={"x": "Drop in R² when shuffled", "y": ""})
        style(fig, 420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""<div class='plain'>We shuffle each feature and measure how much accuracy drops. 
        A bigger drop = that feature matters more for predicting CO₂ capacity.</div>""",
        unsafe_allow_html=True)

    with st.expander("📐 Equations used in the optimizer", expanded=False):
        st.markdown("**1 · Working capacity (ML prediction × regeneration completeness)**")
        st.latex(r"\Delta q = q_{ML}(\text{sorbent},T_{ads},RH,t_{ads})\cdot\left[1-e^{-(T_{des}-40)/25}\right]")
        st.markdown("**2 · Cycle, productivity, sorbent inventory**")
        st.latex(r"N_{cyc}=\frac{525\,600\,A}{t_{ads}+t_{des}+t_{cool}}\qquad "
                 r"P=\Delta q\,M_{CO_2}\,N_{cyc}\qquad m_{sorb}=\frac{\dot m_{CO_2}}{P}")
        st.markdown("**3 · Regeneration heat (sensible + reaction + water)**")
        st.latex(r"E_{th}=\frac{1}{\eta}\left[\frac{c_p(T_{des}-T_{ads})}{\Delta q}+(-\Delta H_{ads})"
                 r"+\frac{w_{H_2O}\,\Delta H_{vap}}{\Delta q}\right]")
        st.markdown("**4 · Air processing requirement**")
        st.latex(r"V_{air}=\frac{n_{CO_2}}{y_{CO_2}\,\varphi}V_m,\quad "
                 r"m_{air}=V_{air}\rho_{air}\approx 2300\text{ t air per t CO}_2")
        st.markdown("**5 · CAPEX (six-tenths rule + Lang factor)**")
        st.latex(r"C=C_{ref}\left(\frac{S}{S_{ref}}\right)^{0.6}\qquad FCI=f_{Lang}\sum PEC")
        st.markdown("**6 · Levelized cost of capture**")
        st.latex(r"CRF=\frac{i(1+i)^n}{(1+i)^n-1}\qquad LCOC=\frac{CAPEX\cdot CRF+OPEX}{\dot m_{CO_2}}")

    with st.expander("🔍 View raw dataset"):
        st.dataframe(df, use_container_width=True, height=300)

# Footer
st.markdown("<div style='text-align:center;color:#475569;font-size:.78rem;padding:10px'>"
            "🏭 S-DAC Plant Optimizer · ML + Techno-Economics · Chemical Technology Round 3"
            "</div>", unsafe_allow_html=True)