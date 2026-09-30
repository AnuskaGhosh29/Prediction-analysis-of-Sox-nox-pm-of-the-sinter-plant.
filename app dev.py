"""
Stack emission prediction monitor - sinter plant.
All data sits inside triple-quoted blocks, so a line lost in a copy-paste
costs you one row, never a syntax error. Run: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
# ---------------------------------------------------------------- data blocks
PAIRS = """
PM 32.161 32.406 | PM 32.641 32.756 | PM 25.253 25.384 | PM 22.962 23.093
PM 23.769 23.930 | PM 38.520 38.486 | PM 26.189 26.169 | PM 21.854 21.865
PM 36.489 36.610 | PM 27.444 27.443 | PM 23.078 23.096 | PM 31.842 31.754
PM 24.521 24.362 | PM 29.179 29.201 | PM 29.854 29.941 | PM 29.183 29.376
PM 30.924 30.979 | PM 21.624 21.804 | PM 26.659 26.659 | PM 36.476 36.463
SOx 276.401 276.403 | SOx 199.183 196.582 | SOx 266.385 268.138
SOx 338.817 338.817 | SOx 377.679 377.672 | SOx 182.714 180.276
SOx 254.149 253.900 | SOx 282.224 282.231 | SOx 201.317 199.733
SOx 259.039 259.126 | SOx 321.966 322.118 | SOx 219.807 218.861
SOx 316.943 315.798 | SOx 191.893 191.909 | SOx 353.488 354.183
SOx 361.845 360.535 | SOx 207.377 207.957 | SOx 324.629 326.406
SOx 207.959 208.543 | SOx 181.208 181.047
NOx 98.953 99.227 | NOx 68.920 68.919 | NOx 137.676 137.682
NOx 69.998 68.617 | NOx 336.228 336.213 | NOx 64.508 64.509
NOx 66.801 66.512 | NOx 166.823 164.084 | NOx 64.239 65.676
NOx 74.100 74.493 | NOx 377.339 379.079 | NOx 69.386 66.421
NOx 4.896 6.873 | NOx 63.214 62.908 | NOx 75.934 75.938
NOx 48.640 46.696 | NOx 67.014 66.980 | NOx 52.542 52.542
NOx 70.534 67.462 | NOx 58.895 61.020
"""
# pollutant ; parameter ; mean absolute SHAP
SHAP = """
PM ; Sinter machine speed ; 0.7757
PM ; Suction / under pressure ; 0.7479
PM ; Moisture content ; 0.7283
PM ; Bed height ; 0.4645
PM ; Air-fuel ratio ; 0.4036
PM ; Granulation index ; 0.1667
PM ; MPV ; 0.1332
PM ; Solid fuel ratio ; 0.1038
PM ; Coke breeze temperature ; 0.0948
PM ; Waste gas flow ; 0.0909
SOx ; Bed height (H) ; 17.0315
SOx ; Coke breeze temperature ; 7.5610
SOx ; Sinter speed ; 7.4592
SOx ; Waste gas flow (Nm3) ; 7.1339
SOx ; Ignition temperature ; 5.5101
SOx ; ESP inlet suction (P) ; 3.0114
SOx ; Total moisture % (M) ; 2.1133
SOx ; Lime consumed ; 1.3819
SOx ; Fe-S % ; 0.8155
NOx ; Flame front speed ; 10.0856
NOx ; ESP inlet suction (P) ; 5.7801
NOx ; CaO/SiO2 ; 4.3882
NOx ; CA flow ; 4.3075
NOx ; Ignition temperature ; 3.2249
NOx ; Total moisture % (M) ; 2.2565
NOx ; Waste gas flow (SM3) ; 2.0994
NOx ; Bed height (H) ; 1.9739
NOx ; Coke breeze temperature ; 1.4594
NOx ; VM concentration ; 0.0268
"""
# pollutant ; parameter ; observed low ; observed high ; fav low ; fav high ; best ; min avg BANDS = """
PM ; Sinter machine speed ; 2.022 ; 2.252 ; 2.022 ; 2.252 ; 2.25 ; 27.57
PM ; Suction / under pressure ; 116.559 ; 134.144 ; 116.559 ; 134.144 ; 117.092 ; 27.318
PM ; Moisture content ; 11.254 ; 15.705 ; 11.254 ; 15.705 ; 11.658 ; 26.801
PM ; Bed height ; 745.006 ; 749.647 ; 745.006 ; 749.647 ; 748.193 ; 28.049
PM ; Air-fuel ratio ; 1.787 ; 1.896 ; 1.787 ; 1.896 ; 1.793 ; 28.067
PM ; Granulation index ; 29.304 ; 34.121 ; 29.304 ; 34.121 ; 31.299 ; 28.341
PM ; MPV ; 0 ; 8.668 ; 0 ; 8.668 ; 7.179 ; 28.329
PM ; Solid fuel ratio ; 0.021 ; 0.028 ; 0.021 ; 0.028 ; 0.022 ; 28.28
PM ; Coke breeze temperature ; 15.592 ; 27.223 ; 15.592 ; 27.223 ; 22.641 ; 28.422
PM ; Waste gas flow ; 989730.114 ; 1087415.924 ; 989730.114 ; 1087415.924 ; 998610.642 ; 28.373
SOx ; Bed height (H) ; 745.006 ; 749.647 ; 748.85 ; 749.647 ; 749.647 ; 232.692
SOx ; Coke breeze temperature ; 0 ; 18.11 ; 15.184 ; 18.11 ; 18.11 ; 258.707
SOx ; Sinter speed ; 2.022 ; 2.252 ; 2.022 ; 2.029 ; 2.022 ; 266.504
SOx ; Waste gas flow (Nm3) ; 989730.114 ; 1087415.924 ; 1063734.516 ; 1087415.924 ; 1084455.748 SOx ; Ignition temperature ; 1034.956 ; 1157.611 ; 1034.956 ; 1057.257 ; 1034.956 ; 266.483
SOx ; ESP inlet suction (P) ; 116.559 ; 134.144 ; 131.302 ; 134.144 ; 134.144 ; 270.136
SOx ; Total moisture % (M) ; 11.254 ; 15.705 ; 11.568 ; 12.648 ; 12.108 ; 272.435
SOx ; Lime consumed ; 143.906 ; 216.589 ; 143.906 ; 154.919 ; 146.109 ; 271.751
SOx ; Fe-S % ; 0.002 ; 0.034 ; 0.002 ; 0.003 ; 0.002 ; 272.224
NOx ; Flame front speed ; 19.324 ; 21.497 ; 19.324 ; 19.85 ; 19.324 ; 85.833
NOx ; ESP inlet suction (P) ; 116.559 ; 134.144 ; 123.842 ; 125.973 ; 124.375 ; 86.379
NOx ; CaO/SiO2 ; 1.522 ; 2.103 ; 1.522 ; 1.634 ; 1.528 ; 87.133
NOx ; CA flow ; 8320.141 ; 9224.63 ; 8676.455 ; 8895.725 ; 8713 ; 88.19
NOx ; Ignition temperature ; 1034.956 ; 1157.611 ; 1096.903 ; 1110.531 ; 1108.053 ; 91.054
NOx ; Total moisture % (M) ; 11.254 ; 15.705 ; 11.254 ; 12.063 ; 11.658 ; 89.119
NOx ; Waste gas flow (SM3) ; 989730.11 ; 1087415.92 ; 1003544.27 ; 1014398.25 ; 1012424.8 ; 91.238
NOx ; Bed height (H) ; 745.006 ; 749.647 ; 748.897 ; 749.647 ; 749.647 ; 91.764
NOx ; Coke breeze temperature ; 17.75 ; 32.741 ; 27.138 ; 28.652 ; 27.744 ; 92.079
NOx ; VM concentration ; 0 ; 0 ; 0 ; 0 ; 0 ; 93.986
"""
# XGBoost: pollutant ; training R2 ; training MAE ; training RMSE ; tolerance
MODELS = """
PM ; 0.9991 ; 0.0891 ; 0.1148 ; 0.15
SOx ; 0.9996 ; 0.8047 ; 1.1543 ; 1.5
NOx ; 0.9994 ; 1.0352 ; 1.5057 ; 2.0
"""
# Consent exceedances - plant records, not model output.
# pollutant ; year ; count ; part-year (yes/no)
EXCEED = """
PM ; FY25 ; 37 ; no
PM ; FY26 ; 18 ; no
PM ; FY27 ; 7 ; yes
"""
EXCEED_META = {"PM": ("Stack dust", "Sinter Plant 3", 30)}
UNIT = "mg/Nm3"
LABEL = {"PM": "PM", "SOx": "SO2", "NOx": "NOx"}
ORDER = ["PM", "SOx", "NOx"]
INK, SLATE, MIST, RULE = "#1B2A33", "#5A6B74", "#94A2A9", "#E1E6E7"
PAPER, STEEL, TEAL, EMBER, MOSS, SAND = "#FAFBFA", "#2F6079", "#2C8C84", "#B14A24", "#4C7A50", PALETTE = ["#2F6079", "#3A7290", "#4785A4", "#2C8C84", "#5AA39B",
"#86B8B2", "#C9A227", "#D2BA63", "#C9CFC9", "#B6BEBD"]
# ---------------------------------------------------------------- parsing
def rows(block, sep=";"):
out = []
for line in block.strip().splitlines():
line = line.strip()
if not line or line.startswith("#"):
continue
out.append([c.strip() for c in line.split(sep)])
return out
def load_pairs():
data = {k: [] for k in ORDER}
for line in PAIRS.strip().splitlines():
for chunk in line.split("|"):
bits = chunk.split()
if len(bits) == 3 and bits[0] in data:
data[bits[0]].append((float(bits[1]), float(bits[2])))
return data
def load_shap():
data = {k: [] for k in ORDER}
for r in rows(SHAP):
if len(r) == 3 and r[0] in data:
data[r[0]].append((r[1], float(r[2])))
return data
def load_bands():
data = {k: [] for k in ORDER}
for r in rows(BANDS):
if len(r) == 8 and r[0] in data:
data[r[0]].append([r[1]] + [float(x) for x in r[2:]])
return data
def load_models():
data = {}
for r in rows(MODELS):
if len(r) == 5:
data[r[0]] = dict(r2=float(r[1]), mae=float(r[2]),
rmse=float(r[3]), tol=float(r[4]))
return data
def load_exceed():
data = {}
for r in rows(EXCEED):
if len(r) == 4:
data.setdefault(r[0], []).append((r[1], int(r[2]), r[3].lower() == "yes"))
return data
PAIR_D, SHAP_D, BAND_D = load_pairs(), load_shap(), load_bands()
MODEL_D, EXCEED_D = load_models(), load_exceed()
# ---------------------------------------------------------------- page
st.set_page_config(page_title="Stack emission monitor", layout="wide")
st.markdown("""
<style>
.stApp { background:%s; }
.block-container { padding-top:1.6rem; max-width:1500px; }
.stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid %s; }
.stTabs [data-baseweb="tab"] { height:44px; padding:0 22px; background:transparent;
color:%s; font-size:14.5px; border-bottom:3px solid transparent; }
.stTabs [aria-selected="true"] { color:%s; font-weight:600; border-bottom-color:%s; }
.stTabs [data-baseweb="tab-highlight"] { background:transparent; }
.statrow { display:grid; gap:1px; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
background:%s; border:1px solid %s; margin-bottom:14px; }
.stat { background:#fff; padding:14px 16px; }
.stat .k { font-size:12px; color:%s; }
.stat .v { font-size:25px; font-weight:600; margin-top:4px; }
.stat .s { font-size:11.5px; color:%s; margin-top:3px; }
</style>
""" % (PAPER, RULE, MIST, INK, SAND, RULE, RULE, SLATE, MIST), unsafe_allow_html=True)
def stat(k, v, s, tone=INK):
return ("<div class='stat'><div class='k'>%s</div>"
"<div class='v' style='color:%s'>%s</div>"
"<div class='s'>%s</div></div>" % (k, tone, v, s))
def frame(fig, height):
fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
plot_bgcolor="white", paper_bgcolor="white",
font=dict(size=11, color=SLATE),
xaxis=dict(gridcolor=RULE, zeroline=False),
yaxis=dict(gridcolor=RULE, zeroline=False))
return fig
def num(v):
if abs(v) >= 10000:
return "{:,.0f}".format(v)
if abs(v) >= 100:
return "{:.1f}".format(v)
return "{:g}".format(v)
st.title("Stack emission - prediction monitor")
st.caption("Sinter plant - model output vs measured value")
tabs = st.tabs([LABEL[k] for k in ORDER])
for tab, key in zip(tabs, ORDER):
with tab:
pairs = PAIR_D[key]
if not pairs:
st.info("No data for " + LABEL[key])
continue
df = pd.DataFrame(pairs, columns=["Measured", "Predicted"])
df["Error"] = df["Measured"] - df["Predicted"]
mdl = MODEL_D.get(key, dict(r2=0, mae=0, rmse=0, tol=1))
st.markdown("<div class='statrow'>"
+ stat("Training R2", "{:.4f}".format(mdl["r2"]), "XGBoost")
+ stat("Training MAE", "{:.4f}".format(mdl["mae"]), UNIT)
+ stat("Training RMSE", "{:.4f}".format(mdl["rmse"]), UNIT)
+ "</div>", unsafe_allow_html=True)
# ---- consent exceedances ----
if key in EXCEED_D:
name, site, limit = EXCEED_META.get(key, ("", "", 0))
st.subheader("Consent exceedances")
st.caption("%s - %s - plant records, not model output" % (name, site))
years = EXCEED_D[key]
c1, c2 = st.columns([1.5, 1])
with c1:
ex = go.Figure()
ex.add_bar(x=[y[0] for y in years], y=[y[1] for y in years],
marker_color=[SAND if y[2] else EMBER for y in years],
text=[y[1] for y in years], textposition="outside")
ex.update_layout(showlegend=False, yaxis_title="events")
st.plotly_chart(frame(ex, 240), use_container_width=True,
key="exc_" + key)
with c2:
first, last = years[0], years[-1]
drop = round((1 - last[1] / first[1]) * 100) if first[1] else 0
st.markdown("<div class='statrow'>"
+ stat("Down since " + first[0], "%d%%" % drop,
"%d to %d events" % (first[1], last[1]), MOSS)
+ stat(last[0] + " run rate", "~%d" % (last[1] * 2),
"annualised from 6 months")
+ "</div>", unsafe_allow_html=True)
st.caption("Recorded breaches of the %d %s limit. %s covers April 2026 to date, "
"so it is not yet comparable with the full years before it."
% (limit, UNIT, years[-1][0]))
# ---- measured vs predicted ----
st.subheader("Measured against predicted")
tol = st.number_input("Tolerance +/- (%s)" % UNIT, min_value=0.0,
value=mdl["tol"], step=mdl["tol"] / 3, key="tol_" + key)
off = df["Error"].abs() > tol
fig = go.Figure()
fig.add_scatter(y=df["Measured"], mode="lines+markers", name="Measured",
line=dict(color=INK, width=2))
fig.add_scatter(y=df["Predicted"], mode="markers", name="Predicted",
marker=dict(color=SAND, size=8))
fig.update_layout(legend=dict(orientation="h", y=1.12, x=0),
xaxis_title="Observation", yaxis_title=UNIT)
st.plotly_chart(frame(fig, 330), use_container_width=True, key="tr_" + key)
left, right = st.columns(2)
with left:
st.subheader("Error by observation")
bar = go.Figure()
bar.add_bar(y=df["Error"], marker_color=np.where(off, EMBER, STEEL))
bar.add_hline(y=tol, line=dict(color=MIST, dash="dot"))
bar.add_hline(y=-tol, line=dict(color=MIST, dash="dot"))
bar.update_layout(showlegend=False, yaxis_title=UNIT)
st.plotly_chart(frame(bar, 265), use_container_width=True, key="res_" + key)
with right:
st.subheader("Prediction against measurement")
lo = float(min(df["Measured"].min(), df["Predicted"].min()))
hi = float(max(df["Measured"].max(), df["Predicted"].max()))
pad = (hi - lo) * 0.06
sc = go.Figure()
sc.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines",
line=dict(color=MIST, dash="dash"))
sc.add_scatter(x=df["Measured"], y=df["Predicted"], mode="markers",
marker=dict(color=STEEL, size=9, opacity=0.75))
sc.update_layout(showlegend=False,
xaxis=dict(title="Measured", range=[lo - pad, hi + pad], gridcolor=yaxis=dict(title="Predicted", range=[lo - pad, hi + pad], gridcolor=st.plotly_chart(frame(sc, 265), use_container_width=True, key="par_" + key)
# ---- drivers ----
def driver_frame(k):
d = pd.DataFrame(SHAP_D[k], columns=["Parameter", "Mean |SHAP|"])
d["Share %"] = d["Mean |SHAP|"] / d["Mean |SHAP|"].sum() * 100
return d.sort_values("Share %", ascending=False).reset_index(drop=True)
drv = driver_frame(key)
a, b = st.columns(2)
with a:
st.subheader("What drives the prediction")
pie = go.Figure(go.Pie(labels=drv["Parameter"], values=drv["Share %"],
hole=0.45, sort=False, textinfo="none",
marker=dict(colors=PALETTE[:len(drv)])))
pie.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(pie, use_container_width=True, key="pie_" + key)
st.caption("Top three account for %.1f%% of the signal"
% drv["Share %"].head(3).sum())
with b:
st.subheader("Driver ranking")
st.dataframe(drv.round(4), hide_index=True, use_container_width=True)
# ---- side by side ----
st.subheader("Contributing factors side by side")
cols = st.columns(3)
for col, k2 in zip(cols, ORDER):
with col:
d2 = driver_frame(k2)
st.markdown("**%s** - top three %.1f%%"
% (LABEL[k2], d2["Share %"].head(3).sum()))
p2 = go.Figure(go.Pie(labels=d2["Parameter"], values=d2["Share %"],
hole=0.5, sort=False, textinfo="none",
marker=dict(colors=PALETTE[:len(d2)])))
p2.update_layout(height=170, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
st.plotly_chart(p2, use_container_width=True, key="mini_%s_%s" % (key, k2))
for i, r in d2.iterrows():
st.markdown("<div style='font-size:12px'><span style='color:%s'>&#9632;</"%s - %.1f%%</div>" % (PALETTE[i % len(PALETTE)],
r["Parameter"], r["Share %"]),
unsafe_allow_html=True)
# ---- operating window ----
st.subheader("Favourable operating window")
bands = BAND_D[key]
if bands:
for nm, olo, ohi, flo, fhi, best, em in bands:
span = ohi - olo
if span <= 0:
st.markdown("**%s** - constant" % nm)
continue
lo_pct = (flo - olo) / span * 100
w_pct = max((fhi - flo) / span * 100, 1.2)
best_pct = (best - olo) / span * 100
st.markdown(
"<div style='display:flex;justify-content:space-between;font-size:13px'>"
"<b>%s</b><span style='color:%s'>%s - %s</span></div>"
"<div style='position:relative;height:10px;background:%s;border:1px solid "<div style='position:absolute;left:%.1f%%;width:%.1f%%;top:0;bottom:0;background:%"<div style='position:absolute;left:%.1f%%;width:2px;top:-3px;bottom:-3px;"<div style='display:flex;justify-content:space-between;font-size:11.5px;"<span>%s</span><span>best %s - holds emission near %s</span><span>%s</span></% (nm, SLATE, num(flo), num(fhi), PAPER, RULE,
lo_pct, w_pct, STEEL, best_pct, INK, MIST,
num(olo), num(best), num(em), num(ohi)),
unsafe_allow_html=True)
st.dataframe(
pd.DataFrame(bands, columns=["Parameter", "Observed low", "Observed high",
"Favourable low", "Favourable high",
"Best value", "Min avg emission"]),
hide_index=True, use_container_width=True)
else:
st.info("No operating window supplied for " + LABEL[key])
with st.expander("Observation record"):
st.dataframe(df.round(3), use_container_width=True)
