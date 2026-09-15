"""
Stack emission prediction monitor — sinter plant.
Single file, no external data. Run: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ data

PAIRS = {
    "PM": [(32.161, 32.406), (32.641, 32.756), (25.253, 25.384), (22.962, 23.093),
           (23.769, 23.930), (38.520, 38.486), (26.189, 26.169), (21.854, 21.865),
           (36.489, 36.610), (27.444, 27.443), (23.078, 23.096), (31.842, 31.754),
           (24.521, 24.362), (29.179, 29.201), (29.854, 29.941), (29.183, 29.376),
           (30.924, 30.979), (21.624, 21.804), (26.659, 26.659), (36.476, 36.463)],
    "SOx": [(276.401, 276.403), (199.183, 196.582), (266.385, 268.138), (338.817, 338.817),
            (377.679, 377.672), (182.714, 180.276), (254.149, 253.900), (282.224, 282.231),
            (201.317, 199.733), (259.039, 259.126), (321.966, 322.118), (219.807, 218.861),
            (316.943, 315.798), (191.893, 191.909), (353.488, 354.183), (361.845, 360.535),
            (207.377, 207.957), (324.629, 326.406), (207.959, 208.543), (181.208, 181.047)],
    "NOx": [(98.953, 99.227), (68.920, 68.919), (137.676, 137.682), (69.998, 68.617),
            (336.228, 336.213), (64.508, 64.509), (66.801, 66.512), (166.823, 164.084),
            (64.239, 65.676), (74.100, 74.493), (377.339, 379.079), (69.386, 66.421),
            (4.896, 6.873), (63.214, 62.908), (75.934, 75.938), (48.640, 46.696),
            (67.014, 66.980), (52.542, 52.542), (70.534, 67.462), (58.895, 61.020)],
}

SHAP = {
    "PM": [("Sinter machine speed", 0.7757), ("Suction / under pressure", 0.7479),
           ("Moisture content", 0.7283), ("Bed height", 0.4645), ("Air-fuel ratio", 0.4036),
           ("Granulation index", 0.1667), ("MPV", 0.1332), ("Solid fuel ratio", 0.1038),
           ("Coke breeze temperature", 0.0948), ("Waste gas flow", 0.0909)],
    "SOx": [("Bed height (H)", 17.0315), ("Coke breeze temperature", 7.5610),
            ("Sinter speed", 7.4592), ("Waste gas flow (Nm3)", 7.1339),
            ("Ignition temperature", 5.5101), ("ESP inlet suction (P)", 3.0114),
            ("Total moisture % (M)", 2.1133), ("Lime consumed", 1.3819), ("Fe-S %", 0.8155)],
    "NOx": [("Flame front speed", 10.0856), ("ESP inlet suction (P)", 5.7801),
            ("CaO/SiO2", 4.3882), ("CA flow", 4.3075), ("Ignition temperature", 3.2249),
            ("Total moisture % (M)", 2.2565), ("Waste gas flow (SM3)", 2.0994),
            ("Bed height (H)", 1.9739), ("Coke breeze temperature", 1.4594),
            ("VM concentration", 0.0268)],
}

BANDS = {
    # name, observed low, observed high, favourable low, favourable high, best value, min avg predicted emission
    "PM": [
        ("Sinter machine speed", 2.022, 2.252, 2.022, 2.252, 2.25, 27.57),
        ("Suction / under pressure", 116.559, 134.144, 116.559, 134.144, 117.092, 27.318),
        ("Moisture content", 11.254, 15.705, 11.254, 15.705, 11.658, 26.801),
        ("Bed height", 745.006, 749.647, 745.006, 749.647, 748.193, 28.049),
        ("Air-fuel ratio", 1.787, 1.896, 1.787, 1.896, 1.793, 28.067),
        ("Granulation index", 29.304, 34.121, 29.304, 34.121, 31.299, 28.341),
        ("MPV", 0.0, 8.668, 0.0, 8.668, 7.179, 28.329),
        ("Solid fuel ratio", 0.021, 0.028, 0.021, 0.028, 0.022, 28.28),
        ("Coke breeze temperature", 15.592, 27.223, 15.592, 27.223, 22.641, 28.422),
        ("Waste gas flow", 989730.114, 1087415.924, 989730.114, 1087415.924, 998610.642, 28.373),
    ],
    "SOx": [
        ("Bed height (H)", 745.006, 749.647, 748.85, 749.647, 749.647, 232.692),
        ("Coke breeze temperature", 0.0, 18.11, 15.184, 18.11, 18.11, 258.707),
        ("Sinter speed", 2.022, 2.252, 2.022, 2.029, 2.022, 266.504),
        ("Waste gas flow (Nm3)", 989730.114, 1087415.924, 1063734.516, 1087415.924, 1084455.748, 264.356),
        ("Ignition temperature", 1034.956, 1157.611, 1034.956, 1057.257, 1034.956, 266.483),
        ("ESP inlet suction (P)", 116.559, 134.144, 131.302, 134.144, 134.144, 270.136),
        ("Total moisture % (M)", 11.254, 15.705, 11.568, 12.648, 12.108, 272.435),
        ("Lime consumed", 143.906, 216.589, 143.906, 154.919, 146.109, 271.751),
        ("Fe-S %", 0.002, 0.034, 0.002, 0.003, 0.002, 272.224),
    ],
    "NOx": [
        ("Flame front speed", 19.324, 21.497, 19.324, 19.85, 19.324, 85.833),
        ("ESP inlet suction (P)", 116.559, 134.144, 123.842, 125.973, 124.375, 86.379),
        ("CaO/SiO2", 1.522, 2.103, 1.522, 1.634, 1.528, 87.133),
        ("CA flow", 8320.141, 9224.63, 8676.455, 8895.725, 8713.0, 88.19),
        ("Ignition temperature", 1034.956, 1157.611, 1096.903, 1110.531, 1108.053, 91.054),
        ("Total moisture % (M)", 11.254, 15.705, 11.254, 12.063, 11.658, 89.119),
        ("Waste gas flow (SM3)", 989730.11, 1087415.92, 1003544.27, 1014398.25, 1012424.8, 91.238),
        ("Bed height (H)", 745.006, 749.647, 748.897, 749.647, 749.647, 91.764),
        ("Coke breeze temperature", 17.75, 32.741, 27.138, 28.652, 27.744, 92.079),
        ("VM concentration", 0.0, 0.0, 0.0, 0.0, 0.0, 93.986),
    ],
}

MODELS = {
    "PM":  dict(name="XGBoost", train_r2=0.9991, train_mae=0.0891, train_rmse=0.1148, gap=0.5396, test_r2=0.4595),
    "SOx": dict(name="XGBoost", train_r2=0.9996, train_mae=0.8047, train_rmse=1.1543, gap=0.5397, test_r2=0.4598),
    "NOx": dict(name="XGBoost", train_r2=0.9994, train_mae=1.0352, train_rmse=1.5057, gap=0.5612, test_r2=0.4382),
}

TOL = {"PM": 0.15, "SOx": 1.5, "NOx": 2.0}
UNIT = "mg/Nm\u00b3"
LABEL = {"PM": "PM", "SOx": "SO\u2082", "NOx": "NO\u2093"}

INK, SLATE, MIST, RULE = "#1B2A33", "#5A6B74", "#94A2A9", "#E1E6E7"
PAPER, STEEL, TEAL, EMBER, MOSS, SAND = "#FAFBFA", "#2F6079", "#2C8C84", "#B14A24", "#4C7A50", "#C9A227"
PALETTE = ["#2F6079", "#3A7290", "#4785A4", "#2C8C84", "#5AA39B",
           "#86B8B2", "#C9A227", "#D2BA63", "#C9CFC9", "#B6BEBD"]

st.set_page_config(page_title="Stack emission monitor", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background:{PAPER}; }}
  .block-container {{ padding-top:1.6rem; max-width:1500px; }}
  html, body, [class*="css"] {{
      font-family:ui-sans-serif,system-ui,"Segoe UI",Helvetica,Arial,sans-serif; }}

  .mast h1 {{ margin:0; font-size:20px; font-weight:600; letter-spacing:-.2px; color:{INK}; }}
  .mast p  {{ margin:4px 0 0; font-size:12.5px; color:{MIST}; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {RULE}; }}
  .stTabs [data-baseweb="tab"] {{
      height:44px; padding:0 22px; background:transparent; color:{MIST};
      font-size:14.5px; border-bottom:3px solid transparent; }}
  .stTabs [aria-selected="true"] {{ color:{INK}; font-weight:600; border-bottom-color:{SAND}; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background:transparent; }}

  .panel {{ background:#fff; border:1px solid {RULE}; padding:14px 16px 6px; margin-bottom:14px; }}
  .panel h2 {{ margin:0; font-size:14.5px; font-weight:600; color:{INK}; }}
  .panel .note {{ margin:3px 0 4px; font-size:12px; color:{MIST}; }}

  .statrow {{ display:grid; gap:1px; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
              background:{RULE}; border:1px solid {RULE}; margin-bottom:14px; }}
  .stat {{ background:#fff; padding:14px 16px; }}
  .stat .k {{ font-size:12px; color:{SLATE}; }}
  .stat .v {{ font-size:25px; font-weight:600; line-height:1.15; margin-top:4px;
              font-variant-numeric:tabular-nums; }}
  .stat .s {{ font-size:11.5px; color:{MIST}; margin-top:3px; }}

  .rank .row {{ display:grid; grid-template-columns:1fr 54px; gap:10px;
                align-items:center; margin-bottom:8px; }}
  .rank .nm {{ font-size:12.5px; color:{INK}; margin-bottom:4px; }}
  .rank .track {{ height:6px; background:{PAPER}; }}
  .rank .pct {{ font-size:12.5px; color:{SLATE}; text-align:right;
                font-variant-numeric:tabular-nums; }}

  .band {{ margin-bottom:16px; }}
  .band .top {{ display:flex; justify-content:space-between; gap:10px; align-items:baseline; }}
  .band .nm {{ font-size:13px; font-weight:600; color:{INK}; }}
  .band .win {{ font-size:12px; color:{SLATE}; font-variant-numeric:tabular-nums; }}
  .band .track {{ position:relative; height:10px; background:{PAPER};
                  border:1px solid {RULE}; margin-top:7px; }}
  .band .foot {{ display:flex; justify-content:space-between; margin-top:5px;
                 font-size:11.5px; color:{MIST}; font-variant-numeric:tabular-nums; }}

  .legend {{ padding:0; margin:0; }}
  .legend li {{ list-style:none; font-size:11.5px; color:{INK}; margin-bottom:3px; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ helpers


def base_layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="system-ui, sans-serif", size=11, color=SLATE),
                      xaxis=dict(gridcolor=RULE, zeroline=False),
                      yaxis=dict(gridcolor=RULE, zeroline=False))
    return fig


def panel_open(title, note=""):
    st.markdown(f"<div class='panel'><h2>{title}</h2><p class='note'>{note}</p>",
                unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


def shap_frame(key):
    d = pd.DataFrame(SHAP[key], columns=["Parameter", "Mean |SHAP|"])
    d["Share %"] = d["Mean |SHAP|"] / d["Mean |SHAP|"].sum() * 100
    return d.sort_values("Share %", ascending=False).reset_index(drop=True)


def fmt(v):
    if abs(v) >= 10000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:g}"


def donut(frame, height, hole=0.45):
    fig = go.Figure(go.Pie(labels=frame["Parameter"], values=frame["Share %"],
                           hole=hole, sort=False, textinfo="none",
                           marker=dict(colors=PALETTE[:len(frame)],
                                       line=dict(color="white", width=1.5)),
                           hovertemplate="%{label}: %{value:.2f}%<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=False, paper_bgcolor="white")
    return fig


def stat(k, v, s, tone=INK):
    return (f"<div class='stat'><div class='k'>{k}</div>"
            f"<div class='v' style='color:{tone}'>{v}</div>"
            f"<div class='s'>{s}</div></div>")

# ------------------------------------------------------------------ page

st.markdown("<div class='mast'><h1>Stack emission — prediction monitor</h1>"
            "<p>Sinter plant · model output vs measured value</p></div>",
            unsafe_allow_html=True)

tabs = st.tabs([LABEL[k] for k in PAIRS])

for tab, key in zip(tabs, PAIRS):
    with tab:
        df = pd.DataFrame(PAIRS[key], columns=["Measured", "Predicted"])
        df["Error"] = df["Measured"] - df["Predicted"]
        mdl = MODELS[key]

        head = st.columns([3, 1])
        with head[1]:
            tol = st.number_input(f"Tolerance ± ({UNIT})", min_value=0.0,
                                  value=TOL[key], step=TOL[key] / 3, key=f"tol_{key}")
        off = df["Error"].abs() > tol

        span = df[["Measured", "Predicted"]].values
        pad = (span.max() - span.min()) * 0.08
        lo_y, hi_y = float(span.min()) - pad, float(span.max()) + pad

        def trend(upto, _lo=lo_y, _hi=hi_y, _df=df):
            f = go.Figure()
            f.add_scatter(y=_df["Measured"][:upto], mode="lines+markers", name="Measured",
                          line=dict(color=INK, width=2), marker=dict(size=5))
            f.add_scatter(y=_df["Predicted"][:upto], mode="lines+markers", name="Predicted",
                          line=dict(color=TEAL, width=2, dash="dash"), marker=dict(size=5))
            f.update_layout(legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11.5)),
                            xaxis=dict(range=[-0.5, len(_df) - 0.5], gridcolor=RULE,
                                       title="Observation", zeroline=False),
                            yaxis=dict(range=[_lo, _hi], gridcolor=RULE,
                                       title=UNIT, zeroline=False))
            return base_layout(f, 330)

        panel_open("Measured against predicted",
                   f"{len(df)} observations · deviation flagged beyond ±{tol:g} {UNIT}")
        slot = st.empty()
        flag = f"drawn_{key}"
        if not st.session_state.get(flag):
            for i in range(2, len(df) + 1):
                slot.plotly_chart(trend(i), use_container_width=True, key=f"tr_{key}_{i}")
                time.sleep(0.045)
            st.session_state[flag] = True
        else:
            slot.plotly_chart(trend(len(df)), use_container_width=True, key=f"tr_{key}_f")
        if st.button("Replay", key=f"replay_{key}"):
            st.session_state[flag] = False
            st.rerun()
        panel_close()

        st.markdown(
            "<div class='statrow'>"
            + stat("Training R²", f"{mdl['train_r2']:.4f}", mdl["name"])
            + stat("Training MAE", f"{mdl['train_mae']:.4f}", UNIT)
            + stat("Training RMSE", f"{mdl['train_rmse']:.4f}", UNIT)
            + "</div>", unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            panel_open("Error by observation", "Measured minus predicted")
            bar = go.Figure()
            bar.add_bar(y=df["Error"], marker_color=np.where(off, EMBER, STEEL),
                        hovertemplate="row %{x}: %{y:.3f}<extra></extra>")
            bar.add_hline(y=tol, line=dict(color=MIST, dash="dot"))
            bar.add_hline(y=-tol, line=dict(color=MIST, dash="dot"))
            bar.add_hline(y=0, line=dict(color=SLATE))
            bar.update_layout(yaxis_title=UNIT, showlegend=False)
            st.plotly_chart(base_layout(bar, 265), use_container_width=True, key=f"res_{key}")
            panel_close()

        with right:
            panel_open("Prediction against measurement", "Points on the line are exact matches")
            lo = float(min(df["Measured"].min(), df["Predicted"].min()))
            hi = float(max(df["Measured"].max(), df["Predicted"].max()))
            p = (hi - lo) * 0.06
            sc = go.Figure()
            sc.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                           line=dict(color=MIST, dash="dash"), hoverinfo="skip")
            sc.add_scatter(x=df["Measured"], y=df["Predicted"], mode="markers",
                           marker=dict(color=STEEL, size=9, opacity=0.75),
                           hovertemplate="measured %{x:.3f}<br>predicted %{y:.3f}<extra></extra>")
            sc.update_layout(showlegend=False,
                             xaxis=dict(title=f"Measured ({UNIT})", range=[lo - p, hi + p],
                                        gridcolor=RULE, zeroline=False),
                             yaxis=dict(title=f"Predicted ({UNIT})", range=[lo - p, hi + p],
                                        gridcolor=RULE, zeroline=False))
            st.plotly_chart(base_layout(sc, 265), use_container_width=True, key=f"par_{key}")
            panel_close()

        drivers = shap_frame(key)
        a, b = st.columns(2)
        with a:
            panel_open("What drives the prediction",
                       f"Top three account for {drivers['Share %'].head(3).sum():.1f}% of the signal")
            st.plotly_chart(donut(drivers, 300), use_container_width=True, key=f"pie_{key}")
            panel_close()
        with b:
            panel_open("Driver ranking", "Share of total mean absolute SHAP")
            top = drivers["Share %"].max()
            rows = "".join(
                f"<div class='row'><div><div class='nm'>{r['Parameter']}</div>"
                f"<div class='track'><div style='width:{r['Share %'] / top * 100:.1f}%;"
                f"height:100%;background:{PALETTE[i % len(PALETTE)]}'></div></div></div>"
                f"<div class='pct'>{r['Share %']:.2f}%</div></div>"
                for i, r in drivers.iterrows())
            st.markdown(f"<div class='rank'>{rows}</div>", unsafe_allow_html=True)
            panel_close()

        panel_open("Contributing factors side by side",
                   "Same drivers, ranked within each pollutant")
        cols = st.columns(3)
        for col, k2 in zip(cols, PAIRS):
            with col:
                d2 = shap_frame(k2)
                st.markdown(
                    f"<div style='font-size:13.5px;font-weight:600;color:{INK}'>{LABEL[k2]}</div>"
                    f"<div style='font-size:11.5px;color:{MIST};margin-bottom:4px'>"
                    f"top three {d2['Share %'].head(3).sum():.1f}%</div>",
                    unsafe_allow_html=True)
                st.plotly_chart(donut(d2, 170, hole=0.5), use_container_width=True,
                                key=f"mini_{key}_{k2}")
                items = "".join(
                    f"<li><span style='color:{PALETTE[i % len(PALETTE)]}'>\u25a0</span> "
                    f"{r['Parameter']} <span style='color:{SLATE}'>{r['Share %']:.1f}%</span></li>"
                    for i, r in d2.iterrows())
                st.markdown(f"<ul class='legend'>{items}</ul>", unsafe_allow_html=True)
        panel_close()

        if BANDS[key]:
            panel_open("Favourable operating window",
                       "Grey is the range seen in the data, blue is the band that keeps emission lowest")
            html = ""
            for nm, olo, ohi, flo, fhi, best, em in BANDS[key]:
                sp = ohi - olo
                flat = sp <= 0
                full = (not flat) and abs(flo - olo) < 1e-9 and abs(fhi - ohi) < 1e-9
                if flat:
                    inner, window = "", "constant"
                else:
                    def pos(v, _olo=olo, _sp=sp):
                        return (v - _olo) / _sp * 100
                    inner = (
                        f"<div style='position:absolute;left:{pos(flo):.1f}%;"
                        f"width:{max(pos(fhi) - pos(flo), 1.2):.1f}%;top:0;bottom:0;"
                        f"background:{STEEL}'></div>"
                        f"<div style='position:absolute;left:calc({pos(best):.1f}% - 1px);"
                        f"width:2px;top:-3px;bottom:-3px;background:{INK}'></div>")
                    window = ("whole observed range" if full
                              else f"{fmt(flo)} – {fmt(fhi)}")
                html += (f"<div class='band'><div class='top'><span class='nm'>{nm}</span>"
                         f"<span class='win'>{window}</span></div>"
                         f"<div class='track'>{inner}</div>"
                         f"<div class='foot'><span>{fmt(olo)}</span>"
                         f"<span style='color:{SLATE}'>best {fmt(best)} · "
                         f"holds emission near {fmt(em)}</span>"
                         f"<span>{fmt(ohi)}</span></div></div>")
            st.markdown(html, unsafe_allow_html=True)
            table = pd.DataFrame(BANDS[key], columns=[
                "Parameter", "Observed low", "Observed high",
                "Favourable low", "Favourable high", "Best model value",
                f"Min avg predicted emission ({UNIT})"])
            st.dataframe(table, hide_index=True, use_container_width=True,
                         key=f"rngtbl_{key}")
            panel_close()
        else:
            panel_open("Favourable operating window",
                       f"No window supplied for {LABEL[key]} yet.")
            panel_close()

        with st.expander("Observation record"):
            st.dataframe(df.round(3), use_container_width=True)
            st.download_button("Download as CSV", df.to_csv(index=False).encode(),
                               file_name=f"{key}_predictions.csv", mime="text/csv",
                               key=f"csv_{key}")
