"""
Stack emission prediction monitor — sinter plant.
Single file, no external data. Run: streamlit run app.py
"""

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

# observed span, favourable window, best value, emission that window holds
BANDS = {
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
    "PM": [],
    "SOx": [],
}

MODELS = {
    "PM":  dict(name="XGBoost", train_r2=0.9991, train_mae=0.0891, train_rmse=0.1148, gap=0.5396, test_r2=0.4595),
    "SOx": dict(name="XGBoost", train_r2=0.9996, train_mae=0.8047, train_rmse=1.1543, gap=0.5397, test_r2=0.4598),
    "NOx": dict(name="XGBoost", train_r2=0.9994, train_mae=1.0352, train_rmse=1.5057, gap=0.5612, test_r2=0.4382),
}

TOL = {"PM": 0.15, "SOx": 1.5, "NOx": 2.0}
UNIT = "mg/Nm\u00b3"
LABEL = {"PM": "PM", "SOx": "SO\u2082", "NOx": "NO\u2093"}

INK, STEEL, TEAL, EMBER, MIST, RULE = "#1B2A33", "#2F6079", "#2C8C84", "#B14A24", "#94A2A9", "#E1E6E7"
PALETTE = ["#2F6079", "#3A7290", "#4785A4", "#2C8C84", "#5AA39B",
           "#86B8B2", "#C9A227", "#D2BA63", "#C9CFC9", "#B6BEBD"]

st.set_page_config(page_title="Stack emission monitor", layout="wide")

# ------------------------------------------------------------------ helpers

def base_layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(gridcolor=RULE), yaxis=dict(gridcolor=RULE))
    return fig


def shap_frame(key):
    df = pd.DataFrame(SHAP[key], columns=["Parameter", "Mean |SHAP|"])
    df["Share %"] = df["Mean |SHAP|"] / df["Mean |SHAP|"].sum() * 100
    return df.sort_values("Share %", ascending=False).reset_index(drop=True)


def fmt(v):
    if abs(v) >= 10000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:g}"

# ------------------------------------------------------------------ page

st.title("Stack emission \u2014 prediction monitor")
st.caption("Sinter plant \u00b7 model output vs measured value")

tabs = st.tabs([LABEL[k] for k in PAIRS])

for tab, key in zip(tabs, PAIRS):
    with tab:
        df = pd.DataFrame(PAIRS[key], columns=["Measured", "Predicted"])
        df["Error"] = df["Measured"] - df["Predicted"]
        mdl = MODELS[key]

        tol = st.number_input(f"Tolerance band ({UNIT})", min_value=0.0,
                              value=TOL[key], step=TOL[key] / 3, key=f"tol_{key}")
        off = df["Error"].abs() > tol
        worst = df["Error"].abs().idxmax()

        c = st.columns(6)
        c[0].metric("Training R\u00b2", f"{mdl['train_r2']:.4f}", mdl["name"])
        c[1].metric("Training MAE", f"{mdl['train_mae']:.4f}", UNIT)
        c[2].metric("Training RMSE", f"{mdl['train_rmse']:.4f}", UNIT)
        c[3].metric("R\u00b2 gap", f"{mdl['gap']:.4f}", f"test R\u00b2 {mdl['test_r2']:.4f}")
        c[4].metric(f"Within \u00b1{tol:g}", f"{(~off).mean() * 100:.0f}%", f"{(~off).sum()} of {len(df)}")
        c[5].metric("Largest miss", f"{df.loc[worst, 'Error']:+.3f}", f"row {worst}")

        # measured vs predicted
        st.subheader("Measured against predicted")
        fig = go.Figure()
        fig.add_scatter(y=df["Measured"], mode="lines+markers", name="Measured",
                        line=dict(color=INK, width=2))
        fig.add_scatter(y=df["Predicted"], mode="lines+markers", name="Predicted",
                        line=dict(color=TEAL, width=2, dash="dash"))
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0),
                          yaxis_title=UNIT, xaxis_title="Observation")
        st.plotly_chart(base_layout(fig, 340), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Error by observation")
            bar = go.Figure()
            bar.add_bar(y=df["Error"], marker_color=np.where(off, EMBER, STEEL))
            bar.add_hline(y=tol, line=dict(color=MIST, dash="dot"))
            bar.add_hline(y=-tol, line=dict(color=MIST, dash="dot"))
            bar.update_layout(yaxis_title=UNIT)
            st.plotly_chart(base_layout(bar, 280), use_container_width=True)

        with right:
            st.subheader("Prediction against measurement")
            lo = float(min(df["Measured"].min(), df["Predicted"].min()))
            hi = float(max(df["Measured"].max(), df["Predicted"].max()))
            pad = (hi - lo) * 0.06
            sc = go.Figure()
            sc.add_scatter(x=df["Measured"], y=df["Predicted"], mode="markers",
                           marker=dict(color=STEEL, size=9, opacity=0.75))
            sc.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                           line=dict(color=MIST, dash="dash"))
            sc.update_layout(showlegend=False,
                             xaxis=dict(title=f"Measured ({UNIT})", range=[lo - pad, hi + pad], gridcolor=RULE),
                             yaxis=dict(title=f"Predicted ({UNIT})", range=[lo - pad, hi + pad], gridcolor=RULE))
            st.plotly_chart(base_layout(sc, 280), use_container_width=True)

        # drivers
        drivers = shap_frame(key)
        a, b = st.columns([1, 1])
        with a:
            st.subheader("What drives the prediction")
            pie = go.Figure(go.Pie(labels=drivers["Parameter"], values=drivers["Share %"],
                                   hole=0.45, sort=False, textinfo="none",
                                   marker=dict(colors=PALETTE[:len(drivers)],
                                               line=dict(color="white", width=1.5))))
            pie.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(pie, use_container_width=True)
            st.caption(f"Top three account for {drivers['Share %'].head(3).sum():.1f}% of the signal")
        with b:
            st.subheader("Driver ranking")
            st.dataframe(drivers.round(4), hide_index=True, use_container_width=True)

        # side by side
        st.subheader("Contributing factors side by side")
        cols = st.columns(3)
        for col, k2 in zip(cols, PAIRS):
            with col:
                d2 = shap_frame(k2)
                st.markdown(f"**{LABEL[k2]}** \u00b7 top three {d2['Share %'].head(3).sum():.1f}%")
                p2 = go.Figure(go.Pie(labels=d2["Parameter"], values=d2["Share %"], hole=0.45,
                                      sort=False, textinfo="none",
                                      marker=dict(colors=PALETTE[:len(d2)],
                                                  line=dict(color="white", width=1.5))))
                p2.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                st.plotly_chart(p2, use_container_width=True, key=f"mini_{key}_{k2}")
                for i, r in d2.iterrows():
                    st.markdown(
                        f"<div style='font-size:12px'><span style='color:{PALETTE[i % len(PALETTE)]}'>\u25a0</span> "
                        f"{r['Parameter']} &mdash; {r['Share %']:.1f}%</div>",
                        unsafe_allow_html=True)

        # operating window
        st.subheader("Favourable operating window")
        if BANDS[key]:
            bands = pd.DataFrame(BANDS[key], columns=[
                "Parameter", "Observed low", "Observed high",
                "Favourable low", "Favourable high", "Best value", f"Emission ({UNIT})"])
            fig2 = go.Figure()
            for _, r in bands.iloc[::-1].iterrows():
                span = r["Observed high"] - r["Observed low"]
                if span <= 0:
                    continue
                norm = lambda v: (v - r["Observed low"]) / span * 100
                fig2.add_trace(go.Bar(
                    y=[r["Parameter"]], x=[100], base=[0], orientation="h",
                    marker_color="#EEF1F1", showlegend=False, hoverinfo="skip", width=0.55))
                fig2.add_trace(go.Bar(
                    y=[r["Parameter"]], x=[max(norm(r["Favourable high"]) - norm(r["Favourable low"]), 1.5)],
                    base=[norm(r["Favourable low"])], orientation="h",
                    marker_color=STEEL, showlegend=False, width=0.55,
                    hovertemplate=f"favourable {fmt(r['Favourable low'])} – {fmt(r['Favourable high'])}"
                                  f"<br>best {fmt(r['Best value'])}<extra></extra>"))
            fig2.update_layout(barmode="overlay", height=34 * len(bands) + 60,
                               margin=dict(l=10, r=10, t=10, b=10),
                               plot_bgcolor="white", paper_bgcolor="white",
                               xaxis=dict(showticklabels=False, range=[0, 100], gridcolor="white"),
                               yaxis=dict(gridcolor="white"))
            st.plotly_chart(fig2, use_container_width=True, key=f"bands_{key}")
            st.caption("Grey is the range seen in the data, blue is the band that keeps emission lowest")
            st.dataframe(bands, hide_index=True, use_container_width=True)
        else:
            st.info(f"No operating window supplied for {LABEL[key]} yet.")

        with st.expander("Observation record"):
            st.dataframe(df.round(3), use_container_width=True)
            st.download_button("Download as CSV", df.to_csv(index=False).encode(),
                               file_name=f"{key}_predictions.csv", mime="text/csv",
                               key=f"csv_{key}")
