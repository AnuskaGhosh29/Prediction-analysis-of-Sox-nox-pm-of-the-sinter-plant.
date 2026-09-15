import { useState, useMemo, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, ZAxis
} from "recharts";

/* ---------- design tokens ---------- */
const C = {
  ink: "#16242E",
  slate: "#54666F",
  mist: "#8C9BA3",
  rule: "#D7DDDF",
  paper: "#F1F3F2",
  surface: "#FFFFFF",
  steel: "#2F6079",
  teal: "#2C8C84",
  ember: "#B14A24",
  moss: "#4C7A50",
  sand: "#C9A227",
};

const FONT = "ui-sans-serif, system-ui, 'Segoe UI', Helvetica, Arial, sans-serif";

/* ---------- seed data: PM (from the notebook outputs) ---------- */
const PM_PAIRS = [
  [32.161, 32.406], [32.641, 32.756], [25.253, 25.384], [22.962, 23.093],
  [23.769, 23.930], [38.520, 38.486], [26.189, 26.169], [21.854, 21.865],
  [36.489, 36.610], [27.444, 27.443], [23.078, 23.096], [31.842, 31.754],
  [24.521, 24.362], [29.179, 29.201], [29.854, 29.941], [29.183, 29.376],
  [30.924, 30.979], [21.624, 21.804], [26.659, 26.659], [36.476, 36.463],
];

const PM_SHAP = [
  ["Sinter machine speed", 0.7757],
  ["Suction / under pressure", 0.7479],
  ["Moisture content", 0.7283],
  ["Bed height", 0.4645],
  ["Air–fuel ratio", 0.4036],
  ["Granulation index", 0.1667],
  ["MPV", 0.1332],
  ["Solid fuel ratio", 0.1038],
  ["Coke breeze temperature", 0.0948],
  ["Waste gas flow", 0.0909],
];

const SO2_PAIRS = [
  [276.401, 276.403], [199.183, 196.582], [266.385, 268.138], [338.817, 338.817],
  [377.679, 377.672], [182.714, 180.276], [254.149, 253.900], [282.224, 282.231],
  [201.317, 199.733], [259.039, 259.126], [321.966, 322.118], [219.807, 218.861],
  [316.943, 315.798], [191.893, 191.909], [353.488, 354.183], [361.845, 360.535],
  [207.377, 207.957], [324.629, 326.406], [207.959, 208.543], [181.208, 181.047],
];

const SO2_SHAP = [
  ["Bed height (H)", 17.0315],
  ["Coke breeze temperature", 7.5610],
  ["Sinter speed", 7.4592],
  ["Waste gas flow (Nm³)", 7.1339],
  ["Ignition temperature", 5.5101],
  ["ESP inlet suction (P)", 3.0114],
  ["Total moisture % (M)", 2.1133],
  ["Lime consumed", 1.3819],
  ["Fe-S %", 0.8155],
];

const NOX_PAIRS = [
  [98.953, 99.227], [68.920, 68.919], [137.676, 137.682], [69.998, 68.617],
  [336.228, 336.213], [64.508, 64.509], [66.801, 66.512], [166.823, 164.084],
  [64.239, 65.676], [74.100, 74.493], [377.339, 379.079], [69.386, 66.421],
  [4.896, 6.873], [63.214, 62.908], [75.934, 75.938], [48.640, 46.696],
  [67.014, 66.980], [52.542, 52.542], [70.534, 67.462], [58.895, 61.020],
];

const NOX_SHAP = [
  ["Flame front speed", 10.0856],
  ["ESP inlet suction (P)", 5.7801],
  ["CaO/SiO2", 4.3882],
  ["CA flow", 4.3075],
  ["Ignition temperature", 3.2249],
  ["Total moisture % (M)", 2.2565],
  ["Waste gas flow (SM³)", 2.0994],
  ["Bed height (H)", 1.9739],
  ["Coke breeze temperature", 1.4594],
  ["VM concentration", 0.0268],
];

/* Observed span, favourable window, best model value and the minimum average
   predicted emission that window delivers. */
const NOX_RANGES = [
  { name: "Coke breeze temperature", obsLo: 17.75, obsHi: 32.741, favLo: 27.138, favHi: 28.652, best: 27.744, emission: 92.079, threshold: 93.549 },
  { name: "VM concentration", obsLo: 0, obsHi: 0, favLo: 0, favHi: 0, best: 0, emission: 93.986, threshold: 93.994 },
  { name: "Total moisture % (M)", obsLo: 11.254, obsHi: 15.705, favLo: 11.254, favHi: 12.063, best: 11.658, emission: 89.119, threshold: 94.023 },
  { name: "Ignition temperature", obsLo: 1034.956, obsHi: 1157.611, favLo: 1096.903, favHi: 1110.531, best: 1108.053, emission: 91.054, threshold: 91.637 },
  { name: "Bed height (H)", obsLo: 745.006, obsHi: 749.647, favLo: 748.897, favHi: 749.647, best: 749.647, emission: 91.764, threshold: 94.011 },
  { name: "ESP inlet suction (P)", obsLo: 116.559, obsHi: 134.144, favLo: 123.842, favHi: 125.973, best: 124.375, emission: 86.379, threshold: 87.275 },
  { name: "Waste gas flow (SM³)", obsLo: 989730.11, obsHi: 1087415.92, favLo: 1003544.27, favHi: 1014398.25, best: 1012424.8, emission: 91.238, threshold: 92.083 },
  { name: "CA flow", obsLo: 8320.141, obsHi: 9224.63, favLo: 8676.455, favHi: 8895.725, best: 8713, emission: 88.19, threshold: 92.038 },
  { name: "Flame front speed", obsLo: 19.324, obsHi: 21.497, favLo: 19.324, favHi: 19.85, best: 19.324, emission: 85.833, threshold: 86.054 },
  { name: "CaO/SiO2", obsLo: 1.522, obsHi: 2.103, favLo: 1.522, favHi: 1.634, best: 1.528, emission: 87.133, threshold: 87.601 },
];

const emptySet = (unit, limit) => ({ pairs: [], shap: [], ranges: [], unit, limit });

const SEED = {
  PM: { pairs: PM_PAIRS, shap: PM_SHAP, ranges: [], unit: "mg/Nm³", limit: 50 },
  SO2: { pairs: SO2_PAIRS, shap: SO2_SHAP, ranges: [], unit: "mg/Nm³", limit: 200 },
  NOx: { pairs: NOX_PAIRS, shap: NOX_SHAP, ranges: NOX_RANGES, unit: "mg/Nm³", limit: 300 },
};

const DEFAULT_TOL = { PM: 0.15, SO2: 1.5, NOx: 2 };

/* XGBoost row from each notebook's model_comparison table */
const MODELS = {
  PM:  { name: "XGBoost", trainR2: 0.9991, trainMAE: 0.0891, trainRMSE: 0.1148, gap: 0.5396, testR2: 0.4595 },
  SO2: { name: "XGBoost", trainR2: 0.9996, trainMAE: 0.8047, trainRMSE: 1.1543, gap: 0.5397, testR2: 0.4598 },
  NOx: { name: "XGBoost", trainR2: 0.9994, trainMAE: 1.0352, trainRMSE: 1.5057, gap: 0.5612, testR2: 0.4382 },
};

const fmt = (v) => {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (Math.abs(n) >= 10000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(n) >= 100) return n.toFixed(1);
  return n.toFixed(3).replace(/\.?0+$/, "");
};

const POLLUTANTS = [
  { key: "PM", label: "PM" },
  { key: "SO2", label: "SO₂" },
  { key: "NOx", label: "NOₓ" },
];

/* ---------- parsing ---------- */
function parsePairs(text) {
  const out = [];
  text.split(/\r?\n/).forEach((line) => {
    const nums = (line.match(/-?\d+\.?\d*/g) || []).map(Number);
    if (nums.length < 2) return;
    let a, p;
    if (nums.length >= 3 && Number.isInteger(nums[0]) && nums[0] === out.length) {
      a = nums[1]; p = nums[2];
    } else if (nums.length >= 4) {
      a = nums[1]; p = nums[2];
    } else {
      a = nums[0]; p = nums[1];
    }
    if (Number.isFinite(a) && Number.isFinite(p)) out.push([a, p]);
  });
  return out;
}

function parseShap(text) {
  const out = [];
  text.split(/\r?\n/).forEach((line) => {
    const nums = (line.match(/-?\d+\.\d+/g) || []).map(Number);
    const name = line
      .replace(/-?\d+\.\d+/g, " ")
      .replace(/^\s*\d+\s+/, "")
      .replace(/[_]/g, " ")
      .trim();
    if (!name || nums.length === 0) return;
    out.push([name, nums[0]]);
  });
  return out;
}

/* ---------- metrics ---------- */
function metrics(pairs) {
  const n = pairs.length;
  if (!n) return null;
  const res = pairs.map(([a, p]) => a - p);
  const abs = res.map(Math.abs);
  const mae = abs.reduce((s, v) => s + v, 0) / n;
  const rmse = Math.sqrt(res.reduce((s, v) => s + v * v, 0) / n);
  const mape = (pairs.reduce((s, [a, p]) => s + Math.abs((a - p) / a), 0) / n) * 100;
  const meanA = pairs.reduce((s, [a]) => s + a, 0) / n;
  const sst = pairs.reduce((s, [a]) => s + (a - meanA) ** 2, 0);
  const sse = res.reduce((s, v) => s + v * v, 0);
  const r2 = sst === 0 ? 0 : 1 - sse / sst;
  const bias = res.reduce((s, v) => s + v, 0) / n;
  const maxIdx = abs.indexOf(Math.max(...abs));
  return { n, mae, rmse, mape, r2, bias, maxIdx, maxDev: res[maxIdx], res };
}

/* ---------- small components ---------- */
function Stat({ label, value, sub, tone }) {
  return (
    <div style={{ padding: "14px 16px", background: C.surface, border: `1px solid ${C.rule}` }}>
      <div style={{ fontSize: 12, color: C.slate, letterSpacing: 0.2 }}>{label}</div>
      <div style={{
        fontSize: 26, fontWeight: 600, color: tone || C.ink,
        fontVariantNumeric: "tabular-nums", lineHeight: 1.15, marginTop: 4,
      }}>{value}</div>
      {sub && <div style={{ fontSize: 11.5, color: C.mist, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function Panel({ title, note, children, right }) {
  return (
    <section style={{ background: C.surface, border: `1px solid ${C.rule}` }}>
      <header style={{
        display: "flex", justifyContent: "space-between", alignItems: "baseline",
        gap: 12, padding: "12px 16px", borderBottom: `1px solid ${C.rule}`,
      }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 14.5, fontWeight: 600, color: C.ink }}>{title}</h2>
          {note && <p style={{ margin: "3px 0 0", fontSize: 12, color: C.mist }}>{note}</p>}
        </div>
        {right}
      </header>
      <div style={{ padding: 16 }}>{children}</div>
    </section>
  );
}

export default function EmissionDashboard() {
  const [sets, setSets] = useState(SEED);
  const [active, setActive] = useState("PM");
  const [tols, setTols] = useState(DEFAULT_TOL);
  const [pasteA, setPasteA] = useState("");
  const [pasteS, setPasteS] = useState("");
  const [openIntake, setOpenIntake] = useState(false);
  const [saved, setSaved] = useState("");

  /* restore any saved session */
  useEffect(() => {
    (async () => {
      try {
        const r = await window.storage.get("stack-emission-sets-v2");
        if (!r || !r.value) return;
        const stored = JSON.parse(r.value);
        const merged = {};
        Object.keys(SEED).forEach((k) => {
          const s = stored[k] || {};
          merged[k] = {
            ...SEED[k],
            pairs: s.pairs && s.pairs.length ? s.pairs : SEED[k].pairs,
            shap: s.shap && s.shap.length ? s.shap : SEED[k].shap,
            ranges: s.ranges || [],
          };
        });
        setSets(merged);
      } catch (e) { /* nothing stored yet */ }
    })();
  }, []);

  const persist = async (next) => {
    setSets(next);
    try {
      await window.storage.set("stack-emission-sets-v2", JSON.stringify(next));
      setSaved("Saved");
      setTimeout(() => setSaved(""), 1800);
    } catch (e) {
      setSaved("Not saved — carry on, data stays for this session");
      setTimeout(() => setSaved(""), 3000);
    }
  };

  const d = sets[active];
  const mdl = MODELS[active];
  const tol = tols[active];
  const setTol = (v) => setTols((t) => ({ ...t, [active]: v }));
  const m = useMemo(() => metrics(d.pairs), [d.pairs]);

  const shapRows = useMemo(() => {
    const total = d.shap.reduce((s, [, v]) => s + v, 0);
    return d.shap
      .map(([name, v]) => ({ name, value: v, pct: total ? (v / total) * 100 : 0 }))
      .sort((a, b) => b.value - a.value);
  }, [d.shap]);

  const series = useMemo(
    () => d.pairs.map(([a, p], i) => ({ i, actual: a, predicted: p, res: a - p })),
    [d.pairs]
  );

  const within = m ? (m.res.filter((v) => Math.abs(v) <= tol).length / m.n) * 100 : 0;
  const top3 = shapRows.slice(0, 3).reduce((s, r) => s + r.pct, 0);

  const parityPad = useMemo(() => {
    const all = d.pairs.flat();
    if (!all.length) return 1;
    return Math.max((Math.max(...all) - Math.min(...all)) * 0.06, 0.05);
  }, [d.pairs]);

  const hasBands = d.ranges.length > 0 && d.ranges[0].favLo !== undefined;
  const orderedRanges = useMemo(() => {
    if (!hasBands) return [];
    const rank = new Map(shapRows.map((r, i) => [r.name, i]));
    return [...d.ranges].sort(
      (a, b) => (rank.get(a.name) ?? 99) - (rank.get(b.name) ?? 99)
    );
  }, [d.ranges, shapRows, hasBands]);

  const pieColors = [C.steel, "#3A7290", "#4785A4", C.teal, "#5AA39B",
    "#86B8B2", C.sand, "#D2BA63", "#C9CFC9", "#B6BEBD"];

  const applyPairs = () => {
    const rows = parsePairs(pasteA);
    if (!rows.length) return;
    persist({ ...sets, [active]: { ...d, pairs: rows } });
    setPasteA("");
  };
  const applyShap = () => {
    const rows = parseShap(pasteS);
    if (!rows.length) return;
    persist({ ...sets, [active]: { ...d, shap: rows } });
    setPasteS("");
  };
  const setRange = (name, field, val) => {
    const others = d.ranges.filter((r) => r.name !== name);
    const cur = d.ranges.find((r) => r.name === name) || { name, lo: "", hi: "" };
    persist({ ...sets, [active]: { ...d, ranges: [...others, { ...cur, [field]: val }] } });
  };
  const rangeOf = (name) => d.ranges.find((r) => r.name === name) || { lo: "", hi: "" };

  const tipStyle = {
    background: C.surface, border: `1px solid ${C.rule}`, fontSize: 12,
    fontFamily: FONT, color: C.ink,
  };

  return (
    <div style={{ fontFamily: FONT, background: C.paper, minHeight: "100%", color: C.ink }}>
      {/* masthead */}
      <div style={{ background: C.ink, color: "#EDF1F1", padding: "18px 20px 0" }}>
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: 10 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 19, fontWeight: 600, letterSpacing: -0.2 }}>
              Stack emission — prediction monitor
            </h1>
            <p style={{ margin: "4px 0 0", fontSize: 12.5, color: "#9FB0B6" }}>
              Sinter plant · model output vs measured value
            </p>
          </div>
          <button
            onClick={() => setOpenIntake((v) => !v)}
            style={{
              alignSelf: "start", background: "transparent", color: "#EDF1F1",
              border: "1px solid #3E5460", padding: "7px 13px", fontSize: 12.5,
              cursor: "pointer", fontFamily: FONT,
            }}
          >
            {openIntake ? "Close data intake" : "Update data"}
          </button>
        </div>

        <nav style={{ display: "flex", gap: 2, marginTop: 16 }}>
          {POLLUTANTS.map((p) => {
            const on = p.key === active;
            const loaded = sets[p.key].pairs.length > 0;
            return (
              <button
                key={p.key}
                onClick={() => setActive(p.key)}
                style={{
                  background: on ? C.paper : "transparent",
                  color: on ? C.ink : "#9FB0B6",
                  border: "none", borderBottom: on ? `3px solid ${C.sand}` : "3px solid transparent",
                  padding: "10px 20px", fontSize: 14.5, fontWeight: on ? 600 : 500,
                  cursor: "pointer", fontFamily: FONT,
                }}
              >
                {p.label}
                {!loaded && <span style={{ fontSize: 11, marginLeft: 7, color: "#7A8C94" }}>no data</span>}
              </button>
            );
          })}
        </nav>
      </div>

      {/* data intake */}
      {openIntake && (
        <div style={{ background: "#E7EBEA", borderBottom: `1px solid ${C.rule}`, padding: 16 }}>
          <p style={{ margin: "0 0 12px", fontSize: 13, color: C.slate, maxWidth: 640 }}>
            Paste the printed output straight from the notebook for {POLLUTANTS.find(p => p.key === active).label}.
            Index and error columns are ignored, so a copied dataframe works as-is.
          </p>
          <div style={{ display: "grid", gap: 14, gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
            <div>
              <label style={{ fontSize: 12.5, color: C.ink, fontWeight: 600 }}>Actual and predicted values</label>
              <textarea
                value={pasteA} onChange={(e) => setPasteA(e.target.value)} rows={5}
                placeholder={"0   32.161   31.158   1.003\n1   32.641   30.280   2.361"}
                style={{
                  width: "100%", marginTop: 6, padding: 9, fontSize: 12.5, fontFamily: FONT,
                  border: `1px solid ${C.rule}`, background: C.surface, color: C.ink, boxSizing: "border-box",
                }}
              />
              <button onClick={applyPairs} style={btn}>Load predictions</button>
            </div>
            <div>
              <label style={{ fontSize: 12.5, color: C.ink, fontWeight: 600 }}>SHAP contributions</label>
              <textarea
                value={pasteS} onChange={(e) => setPasteS(e.target.value)} rows={5}
                placeholder={"0  Sinter_Machine_Speed  0.7757  20.91\n1  Suction/under_pressure  0.7479  20.16"}
                style={{
                  width: "100%", marginTop: 6, padding: 9, fontSize: 12.5, fontFamily: FONT,
                  border: `1px solid ${C.rule}`, background: C.surface, color: C.ink, boxSizing: "border-box",
                }}
              />
              <button onClick={applyShap} style={btn}>Load drivers</button>
            </div>
          </div>
          {saved && <div style={{ fontSize: 12, color: C.moss, marginTop: 10 }}>{saved}</div>}
        </div>
      )}

      <main style={{ padding: 16, display: "grid", gap: 14 }}>
        {!m ? (
          <div style={{
            background: C.surface, border: `1px dashed ${C.rule}`, padding: "44px 24px", textAlign: "center",
          }}>
            <p style={{ margin: 0, fontSize: 15, color: C.ink }}>
              No {POLLUTANTS.find(p => p.key === active).label} run loaded yet.
            </p>
            <p style={{ margin: "6px 0 16px", fontSize: 13, color: C.mist }}>
              Paste the actual and predicted columns from the notebook and this page fills itself in.
            </p>
            <button onClick={() => setOpenIntake(true)} style={{ ...btn, marginTop: 0 }}>Add data</button>
          </div>
        ) : (
          <>
            {/* headline chart */}
            <Panel
              title="Measured against predicted"
              note={`${m.n} observations · reconstructed at training error · band ±${tol} ${d.unit}`}
              right={
                <label style={{ fontSize: 12, color: C.slate, display: "flex", alignItems: "center", gap: 7 }}>
                  Tolerance ±
                  <input
                    type="number" value={tol} min={0} step={tol >= 10 ? 5 : 0.5}
                    onChange={(e) => setTol(Number(e.target.value) || 0)}
                    style={{
                      width: 52, padding: "4px 6px", fontSize: 12.5, fontFamily: FONT,
                      border: `1px solid ${C.rule}`, color: C.ink,
                    }}
                  />
                </label>
              }
            >
              <div style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series} margin={{ top: 6, right: 8, left: -14, bottom: 4 }}>
                    <CartesianGrid stroke={C.rule} vertical={false} />
                    <XAxis dataKey="i" tick={{ fontSize: 11, fill: C.mist }} stroke={C.rule} />
                    <YAxis tick={{ fontSize: 11, fill: C.mist }} stroke={C.rule}
                      domain={["dataMin - 3", "dataMax + 3"]} />
                    <Tooltip contentStyle={tipStyle} formatter={(v) => Number(v).toFixed(3)} />
                    <Line type="monotone" dataKey="actual" stroke={C.ink} strokeWidth={2}
                      dot={{ r: 2.5, fill: C.ink }} name="Measured" />
                    <Line type="monotone" dataKey="predicted" stroke={C.teal} strokeWidth={2}
                      strokeDasharray="5 3" dot={{ r: 2.5, fill: C.teal }} name="Predicted" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display: "flex", gap: 18, marginTop: 8, fontSize: 12, color: C.slate }}>
                <Key color={C.ink} label="Measured" />
                <Key color={C.teal} label="Predicted" dashed />
              </div>
            </Panel>

            {/* stats */}
            <div style={{ display: "grid", gap: 1, gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", background: C.rule, border: `1px solid ${C.rule}` }}>
              <Stat label="Training R²" value={mdl.trainR2.toFixed(4)} sub={mdl.name} />
              <Stat label="Training MAE" value={mdl.trainMAE.toFixed(4)} sub={d.unit} />
              <Stat label="Training RMSE" value={mdl.trainRMSE.toFixed(4)} sub={d.unit} />
              <Stat label="R² gap" value={mdl.gap.toFixed(4)}
                sub={`test R² ${mdl.testR2.toFixed(4)}`} tone={mdl.gap > 0.3 ? C.ember : C.ink} />
              <Stat label={`Within ±${tol}`} value={within.toFixed(0) + "%"}
                sub={`${m.res.filter(v => Math.abs(v) <= tol).length} of ${m.n} points`}
                tone={within >= 80 ? C.moss : C.ember} />
              <Stat label="Largest miss" value={(m.worstVal ?? m.maxDev) >= 0 ? "+" + m.maxDev.toFixed(3) : m.maxDev.toFixed(3)}
                sub={`row ${m.maxIdx}`} tone={C.ember} />
            </div>

            {/* residuals + parity */}
            <div style={{ display: "grid", gap: 14, gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
              <Panel title="Error by observation" note="Measured minus predicted">
                <div style={{ height: 210 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={series} margin={{ top: 6, right: 8, left: -18, bottom: 4 }}>
                      <CartesianGrid stroke={C.rule} vertical={false} />
                      <XAxis dataKey="i" tick={{ fontSize: 11, fill: C.mist }} stroke={C.rule} />
                      <YAxis tick={{ fontSize: 11, fill: C.mist }} stroke={C.rule} />
                      <Tooltip contentStyle={tipStyle} formatter={(v) => Number(v).toFixed(3)} />
                      <ReferenceLine y={tol} stroke={C.mist} strokeDasharray="3 3" />
                      <ReferenceLine y={-tol} stroke={C.mist} strokeDasharray="3 3" />
                      <ReferenceLine y={0} stroke={C.slate} />
                      <Bar dataKey="res" name="Error">
                        {series.map((s, i) => (
                          <Cell key={i} fill={Math.abs(s.res) > tol ? C.ember : C.steel} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>

              <Panel title="Prediction against measurement" note="Points on the line are exact matches">
                <div style={{ height: 210 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 6, right: 10, left: -18, bottom: 4 }}>
                      <CartesianGrid stroke={C.rule} />
                      <XAxis type="number" dataKey="actual" name="Measured"
                        domain={[(v) => v - parityPad, (v) => v + parityPad]}
                        tick={{ fontSize: 11, fill: C.mist }} stroke={C.rule} />
                      <YAxis type="number" dataKey="predicted" name="Predicted"
                        domain={[(v) => v - parityPad, (v) => v + parityPad]}
                        tick={{ fontSize: 11, fill: C.mist }} stroke={C.rule} />
                      <ZAxis range={[46, 46]} />
                      <Tooltip contentStyle={tipStyle} cursor={{ strokeDasharray: "3 3" }}
                        formatter={(v) => Number(v).toFixed(3)} />
                      <Scatter data={series} fill={C.steel} fillOpacity={0.75} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            </div>

            {/* drivers */}
            <div style={{ display: "grid", gap: 14, gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
              <Panel
                title="What drives the prediction"
                note={shapRows.length ? `Top three account for ${top3.toFixed(1)}% of the signal` : "Load SHAP output to fill this"}
              >
                {shapRows.length ? (
                  <div style={{ height: 230 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={shapRows} dataKey="pct" nameKey="name"
                          cx="50%" cy="50%" innerRadius={48} outerRadius={92}
                          paddingAngle={1} stroke={C.surface} strokeWidth={1.5}>
                          {shapRows.map((r, i) => (
                            <Cell key={r.name} fill={pieColors[i % pieColors.length]} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={tipStyle}
                          formatter={(v, n) => [Number(v).toFixed(2) + "%", n]} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p style={{ fontSize: 13, color: C.mist, margin: 0 }}>
                    Paste the global SHAP table and the split appears here.
                  </p>
                )}
              </Panel>

              <Panel title="Driver ranking" note="Share of total mean absolute SHAP">
                {shapRows.length ? (
                  <div style={{ display: "grid", gap: 8 }}>
                    {shapRows.map((r, i) => (
                      <div key={r.name} style={{ display: "grid", gridTemplateColumns: "1fr 46px", gap: 10, alignItems: "center" }}>
                        <div>
                          <div style={{ fontSize: 12.5, color: C.ink, marginBottom: 4 }}>{r.name}</div>
                          <div style={{ height: 6, background: C.paper }}>
                            <div style={{
                              width: `${(r.pct / shapRows[0].pct) * 100}%`, height: "100%",
                              background: pieColors[i % pieColors.length],
                            }} />
                          </div>
                        </div>
                        <div style={{ fontSize: 12.5, color: C.slate, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                          {r.pct.toFixed(2)}%
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p style={{ fontSize: 13, color: C.mist, margin: 0 }}>Nothing loaded yet.</p>
                )}
              </Panel>
            </div>

            {/* drivers across all three */}
            <Panel
              title="Contributing factors side by side"
              note="Same drivers, ranked within each pollutant"
            >
              <div style={{ display: "grid", gap: 22, gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))" }}>
                {POLLUTANTS.map((p) => (
                  <MiniPie key={p.key} label={p.label} shap={sets[p.key].shap} />
                ))}
              </div>
            </Panel>

            {/* operating window */}
            <Panel
              title="Favourable operating window"
              note={hasBands
                ? "Grey is the range seen in the data, blue is the band that keeps emission lowest"
                : "The band each driver should sit in to keep emission low"}
            >
              {hasBands ? (
                <div style={{ display: "grid", gap: 16 }}>
                  {orderedRanges.map((r) => <RangeBand key={r.name} r={r} unit={d.unit} />)}
                </div>
              ) : shapRows.length ? (
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5, minWidth: 380 }}>
                    <thead>
                      <tr style={{ color: C.slate, textAlign: "left" }}>
                        <th style={th}>Parameter</th>
                        <th style={{ ...th, width: 86 }}>Low</th>
                        <th style={{ ...th, width: 86 }}>High</th>
                        <th style={{ ...th, width: 74, textAlign: "right" }}>Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {shapRows.map((r) => {
                        const rg = rangeOf(r.name);
                        return (
                          <tr key={r.name} style={{ borderTop: `1px solid ${C.rule}` }}>
                            <td style={td}>{r.name}</td>
                            <td style={td}>
                              <input value={rg.lo} onChange={(e) => setRange(r.name, "lo", e.target.value)}
                                placeholder="—" style={cellInput} />
                            </td>
                            <td style={td}>
                              <input value={rg.hi} onChange={(e) => setRange(r.name, "hi", e.target.value)}
                                placeholder="—" style={cellInput} />
                            </td>
                            <td style={{ ...td, textAlign: "right", color: C.slate, fontVariantNumeric: "tabular-nums" }}>
                              {r.pct.toFixed(1)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <p style={{ fontSize: 12, color: C.mist, margin: "12px 0 0" }}>
                    Values are kept between visits, so the window only needs entering once per pollutant.
                  </p>
                </div>
              ) : (
                <p style={{ fontSize: 13, color: C.mist, margin: 0 }}>
                  Load the drivers first — the window is built against them.
                </p>
              )}
            </Panel>

            {/* record */}
            <Panel title="Observation record" note="Every point behind the charts">
              <div style={{ overflowX: "auto", maxHeight: 320, overflowY: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5, minWidth: 340 }}>
                  <thead>
                    <tr style={{ color: C.slate, textAlign: "right" }}>
                      <th style={{ ...th, textAlign: "left" }}>#</th>
                      <th style={th}>Measured</th>
                      <th style={th}>Predicted</th>
                      <th style={th}>Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {series.map((s) => {
                      const off = Math.abs(s.res) > tol;
                      return (
                        <tr key={s.i} style={{ borderTop: `1px solid ${C.rule}` }}>
                          <td style={{ ...td, color: C.mist }}>{s.i}</td>
                          <td style={tdNum}>{s.actual.toFixed(3)}</td>
                          <td style={tdNum}>{s.predicted.toFixed(3)}</td>
                          <td style={{ ...tdNum, color: off ? C.ember : C.slate, fontWeight: off ? 600 : 400 }}>
                            {s.res >= 0 ? "+" : ""}{s.res.toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </>
        )}
      </main>
    </div>
  );
}

function MiniPie({ label, shap }) {
  const total = shap.reduce((s, [, v]) => s + v, 0);
  const rows = shap
    .map(([name, v]) => ({ name, pct: total ? (v / total) * 100 : 0 }))
    .sort((a, b) => b.pct - a.pct);
  const colors = ["#2F6079", "#3A7290", "#4785A4", "#2C8C84", "#5AA39B",
    "#86B8B2", "#C9A227", "#D2BA63", "#C9CFC9", "#B6BEBD"];

  if (!rows.length) {
    return (
      <div>
        <h3 style={{ margin: "0 0 8px", fontSize: 13.5, fontWeight: 600 }}>{label}</h3>
        <p style={{ fontSize: 12.5, color: C.mist, margin: 0 }}>No drivers loaded.</p>
      </div>
    );
  }

  return (
    <div>
      <h3 style={{ margin: "0 0 2px", fontSize: 13.5, fontWeight: 600 }}>{label}</h3>
      <div style={{ fontSize: 11.5, color: C.mist, marginBottom: 6 }}>
        top three {rows.slice(0, 3).reduce((s, r) => s + r.pct, 0).toFixed(1)}%
      </div>
      <div style={{ height: 150 }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={rows} dataKey="pct" nameKey="name" cx="50%" cy="50%"
              innerRadius={32} outerRadius={62} paddingAngle={1}
              stroke={C.surface} strokeWidth={1.5}>
              {rows.map((r, i) => <Cell key={r.name} fill={colors[i % colors.length]} />)}
            </Pie>
            <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.rule}`, fontSize: 12 }}
              formatter={(v, n) => [Number(v).toFixed(2) + "%", n]} />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <ol style={{ margin: "8px 0 0", padding: 0, listStyle: "none", display: "grid", gap: 4 }}>
        {rows.map((r, i) => (
          <li key={r.name} style={{
            display: "grid", gridTemplateColumns: "9px 1fr auto", gap: 7,
            alignItems: "baseline", fontSize: 11.5,
          }}>
            <span style={{ width: 9, height: 9, background: colors[i % colors.length], display: "inline-block" }} />
            <span style={{ color: C.ink }}>{r.name}</span>
            <span style={{ color: C.slate, fontVariantNumeric: "tabular-nums" }}>{r.pct.toFixed(1)}%</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function RangeBand({ r, unit }) {
  const span = r.obsHi - r.obsLo;
  const flat = span <= 0;
  const pos = (v) => (flat ? 0 : ((v - r.obsLo) / span) * 100);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>{r.name}</span>
        <span style={{ fontSize: 12, color: C.slate, fontVariantNumeric: "tabular-nums" }}>
          {flat ? "constant" : `${fmt(r.favLo)} – ${fmt(r.favHi)}`}
        </span>
      </div>
      <div style={{ position: "relative", height: 10, background: C.paper, marginTop: 7 }}>
        {!flat && (
          <>
            <div style={{
              position: "absolute", left: `${pos(r.favLo)}%`,
              width: `${Math.max(pos(r.favHi) - pos(r.favLo), 1.2)}%`,
              top: 0, bottom: 0, background: C.steel,
            }} />
            <div style={{
              position: "absolute", left: `calc(${pos(r.best)}% - 1px)`,
              width: 2, top: -3, bottom: -3, background: C.ink,
            }} />
          </>
        )}
      </div>
      <div style={{
        display: "flex", justifyContent: "space-between", marginTop: 5,
        fontSize: 11.5, color: C.mist, fontVariantNumeric: "tabular-nums",
      }}>
        <span>{fmt(r.obsLo)}</span>
        <span style={{ color: C.slate }}>
          best {fmt(r.best)} · holds emission near {fmt(r.emission)} {unit}
        </span>
        <span>{fmt(r.obsHi)}</span>
      </div>
    </div>
  );
}

function Key({ color, label, dashed }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
      <span style={{
        width: 20, height: 0, borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}`,
      }} />
      {label}
    </span>
  );
}

const btn = {
  marginTop: 8, background: C.ink, color: "#EDF1F1", border: "none",
  padding: "8px 15px", fontSize: 12.5, cursor: "pointer", fontFamily: FONT,
};
const th = { padding: "6px 8px", fontWeight: 600, fontSize: 11.5 };
const td = { padding: "7px 8px", color: C.ink };
const tdNum = { padding: "7px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums", color: C.ink };
const cellInput = {
  width: "100%", padding: "4px 6px", fontSize: 12.5, fontFamily: FONT,
  border: `1px solid ${C.rule}`, color: C.ink, boxSizing: "border-box",
};
