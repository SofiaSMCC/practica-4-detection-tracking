// Genera presentacion/Practica4_Deteccion_Tracking.pptx con el estilo de template.pdf
// (banda azul marino, pie "Página N", tablas estilo LaTeX, diagramas de cajas redondeadas).
// Uso (desde la raiz del proyecto):  python -I presentacion/make_figures.py && python -I presentacion/collect_results.py
//   && python -I presentacion/collect_m4.py && node presentacion/build_deck.js
// Las laminas marcadas "Por completar" son la estructura de resultados: se rellenan con results/comparison_mean.csv.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
// apply_theme.js viene del skill pptx; su ruta cambia por maquina: se puede pasar con PPTX_SKILL=<carpeta del skill>
const SKILL_DIRS = [process.env.PPTX_SKILL, "/home/marcelino.maita/.claude/skills/synced/21ca2af7-4662-4d2f-a2a2-c6a78ffddc45_c033fe8a-d2df-4226-86e8-d0d0ad902e7c/pptx"].filter(Boolean);
const { applyTheme } = require(path.join(SKILL_DIRS.find((d) => fs.existsSync(path.join(d, "scripts", "apply_theme.js"))), "scripts", "apply_theme.js"));

const HERE = __dirname;
const FIG = (f) => path.join(HERE, "fig", f);
const OUT = path.join(HERE, "Practica4_Deteccion_Tracking.pptx");
const stats = JSON.parse(fs.readFileSync(path.join(HERE, "stats.json"), "utf8"));
const RS = JSON.parse(fs.readFileSync(path.join(HERE, "results.json"), "utf8"));
const gmc = JSON.parse(fs.readFileSync(path.join(HERE, "gmc_stats.json"), "utf8"));
const EQS = JSON.parse(fs.readFileSync(path.join(HERE, "fig", "eq", "eq.json"), "utf8")); // tamaños (in) de cada ecuacion a 18 pt
const fm = (n, d = 1) => Number(n).toFixed(d).replace(".", ",").replace("-", "−");

// ---- paleta tomada de template.pdf --------------------------------------------------------
const NAVY = "1B3A66", FOOT = "E9EBF7", HL = "FFC04D", HL2 = "FFDE91";
const INK = "1A1A1A", MUTED = "6B7280", RULE = "9AA0A6", GREEN = "2E7D32", RED = "C62828", LIGHTBLUE = "A9C4E4";
const SANS = "Calibri", SERIF = "Cambria", MONO = "Courier New", DIAG = "Times New Roman";
const FOOTER_TEXT = "Detección y tracking de vehículos con drones";

const THEME = {
  name: "UTEC Vision",
  headFontFace: SANS, bodyFontFace: SANS,
  colors: { dk1: INK, lt1: "FFFFFF", dk2: NAVY, lt2: FOOT, accent1: NAVY, accent2: HL, accent3: GREEN,
            accent4: RED, accent5: "5B7DB1", accent6: "8A8F98", hlink: NAVY, folHlink: "5B7DB1" },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: SANS, bodyFontFace: SANS };
pres.title = "Detección y tracking de vehículos con drones";
pres.author = "Visión por Computador";

// ---- layouts (el "marco" de cada tipo de lamina) -------------------------------------------
const footerObjs = () => [
  { rect: { x: 0, y: 5.4, w: 10, h: 0.225, fill: { color: FOOT }, line: { color: FOOT, width: 0 } } },
  { text: { text: FOOTER_TEXT, options: { x: 0.12, y: 5.4, w: 6, h: 0.225, fontFace: SANS, fontSize: 8, color: "4B5563", valign: "middle", margin: 0 } } },
  { text: { text: "Página", options: { x: 8.85, y: 5.4, w: 0.6, h: 0.225, fontFace: SANS, fontSize: 8, color: "4B5563", align: "right", valign: "middle", margin: 0 } } },
];
const slideNum = { x: 9.5, y: 5.4, w: 0.4, h: 0.225, fontFace: SANS, fontSize: 8, color: "4B5563", align: "left", valign: "middle", margin: 0 };

pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" }, slideNumber: slideNum,
  objects: [
    { rect: { x: 0, y: 0, w: 10, h: 0.6, fill: { color: NAVY }, line: { color: NAVY, width: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.2, y: 0.04, w: 9.6, h: 0.52, fontFace: SANS, fontSize: 24, color: "FFFFFF", align: "left", valign: "middle", margin: 0 }, text: "" } },
    ...footerObjs(),
  ],
});
pres.defineSlideMaster({
  title: "TITLE", background: { color: "FFFFFF" },
  objects: [
    { rect: { x: 0.5, y: 0.85, w: 9, h: 1.3, fill: { color: NAVY }, line: { color: NAVY, width: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.95, w: 8.8, h: 0.65, fontFace: SANS, fontSize: 30, color: "FFFFFF", align: "center", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 1.6, w: 8.8, h: 0.4, fontFace: SANS, fontSize: 16, color: "FFFFFF", align: "center", valign: "middle", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CLOSING", background: { color: "FFFFFF" }, slideNumber: slideNum,
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 1, y: 2.2, w: 8, h: 1.1, fontFace: SANS, fontSize: 44, color: NAVY, align: "center", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Página", options: { x: 8.85, y: 5.4, w: 0.6, h: 0.225, fontFace: SANS, fontSize: 8, color: "4B5563", align: "right", valign: "middle", margin: 0 } } },
  ],
});

// ---- ayudantes -----------------------------------------------------------------------------
const content = (title, section) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};
const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: SANS, color: INK, margin: 0, valign: "top", ...o });

// lista con vinetas circulares (como la plantilla)
const bullets = (items, o = {}) =>
  items.map((t, i) => ({
    text: t,
    options: { bullet: { code: "25CF" }, breakLine: i < items.length - 1, paraSpaceAfter: o.gap ?? 8, fontSize: o.size ?? 16, color: INK, fontFace: SANS },
  }));

// tabla estilo LaTeX (booktabs): regla gruesa arriba y abajo, fina bajo el encabezado
const none = { type: "none" };
const rule = (pt) => ({ type: "solid", pt, color: INK });
function latexTable(s, header, rows, o) {
  const size = o.size ?? 13;
  const mk = (txt, r, c, last) => {
    const head = r === 0;
    const cell = typeof txt === "object" && txt !== null && !Array.isArray(txt) ? txt : { text: String(txt) };
    return {
      text: cell.text,
      options: {
        fontFace: SERIF, fontSize: size, color: cell.color ?? INK, bold: head || cell.bold || false,
        align: cell.align ?? (c === 0 ? "left" : "center"), valign: "middle",
        fill: cell.fill ? { color: cell.fill } : undefined, margin: [0.05, 0.08, 0.05, 0.08],
        border: [head ? rule(1.5) : none, none, head ? rule(0.75) : last ? rule(1.5) : none, none],
      },
    };
  };
  const data = [header.map((h, c) => mk(h, 0, c, false)), ...rows.map((row, r) => row.map((v, c) => mk(v, r + 1, c, r === rows.length - 1)))];
  s.addTable(data, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH ?? 0.4, autoPage: false });
}
const dash = "—";
const pending = (v = dash) => ({ text: v, color: RULE });
function pill(s) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.2, y: 0.72, w: 1.45, h: 0.28, rectRadius: 0.1, fill: { color: HL }, line: { color: HL, width: 0 } });
  T(s, "Por completar", { x: 8.2, y: 0.72, w: 1.45, h: 0.28, fontSize: 11, bold: true, color: NAVY, align: "center", valign: "middle" });
}
function placeholderBox(s, x, y, w, h, label, color = RULE) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "FFFFFF" }, line: { color, width: 1.5, dashType: "dash" } });
  T(s, label, { x, y, w, h, fontSize: 13, italic: true, color: MUTED, align: "center", valign: "middle" });
}
// diagrama tipo TikZ: caja redondeada + texto serif
function box(s, x, y, w, h, text, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: o.fill ?? "FFFFFF" }, line: { color: INK, width: 1, dashType: o.dashed ? "dash" : "solid" } });
  T(s, text, { x: x + 0.05, y, w: w - 0.1, h, fontFace: DIAG, fontSize: o.size ?? 12, align: "center", valign: "middle" });
}
function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: INK, width: 1.25, endArrowType: "triangle" } });
}
// ecuacion (PNG con fuente Computer Modern); scale 1 = 18 pt. align "center" centra dentro de [x, x+w]
function eq(s, name, x, y, o = {}) {
  const [w0, h0] = EQS[name], k = o.scale ?? 1, w = w0 * k, h = h0 * k;
  const xx = o.align === "center" ? x + (o.w - w) / 2 : x;
  s.addImage({ path: path.join(HERE, "fig", "eq", name + ".png"), x: xx, y, w, h, altText: name });
  return h;
}
const label = (s, text, x, y, w = 6) => T(s, text, { x, y, w, h: 0.3, fontSize: 14, bold: true, color: NAVY });
const cite = (s, text) => T(s, text, { x: 0.4, y: 5.07, w: 9.2, h: 0.28, fontSize: 10, italic: true, color: MUTED });
const caption = (s, text, x, y, w) => T(s, text, { x, y, w, h: 0.3, fontSize: 11, italic: true, color: MUTED });

// =============================================================================================
// 1. Portada
pres.addSection({ title: "Introducción" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introducción" });
  s.addText("Detección y tracking de vehículos con drones", { placeholder: "title" });
  s.addText("YOLOv11, YOLO26 y RT-DETR con SORT, ByteTrack y BoT-SORT sobre UAVDT", { placeholder: "subtitle" });
  ["Diego Quispe", "Brigitte Rojas", "Alvaro Garcia", "Sofia Salazar", "Marcelino Maita"].forEach((n, i) =>
    T(s, n, { x: 0.5 + i * 1.8, y: 2.55, w: 1.8, h: 0.4, fontSize: 15, align: "center" }));
  T(s, "Visión por Computador\nPráctica 4", { x: 0.5, y: 3.3, w: 9, h: 0.6, fontSize: 12, align: "center" });
  T(s, "Octubre de 2026", { x: 0.5, y: 4.2, w: 9, h: 0.4, fontSize: 15, align: "center" });
  s.addNotes("Tema: seguimiento multi-objeto de vehículos en video aéreo (UAVDT), separando detección y asociación.");
}

// 2. Indice
{
  const s = content("Índice", "Introducción");
  const left = ["Problema y dataset", "Pipeline y las 4 configuraciones", "YOLO26 frente a RT-DETR", "Trackers: geometría y apariencia"];
  const right = ["SAHI, fusión y ajuste", "Protocolo de evaluación", "Resultados y demos visuales", "Conclusiones y referencias"];
  const col = (items, x) => items.forEach((t, i) => {
    const y = 1.4 + i * 0.7;
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.06, w: 0.2, h: 0.2, fill: { color: NAVY }, line: { color: NAVY, width: 0 } });
    T(s, t, { x: x + 0.38, y, w: 4.2, h: 0.34, fontSize: 18, valign: "middle" });
  });
  col(left, 0.6); col(right, 5.1);
}

// 3. Problema y dataset
{
  const s = content("Problema y dataset", "Introducción");
  T(s, "UAVDT-Benchmark (subset de test)", { x: 0.4, y: 0.95, w: 4.4, h: 0.35, fontSize: 18 });
  T(s, bullets(["20 de las 50 secuencias (split oficial)", "16 592 frames de 1024 × 540 px", "340 906 cajas: car, truck y bus"], { size: 16, gap: 6 }), { x: 0.55, y: 1.4, w: 4.2, h: 1.5 });
  T(s, "Cada secuencia trae sus imágenes, un gt.txt con identidades y regiones ignoradas.", { x: 0.4, y: 3.15, w: 4.2, h: 0.9, fontSize: 16 });
  T(s, "Retos de la vista aérea: vehículos de pocos píxeles, alta densidad, cámara en movimiento y oclusiones.", { x: 0.4, y: 4.15, w: 4.2, h: 0.9, fontSize: 14, color: MUTED });
  s.addImage({ path: FIG("collage.jpg"), x: 4.95, y: 1.0, w: 4.75, h: 2.52 });
  caption(s, "Secuencias M0203, M0403, M0701 y M1301", 4.95, 3.58, 4.75);
  cite(s, "Du et al. [1]");
  s.addNotes("El PDF obliga a usar el subset de test de UAVDT: las 20 secuencias de M_attr/test. La vista aérea trae vehículos pequeños, mucha densidad y movimiento de cámara.");
}

// 4. Tamaño y densidad (graficos nativos con datos reales)
{
  const s = content("Secuencias de test: tamaño y densidad", "Introducción");
  const labels = stats.map((r) => r.seq);
  const base = { barDir: "col", showLegend: false, showTitle: true, titleFontFace: SANS, titleFontSize: 12, titleColor: INK,
    catAxisLabelFontFace: SANS, valAxisLabelFontFace: SANS, catAxisLabelFontSize: 8, valAxisLabelFontSize: 9,
    catAxisLabelColor: "4B5563", valAxisLabelColor: "4B5563", valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
    catAxisLabelRotate: -90, barGapWidthPct: 40 };
  s.addChart(pres.charts.BAR, [{ name: "Frames", labels, values: stats.map((r) => r.frames) }],
    { ...base, x: 0.3, y: 0.8, w: 4.7, h: 3.7, chartColors: [LIGHTBLUE], title: "Frames por secuencia" });
  s.addChart(pres.charts.BAR, [{ name: "Vehículos por frame", labels, values: stats.map((r) => r.per_frame) }],
    { ...base, x: 5.0, y: 0.8, w: 4.7, h: 3.7, chartColors: [NAVY], title: "Vehículos anotados por frame" });
  const lo = stats.reduce((a, b) => (a.per_frame < b.per_frame ? a : b)), hi = stats.reduce((a, b) => (a.per_frame > b.per_frame ? a : b));
  T(s, `La densidad va de ${fm(lo.per_frame)} vehículos por frame (${lo.seq}) a ${fm(hi.per_frame)} (${hi.seq}): el tracker se enfrenta a escenas muy distintas.`,
    { x: 0.4, y: 4.6, w: 9.2, h: 0.65, fontSize: 14 });
}

// ---------------------------------------------------------------------------------------------
// METODOLOGIA
pres.addSection({ title: "Metodología" });
// 5. Formato MOT y GT
{
  const s = content("Formato MOT Challenge y ground truth", "Metodología");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 0.8, w: 9.2, h: 0.4, fill: { color: FOOT }, line: { color: FOOT, width: 0 } });
  T(s, "<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<vis>", { x: 0.4, y: 0.8, w: 9.2, h: 0.4, fontFace: MONO, fontSize: 12, align: "center", valign: "middle" });
  s.addImage({ path: FIG("gt_frame.jpg"), x: 0.4, y: 1.4, w: 5.5, h: 2.9 });
  caption(s, "Frame 200 de M0403. Verde: GT. Rojo: región ignorada.", 0.4, 4.36, 5.5);
  T(s, bullets(["El GT solo usa cajas con score = 1", "Las predicciones siguen el mismo formato", "class: 1 car, 2 truck, 3 bus; vis = 1", "Se descartan las predicciones dentro de regiones ignoradas"], { size: 14, gap: 8 }), { x: 6.1, y: 1.4, w: 3.5, h: 3.4 });
  cite(s, "Formato de Milan et al. [17]");
}

// 6. Pipeline 1/2
{
  const s = content("Pipeline general (1/2): detección y asociación", "Metodología");
  const labels = ["Frame t\n(img######.jpg)", "Detector\n(pesos preentrenados)", "Detecciones\n(x1, y1, x2, y2,\nconf, clase)", "Tracker\n(asociación)", "Tracks\ncon ID", "Archivo MOT\n(.txt)"];
  const w = 1.32, gap = 0.3, x0 = 0.4, y = 1.15, h = 0.9;
  labels.forEach((t, i) => {
    const x = x0 + i * (w + gap);
    box(s, x, y, w, h, t, { size: 11, fill: i === 5 ? "DCE6F5" : "FFFFFF" });
    if (i < labels.length - 1) arrow(s, x + w, y + h / 2, x + w + gap, y + h / 2);
  });
  const brace = (i, txt) => {
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.LINE, { x, y: y + h + 0.12, w, h: 0, line: { color: NAVY, width: 1.5 } });
    T(s, txt, { x: x - 0.2, y: y + h + 0.18, w: w + 0.4, h: 0.3, fontFace: DIAG, fontSize: 12, italic: true, color: NAVY, align: "center" });
  };
  brace(1, "t_det"); brace(3, "t_trk");
  eq(s, "fps", 0.4, 2.7, { w: 9.2, align: "center", scale: 0.85 });
  const y2 = 3.55, w2 = 2.3;
  box(s, 0.4, y2, w2, 0.85, "Archivo MOT\n(predicciones)", { size: 12 });
  box(s, 3.85, y2, w2, 0.85, "motmetrics + GT\n(IoU ≥ 0,5)", { size: 12 });
  box(s, 7.3, y2, 2.3, 0.85, "MOTA · IDF1\nIDS · FPS", { size: 12, fill: "DCE6F5" });
  arrow(s, 0.4 + w2, y2 + 0.425, 3.85, y2 + 0.425); arrow(s, 3.85 + w2, y2 + 0.425, 7.3, y2 + 0.425);
  s.addNotes("Dos pasos desacoplados: primero detección con pesos preentrenados y luego asociación. Medimos t_det y t_trk por separado para analizar el efecto de quitar NMS.");
}

// 7. Pipeline 2/2: las cuatro configuraciones
{
  const s = content("Pipeline general (2/2): las cuatro configuraciones", "Metodología");
  latexTable(s, ["Método", "Detector", "Tracker", "Qué se evalúa"], [
    ["1  Baseline", "YOLOv11", "SORT", "Asociación puramente geométrica"],
    ["2  Eficiencia", "YOLO26 (sin NMS)", "ByteTrack", "Latencia sin NMS; uso de detecciones de baja confianza"],
    ["3  Precisión", "RT-DETRv4", "BoT-SORT + GMC", "Atención global y compensación del movimiento de cámara"],
    ["4  Resolución", "YOLO26 + SAHI, fusionado con RT-DETRv4", "ByteTrack + filtro de escala", "Vehículos pequeños (SAHI) y heurísticas propias: fusión de detectores y filtro de escala"],
  ], { x: 0.4, y: 1.0, w: 9.2, colW: [1.7, 2.0, 1.9, 3.6], rowH: 0.62, size: 14 });
  T(s, "En todos los métodos la detección usa pesos preentrenados (sin reentrenar) y la asociación se ejecuta como un paso aparte.", { x: 0.4, y: 4.4, w: 9.2, h: 0.7, fontSize: 14, color: MUTED });
}

// 8. YOLO26: arquitectura
{
  const s = content("YOLO26: arquitectura", "Metodología");
  const items = ["Imagen\n1024 × 1024", "Backbone\nConv, C3k2, SPPF,\nC2PSA (atención en P5)", "Neck tipo PAN\nfusión multiescala\nP3 / P4 / P5", "Cabeza Detect\nend-to-end:\nuna caja por objeto"];
  items.forEach((t, i) => {
    const x = 0.4 + i * 2.4;
    box(s, x, 0.85, 2.0, 0.95, t, { size: 11.5, fill: i === 3 ? "DCE6F5" : "FFFFFF" });
    if (i < 3) arrow(s, x + 2.0, 1.325, x + 2.4, 1.325);
  });
  T(s, bullets([
    "Backbone y neck convolucionales; C2PSA añade atención parcial solo en el nivel más profundo (P5)",
    "Sin DFL: la caja se regresa directamente (reg_max = 1), con una cabeza más simple",
    "Dos cabezas al entrenar: uno-a-muchos (top-k = 10) y uno-a-uno (top-k = 1). En inferencia solo la uno-a-uno: sin NMS",
    "ProgLoss y STAL refuerzan las etiquetas positivas de objetos pequeños; optimizador MuSGD",
    "YOLO26-m: ≈ 22 M parámetros y 75 GFLOPs (configuración COCO)",
  ], { size: 14, gap: 7 }), { x: 0.5, y: 2.1, w: 9.1, h: 2.9 });
  cite(s, "Ultralytics YOLO26 [13]; configuración yolo26.yaml de ultralytics 8.4");
  s.addNotes("La clave para el tracking: sin NMS, la latencia de detección es más estable. Con --with_nms medimos la diferencia.");
}

// 9. RT-DETRv4: arquitectura
{
  const s = content("RT-DETRv4: arquitectura", "Metodología");
  const items = ["Imagen\n960 × 960", "Backbone\nHGNetv2\n(S3, S4, S5)", "Codificador híbrido\nAIFI (atención en S5)\n+ fusión CNN", "Decodificador\nD-FINE\n4 capas, queries", "Cajas y clases\n(sin NMS)"];
  items.forEach((t, i) => {
    const x = 0.4 + i * 1.9;
    box(s, x, 0.85, 1.6, 0.95, t, { size: 10.5, fill: i === 4 ? "DCE6F5" : "FFFFFF" });
    if (i < 4) arrow(s, x + 1.6, 1.325, x + 1.9, 1.325);
  });
  T(s, bullets([
    "Cada token atiende a todos los demás: contexto global",
    "El decodificador usa queries aprendidas; cada una predice un objeto",
    "Entrenamiento con emparejamiento húngaro uno-a-uno: sin duplicados, sin NMS",
    "RT-DETRv4 destila un modelo de visión fundacional (DSI y GAM) solo al entrenar; la inferencia no cambia",
  ], { size: 13, gap: 7 }), { x: 0.5, y: 2.1, w: 5.0, h: 3.0 });
  label(s, "Atención (Vaswani et al. [9])", 5.7, 2.15, 4);
  eq(s, "attn", 5.7, 2.5, { scale: 0.88 });
  label(s, "Emparejamiento uno-a-uno (DETR [8])", 5.7, 3.5, 4);
  eq(s, "match", 5.7, 3.85, { scale: 1.0 });
  cite(s, "RT-DETR [10], D-FINE [11], RT-DETRv4 [12]; configuración config.yaml de los pesos VisDrone");
}

// 10. YOLO26 vs RT-DETR
{
  const s = content("YOLO26 frente a RT-DETR", "Metodología");
  latexTable(s, ["Aspecto", "YOLOv11", "YOLO26", "RT-DETRv4"], [
    ["Entrenado en", "UAVDT", "UAVDT", "VisDrone [14]"],
    ["Atención", "C2PSA (solo P5)", "C2PSA (solo P5)", "AIFI en S5 y cross-attention sobre 3 escalas"],
    ["Predicción", "Cuadrícula densa", "Cuadrícula densa", "Queries aprendidas"],
    ["Duplicados", "NMS", "Uno-a-uno, sin NMS", "Húngaro, sin NMS"],
    ["Regresión de caja", "Con DFL", "Directa (sin DFL)", "Distribución refinada"],
    ["Entrada", "1024 px", "1024 px", "960 px"],
  ], { x: 0.4, y: 0.9, w: 9.2, colW: [1.9, 2.0, 2.2, 3.1], rowH: 0.46, size: 12.5 });
  T(s, "Hipótesis a contrastar en los resultados: STAL (YOLO26) ayuda con objetos pequeños; la atención global de RT-DETR ayuda con contexto y oclusiones.", { x: 0.4, y: 4.35, w: 9.2, h: 0.7, fontSize: 14, color: MUTED });
  cite(s, "RT-DETR usa 10 clases de VisDrone: car y van pasan a car, truck a truck y bus a bus");
}

// 11. Asociacion: IoU y asignacion
{
  const s = content("Asociación: IoU y asignación óptima", "Metodología");
  label(s, "Superposición entre cajas", 0.4, 0.9);
  eq(s, "iou", 0.4, 1.25, { scale: 1.1 });
  label(s, "Costo entre la caja predicha b̂ᵢ del track i y la detección dⱼ", 0.4, 2.15, 6);
  eq(s, "cost", 0.4, 2.5);
  label(s, "Asignación óptima (algoritmo húngaro [3])", 0.4, 3.1);
  eq(s, "hungarian", 0.4, 3.45, { scale: 1.1 });
  eq(s, "accept", 0.4, 4.35);
  // diagrama de IoU
  s.addShape(pres.shapes.RECTANGLE, { x: 6.5, y: 1.0, w: 1.7, h: 1.2, fill: { color: HL, transparency: 25 }, line: { color: INK, width: 1 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.4, y: 1.55, w: 1.7, h: 1.2, fill: { color: "A9C4E4", transparency: 25 }, line: { color: INK, width: 1 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.4, y: 1.55, w: 0.8, h: 0.65, fill: { color: NAVY, transparency: 15 }, line: { color: NAVY, width: 1 } });
  T(s, "A", { x: 6.6, y: 1.05, w: 0.4, h: 0.3, fontFace: DIAG, fontSize: 14, italic: true });
  T(s, "B", { x: 8.75, y: 2.4, w: 0.4, h: 0.3, fontFace: DIAG, fontSize: 14, italic: true });
  T(s, "A ∩ B", { x: 7.4, y: 1.55, w: 0.8, h: 0.65, fontFace: DIAG, fontSize: 11, color: "FFFFFF", align: "center", valign: "middle" });
  T(s, "SORT, ByteTrack y BoT-SORT comparten esta asociación; cambian la caja predicha b̂ y qué detecciones participan.", { x: 6.4, y: 3.1, w: 3.2, h: 1.4, fontSize: 13, color: "374151" });
  cite(s, "Kuhn [3]");
}

// 12. SORT: Kalman
{
  const s = content("SORT: filtro de Kalman", "Metodología");
  label(s, "Estado de cada track", 0.4, 0.88);
  eq(s, "kstate", 0.4, 1.2);
  label(s, "Predicción", 0.4, 1.7);
  eq(s, "kpred1", 0.4, 2.02); eq(s, "kpred2", 2.9, 2.0);
  label(s, "Actualización con la detección zₖ", 0.4, 2.6);
  eq(s, "kgain", 0.4, 2.92);
  eq(s, "kupd1", 0.4, 3.4); eq(s, "kupd2", 4.4, 3.4);
  T(s, "Sin detección, el track solo predice. Muere tras max_age = 30 frames sin pareja y se confirma con min_hits = 3.", { x: 0.4, y: 4.1, w: 6.0, h: 0.8, fontSize: 13, color: "374151" });
  T(s, bullets(["Asociación: IoU + húngaro", "Nace un track con cada detección sin pareja", "Solo geometría: ni apariencia ni cámara"], { size: 13, gap: 8 }), { x: 6.7, y: 1.2, w: 2.95, h: 3.0 });
  cite(s, "Bewley et al. [4]; Kalman [2]");
  s.addNotes("F es el modelo de velocidad constante; H observa solo [u, v, s, r]. Q y R son las covarianzas del proceso y de la medición.");
}

// 13. ByteTrack
{
  const s = content("ByteTrack: asociar toda detección", "Metodología");
  const items = ["Ronda 1\ntracks × detecciones D_alta", "Ronda 2\ntracks sin pareja × D_baja\n(solo IoU)", "Tracks nuevos\nD_alta sin pareja\n(confianza ≥ 0,25)"];
  items.forEach((t, i) => {
    const x = 0.4 + i * 3.2;
    box(s, x, 0.85, 2.8, 0.9, t, { size: 11.5, fill: i === 2 ? "DCE6F5" : "FFFFFF" });
    if (i < 2) arrow(s, x + 2.8, 1.3, x + 3.2, 1.3);
  });
  label(s, "Dos grupos de detecciones según su confianza s", 0.4, 2.0, 8);
  eq(s, "bt_high", 0.4, 2.35);
  label(s, "Costo que incorpora la confianza (fuse_score)", 0.4, 3.0, 8);
  eq(s, "bt_fuse", 0.4, 3.35);
  T(s, bullets(["τ_alta = 0,25 y τ_baja = 0,1 (valores por defecto de Ultralytics)", "Una detección dudosa (sombra, árbol, oclusión parcial) rescata el track en vez de perderlo", "Costo adicional casi nulo; sigue sin usar apariencia"], { size: 13, gap: 6 }), { x: 3.4, y: 3.3, w: 6.2, h: 1.7 });
  cite(s, "Zhang et al. [5]");
}

// 14. BoT-SORT + GMC
{
  const s = content("BoT-SORT: compensación del movimiento de cámara", "Metodología");
  s.addImage({ path: FIG("gmc_flow.jpg"), x: 0.4, y: 0.85, w: 5.2, h: 2.74 });
  caption(s, `${gmc.seq}, frames ${gmc.f0} → ${gmc.f0 + gmc.dt}. Verde: fondo. Rojo: vehículos en movimiento.`, 0.4, 3.62, 5.3);
  T(s, bullets([
    "Esquinas (Shi–Tomasi) seguidas con flujo óptico de Lucas–Kanade",
    `RANSAC ajusta A: aquí t = (${fm(gmc.tx)}; ${fm(gmc.ty)}) px y giro de ${fm(gmc.rot_deg)}° en ${gmc.dt} frames (${gmc.inliers} de ${gmc.pts} puntos son inliers)`,
    "A corrige la media y la covarianza de cada track antes de asociar",
  ], { size: 12.5, gap: 7 }), { x: 5.85, y: 0.9, w: 3.8, h: 3.1 });
  eq(s, "gmc_pos", 0.4, 4.15, { scale: 0.95 }); eq(s, "gmc_cov", 4.5, 4.15, { scale: 0.95 });
  cite(s, "Aharon et al. [6]; flujo óptico y RANSAC con OpenCV. R rota cada par (posición, velocidad) del estado (x, y, w, h)");
  s.addNotes("Un dron en movimiento desplaza a todos los vehículos a la vez. GMC estima ese desplazamiento global con el fondo y lo resta antes de asociar.");
}

// 15. Apariencia vs geometria
{
  const s = content("Apariencia (embeddings) frente a geometría", "Metodología");
  latexTable(s, ["Situación", "Qué le pasa a la geometría"], [
    ["Oclusión larga", "El Kalman deriva y el track se pierde"],
    ["Vehículos que se cruzan", "El IoU es ambiguo entre dos tracks"],
    ["Cámara en movimiento", "La predicción queda desplazada (GMC)"],
    ["Detección intermitente", "El track muere y reaparece con otro ID"],
  ], { x: 0.4, y: 0.9, w: 5.1, colW: [1.9, 3.2], rowH: 0.48, size: 12 });
  label(s, "Distancia de apariencia (BoT-SORT)", 5.75, 0.88, 4);
  eq(s, "cos", 5.75, 1.2, { scale: 0.95 });
  eq(s, "fusion", 5.75, 1.95, { scale: 0.95 });
  eq(s, "ema", 5.75, 2.55, { scale: 0.9 });
  T(s, bullets([
    "La apariencia re-identifica tras una oclusión larga, pero exige un embedding por detección",
    "En vista aérea los vehículos miden unas decenas de píxeles y muchos son parecidos: el embedding puede discriminar poco (hipótesis)",
    "Aquí: BoT-SORT con GMC y sin ReID (with_reid = False). Comparamos geometría pura (SORT, ByteTrack) con geometría + GMC; ReID queda como extensión",
  ], { size: 12, gap: 5 }), { x: 0.5, y: 3.65, w: 9.1, h: 1.4 });
  cite(s, "BoT-SORT [6]; embeddings f normalizados, e: promedio móvil por track");
  s.addNotes("Justificación del uso de apariencia: re-identificación y oclusiones largas. Decisión de este trabajo: no activar ReID por costo y por la poca resolución de los vehículos; lo evaluamos comparando contra los métodos geométricos.");
}

// 16. SAHI
{
  const s = content("SAHI: detección por slices", "Metodología");
  s.addImage({ path: FIG("sahi_slices.jpg"), x: 0.4, y: 0.95, w: 5.6, h: 2.95 });
  caption(s, "Configuración usada: slices de 640 px con 20 % de solape", 0.4, 3.97, 5.6);
  T(s, bullets(["Corta el frame en recuadros solapados y detecta en cada uno", "Los vehículos pequeños ganan tamaño relativo", "Se fusionan los duplicados (GREEDYNMM, IOS 0,5)", "Costo: varias inferencias por frame, además de la pasada completa"], { size: 14, gap: 8 }), { x: 6.2, y: 1.0, w: 3.45, h: 3.9 });
  cite(s, "Akyon et al. [7]");
  s.addNotes("Con frames de 1024x540 y slices de 640 salen solo 2 recuadros, así que el aumento es modesto. Slices más chicos dan más aumento a costa de más inferencias: es un parámetro a ajustar.");
}

// 17. SAHI: matematica y costo
{
  const s = content("SAHI: ampliación y costo", "Metodología");
  latexTable(s, ["Slice S", "Solape o", "Slices", "Inferencias por frame", "Ampliación"], [
    ["640 px", "0,2", "2", "3", "1,6×"],
    ["480 px", "0,2", "6", "7", "2,1×"],
    ["320 px", "0,2", "8", "9", "3,2×"],
    ["320 px", "0,3", "10", "11", "3,2×"],
  ], { x: 0.4, y: 0.9, w: 9.2, colW: [1.6, 1.6, 1.4, 2.6, 2.0], rowH: 0.42, size: 13 });
  caption(s, "Frames de 1024 × 540, imgsz = 1024. Inferencias = slices + 1 (pasada sobre la imagen completa). Conteo con sahi.slicing.", 0.4, 3.0, 9.2);
  eq(s, "nslices", 0.4, 3.4, { scale: 1.0 });
  eq(s, "zoom", 5.9, 3.45, { scale: 1.0 });
  eq(s, "ios", 0.4, 4.25, { scale: 0.95 });
  T(s, "El costo crece con el número de inferencias; la ganancia depende de la ampliación.", { x: 5.9, y: 4.3, w: 3.7, h: 0.7, fontSize: 12, color: "374151" });
}

// 18. Hiperparametros
{
  const s = content("Hiperparámetros a ajustar", "Metodología");
  latexTable(s, ["Parámetro", "Qué controla", "Valor inicial"], [
    ["conf (detector)", "Umbral de confianza de las detecciones", "0,25 (0,1 con ByteTrack y BoT-SORT)"],
    ["min_hits", "Detecciones para confirmar un track (SORT)", "3"],
    ["max_age / track_buffer", "Frames que sobrevive un track perdido", "30"],
    ["iou / match_thresh", "Umbral de emparejamiento", "0,3 (SORT) · 0,8 (ByteTrack, BoT-SORT)"],
    ["track_high / track_low", "Umbrales de las dos rondas", "0,25 / 0,1"],
    ["gmc_method", "Compensación de cámara", "sparseOptFlow"],
    ["slice / overlap", "Tamaño y solape de los recuadros SAHI", "640 px / 0,2"],
    ["IoU fusión · ρ · largo mín.", "Fusión de cajas y post-proceso (método 4)", "0,55 · 0,6 · 20 frames"],
  ], { x: 0.4, y: 0.9, w: 9.2, colW: [2.4, 3.6, 3.2], rowH: 0.46, size: 13 });
}

// 19. Metodo 4: SAHI + heuristicas (fusion con votos)
{
  const s = content("Método 4: SAHI + heurísticas", "Metodología");
  box(s, 0.4, 0.8, 1.75, 0.55, "YOLO26 (imagen)", { size: 11 });
  box(s, 0.4, 1.45, 1.75, 0.55, "SAHI 640 + YOLO26", { size: 11 });
  box(s, 0.4, 2.1, 1.75, 0.55, "RT-DETRv4 (VisDrone)", { size: 11 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.3, y: 0.74, w: 1.95, h: 1.32, rectRadius: 0.12, fill: { color: "FFFFFF", transparency: 100 }, line: { color: NAVY, width: 1.25, dashType: "dash" } });
  const chain = [["Fusión de cajas\n(votos)", 2.65], ["ByteTrack", 4.55], ["Filtro de escala\n+ largo ≥ 20", 6.45]];
  chain.forEach(([t, x], i) => box(s, x, 1.15, 1.6, 0.85, t, { size: 11.5, fill: i === 0 || i === 2 ? "DCE6F5" : "FFFFFF" }));
  arrow(s, 2.15, 1.075, 2.65, 1.5); arrow(s, 2.15, 2.375, 2.65, 1.65);
  arrow(s, 4.25, 1.575, 4.55, 1.575); arrow(s, 6.15, 1.575, 6.45, 1.575); arrow(s, 8.05, 1.575, 8.35, 1.575);
  T(s, "MOT", { x: 8.4, y: 1.4, w: 1.2, h: 0.35, fontFace: DIAG, fontSize: 13, valign: "middle" });
  caption(s, "Recuadro punteado: YOLO26 y SAHI usan los mismos pesos y cuentan como un solo voto", 2.65, 2.2, 7.0);
  T(s, bullets([
    "SAHI como respaldo: entra solo donde YOLO26 no vio nada; si ambos ven la caja, se queda la de mayor confianza",
    "Como tercer votante empeoraba (MOTA 0,485): el acuerdo YOLO26–SAHI no es consenso entre modelos distintos",
    "Caja fusionada: promedio ponderado por confianza (IoU ≥ 0,55). Una caja vista por un solo voto queda con la mitad de confianza",
  ], { size: 12.5, gap: 6 }), { x: 0.5, y: 2.75, w: 5.3, h: 2.3 });
  label(s, "Caja fusionada", 6.0, 2.75, 3.6);
  eq(s, "fuse_box", 6.0, 3.05, { scale: 0.8 });
  label(s, "Confianza", 6.0, 3.95, 3.6);
  eq(s, "fuse_vote", 6.0, 4.25, { scale: 0.8 });
  cite(s, "Inspirada en Weighted Boxes Fusion [18]; implementación propia (scripts/m4_lib.py)");
  s.addNotes("La fusión del método 5 anterior pasa a ser parte del método 4. La novedad es cómo entra SAHI: comparte el voto con YOLO26 porque son el mismo modelo.");
}

// 20. Heuristica: filtro de escala relativa
{
  const s = content("Heurística: filtro de escala relativa", "Metodología");
  const fa = RS.fp_analysis.seqs, ks = ["M1101", "M1007", "M0701", "M0601"];
  s.addChart(pres.charts.BAR, [
    { name: "Vehículos del GT", labels: ks, values: ks.map((q) => fa[q].gt_side) },
    { name: "Falsos positivos persistentes", labels: ks, values: ks.map((q) => fa[q].fp_side) },
  ], { x: 0.3, y: 0.75, w: 4.9, h: 3.4, barDir: "col", barGrouping: "clustered", chartColors: [NAVY, HL],
    showLegend: true, legendPos: "b", legendFontFace: SANS, legendFontSize: 10, showTitle: true, title: "Lado típico √(w·h), px (mediana)",
    titleFontFace: SANS, titleFontSize: 12, titleColor: INK, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelFormatCode: "0",
    catAxisLabelFontFace: SANS, valAxisLabelFontFace: SANS, catAxisLabelFontSize: 10, valAxisLabelFontSize: 9,
    catAxisLabelColor: "4B5563", valAxisLabelColor: "4B5563", valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" } });
  T(s, bullets([
    `Los falsos positivos que hunden M1101 y M1007 miden la mitad que los vehículos reales (${fm(fa.M1101.fp_side, 0)} frente a ${fm(fa.M1101.gt_side, 0)} px en M1101)`,
    `En general no son autos estacionados: con la cámara compensada (GMC), solo el ${fm(RS.fp_analysis.static_pct_total, 0)} % de los frames con falsos positivos (20 secuencias) está quieto`,
    "Un dron vuela a altura casi constante: los vehículos de una escena tienen escala parecida",
  ], { size: 12.5, gap: 6 }), { x: 5.45, y: 0.85, w: 4.2, h: 2.6 });
  eq(s, "relsize", 5.45, 3.55, { scale: 0.7 });
  T(s, "La mediana sale de los propios tracks (≥ 10 frames): no usa el GT. ρ se eligió por validación cruzada.", { x: 5.45, y: 3.95, w: 4.2, h: 0.6, fontSize: 11.5, color: MUTED });
  cite(s, "Tracks antes del filtro (fusión + ByteTrack). Un track es falso positivo si menos de la mitad de sus cajas empareja con el GT (scripts/m4_fp_analysis.py)");
  s.addNotes("Este análisis cambió la hipótesis inicial: pensábamos en autos estacionados, pero los FP no son estáticos sino demasiado pequeños para la escena. El filtro estático no mejoró en validación cruzada; el de escala sí (+0,015 de MOTA).");
}

// ---------------------------------------------------------------------------------------------
// EVALUACION
pres.addSection({ title: "Evaluación" });
{
  const s = content("Protocolo de evaluación", "Evaluación");
  T(s, bullets(["Biblioteca motmetrics", "Emparejamiento por IoU ≥ 0,5, como el toolkit oficial", "Se ignoran las regiones marcadas en el GT", "Promedio sobre las 20 secuencias de test"], { size: 15, gap: 10 }), { x: 0.5, y: 1.0, w: 4.0, h: 3.6 });
  const rows = [["MOTA", "mota", 1.05], ["IDF1", "idf1", 1.95], ["FPS", "fps", 3.7]];
  rows.forEach(([n, e, y]) => { label(s, n, 4.8, y + 0.1, 1.0); eq(s, e, 5.9, y, { scale: 1.0 }); });
  label(s, "IDS", 4.8, 2.85, 1.0);
  T(s, "Número de veces que un vehículo cambia de identificador", { x: 5.9, y: 2.85, w: 3.7, h: 0.6, fontSize: 13 });
  T(s, "MOTA penaliza errores de detección y de identidad; IDF1 mide cuánto se conserva el mismo ID a lo largo del video.", { x: 4.8, y: 4.4, w: 4.8, h: 0.7, fontSize: 12, color: MUTED });
  cite(s, "Bernardin y Stiefelhagen [15]; Ristani et al. [16]");
}

// ---------------------------------------------------------------------------------------------
// RESULTADOS (numeros reales: presentacion/results.json, generado por collect_results.py)
pres.addSection({ title: "Resultados" });
const MN = { m1: "1  YOLOv11 + SORT", m2: "2  YOLO26 + ByteTrack", m3: "3  RT-DETRv4 + BoT-SORT", m4: "SAHI + YOLO26 (versión inicial)", m5: "Fusión YOLO26 + RT-DETR (sin SAHI)", m4n: "4  SAHI + fusión + filtro de escala" };
const W4 = ["m1", "m2", "m3", "m4n"];
const inte = (x) => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const hl = (txt, on) => (on ? { text: txt, fill: HL2, bold: true } : { text: txt });
const argbest = (keys, f, dir = 1) => keys.reduce((a, b) => (dir * (f(b) - f(a)) > 0 ? b : a));
{
  const s = content("Comparación de los cuatro métodos", "Resultados");
  const me = RS.methods, cv = RS.cv;
  const bM = argbest(W4, (k) => me[k].mota), bI = argbest(W4, (k) => me[k].idf1), bS = argbest(W4, (k) => me[k].ids, -1);
  const bF = argbest(W4, (k) => me[k].fps), bC = argbest(W4, (k) => cv[k].tuned);
  latexTable(s, ["Método", "MOTA", "IDF1", "IDS", "FPS (GPU)", "MOTA (val. cruzada)"],
    W4.map((k) => [MN[k], hl(fm(me[k].mota, 3), k === bM), hl(fm(me[k].idf1, 3), k === bI), hl(inte(me[k].ids), k === bS),
      hl(fm(me[k].fps, 1), me[k].fps >= me[bF].fps - 0.1), hl(fm(cv[k].tuned, 3) + (k === "m4n" ? " †" : ""), k === bC)]),
    { x: 0.4, y: 1.0, w: 9.2, colW: [3.2, 1.0, 1.0, 0.9, 1.3, 1.8], rowH: 0.55, size: 14 });
  T(s, bullets([
    "Promedio sobre las 20 secuencias de test, con hiperparámetros ajustados por validación cruzada. Resaltado: mejor de cada columna",
    "FPS en GPU sobre las 5 primeras secuencias; el método 4 se midió de punta a punta (3 detectores + fusión + tracking)",
    "† Validación cruzada sobre secuencias completas (ByteTrack fijo + grilla de post-proceso); los métodos 1–3 usan ventanas de 300 frames",
  ], { size: 12, gap: 5 }), { x: 0.5, y: 4.0, w: 9.1, h: 1.2 });
  s.addNotes("MOTA promedio = promedio simple de las 20 secuencias (lo que pide el PDF). La última columna es la estimación sin sesgo: MOTA en las secuencias que no se usaron para ajustar. La corrida en GPU del método 4 reproduce la de las detecciones cacheadas (MOTA global 0,548 en ambas).");
}
{
  const s = content("Resultados por secuencia", "Resultados");
  const labels = RS.seqs, series = [["m1", LIGHTBLUE], ["m2", NAVY], ["m3", HL], ["m4n", GREEN]];
  s.addChart(pres.charts.BAR, series.map(([k]) => ({ name: MN[k].replace(/^\d\s+/, ""), labels, values: labels.map((q) => RS.per_seq[k][q]) })), {
    x: 0.3, y: 0.75, w: 9.4, h: 4.4, barDir: "col", barGrouping: "clustered", chartColors: series.map(([, c]) => c),
    showLegend: true, legendPos: "b", legendFontFace: SANS, legendFontSize: 10, showTitle: false,
    catAxisLabelFontFace: SANS, valAxisLabelFontFace: SANS, catAxisLabelFontSize: 9, valAxisLabelFontSize: 9, catAxisLabelRotate: -90,
    catAxisLabelColor: "4B5563", valAxisLabelColor: "4B5563", valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
    valAxisMinVal: -0.6, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0", barGapWidthPct: 30,
  });
  s.addNotes("MOTA por secuencia de los 4 métodos ajustados. Las secuencias M0403, M0205, M0801 y M1009 salen bien; M1101, M1007, M0601 y M0602 salen mal (noche con destellos y vehículos de ~12 px). El método 4 es el único que no cae por debajo de −0,2 en M1101.");
}
{
  const s = content("Efecto del ajuste: validación cruzada", "Resultados");
  const cfgTxt = (k) => {
    const c = RS.cfg[k];
    if (k === "m1") return `conf ${fm(c.conf, 2)} · min_hits ${c.min_hits} · iou ${fm(c.iou, 1)}`;
    if (k === "m4n") return `alta ${fm(c.tracker.high, 2)} · baja ${fm(c.tracker.low, 2)} · ρ ${fm(c.post.rel_size, 1)} · largo ≥ ${c.post.min_len}`;
    return `alta ${fm(c.high, 2)} · baja ${fm(c.low, 2)} · buffer ${c.buffer} · match ${fm(c.match, 1)}` + (k === "m3" && c.gmc === false ? " · sin GMC" : "");
  };
  latexTable(s, ["Método", "Por defecto", "Ajustado", "Parámetros finales"],
    ["m1", "m2", "m3", "m4", "m5", "m4n"].map((k) => [MN[k], RS.cv[k].default == null ? pending() : fm(RS.cv[k].default, 3), hl(fm(RS.cv[k].tuned, 3) + (k === "m4n" ? " †" : ""), true), { text: cfgTxt(k), align: "left" }]),
    { x: 0.4, y: 0.85, w: 9.2, colW: [3.0, 1.2, 1.1, 3.9], rowH: 0.45, size: 12 });
  T(s, bullets([
    "MOTA en secuencias no vistas al ajustar: 4 pliegues de 5 secuencias (se ajusta con 15 y se mide en 5). † secuencias completas",
    "RT-DETR y SAHI cambian mucho: por defecto dejan pasar demasiados falsos positivos",
    "Ajustar muchos parámetros a la vez sobreajusta: en el método 4 se fijó ByteTrack y solo se ajustó ρ y el largo mínimo",
  ], { size: 12, gap: 4 }), { x: 0.5, y: 4.15, w: 9.1, h: 1.1 });
  s.addNotes("Un primer ajuste sobre solo 5 secuencias dio MOTA 0,70 ahí y 0,22 en las 15 restantes: sobreajuste. La validación cruzada evita ese espejismo.");
}
{
  const s = content("Latencia: con y sin NMS en YOLO26", "Resultados");
  const a = RS.nms.sin_nms, b = RS.nms.con_nms;
  latexTable(s, ["Configuración", "Detección (ms/frame)", "Asociación (ms/frame)", "FPS", "MOTA", "IDS"], [
    ["YOLO26 + ByteTrack, con NMS", fm(b.det_ms, 1), fm(b.trk_ms, 1), fm(b.fps, 1), fm(b.mota, 3), inte(b.ids)],
    ["YOLO26 + ByteTrack, sin NMS", hl(fm(a.det_ms, 1), true), fm(a.trk_ms, 1), hl(fm(a.fps, 1), true), hl(fm(a.mota, 3), true), hl(inte(a.ids), true)],
  ], { x: 0.4, y: 1.2, w: 9.2, colW: [3.2, 1.5, 1.5, 0.8, 0.9, 0.8], rowH: 0.75, size: 13 });
  T(s, bullets([
    `Quitar el NMS reduce la latencia de detección en ${fm(100 * (1 - a.det_ms / b.det_ms), 0)} % (${fm(b.det_ms, 1)} → ${fm(a.det_ms, 1)} ms) sin perder precisión`,
    "Latencia y FPS: GPU, 5 secuencias. MOTA e IDS: 20 secuencias, valores por defecto",
  ], { size: 14, gap: 8 }), { x: 0.5, y: 3.9, w: 9.1, h: 1.3 });
}
{
  const s = content("Impacto de SAHI en vehículos pequeños", "Resultados");
  const r = RS.recall, c = RS.recall_cut, lab = [`Pequeño (< ${fm(c[0], 0)} px)`, `Mediano (${fm(c[0], 0)} a ${fm(c[1], 0)} px)`, `Grande (> ${fm(c[1], 0)} px)`];
  const v = (k, t, i) => r[k][t][i];
  latexTable(s, ["Tamaño (√área)", "YOLO26 (por defecto)", "+ SAHI (por defecto)", "YOLO26 (ajustado)", "+ SAHI (ajustado)"],
    [0, 1, 2].map((i) => [lab[i], fm(v("m2", "default", i), 3), hl(fm(v("m4", "default", i), 3), true), fm(v("m2", "tuned", i), 3), hl(fm(v("m4", "tuned", i), 3), true)]),
    { x: 0.4, y: 1.0, w: 9.2, colW: [2.4, 1.7, 1.7, 1.7, 1.7], rowH: 0.62, size: 13 });
  const d0 = (v("m4", "default", 0) - v("m2", "default", 0)) * 100, d1 = (v("m4", "tuned", 0) - v("m2", "tuned", 0)) * 100;
  T(s, bullets([
    `SAHI sube el recall de los pequeños en ${fm(d0, 1)} puntos por defecto y ${fm(d1, 1)} puntos ajustado`,
    "Costo: cuatro veces menos FPS (9,1 frente a 39,0) y más falsos positivos",
    "Ajustar los umbrales para MOTA baja el recall de los pequeños: se descartan detecciones de baja confianza",
  ], { size: 13, gap: 6 }), { x: 0.5, y: 3.95, w: 9.1, h: 1.1 });
  cite(s, "Recall de detección (IoU ≥ 0,5, sin mirar IDs) sobre las 20 secuencias; cortes = terciles de tamaño del GT. Akyon et al. [7]");
}
{
  const s = content("SAHI con slices más finos", "Resultados");
  const sc = RS.sahi_compare, names = Object.keys(sc);
  const bM = argbest(names, (n) => sc[n].MOTA);
  latexTable(s, ["Configuración", "MOTA", "IDF1", "FN", "FP", "Recall pequeño"],
    names.map((n) => [n, hl(fm(sc[n].MOTA, 3), n === bM), fm(sc[n].IDF1, 3), inte(sc[n].FN), inte(sc[n].FP), fm(sc[n].recall[0], 3)]),
    { x: 0.4, y: 1.0, w: 9.2, colW: [2.8, 1.2, 1.2, 1.3, 1.3, 1.4], rowH: 0.6, size: 13.5 });
  T(s, bullets([
    "Cinco secuencias con más vehículos pequeños (M0602, M0601, M1301, M1004, M1302), primeros 300 frames",
    "Slices de 320 px (ampliación 3,2×): mejora modesta en MOTA y casi nada en el recall de pequeños",
    "Hipótesis: con estos pesos, vehículos de ≈ 12 px quedan por debajo de lo que el detector resuelve",
  ], { size: 13, gap: 7 }), { x: 0.5, y: 3.85, w: 9.1, h: 1.3 });
}
{
  const s = content("Método 4: ablación paso a paso", "Resultados");
  const ab = RS.ablation_m4.filter((r) => r.paso !== "control" || r.cfg === "paso 6 sin SAHI");
  const fin = ab.findIndex((r) => r.paso === "6");
  latexTable(s, ["Paso", "Configuración", "MOTA (val. cruzada)", "IDF1", "IDS"],
    ab.map((r, i) => [r.paso === "control" ? "ctrl." : r.paso, { text: r.cfg.replace(" = MÉTODO 4 FINAL", ""), align: "left", bold: i === fin },
      hl(fm(r.cv_mota, 3), i === fin), fm(r.cv_idf1, 3), inte(r.ids)]),
    { x: 0.4, y: 0.75, w: 9.2, colW: [0.7, 4.9, 1.5, 1.0, 1.1], rowH: 0.34, size: 12 });
  const rc = RS.recall, d = (rc.m4n.tuned[0] - rc.m4n_sin_sahi.tuned[0]) * 100;
  const wins = RS.seqs.filter((q) => RS.per_seq.m4n[q] > RS.per_seq.m5[q] + 0.001).length;
  T(s, bullets([
    `La fusión aporta lo más grande (+0,08 sobre SAHI solo); el filtro de escala suma +0,015 y el largo mínimo baja los IDS de ${RS.ablation_m4[4].ids} a ${RS.ablation_m4[5].ids}`,
    `SAHI no cambia el MOTA promedio (control sin SAHI: dentro del ruido), pero sube el recall de pequeños ${fm(d, 1)} puntos y el MOTA global (${fm(RS.methods.m4n.mota_g, 3)})`,
    `Frente a la fusión sin SAHI ni filtros, el método 4 gana en ${wins} de 20 secuencias. Interpolación, filtro por confianza y filtro estático: sin ganancia`,
  ], { size: 11.5, gap: 4 }), { x: 0.5, y: 3.7, w: 9.1, h: 1.3 });
  cite(s, "Mismo protocolo en todas las filas: validación cruzada en 4 pliegues sobre secuencias completas (results/m4_exp/ablation_m4.csv)");
}
{
  const s = content("¿Dónde funciona mejor el seguimiento?", "Resultados");
  const g = RS.groups, order = ["vista cenital", "baja altura", "día", "media altura", "vista lateral", "niebla", "vista frontal", "noche", "gran altura"];
  const mean4 = (x) => (x.m1 + x.m2 + x.m3 + x.m4n) / 4;
  const rows = order.filter((n) => g[n]).sort((a, b) => mean4(g[b]) - mean4(g[a]));
  latexTable(s, ["Escena", "n", "Método 1", "Método 2", "Método 3", "Método 4"],
    rows.map((n) => { const b = argbest(["m1", "m2", "m3", "m4n"], (k) => g[n][k]); return [n, String(g[n].n), ...["m1", "m2", "m3", "m4n"].map((k) => hl(fm(g[n][k], 2), k === b))]; }),
    { x: 0.4, y: 0.85, w: 9.2, colW: [2.6, 0.6, 1.5, 1.5, 1.5, 1.5], rowH: 0.34, size: 12.5 });
  T(s, "MOTA promedio por tipo de escena. Los vehículos grandes y la vista cenital favorecen a todos; la noche, la vista frontal y los vehículos de ≈ 12 px los perjudican. La densidad de tráfico casi no influye.", { x: 0.5, y: 4.5, w: 9.1, h: 0.7, fontSize: 12, color: "374151" });
  s.addNotes("Con 20 secuencias, gran altura (2) y vista cenital (4) son medias frágiles. La correlación entre el MOTA y el tamaño mediano del vehículo es +0,48; con la densidad, casi cero.");
}
{
  const s = content("Demo visual: caso exitoso", "Resultados");
  s.addImage({ path: FIG("demo_exito_M0403_cv.jpg"), x: 0.4, y: 0.85, w: 9.2, h: 3.04, altText: "M0403 con IDs y estela" });
  caption(s, "M0403, frames 100, 130 y 160. YOLO26 + ByteTrack ajustado. Cada color es un ID; la línea es su estela de 25 frames.", 0.4, 3.95, 9.2);
  T(s, bullets(["Los IDs se mantienen en una escena densa (≈ 60 vehículos): los dos buses y los autos del carril central", "MOTA 0,90 en esta secuencia con los cuatro métodos"], { size: 14, gap: 6 }), { x: 0.5, y: 4.3, w: 9.1, h: 0.85 });
  s.addNotes("python -I scripts/make_demos.py --method method2_yolo26_bytetrack_cv --seq M0403 --frames 100 130 160 --crop 250 40 750 540 --trail 25");
}
{
  const s = content("Demo visual: caso de fallo (M0209)", "Resultados");
  s.addImage({ path: FIG("compare_M0209_cv.jpg"), x: 1.05, y: 0.72, w: 7.9, h: 3.92, altText: "Comparación en M0209" });
  T(s, "M0209 (peaje, vista oblicua). YOLO26 pierde casi todos los vehículos (rojo); RT-DETR y la fusión los detectan, a costa de falsos positivos en las casetas (naranja).", { x: 0.4, y: 4.7, w: 9.2, h: 0.6, fontSize: 13, color: "374151" });
  s.addNotes("YOLO perdía el 78 % de esta secuencia con vehículos de tamaño normal (mediana 36 px): no es un problema de resolución sino de dominio, vista oblicua frente a cenital.");
}
{
  const s = content("Demo visual: caso de fallo (noche, M1101)", "Resultados");
  s.addImage({ path: FIG("compare_M1101_cv.jpg"), x: 1.05, y: 0.72, w: 7.9, h: 3.92, altText: "Comparación en M1101" });
  T(s, `M1101 (noche). Los vehículos de la calzada se detectan bien, pero aparecen cajas pequeñas que el GT no anota (luces, objetos lejanos). MOTA: −0,13 / −0,51 / −0,42; con el filtro de escala, el método 4 llega a ${fm(RS.per_seq.m4n.M1101, 2)}.`, { x: 0.4, y: 4.7, w: 9.2, h: 0.6, fontSize: 13, color: "374151" });
}

// ---------------------------------------------------------------------------------------------
// CIERRE
pres.addSection({ title: "Cierre" });
{
  const s = content("Conclusiones", "Cierre");
  const me = RS.methods;
  T(s, bullets([
    `Mejor resultado: método 4, SAHI + fusión + filtro de escala (MOTA ${fm(me.m4n.mota, 3)}, IDF1 ${fm(me.m4n.idf1, 3)}, ${inte(me.m4n.ids)} cambios de ID), a ${fm(me.m4n.fps, 0)} FPS`,
    `Mejor equilibrio velocidad–precisión: YOLO26 + ByteTrack (MOTA ${fm(me.m2.mota, 3)}, ${inte(me.m2.ids)} cambios de ID, ${fm(me.m2.fps, 0)} FPS)`,
    `Ajustar pesó más que la arquitectura: con validación cruzada, RT-DETR pasó de MOTA ${fm(RS.cv.m3.default, 2)} a ${fm(RS.cv.m3.tuned, 2)}; ByteTrack cambia mucho menos de ID que SORT (${inte(me.m2.ids)} frente a ${inte(me.m1.ids)})`,
    "SAHI solo no basta: rinde como respaldo de otro detector, y su aporte se ve en los vehículos pequeños más que en el MOTA promedio",
    "Analizar los errores guió la heurística: los falsos positivos eran cajas demasiado pequeñas para la escena, no autos estacionados",
    "Contexto: trackers recientes reportan MOTA 0,47–0,55 en UAVDT [19, 20], con otro protocolo y pesos propios, así que no es comparable directamente",
    "Limitaciones: split de entrenamiento de los pesos YOLO sin verificar, sin ReID, y mucha variación entre escenas",
  ], { size: 14, gap: 7 }), { x: 0.6, y: 0.85, w: 8.8, h: 4.45 });
}
{
  const s = pres.addSlide({ masterName: "CLOSING", sectionTitle: "Cierre" });
  s.addText("Gracias", { placeholder: "title" });
}
{
  const refs = [
    "[1] D. Du et al., “The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking,” in ECCV, 2018.",
    "[2] R. E. Kalman, “A new approach to linear filtering and prediction problems,” J. Basic Eng., vol. 82, no. 1, 1960.",
    "[3] H. W. Kuhn, “The Hungarian method for the assignment problem,” Naval Research Logistics Quarterly, vol. 2, 1955.",
    "[4] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, “Simple online and realtime tracking,” in ICIP, 2016.",
    "[5] Y. Zhang et al., “ByteTrack: Multi-object tracking by associating every detection box,” in ECCV, 2022.",
    "[6] N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, “BoT-SORT: Robust associations multi-pedestrian tracking,” arXiv:2206.14651, 2022.",
    "[7] F. C. Akyon, S. O. Altinuc, and A. Temizel, “Slicing aided hyper inference and fine-tuning for small object detection,” in ICIP, 2022.",
    "[8] N. Carion et al., “End-to-end object detection with transformers,” in ECCV, 2020.",
    "[9] A. Vaswani et al., “Attention is all you need,” in NeurIPS, 2017.",
    "[10] W. Lv et al., “RT-DETR: DETRs beat YOLOs on real-time object detection,” in CVPR, 2024.",
    "[11] Y. Peng et al., “D-FINE: Redefine regression task in DETRs as fine-grained distribution refinement,” arXiv:2410.13842, 2024.",
    "[12] Z. Liao et al., “RT-DETRv4: Painlessly furthering real-time object detection with vision foundation models,” arXiv:2510.25257, 2025.",
    "[13] G. Jocher and J. Qiu, “Ultralytics YOLO11,” 2024; Ultralytics, “YOLO26,” docs.ultralytics.com/models/yolo26, 2025.",
    "[14] P. Zhu et al., “Detection and tracking meet drones challenge,” IEEE TPAMI, 2021.",
    "[15] K. Bernardin and R. Stiefelhagen, “Evaluating multiple object tracking performance: The CLEAR MOT metrics,” EURASIP J. Image Video Process., 2008.",
    "[16] E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi, “Performance measures and a data set for multi-target, multi-camera tracking,” in ECCV Workshops, 2016.",
    "[17] A. Milan, L. Leal-Taixé, I. Reid, S. Roth, and K. Schindler, “MOT16: A benchmark for multi-object tracking,” arXiv:1603.00831, 2016.",
    "[18] R. Solovyev, W. Wang, and T. Gabruseva, “Weighted boxes fusion: Ensembling boxes from different object detection models,” Image and Vision Computing, 2021.",
    "[19] I. Song and J. Lee, “SFTrack: A robust scale and motion adaptive algorithm for tracking small and fast moving objects,” arXiv:2410.20079, 2024.",
    "[20] J. Ma et al., “Tracking the unstable: Appearance-guided motion modeling for robust multi-object tracking in UAV-captured videos,” arXiv:2508.01730, 2025.",
  ];
  [["Referencias (1/2)", refs.slice(0, 10)], ["Referencias (2/2)", refs.slice(10)]].forEach(([title, list]) => {
    const s = content(title, "Cierre");
    T(s, list.map((t, i) => ({ text: t, options: { breakLine: i < list.length - 1, paraSpaceAfter: 6, fontSize: 11, fontFace: SANS, color: "374151" } })), { x: 0.5, y: 0.85, w: 9, h: 4.45 });
  });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("ok", OUT);
})();
