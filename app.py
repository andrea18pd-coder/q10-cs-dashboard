# =========================================================
# Q10 – Gestión Customer Support (Streamlit)
# FIXES IMPORTANTES:
# - Tickets: usa EXACTAMENTE:
#   Date entered "Nuevo (Pipeline CSS)"   -> Tickets ENTRADOS
#   Date entered "Cerrado (Pipeline CSS)" -> Tickets CERRADOS
# - Debug: muestra columna elegida + parseo + min/max + no-NA
# - CSAT: Customer Satisfaction (score) y Date (fecha)
# - SLA (horas hábiles): entre ENTRADO -> CERRADO (L-V 08:00–17:00)
#   Excluye: 24/12/2025, 25/12/2025, 31/12/2025, 01/01/2026
# - Módulo: "Módulo(s) afectados o dónde se presenta la novedad"
# - TIER: "Etiquetas de tickets"
# - Texto: (Asunto + cuerpo del ticket limpio + Módulo)
#   * Si hay sklearn: TF-IDF + KMeans + TITULADO + Resumen ejecutivo (tipo Breeze)
#   * Si NO hay sklearn: n-grams
#
# Requisitos:
#   pip install streamlit pandas numpy plotly openpyxl
#   (Opcional ML) pip install scikit-learn
#
# Ejecutar:
#   streamlit run app.py
# =========================================================

import re
import math
from datetime import time
from collections import Counter

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# =========================================================
# 0) RUTAS (AJUSTA AQUÍ)
# =========================================================
st.sidebar.subheader("Archivos de entrada")

up_tickets  = st.sidebar.file_uploader("Tickets (HubSpot export .xlsx)", type=["xlsx"])
up_csat     = st.sidebar.file_uploader("CSAT (HubSpot export .xlsx)", type=["xlsx"])
up_llamadas = st.sidebar.file_uploader("Gestión llamadas (Excel .xlsx)", type=["xlsx"])
up_wapp     = st.sidebar.file_uploader("Gestión WhatsApp (Excel .xlsx)", type=["xlsx"])

DEFAULT_START = pd.Timestamp("2025-12-22")
DEFAULT_END   = pd.Timestamp("2026-01-12 23:59:59")

# =========================================================
# 1) CALENDARIO LABORAL PARA SLA (HORAS HÁBILES)
# =========================================================
WORK_START = time(8, 0)
WORK_END   = time(17, 0)

HOLIDAYS = {
    pd.Timestamp("2025-12-24").date(),
    pd.Timestamp("2025-12-25").date(),
    pd.Timestamp("2025-12-31").date(),
    pd.Timestamp("2026-01-01").date(),
}

def business_minutes_between(start, end):
    """Minutos hábiles entre dos timestamps (L-V 08:00–17:00, excluye festivos)."""
    if pd.isna(start) or pd.isna(end) or end <= start:
        return np.nan

    total = 0.0
    day = start.normalize()

    while day <= end.normalize():
        d = day.date()
        if day.weekday() >= 5 or d in HOLIDAYS:
            day += pd.Timedelta(days=1)
            continue

        day_start = pd.Timestamp.combine(d, WORK_START)
        day_end   = pd.Timestamp.combine(d, WORK_END)

        interval_start = max(start, day_start)
        interval_end   = min(end, day_end)

        if interval_end > interval_start:
            total += (interval_end - interval_start).total_seconds() / 60

        day += pd.Timedelta(days=1)

    return total

def minutes_to_hhmm(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    x = int(round(float(x)))
    hh = x // 60
    mm = x % 60
    return f"{hh:02d}:{mm:02d}"

# =========================================================
# 2) NORMALIZACIÓN DE RESPONSABLES (SOLO 5)
# =========================================================
RESP_MAP = {
    "ana": "Ana",
    "ana maria": "Ana",
    "ana maría": "Ana",
    "ana maria ramirez": "Ana",
    "ana maría ramirez": "Ana",
    "ana maria ramirez cortes": "Ana",
    "ana maría ramirez cortes": "Ana",
    "ana maría ramírez cortés": "Ana",

    "helen": "Helen",
    "helen acevedo": "Helen",
    "helen dayana": "Helen",
    "helen dayana acevedo": "Helen",
    "helen dayana acevedo gonzalez": "Helen",
    "helen dayana acevedo gonzález": "Helen",
    "helen acevedo gonzalez": "Helen",
    "helen acevedo gonzález": "Helen",

    "valentina": "Valentina",
    "valentina taborda": "Valentina",
    "valentina taborda zapata": "Valentina",

    "edwyn": "Edwyn",
    "edwyn henao": "Edwyn",
    "edwyn henao toro": "Edwyn",

    "robinson": "Robinson",
    "robinson munoz": "Robinson",
    "robinson muñoz": "Robinson",
}

def _strip_accents(s: str) -> str:
    return (s.replace("á","a").replace("é","e").replace("í","i")
             .replace("ó","o").replace("ú","u").replace("ñ","n"))

def normalize_responsable(name):
    if pd.isna(name):
        return "Sin responsable"
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = _strip_accents(s)

    if s in RESP_MAP:
        return RESP_MAP[s]

    for key, val in RESP_MAP.items():
        if _strip_accents(key) in s:
            return val

    return "Otro"

# =========================================================
# 3) HELPERS ROBUSTOS
# =========================================================
def normalize_colname(c):
    s = str(c).replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def canon(s: str) -> str:
    """Canoniza nombre de columna: normaliza comillas, lower, sin tildes."""
    s = normalize_colname(s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = _strip_accents(s.lower())
    return s

def find_col_exact(df, exact_name):
    target = canon(exact_name)
    for c in df.columns:
        if canon(c) == target:
            return c
    return None

def find_col_regex(df, pattern):
    rx = re.compile(pattern)
    for c in df.columns:
        if rx.search(canon(c)):
            return c
    return None

def read_best_sheet(path, must_have_any=("fecha",), nice_to_have_any=("tipo","css","usuario","responsable")):
    """Elige la hoja con mayor probabilidad de ser log (evita caer en 'Entidades')."""
    sheets = pd.read_excel(path, sheet_name=None)
    best_name, best_df, best_score = None, None, -1

    for name, df in sheets.items():
        df = df.copy()
        df.columns = [normalize_colname(c) for c in df.columns]
        cols_low = [c.lower() for c in df.columns]

        if not any(any(k in c for c in cols_low) for k in must_have_any):
            continue

        score = 0
        for k in must_have_any:
            score += sum(k in c for c in cols_low) * 10
        for k in nice_to_have_any:
            score += sum(k in c for c in cols_low) * 2
        score += min(len(df) / 500, 15)

        if score > best_score:
            best_score = score
            best_name, best_df = name, df

    if best_df is None:
        first_name = list(sheets.keys())[0]
        df = sheets[first_name].copy()
        df.columns = [normalize_colname(c) for c in df.columns]
        return first_name, df

    return best_name, best_df

def find_col_contains(df, keywords):
    cols = list(df.columns)
    cols_low = [c.lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(cols_low):
            if kw in c:
                return cols[i]
    return None

def parse_excel_datetime(series):
    """
    Parseo robusto:
    - datetime -> ok
    - serial Excel numérico -> origin 1899-12-30
    - texto -> dayfirst True/False (elige el que más parsea)
    """
    s = series.copy()

    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce")

    if np.issubdtype(s.dtype, np.number):
        return pd.to_datetime(s, errors="coerce", unit="D", origin="1899-12-30")

    s1 = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    s2 = pd.to_datetime(s, errors="coerce", dayfirst=False, utc=False)
    return s2 if s2.notna().sum() > s1.notna().sum() else s1

def td_to_minutes(series):
    td = pd.to_timedelta(series, errors="coerce")
    return td.dt.total_seconds() / 60

def iso_week(dt_series):
    iso = dt_series.dt.isocalendar()
    return iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)

def kpi_card(label, value):
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:16px;background:#f3f3f3;border:1px solid #e6e6e6;">
            <div style="font-size:14px;color:#666;">{label}</div>
            <div style="font-size:34px;font-weight:700;color:#111;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def fmt_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return "-"

def fmt_float(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"

# =========================================================
# 4) TEXTO + LIMPIEZA DE CUERPO (quita saludos/despedidas/firma/citas)
#    + FILTRO ANTI-IMÁGENES/ADJUNTOS + FILTRO ANTI-IDENTIFICADORES (CE/BD/nombres)
# =========================================================
SPANISH_STOPWORDS = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no","una","su","al","lo",
    "como","mas","pero","sus","le","ya","o","este","si","porque","esta","entre","cuando","muy","sin","sobre",
    "tambien","me","hasta","hay","donde","quien","desde","todo","nos","durante","todos","uno","les","ni","contra",
    "otros","ese","eso","ante","ellos","e","esto","mi","mis","tu","tus","favor","buenos","buenas","hola","gracias",
    "dia","cordial","saludo","adjunto","adjuntar","porfa","porfavor","ayuda","apoyo",
    "estudiante","docente","usuario","plataforma","q10"
}

BOILERPLATE_PHRASES = {
    "hola", "buen dia", "buenos dias", "buenas tardes", "buenas noches",
    "cordial saludo", "cordialmente", "atentamente", "saludos", "saludos cordiales",
    "espero se encuentren bien", "espero estes bien", "espero se encuentre bien",
    "gracias", "muchas gracias", "mil gracias", "de antemano gracias",
    "quedo atenta", "quedo atento", "quedamos atentos", "quedamos atentas",
    "quedo pendiente", "quedo pendiente de tu respuesta",
    "quedo atento a tus comentarios", "quedo atenta a tus comentarios",
    "agradezco su apoyo", "agradezco tu apoyo",
}

SPANISH_STOPWORDS = SPANISH_STOPWORDS.union({
    "cordialmente","atentamente","saludos","gracias",
    "quedo","atenta","atento","pendiente","comentarios",
    "buen","dia","dias","tardes","noches",
    "adjunto","adjuntamos","anexo","anexamos",
    "cualquier","cosa","atencion"
})

def clean_text(s):
    s = str(s).lower()
    s = _strip_accents(s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

ATTACHMENT_PATTERNS = [
    r"\bpng\b", r"\bjpg\b", r"\bjpeg\b", r"\bgif\b", r"\bwebp\b",
    r"\bpdf\b", r"\bxlsx\b", r"\bdocx\b",
    r"adjunt", r"captura", r"pantallaz", r"imagen", r"foto", r"evidenc",
    r"http", r"www"
]

# Identificadores que NO queremos como señal principal del “problema”
IDENTIFIER_PATTERNS = [
    r"\bce\b", r"\bcodigo\s*entidad\b", r"\bentidad\b", r"\bbase\s*de\s*datos\b", r"\bbd\b", r"\budbz\w+\b",
    r"\btelefono\b", r"\bcorreo\b", r"\bemail\b", r"\bext\b"
]

def is_attachment_noise(text: str) -> bool:
    """Filtra descripciones que son básicamente adjuntos/capturas/links o muy cortas."""
    t = clean_text(text)
    if len(t) < 20:
        return True

    toks = [w for w in t.split() if w not in SPANISH_STOPWORDS and len(w) >= 3]
    if len(toks) < 5:
        return True

    hits = sum(1 for p in ATTACHMENT_PATTERNS if re.search(p, t))
    if hits >= 3 and len(toks) < 10:
        return True

    return False

def strip_email_boilerplate(raw: str) -> str:
    """
    Se queda con el "cuerpo" del problema:
    - elimina firmas/bloques de contacto
    - elimina saludos/despedidas
    - elimina texto citado/reenvíos típicos
    - elimina líneas cortas de cortesía
    """
    if raw is None:
        return ""

    text = str(raw).replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]

    cleaned = []
    for ln in lines:
        low = _strip_accents(ln.lower())

        # corta cuando empieza un hilo citado/reenvío
        if re.search(r"^(de:|from:|para:|to:|cc:|asunto:|subject:|enviado:|sent:|fecha:|date:)\b", low):
            break
        if "-----original message-----" in low or "mensaje original" in low:
            break
        if low.startswith(">"):
            break

        # elimina urls / adjuntos
        if re.search(r"http\S+|www\.\S+", low):
            continue
        if re.search(r"\b(png|jpg|jpeg|gif|webp|pdf|xlsx|docx)\b", low):
            continue

        # elimina firmas/contacto típicos si son líneas cortas
        if re.search(r"\b(ext|telefono|tel|cel|correo|email|mail|direccion|address|website)\b", low):
            if len(low) <= 55:
                continue

        # elimina cortesía (si la línea es básicamente eso)
        if len(low) <= 60 and any(p in low for p in BOILERPLATE_PHRASES):
            continue

        cleaned.append(ln)

    return " ".join(cleaned).strip()

def strip_identifiers(text: str) -> str:
    """Remueve señales de CE/BD/udbz y similares para evitar tópicos “institución/código”."""
    if text is None:
        return ""
    t = str(text)

    # elimina tokens típicos de BD
    t = re.sub(r"\budbz[a-z0-9_]+\b", " ", t, flags=re.IGNORECASE)
    # elimina números largos (códigos, cédulas, etc.)
    t = re.sub(r"\b\d{6,}\b", " ", t)
    # elimina patrones tipo CE: 123 / BD: xxx
    t = re.sub(r"\b(ce|bd|codigo\s*entidad|c[oó]digo\s*de\s*la\s*entidad|base\s*de\s*datos)\s*[:=-]\s*[\w\-_.]+\b", " ", t, flags=re.IGNORECASE)

    t = re.sub(r"\s+", " ", t).strip()
    return t

def top_ngrams(texts, n=1, topk=20):
    tokens_list = []
    for t in texts:
        tt = clean_text(t)
        toks = [w for w in tt.split() if w and w not in SPANISH_STOPWORDS and len(w) >= 3]
        if len(toks) >= n:
            tokens_list.append(toks)

    counter = Counter()
    for toks in tokens_list:
        if n == 1:
            counter.update(toks)
        else:
            grams = zip(*[toks[i:] for i in range(n)])
            counter.update([" ".join(g) for g in grams])

    out = counter.most_common(topk)
    return pd.DataFrame(out, columns=["termino", "frecuencia"])

# =========================================================
# 5) TIER desde "Etiquetas de tickets"
# =========================================================
def extract_tier(tags):
    if pd.isna(tags):
        return "Sin TIER"
    s = _strip_accents(str(tags).lower())

    if "vip" in s or "enterprise" in s:
        return "VIP/ENTERPRISE"
    if "mid" in s or "mid-market" in s or "mid market" in s:
        return "MID MARKET"
    if "small" in s or "smb" in s or "small business" in s:
        return "SMALL BUSINESS"

    m = re.search(r"tier\s*[:=-]\s*([a-z0-9/\s-]+)", s)
    if m:
        val = re.sub(r"\s+", " ", m.group(1).strip()).upper()
        return val[:40] if val else "Sin TIER"

    return "Sin TIER"

# =========================================================
# 6) TOPICS ML (si hay sklearn) + TITULADO + RESUMEN EJECUTIVO
# =========================================================
def breeze_like_summary(df_topics: pd.DataFrame) -> str:
    if df_topics is None or df_topics.empty:
        return "No se identificaron patrones suficientes para generar un resumen."

    total = int(df_topics["Tickets"].sum())
    top3 = df_topics.head(3)

    bullets = []
    for _, r in top3.iterrows():
        title = r.get("Título sugerido", "Tema")
        n = int(r.get("Tickets", 0))
        pct = (n / total * 100) if total > 0 else 0.0
        bullets.append(f"- **{title}**: {n} tickets ({pct:.1f}%).")

    txt = (
        "Entre los tickets con texto útil (Asunto + cuerpo limpio + Módulo, sin adjuntos/links/saludos/identificadores), "
        "se observan patrones recurrentes concentrados en:\n"
        + "\n".join(bullets)
        + "\n\n"
        "Recomendación: priorizar los tópicos con mayor volumen y validar con los ejemplos representativos "
        "para confirmar el patrón y definir acciones (guías, mejoras de producto, ajustes operativos)."
    )
    return txt

def build_topic_summary(text_series: pd.Series, n_topics: int = 6, top_terms: int = 10, top_bigrams: int = 6, examples_per_topic: int = 4):
    """
    - TF-IDF (1-2gramas) + KMeans
    - Filtra ruido de adjuntos/imagenes/links + cortesía + identificadores
    - Para cada tópico:
        * Título sugerido (basado en bigramas + términos)
        * Frases clave
        * # tickets
        * Ejemplos reales (para validar)
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None, None, "No se encontró scikit-learn."

    docs_raw_all = text_series.fillna("").astype(str).tolist()

    # 1) Filtra ruido (adjuntos/imagenes/links) + textos cortos
    docs_raw = [d for d in docs_raw_all if not is_attachment_noise(d)]

    if len(docs_raw) < max(20, n_topics * 5):
        return None, None, "No hay suficientes textos útiles tras filtrar adjuntos/imagenes."

    # 2) Limpieza fuerte adicional: quita identificadores y boilerplate residual
    docs_raw2 = [strip_identifiers(d) for d in docs_raw]

    docs_clean = [clean_text(x) for x in docs_raw2]
    docs_clean = [" ".join([w for w in d.split() if w and w not in SPANISH_STOPWORDS and len(w) >= 3]) for d in docs_clean]

    # filtra vacíos reales
    idx_keep = [i for i, d in enumerate(docs_clean) if len(d.split()) >= 5]
    if len(idx_keep) < max(30, n_topics * 5):
        return None, None, "No hay suficiente texto limpio (mínimo ~30 descripciones útiles)."

    docs_clean = [docs_clean[i] for i in idx_keep]
    docs_raw2  = [docs_raw2[i] for i in idx_keep]

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True
    )
    X = vec.fit_transform(docs_clean)
    terms = np.array(vec.get_feature_names_out())

    k = min(n_topics, max(2, int(len(docs_clean) / 12)))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    centroids = km.cluster_centers_
    sims = cosine_similarity(X, centroids)

    rows = []
    for topic_id in range(k):
        idx_topic = np.where(labels == topic_id)[0]
        n_docs_topic = len(idx_topic)
        if n_docs_topic == 0:
            continue

        centroid = centroids[topic_id]
        top_idx = np.argsort(centroid)[::-1][:top_terms * 8]

        top_terms_list = []
        top_bigrams_list = []

        banned = {
            "image","png","jpeg","jpg","http","www","adjunt","captura","pantallaz","foto","pdf","xlsx","docx",
            "cordial","saludo","saludos","atentamente","cordialmente","gracias","quedo","atenta","atento","pendiente",
            "companeros","compañeros","espero","buenos","dias","buenas","tardes","noches",
            "sede","institucion","institución","entidad","codigo","código","bd","ce"
        }

        for j in top_idx:
            t = terms[j]
            t_low = t.lower()

            if any(b in t_low for b in banned):
                continue
            if len(t) < 3:
                continue

            if " " in t:
                if len(top_bigrams_list) < top_bigrams:
                    top_bigrams_list.append(t)
            else:
                if len(top_terms_list) < top_terms:
                    top_terms_list.append(t)

            if len(top_terms_list) >= top_terms and len(top_bigrams_list) >= top_bigrams:
                break

        topic_sims = sims[idx_topic, topic_id]
        top_example_local_idx = np.argsort(topic_sims)[::-1][:examples_per_topic]
        examples = [docs_raw2[idx_topic[i]] for i in top_example_local_idx]

        if len(top_bigrams_list) > 0:
            title = " / ".join(top_bigrams_list[:2])
        else:
            title = " / ".join(top_terms_list[:3])

        rows.append({
            "Tópico": f"Tópico {topic_id+1}",
            "Título sugerido": title if title else "Tema sin identificar",
            "Tickets": n_docs_topic,
            "Frases clave (2-gramas)": ", ".join(top_bigrams_list[:8]),
            "Términos clave": ", ".join(top_terms_list[:10]),
            "Ejemplos (texto real)": "\n---\n".join([e[:360] + ("..." if len(e) > 360 else "") for e in examples]),
        })

    df_topics = pd.DataFrame(rows).sort_values("Tickets", ascending=False).reset_index(drop=True)
    return df_topics, (idx_keep, labels), None

# =========================================================
# 7) CARGA + PREP
# =========================================================
@st.cache_data(show_spinner=False)
def load_data_from_uploads(up_tickets, up_csat, up_llamadas, up_wapp):
    if up_tickets is None or up_csat is None or up_llamadas is None or up_wapp is None:
        return None

    tickets = pd.read_excel(up_tickets)
    csat    = pd.read_excel(up_csat)

    # Estas funciones ya las tienes:
    sheet_ll, llamadas = read_best_sheet(up_llamadas, must_have_any=("fecha",))
    sheet_wa, wapp     = read_best_sheet(up_wapp, must_have_any=("fecha",))

    tickets.columns  = [normalize_colname(c) for c in tickets.columns]
    csat.columns     = [normalize_colname(c) for c in csat.columns]
    llamadas.columns = [normalize_colname(c) for c in llamadas.columns]
    wapp.columns     = [normalize_colname(c) for c in wapp.columns]

    return tickets, csat, llamadas, wapp, sheet_ll, sheet_wa


def prep_tickets(tickets, start, end):
    exact_nuevo   = 'Date entered "Nuevo (Pipeline CSS)"'
    exact_cerrado = 'Date entered "Cerrado (Pipeline CSS)"'

    col_nuevo   = find_col_exact(tickets, exact_nuevo) or find_col_regex(tickets, r'date entered\s*"?nuevo.*pipeline css"?')
    col_cerrado = find_col_exact(tickets, exact_cerrado) or find_col_regex(tickets, r'date entered\s*"?cerrado.*pipeline css"?')

    if col_nuevo is None or col_cerrado is None:
        raise ValueError(
            "No pude ubicar las columnas de pipeline.\n"
            f"Encontré Nuevo: {col_nuevo}\nEncontré Cerrado: {col_cerrado}\n"
            "Activa Debug y revisa el listado de columnas."
        )

    tickets["_fecha_nuevo"]   = parse_excel_datetime(tickets[col_nuevo])
    tickets["_fecha_cerrado"] = parse_excel_datetime(tickets[col_cerrado])

    t_in  = tickets[(tickets["_fecha_nuevo"]   >= start) & (tickets["_fecha_nuevo"]   <= end)].copy()
    t_out = tickets[(tickets["_fecha_cerrado"] >= start) & (tickets["_fecha_cerrado"] <= end)].copy()

    resp_col = None
    for c in ["Agente de CS", "Propietario del ticket", "Propietario"]:
        if c in tickets.columns:
            resp_col = c
            break
    if resp_col:
        t_in["_responsable"]  = t_in[resp_col].apply(normalize_responsable)
        t_out["_responsable"] = t_out[resp_col].apply(normalize_responsable)
    else:
        t_in["_responsable"] = "Sin responsable"
        t_out["_responsable"] = "Sin responsable"

    mod_col = find_col_exact(tickets, "Módulo(s) afectados o dónde se presenta la novedad") \
              or find_col_contains(tickets, ["modulo", "módulo", "afectados", "novedad"])
    tier_col = find_col_exact(tickets, "Etiquetas de tickets") \
              or find_col_contains(tickets, ["etiquetas", "labels", "tags"])
    desc_col = find_col_exact(tickets, "Descripción del ticket") \
              or find_col_contains(tickets, ["descripcion del ticket", "descripción del ticket", "descripcion", "descripción", "detalle"])

    # Asunto: intentamos varias opciones
    subject_col = find_col_exact(tickets, "Asunto del ticket") \
                  or find_col_exact(tickets, "Asunto") \
                  or find_col_exact(tickets, "Subject") \
                  or find_col_contains(tickets, ["asunto", "subject", "titulo", "título", "nombre del ticket"])

    for df in (t_in, t_out):
        df["_modulo"] = df[mod_col].astype(str) if mod_col and mod_col in df.columns else "Sin módulo"
        df["_tags"] = df[tier_col].astype(str) if tier_col and tier_col in df.columns else ""
        df["_tier"] = df["_tags"].apply(extract_tier)

        desc_raw = df[desc_col].astype(str) if desc_col and desc_col in df.columns else ""
        subj = df[subject_col].astype(str) if subject_col and subject_col in df.columns else ""

        # cuerpo limpio (sin saludo/despedida/firma/citas)
        desc_clean = desc_raw.apply(strip_email_boilerplate)

        # 🔥 NUEVO: quitamos identificadores (CE/BD/udbz/números largos)
        desc_clean2 = desc_clean.apply(strip_identifiers)
        subj2 = subj.apply(strip_identifiers)

        # 🔥 NUEVO: incluimos módulo como contexto (pero lo limpiamos también)
        mod2 = df["_modulo"].fillna("").astype(str).apply(strip_identifiers)

        # texto final ML: (Módulo) + Asunto + Cuerpo limpio
        df["_texto_ml"] = ("[" + mod2 + "] " + subj2.fillna("").astype(str) + ". " + desc_clean2.fillna("").astype(str)).str.strip()

        # deja descripción original para auditoría/validación
        df["_descripcion"] = desc_raw

    t_out["_sla_habil_min"] = t_out.apply(
        lambda r: business_minutes_between(r["_fecha_nuevo"], r["_fecha_cerrado"]),
        axis=1
    )

    ttr_col = "Tiempo para la primera asignación de representantes (HH:mm:ss)"
    t_out["_ttr_min"] = td_to_minutes(t_out[ttr_col]) if ttr_col in t_out.columns else np.nan

    meta = {
        "col_nuevo": col_nuevo,
        "col_cerrado": col_cerrado,
        "resp_col": resp_col,
        "mod_col": mod_col,
        "tier_col": tier_col,
        "desc_col": desc_col,
        "subject_col": subject_col,
    }
    return t_in, t_out, meta

def prep_gestiones(df, start, end, canal_label):
    out = df.copy()

    fecha_col = None
    for candidate in ["FECHA", "Fecha", "fecha", "Fecha gestión", "Fecha y hora", "Fecha y Hora"]:
        if candidate in out.columns:
            fecha_col = candidate
            break
    if fecha_col is None:
        fecha_col = find_col_contains(out, ["fech"])

    out["_fecha"] = parse_excel_datetime(out[fecha_col]) if fecha_col else pd.NaT
    out = out[(out["_fecha"] >= start) & (out["_fecha"] <= end)].copy()

    resp_col = find_col_contains(out, ["css", "responsable", "agente"])
    if resp_col:
        out["_responsable"] = out[resp_col].apply(normalize_responsable)
    elif "CSS" in out.columns:
        out["_responsable"] = out["CSS"].apply(normalize_responsable)
    else:
        out["_responsable"] = "Sin responsable"

    out["_canal"] = canal_label
    return out

def prep_csat(csat, start, end):
    date_col  = find_col_exact(csat, "Date") or find_col_contains(csat, ["date", "fecha"])
    score_col = find_col_exact(csat, "Customer Satisfaction") or find_col_contains(csat, ["customer satisfaction", "csat", "satisfaction"])

    csat["_fecha"] = parse_excel_datetime(csat[date_col]) if date_col else pd.NaT
    c = csat[(csat["_fecha"] >= start) & (csat["_fecha"] <= end)].copy()
    c["_score"] = pd.to_numeric(c[score_col], errors="coerce") if score_col and score_col in c.columns else np.nan

    meta = {"date_col": date_col, "score_col": score_col}
    return c, meta

# =========================================================
# 8) UI
# =========================================================
st.set_page_config(page_title="Q10 - Gestión Customer Support", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem;}
      h1, h2, h3 {margin-bottom: 0.3rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Q10 – Gestión Customer Support")

data = load_data_from_uploads(up_tickets, up_csat, up_llamadas, up_wapp)

if data is None:
    st.info("Carga los 4 archivos en la barra lateral para iniciar el dashboard.")
    st.stop()

tickets, csat, llamadas_raw, wapp_raw, sheet_ll, sheet_wa = data


with st.sidebar:
    st.header("Filtros")
    start = st.date_input("Desde", DEFAULT_START.date())
    end   = st.date_input("Hasta", DEFAULT_END.date())
    start = pd.Timestamp(start)
    end   = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    st.caption(f"Hojas detectadas: Llamadas = **{sheet_ll}** | WhatsApp = **{sheet_wa}**")
    debug = st.checkbox("Debug (recomendado)", value=True)

# -------- Preparación --------
t_in, t_out, meta_t = prep_tickets(tickets, start, end)
ll = prep_gestiones(llamadas_raw, start, end, "Llamadas")
wa = prep_gestiones(wapp_raw, start, end, "WhatsApp")
c, meta_csat = prep_csat(csat, start, end)

# -------- Debug fuerte --------
if debug:
    with st.sidebar:
        st.markdown("### Debug Tickets (Pipeline CSS)")
        st.write("Col Nuevo:", meta_t["col_nuevo"])
        st.write("Col Cerrado:", meta_t["col_cerrado"])
        st.write("Col Asunto:", meta_t.get("subject_col"))
        st.write("Col Descripción:", meta_t.get("desc_col"))
        st.write("Col Módulo:", meta_t.get("mod_col"))

        all_nuevo = parse_excel_datetime(tickets[meta_t["col_nuevo"]])
        all_cerr  = parse_excel_datetime(tickets[meta_t["col_cerrado"]])

        st.write("Nuevo no-NA:", int(all_nuevo.notna().sum()))
        st.write("Nuevo min/max:", all_nuevo.min(), " / ", all_nuevo.max())
        st.write("Cerrado no-NA:", int(all_cerr.notna().sum()))
        st.write("Cerrado min/max:", all_cerr.min(), " / ", all_cerr.max())

        st.write("Entrados en filtro:", len(t_in))
        st.write("Cerrados en filtro:", len(t_out))

        st.markdown("### Debug CSAT")
        st.write("Col Date:", meta_csat["date_col"])
        st.write("Col Score:", meta_csat["score_col"])
        st.write("CSAT respuestas en filtro:", int(c["_score"].notna().sum()))

        st.markdown("### Debug Llamadas/WA")
        st.write("Llamadas filtradas:", len(ll))
        st.write("WA filtradas:", len(wa))

        st.markdown("### Columnas tickets (muestra)")
        st.write(list(tickets.columns)[:35])

# -------- Gestiones unificadas --------
t_g = t_in[["_fecha_nuevo", "_responsable"]].copy().rename(columns={"_fecha_nuevo": "_fecha"})
t_g["_canal"] = "Tickets (Entrados)"

gestiones = pd.concat(
    [t_g, ll[["_fecha","_responsable","_canal"]], wa[["_fecha","_responsable","_canal"]]],
    ignore_index=True
).dropna(subset=["_fecha"])

# -------- KPIs --------
n_in   = len(t_in)
n_out  = len(t_out)
n_ll   = len(ll)
n_wa   = len(wa)
total_g = n_in + n_ll + n_wa

sla_prom_min = float(np.nanmean(t_out["_sla_habil_min"])) if t_out["_sla_habil_min"].notna().any() else np.nan
ttr_prom_min = float(np.nanmean(t_out["_ttr_min"])) if t_out["_ttr_min"].notna().any() else np.nan
ttr_p90_min  = float(np.nanpercentile(t_out["_ttr_min"].dropna(), 90)) if t_out["_ttr_min"].notna().any() else np.nan

# CSAT % Top-Box (4–5)
csat_valid = c["_score"].dropna()
csat_n = int(csat_valid.shape[0])
csat_topbox_n = int((csat_valid.isin([4, 5])).sum())
csat_pct = (csat_topbox_n / csat_n * 100) if csat_n > 0 else np.nan

# =========================================================
# 9) LAYOUT
# =========================================================
colA, colB, colC, colD, colE, colF, colG = st.columns([1.35,1,1,1,1,1,1])
with colA: kpi_card("Total gestiones (Entrados + Llamadas + WA)", fmt_int(total_g))
with colB: kpi_card("Tickets entrados", fmt_int(n_in))
with colC: kpi_card("Tickets cerrados", fmt_int(n_out))
with colD: kpi_card("Llamadas", fmt_int(n_ll))
with colE: kpi_card("WhatsApp", fmt_int(n_wa))
with colF: kpi_card("SLA prom (HH:MM)", minutes_to_hhmm(sla_prom_min))
with colG: kpi_card("TTR prom / p90 (min)", f"{fmt_float(ttr_prom_min,1)} / {fmt_float(ttr_p90_min,1)}")

st.divider()

col1, col2 = st.columns([1.7, 1.0])
with col1:
    st.subheader("Atenciones por semana (Tickets entrados + Llamadas + WhatsApp)")
    tmp = gestiones.copy()
    tmp["semana"] = iso_week(tmp["_fecha"])
    week = tmp.groupby(["semana","_canal"]).size().reset_index(name="atenciones")
    fig = px.bar(week, x="semana", y="atenciones", color="_canal", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("CSAT (Customer Satisfaction)")
    if csat_n == 0:
        st.info("No hay CSAT en el periodo (o no se detectaron columnas Date / Customer Satisfaction).")
    else:
        kpi_card("CSAT % (4–5)", f"{fmt_float(csat_pct, 2)}%")
        st.caption(f"Respuestas: {csat_n} | 4–5: {csat_topbox_n}")
        dist = c["_score"].dropna().value_counts().sort_index().reset_index()
        dist.columns = ["score", "respuestas"]
        fig2 = px.bar(dist, x="score", y="respuestas")
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 9.1) GRÁFICAS EN VERTICAL + EJE X DIARIO COMPLETO
# =========================================================
st.subheader("Atenciones por día")
day = gestiones.groupby([gestiones["_fecha"].dt.date, "_canal"]).size().reset_index(name="atenciones")
day.columns = ["dia","canal","atenciones"]
day["dia"] = pd.to_datetime(day["dia"])

fig3 = px.bar(day, x="dia", y="atenciones", color="canal", barmode="group")
fig3.update_xaxes(
    tickmode="linear",
    dtick=24*60*60*1000,
    tickformat="%b %d",
    tickangle=-45
)
fig3.update_layout(height=520)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Atenciones por responsable")
top = (gestiones.groupby("_responsable").size()
       .reset_index(name="atenciones")
       .sort_values("atenciones", ascending=False))
fig4 = px.bar(top, x="atenciones", y="_responsable", orientation="h")
fig4.update_layout(yaxis={"categoryorder":"total ascending"})
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# =========================================================
# 10) MÓDULOS + TIER (sobre tickets ENTRADOS)
# =========================================================
col5, col6 = st.columns([1.2, 1.2])
with col5:
    st.subheader("Top módulos (Tickets entrados)")
    top_mod = (t_in.groupby("_modulo").size().reset_index(name="tickets")
               .sort_values("tickets", ascending=False).head(15))
    fig_mod = px.bar(top_mod, x="tickets", y="_modulo", orientation="h")
    fig_mod.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig_mod, use_container_width=True)

with col6:
    st.subheader("Análisis por TIER (Etiquetas de tickets)")
    tier_count = (t_in.groupby("_tier").size().reset_index(name="tickets")
                  .sort_values("tickets", ascending=False))
    fig_tier = px.bar(tier_count, x="tickets", y="_tier", orientation="h")
    fig_tier.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig_tier, use_container_width=True)

st.subheader("TIER x Módulo (Top 10 módulos)")
top10_mods = set(top_mod["_modulo"].head(10).tolist())
cross = t_in[t_in["_modulo"].isin(top10_mods)].groupby(["_tier","_modulo"]).size().reset_index(name="tickets")
fig_cross = px.bar(cross, x="tickets", y="_modulo", color="_tier", orientation="h", barmode="group")
fig_cross.update_layout(yaxis={"categoryorder":"total ascending"})
st.plotly_chart(fig_cross, use_container_width=True)

st.divider()

# =========================================================
# 11) ¿QUÉ FUE LO QUE MÁS PREGUNTARON?
#     -> ML "tipo Breeze": usa (Módulo + Asunto + cuerpo limpio) y quita identificadores
# =========================================================
st.subheader("¿Qué fue lo que más preguntaron? (Módulo + asunto + cuerpo del ticket - Tickets entrados)")

desc = t_in["_texto_ml"].fillna("").astype(str)
desc = desc[desc.str.len() >= 50]

if len(desc) < 20:
    st.info("No hay suficientes textos (>=20) para un análisis representativo en el período.")
else:
    df_topics, labels_info, err = build_topic_summary(desc, n_topics=6, top_terms=10, top_bigrams=6, examples_per_topic=4)

    if err is None and df_topics is not None and labels_info is not None:
        st.markdown("### Resumen")
        st.write(breeze_like_summary(df_topics))

        colT1, colT2 = st.columns([1.35, 1.0])

        with colT1:
            st.markdown("**Tópicos con título sugerido + ejemplos**")
            st.dataframe(df_topics, use_container_width=True, height=360)

        with colT2:
            _, labels = labels_info
            vc = pd.Series(labels).value_counts().sort_index()
            dist_topics = pd.DataFrame({
                "tópico": [f"Tópico {i+1}" for i in vc.index],
                "tickets": vc.values
            })
            fig_topic = px.bar(dist_topics, x="tickets", y="tópico", orientation="h")
            fig_topic.update_layout(yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig_topic, use_container_width=True)

        with st.expander("Ver tickets por tópico (validación)"):
            idx_keep, labels = labels_info
            t_dbg = t_in.copy()
            t_dbg = t_dbg.loc[desc.index].copy()
            t_dbg = t_dbg.iloc[idx_keep].copy()
            t_dbg["_topico"] = [f"Tópico {x+1}" for x in labels]
            st.dataframe(
                t_dbg[["_fecha_nuevo","_fecha_cerrado","_topico","_tier","_modulo","_responsable","_texto_ml","_descripcion"]]
                .sort_values("_fecha_nuevo", ascending=False)
                .head(300),
                use_container_width=True,
                height=420
            )

    else:
        st.warning("No se pudo ejecutar análisis ML (probablemente falta scikit-learn o no hay texto útil tras filtrar). Mostrando n-grams.")
        c1, c2 = st.columns([1.1, 1.1])
        uni = top_ngrams(desc.tolist(), n=1, topk=25)
        bi  = top_ngrams(desc.tolist(), n=2, topk=25)

        with c1:
            st.markdown("**Top palabras (unigramas)**")
            fig_u = px.bar(uni.sort_values("frecuencia"), x="frecuencia", y="termino", orientation="h")
            st.plotly_chart(fig_u, use_container_width=True)

        with c2:
            st.markdown("**Top frases (bigramas)**")
            fig_b = px.bar(bi.sort_values("frecuencia"), x="frecuencia", y="termino", orientation="h")
            st.plotly_chart(fig_b, use_container_width=True)

st.divider()

# =========================================================
# 12) DETALLE
# =========================================================
st.subheader("Detalle (validación rápida)")
st.caption("Gestiones unificadas (Tickets entrados + Llamadas + WhatsApp)")
st.dataframe(
    gestiones.sort_values("_fecha", ascending=False).head(200),
    use_container_width=True,
    height=320
)

st.caption(
    "Notas: Tickets entrados/cerrados usan columnas de pipeline (Nuevo/Cerrado). "
    "SLA es horas hábiles entre ENTRADO->CERRADO. "
    "Texto ML usa Módulo + Asunto + cuerpo limpio (sin saludos/despedidas) y filtra adjuntos/links/identificadores (CE/BD/udbz/números largos). "
)

