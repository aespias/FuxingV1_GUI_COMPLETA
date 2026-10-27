# ================================================================
# FUXING v3 MAXIMUM — BONOLOTO AI RESEARCH LAB
# ================================================================
# Versión para PC: almacenamiento persistente en Escritorio\FuxingV1.
# Adaptación de rutas (25/10/2026): el motor predictivo se conserva; en Windows
# se usa el Escritorio local del perfil, sin seguir redirecciones de OneDrive.
# Los datos, historiales, registros e informes se centralizan en FuxingV1.
# Candidata v3 tras revisión de auditoría externa (24/10/2026).
# Nota: el motor predictivo se conserva; las ampliaciones se limitan a datos,
# persistencia, validación temporal y salidas.
# Correcciones de esta revisión:
# - Bonoloto se trata como sorteo diario también en domingo (calendario actual SELAE).
# - Cobertura reciente desde las 21:30 y reparación SELAE limitada a 2 workers con backoff.
# - Europe/Madrid requiere tzdata si la base de zonas del sistema no está disponible.
# - Eliminado el flag estructural inalcanzable/no-op y su rama muerta.
# - Excel exportado sin fsync sobre descriptor de solo lectura.
# - Objetivos de predicción basados en hora real de generación (corte 21:30).
#
# Cambios respecto a la versión anterior:
#   1.  parse_fecha_es: soporta "lunes, 4 de marzo de 2024" y variantes.
#   2.  Conformal: misma escala en fit y predict (normalize_rows).
#   3.  scrape_with_cache: fuerza actualización del año en curso.
#   4.  number_features: vectorizado con cum precomputada (pasada como
#       argumento). Ya no reconstruye la matriz 7500 veces.
#   5.  Barra de progreso con ETA estimado.
#   6.  Off-by-one: features incluyen el sorteo más reciente (index+1).
#   7.  Limpieza de código muerto y funciones no usadas.
#   8.  normalize_rows: garantía matemática de suma 6 y valores en [0,1].
#   9.  _compute_components: alineación robusta y cache de resultados.
#  10.  La cobertura histórica no presupone un número fijo de sorteos por año.
#  11.  _exponential_frequency_from_counts: pesos CORREGIDOS (más peso
#       a sorteos recientes) e INCLUYE el sorteo más reciente.
#  12.  _scrape_year_loteriadehoy: elimina el patrón de fecha del bloque
#       antes de extraer números (no borra números válidos del sorteo).
#  13.  Resultados/: informe TXT y JSON por ejecución, actualizado al recuperar
#       el resultado real; conserva separados todos los run_id.
#  14.  Valida que la predicción generada después de las 21:30 no use un histórico
#       de entrenamiento que todavía termina en el día anterior.
#  15.  Usa la misma marca temporal para asignar el objetivo y registrar predicción.
#
# ADVERTENCIA HONESTA:
# La Bonoloto es aleatoria: no hay garantías predictivas. Cualquier posible
# ventaja del modelo debe demostrarse fuera de muestra frente a una referencia.
# ================================================================

from __future__ import annotations

import os
import sys
import re
import csv
import io
import json
import shutil
import math
import time
import pickle
import random
import sqlite3
import warnings
import threading
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from itertools import combinations
from pathlib import Path
import argparse
import uuid
from contextlib import contextmanager
from urllib.parse import urljoin, unquote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------
# Dependencias opcionales
# ----------------------------------------------------------------
TORCH_AVAILABLE = False
SKLEARN_AVAILABLE = False
XGB_AVAILABLE = False
LGBM_AVAILABLE = False
NUMBA_AVAILABLE = False
TQDM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        ExtraTreesClassifier,
        RandomForestClassifier,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGBClassifier = None

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    lgb = None

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    njit = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    tqdm = None


# ================================================================
# CONFIGURACIÓN GLOBAL
# ================================================================

MAX_NUM = 49
NUMBERS_PER_DRAW = 6
START_YEAR = 1988
SEQ_LEN = 120

ULTIMATE_SEED = 20260901

def _resolve_desktop_directory():
    """Devuelve la carpeta Escritorio local del usuario actual.

    En Windows se usa directamente la carpeta Desktop del perfil local (o
    la carpeta Escritorio si esa es la que existe). Deliberadamente
    NO se consulta el registro de redirección de carpetas ni variables de
    OneDrive: este programa debe crear FuxingV1 en el perfil local del usuario,
    no depender de la cuenta de nube que pueda tener configurada ese equipo.

    En Linux se respeta XDG_DESKTOP_DIR; en macOS y como alternativa, se usa
    la carpeta Desktop del directorio personal.
    """
    home = Path.home()

    if os.name == "nt":
        profile_text = os.environ.get("USERPROFILE", "").strip()
        profile = Path(profile_text).expanduser() if profile_text else home

        # Ruta local del perfil. No consultar User Shell Folders, OneDrive ni
        # otras ubicaciones sincronizadas, aunque estén configuradas en Windows.
        local_candidates = (profile / "Desktop", profile / "Escritorio")
        for candidate in local_candidates:
            try:
                if candidate.is_dir():
                    # No usar resolve(): se conserva la ruta del perfil indicada.
                    return candidate.absolute()
            except OSError:
                continue

        # En Windows, Desktop es el nombre físico habitual incluso en sistemas
        # con interfaz en español. BASE_DIR.mkdir(parents=True) creará la ruta
        # local si todavía no existe.
        return (profile / "Desktop").absolute()

    candidates = []

    if sys.platform.startswith("linux"):
        # En Linux, respeta el directorio definido por XDG cuando exista.
        xdg_file = home / ".config" / "user-dirs.dirs"
        try:
            if xdg_file.is_file():
                for line in xdg_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    match = re.match(r"\s*XDG_DESKTOP_DIR\s*=\s*(.*?)\s*$", line)
                    if not match:
                        continue
                    value = match.group(1).strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                        value = value[1:-1]
                    value = value.replace(r"\$", "$").replace("$HOME", str(home))
                    value = os.path.expandvars(value)
                    if value:
                        candidates.append(Path(value).expanduser())
                    break
        except Exception:
            pass

    candidates.append(home / "Desktop")
    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except Exception:
            continue

    return candidates[0].expanduser().resolve() if candidates else (home / "Desktop").resolve()

# La carpeta de datos no depende de dónde esté el .py, de VS Code ni del .exe.
# Cada PC tendrá su propia base en Escritorio\FuxingV1. El histórico de sorteos
# se descarga/valida allí; el registro de predicciones empieza vacío en esa PC.
DESKTOP_DIR = _resolve_desktop_directory()
BASE_DIR = DESKTOP_DIR / "FuxingV1"
BASE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "bonoloto_data"
MODEL_DIR = BASE_DIR / "bonoloto_models"
LOG_DIR = BASE_DIR / "bonoloto_logs"
RESULTS_DIR = BASE_DIR / "Predicciones y aciertos del programa"
RUN_RESULTS_DIR = BASE_DIR / "Resultados"  # Un informe por ejecución, actualizado al llegar el sorteo real.

for d in (DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR, RUN_RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SQLITE_FILE = DATA_DIR / "bonoloto.db"
CACHE_FILE = DATA_DIR / "bonoloto_history.pkl"
CANDIDATE_CACHE_FILE = DATA_DIR / "bonoloto_history_candidate.pkl"
CSV_FILE = DATA_DIR / "bonoloto_history.csv"
AUDIT_FILE = DATA_DIR / "dataset_audit.json"
BACKTEST_FILE = DATA_DIR / "backtest_results.json"
FEATURE_AUDIT_FILE = DATA_DIR / "feature_audit.json"

RESULT_FILE = BASE_DIR / "fuxing_resultado.json"
RUN_REPORT_FILE = BASE_DIR / "fuxing_run_report.json"
EXEC_LOG_FILE = LOG_DIR / f"fuxing_{datetime.now():%Y%m%d_%H%M%S}.log"

DEVICE = "cpu"
if TORCH_AVAILABLE and torch.cuda.is_available():
    DEVICE = "cuda"

CPU_COUNT = os.cpu_count() or 2
MAX_CPU_WORKERS = max(1, CPU_COUNT - 1)

NEURAL_ENSEMBLE_SIZE = 5
TREE_ESTIMATORS = 600
XGB_ESTIMATORS = 500
LGBM_ESTIMATORS = 700
MC_ITERATIONS = 1_000_000
MC_ITERATIONS_SEMANAL = 2_000_000

FEATURE_WINDOWS = (5, 10, 20, 40, 80, 160, 320, 640)
FEATURE_HALF_LIFE = 60.0

# Nota de integridad histórica: NO se estima la cobertura histórica usando días
# naturales. El calendario de sorteos cambió con el tiempo. La cobertura se
# verifica contra el contador oficial de SELAE cuando está disponible.

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5,
    "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10,
    "nov": 11, "dic": 12,
}


# ================================================================
# LOGGING CON BARRA DE PROGRESO Y ETA
# ================================================================

class FuxingLogger:
    LEVELS = {
        "INFO": "", "DATA": "\033[36m", "NETWORK": "\033[35m",
        "MODEL": "\033[34m", "OPTIM": "\033[33m", "PRED": "\033[32m",
        "AUDIT": "\033[37m", "WARNING": "\033[33m", "ERROR": "\033[31m",
        "OK": "\033[92m",
    }
    RESET = "\033[0m"

    def __init__(self, log_file: Path):
        self.start = datetime.now()
        self.log_file = log_file
        self._lock = threading.Lock()
        self._fh = open(self.log_file, "a", encoding="utf-8")
        self._last_progress_len = 0

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(self):
        return (datetime.now() - self.start).total_seconds()

    def _color(self, level):
        return self.LEVELS.get(level, "")

    def _emit(self, text):
        with self._lock:
            try:
                print(text, flush=True)
            except Exception:
                pass
            try:
                clean = re.sub(r"\033\[[0-9;]*m", "", text)
                self._fh.write(clean + "\n")
                self._fh.flush()
            except Exception:
                pass

    def log(self, msg, level="INFO"):
        c = self._color(level)
        r = self.RESET
        line = f"[{self._ts()}] {c}[{level:7s}]{r} {msg}  \033[90m(t {self._elapsed():6.1f}s)\033[0m"
        self._emit(line)

    def banner(self, text, char="="):
        line = char * 78
        self._emit("")
        self._emit(line)
        self._emit(f"  {text}")
        self._emit(line)

    def progress(self, pct, msg, width=30):
        """Barra de progreso con ETA estimado."""
        pct = max(0.0, min(100.0, float(pct)))
        filled = int(width * pct / 100.0)
        bar = "#" * filled + "." * (width - filled)
        status = "OK" if pct >= 99.9 else ("RUN" if pct > 0 else "WAIT")

        elapsed = self._elapsed()
        if 0.1 < pct < 99.9:
            eta_sec = elapsed * (100.0 - pct) / pct
            eta_str = f" ETA {int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"
        elif pct >= 99.9:
            eta_str = " DONE"
        else:
            eta_str = ""

        text = (
            f"[{self._ts()}] \033[96m[{status}]\033[0m "
            f"\033[90m[{bar}]\033[0m {pct:6.2f}%{eta_str}  {msg}"
        )
        with self._lock:
            try:
                clean_len = len(re.sub(r"\033\[[0-9;]*m", "", text))
                pad = " " * max(0, self._last_progress_len - clean_len)
                print(text + pad, flush=True)
                self._last_progress_len = clean_len
            except Exception:
                pass
            try:
                clean = re.sub(r"\033\[[0-9;]*m", "", text)
                self._fh.write(clean + "\n")
                self._fh.flush()
            except Exception:
                pass

    def ok(self, msg):
        self.log(msg, "OK")

    def warn(self, msg):
        self.log(msg, "WARNING")

    def error(self, msg):
        self.log(msg, "ERROR")

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


log = FuxingLogger(EXEC_LOG_FILE)


def print_system_status():
    log.banner("FUXING v3 MAXIMUM — ESTADO DEL SISTEMA")
    log.log(f"Python: {os.sys.version.split()[0]}", "INFO")
    log.log(f"Carpeta de trabajo y datos: {BASE_DIR}", "INFO")
    log.log(f"CPU cores: {CPU_COUNT}  |  workers: {MAX_CPU_WORKERS}", "INFO")
    log.log(f"Dispositivo de cómputo: {DEVICE.upper()}", "INFO")
    log.log(f"PyTorch: {'SI' if TORCH_AVAILABLE else 'NO'}  |  "
            f"Scikit-learn: {'SI' if SKLEARN_AVAILABLE else 'NO'}  |  "
            f"XGBoost: {'SI' if XGB_AVAILABLE else 'NO'}  |  "
            f"LightGBM: {'SI' if LGBM_AVAILABLE else 'NO'}", "INFO")
    log.log(f"Numba: {'SI' if NUMBA_AVAILABLE else 'NO'}  |  "
            f"tqdm: {'SI' if TQDM_AVAILABLE else 'NO'}", "INFO")
    log.log(f"Ensemble neuronal: {NEURAL_ENSEMBLE_SIZE} redes Transformer", "INFO")
    log.log(f"Monte Carlo diario: {MC_ITERATIONS:,} iter  |  "
            f"semanal: {MC_ITERATIONS_SEMANAL:,} iter", "INFO")
    log.log("Bonoloto: sorteo diario actualmente; integridad histórica contrastada con el contador oficial cuando está disponible", "INFO")


# ================================================================
# UTILIDADES BÁSICAS
# ================================================================

def parse_fecha_es(texto):
    """Parsea fechas en español con soporte para:
       - "lunes, 4 de marzo de 2024"
       - "4 de marzo de 2024"
       - "lunes 4 marzo 2024"
       - "4 marzo 2024"
    """
    if not texto:
        return None
    try:
        txt = texto.lower().strip()
        txt = re.sub(
            r"^(lun(es)?|mar(tes)?|mi[eé]rcoles|jue(ves)?|vie(rnes)?|"
            r"s[aá]b(ado)?|dom(ingo)?)[,.\s]+",
            "", txt,
        )
        txt = re.sub(r"\s+de\s+", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        partes = txt.split()
        if len(partes) != 3:
            return None
        dia = int(re.sub(r"\D", "", partes[0]))
        mes = MESES.get(partes[1].strip("."))
        anio = int(partes[2])
        if mes is None:
            return None
        return datetime(anio, mes, dia)
    except Exception:
        return None


def validate_draw(draw):
    if draw is None or len(draw) != 6:
        return False
    try:
        nums = [int(x) for x in draw]
    except Exception:
        return False
    return len(set(nums)) == 6 and all(1 <= x <= 49 for x in nums)


def normalize_draw(draw):
    return tuple(sorted(int(x) for x in draw))


def one_hot_draw(draw):
    v = np.zeros(MAX_NUM, dtype=np.float32)
    for n in draw:
        if 1 <= n <= MAX_NUM:
            v[n - 1] = 1.0
    return v


def even_count(draw):
    return sum(x % 2 == 0 for x in draw)


def low_count(draw):
    return sum(x <= 24 for x in draw)


def draw_range(draw):
    return max(draw) - min(draw)


def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    std = np.std(x)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=float)
    temperature = max(float(temperature), 1e-6)
    z = (x - np.max(x)) / temperature
    e = np.exp(np.clip(z, -60, 60))
    return e / (e.sum() + 1e-12)


def normalize_mass(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=1.0 / MAX_NUM, posinf=1.0, neginf=1.0)
    p = np.clip(p, 1e-12, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(MAX_NUM, 1.0 / MAX_NUM, dtype=np.float64)
    return p / s


def normalize_rows(a):
    """Proyecta probabilidades marginales sobre el simplex acotado:
    0 <= p_i <= 1 y sum(p_i) = NUMBERS_PER_DRAW.

    La proyección por bisección evita que el recorte final pierda masa,
    incluso con entradas extremas, sesgadas o no finitas.
    """
    arr = np.asarray(a, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] < NUMBERS_PER_DRAW:
        raise ValueError("La matriz debe tener al menos NUMBERS_PER_DRAW columnas")
    if arr.shape[1] != MAX_NUM:
        raise ValueError(f"Se esperaban {MAX_NUM} columnas y se recibieron {arr.shape[1]}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    target = float(NUMBERS_PER_DRAW)
    out = np.empty_like(arr, dtype=float)

    for i, row in enumerate(arr):
        # La escala de las puntuaciones no debe alterar el resultado de forma
        # arbitraria: la proyección conserva el orden relativo y acota [0, 1].
        row = np.clip(row, 0.0, None)
        if not np.any(row > 0):
            out[i] = target / MAX_NUM
            continue

        lo = float(np.min(row) - 1.0)
        hi = float(np.max(row))
        for _ in range(80):
            mid = (lo + hi) / 2.0
            total = float(np.clip(row - mid, 0.0, 1.0).sum())
            if total > target:
                lo = mid
            else:
                hi = mid
        projected = np.clip(row - (lo + hi) / 2.0, 0.0, 1.0)

        # Corrección de redondeo residual, repartida sólo entre columnas libres.
        residual = target - float(projected.sum())
        if abs(residual) > 1e-10:
            if residual > 0:
                free = np.flatnonzero(projected < 1.0 - 1e-12)
                for j in free:
                    delta = min(residual, 1.0 - projected[j])
                    projected[j] += delta
                    residual -= delta
                    if residual <= 1e-10:
                        break
            else:
                free = np.flatnonzero(projected > 1e-12)
                for j in free:
                    delta = min(-residual, projected[j])
                    projected[j] -= delta
                    residual += delta
                    if residual >= -1e-10:
                        break
        out[i] = np.clip(projected, 0.0, 1.0)

    if was_1d:
        return out[0]
    return out


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def chi_square(freq):
    total = float(np.sum(freq))
    if total <= 0:
        return 0.0
    expected = total / MAX_NUM
    return float(np.sum((freq - expected) ** 2 / expected))

# ================================================================
# PARTE 2 — SCRAPING MULTI-FUENTE
# ================================================================

class DataScraper:
    """Descarga el histórico completo desde 1988 y lo mantiene localmente."""

    ELGORDO = "https://www.elgordo.com/es/resultados/bonoloto--a%C3%B1o-{year}"
    LOTOCRACK = "https://lotocrack.com/historico-de-resultados/bonoloto/resultados-{year}/"
    LOTERIADEHOY = "https://www.loteriadehoy.com/bonoloto/resultados/{year}/"
    RESULTADOSBONOLOTO = "https://www.resultadosbonoloto.com/bonoloto/{year}/"
    LAWEB_HISTORICO = "https://lawebdelaprimitiva.com/Bonoloto/Historico%20de%20sorteos"
    SELAE_STATS = "https://www.loteriasyapuestas.es/es/bonoloto/estadisticas"
    SELAE_RESULTS_LIST = "https://www.loteriasyapuestas.es/es/bonoloto/resultados"
    # Fichero oficial documentado para consultar un sorteo concreto.
    SELAE_RESULT_WIDGET = "https://www.loteriasyapuestas.es/f/loterias/resultados/bonoloto.html"
    # Archivos históricos descargables indicados por Lotoideas (1988-2012 y 2013-2026).
    LOTOIDEAS_ARCHIVES = {
        "lotoideas_1988_2012": (
            "https://docs.google.com/spreadsheets/d/e/2PACX-1vQALTRaLDFfhXOAQmeONPqmFKm9yOiQ4W97rhWgR41BZ7czFsjK5YktD6fnETKHGB9YUnyQ4XBSbhZx/pub?gid=1&output=csv&single=true"
        ),
        "lotoideas_2013_2026": (
            "https://docs.google.com/spreadsheets/d/e/2PACX-1vQALTRaLDFfhXOAQmeONPqmFKm9yOiQ4W97rhWgR41BZ7czFsjK5YktD6fnETKHGB9YUnyQ4XBSbhZx/pub?gid=0&output=csv&single=true"
        ),
    }

    def __init__(self, start_year=START_YEAR, end_year=None):
        self.start_year = int(start_year)
        self.end_year = int(end_year or _madrid_now().year)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "es-ES,es;q=0.9",
        })
        self.last_history_audit = {}
        self.last_source_counts = {}
        self.last_source_conflicts = []
        self.last_annual_source_conflicts = []
        self.last_annual_unresolved_dates = []
        self.last_candidate_cache_conflicts = []
        self._source_audit_lock = threading.Lock()
        self.official_expected_total = None
        self.official_selae_results = {}  # ISO date -> (datetime, sorted six-number list)
        self.selae_results_page_ok = False
        self.selae_results_page_latest_date = None
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(SQLITE_FILE)
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS draws (
                    fecha TEXT NOT NULL,
                    n1 INTEGER NOT NULL, n2 INTEGER NOT NULL, n3 INTEGER NOT NULL,
                    n4 INTEGER NOT NULL, n5 INTEGER NOT NULL, n6 INTEGER NOT NULL,
                    source TEXT,
                    PRIMARY KEY (fecha, n1, n2, n3, n4, n5, n6)
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_fecha ON draws(fecha)")
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _backup_file(path):
        """Mantiene una copia .bak antes de sustituir un archivo de datos."""
        path = Path(path)
        if path.exists() and path.is_file():
            backup = path.with_suffix(path.suffix + ".bak")
            try:
                shutil.copy2(path, backup)
            except Exception as exc:
                raise RuntimeError(f"No se pudo crear copia de seguridad de {path}: {exc}") from exc

    def _save_sqlite(self, data):
        if not data:
            return
        rows = []
        seen_dates = set()
        for fecha, draw in data:
            date_key = fecha.strftime("%Y-%m-%d")
            if date_key in seen_dates:
                raise ValueError(f"No se puede guardar SQLite: fecha duplicada {date_key}")
            seen_dates.add(date_key)
            normalized = tuple(sorted(int(x) for x in draw))
            if not validate_draw(normalized):
                raise ValueError(f"No se puede guardar combinación inválida para {date_key}: {draw}")
            rows.append((date_key, *normalized, "merged"))

        self._backup_file(SQLITE_FILE)
        # La sustitución completa evita dejar filas antiguas en fechas que ya no
        # estén en el conjunto validado. SQLite hace rollback si falla la operación.
        con = sqlite3.connect(SQLITE_FILE)
        try:
            con.execute("BEGIN IMMEDIATE")
            con.execute("DELETE FROM draws")
            con.executemany(
                "INSERT INTO draws (fecha,n1,n2,n3,n4,n5,n6,source) VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _load_sqlite(self):
        if not SQLITE_FILE.exists():
            return []
        out = []
        con = None
        try:
            con = sqlite3.connect(SQLITE_FILE)
            cur = con.execute(
                "SELECT fecha,n1,n2,n3,n4,n5,n6 FROM draws ORDER BY fecha, source"
            )
            grouped = defaultdict(Counter)
            date_objects = {}
            for row in cur:
                try:
                    fecha = datetime.strptime(row[0], "%Y-%m-%d")
                    draw = tuple(sorted(int(x) for x in row[1:7]))
                except (TypeError, ValueError):
                    continue
                if validate_draw(draw):
                    grouped[row[0]][draw] += 1
                    date_objects[row[0]] = fecha
            by_date = {}
            for date_key, votes in grouped.items():
                ranked = votes.most_common()
                if not ranked:
                    continue
                if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                    log.warn(f"SQLite: conflicto no resuelto para {date_key}; se omite esa fecha.")
                    continue
                by_date[date_key] = (date_objects[date_key], list(ranked[0][0]))
            out = sorted(by_date.values(), key=lambda x: x[0])
        except Exception as e:
            log.warn(f"SQLite no utilizable: {e}")
            return []
        finally:
            if con is not None:
                con.close()
        return out

    def _request(self, url, attempts=4):
        last = None
        for attempt in range(attempts):
            pause_checkpoint(f"petición de red, intento {attempt + 1}/{attempts}")
            try:
                r = self.session.get(url, timeout=40)
                if r.status_code == 200 and r.text:
                    return r
                last = RuntimeError(f"HTTP {r.status_code}")
            except Exception as e:
                last = e
            time.sleep(1.5 * (attempt + 1))
        raise last or RuntimeError("Error HTTP")

    def _clean_data(self, data):
        """Deduplica por (fecha, draw) con votación por mayoría entre fuentes."""
        by_fecha_meta = defaultdict(list)

        for row in data or []:
            try:
                fecha, draw = row
                if not isinstance(fecha, datetime):
                    fecha = pd.to_datetime(fecha).to_pydatetime()
                draw = tuple(normalize_draw(draw))
                if validate_draw(draw):
                    by_fecha_meta[fecha.date().isoformat()].append((fecha, draw))
            except Exception:
                continue

        result = {}
        for fecha_iso, rows in by_fecha_meta.items():
            vote_count = Counter(draw for _, draw in rows)
            ranked = vote_count.most_common()
            if not ranked:
                continue
            # No escoger arbitrariamente si dos combinaciones distintas tienen
            # exactamente el mismo apoyo. Se informa y se excluye la fecha dudosa.
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                log.warn(f"Histórico: conflicto entre fuentes sin mayoría para {fecha_iso}; fecha excluida.")
                continue
            best_draw, _ = ranked[0]
            fecha = rows[0][0]
            result[fecha_iso] = (fecha, list(best_draw))

        return sorted(result.values(), key=lambda x: x[0])

    def _merge(self, *lists):
        combined = []
        for lst in lists:
            combined.extend(lst or [])
        return self._clean_data(combined)

    # ------------------------------------------------------------
    # Parsers por fuente
    # ------------------------------------------------------------
    def _scrape_year_lotocrack(self, year):
        url = self.LOTOCRACK.format(year=int(year))
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        lines = [
            re.sub(r"\s+", " ", x.strip())
            for x in soup.get_text("\n", strip=True).splitlines()
            if x.strip()
        ]
        months = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "setiembre": 9, "octubre": 10,
            "noviembre": 11, "diciembre": 12,
        }
        date_re = re.compile(
            r"\b(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚáéíóúñÑ]+)\s+de\s+(\d{4})\b",
            re.IGNORECASE,
        )
        combo_re = re.compile(
            r"^\s*(\d{1,2}(?:\s+\d{1,2}){5})\s+C-\d{1,2}\s+R",
            re.IGNORECASE,
        )
        rows = []
        for i, line in enumerate(lines):
            dm = date_re.search(line)
            if not dm:
                continue
            try:
                day = int(dm.group(1))
                month = months.get(dm.group(2).lower())
                y = int(dm.group(3))
                if month is None or y != int(year):
                    continue
                fecha = datetime(y, month, day)
            except Exception:
                continue
            for nxt in lines[i + 1:i + 4]:
                cm = combo_re.match(nxt)
                if not cm:
                    continue
                nums = [int(x) for x in re.findall(r"\d{1,2}", cm.group(1))]
                if validate_draw(nums):
                    rows.append((fecha, sorted(nums)))
                break
        return self._clean_data(rows)

    def _scrape_year_elgordo(self, year):
        url = self.ELGORDO.format(year=year)
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        text_lines = [
            re.sub(r"\s+", " ", x.strip())
            for x in soup.get_text("\n", strip=True).splitlines()
            if x.strip()
        ]
        combo_re = re.compile(
            r"^((?:\d{1,2}-){5}\d{1,2})(?:-C\d{1,2})?(?:\s+-?R\d{1,2})?$"
        )
        for i, line in enumerate(text_lines):
            m = combo_re.match(line)
            if not m:
                continue
            nums = [int(x) for x in m.group(1).split("-")]
            if not validate_draw(nums):
                continue
            fecha = None
            for nxt in text_lines[i + 1:i + 6]:
                fecha = parse_fecha_es(nxt)
                if fecha:
                    break
            if fecha and fecha.year == int(year):
                rows.append((fecha, sorted(nums)))
        for fila in soup.select("div.pseudo-table-row"):
            try:
                fe = fila.select_one(".history-results-date")
                ce = fila.select_one(".history-results-combi")
                if not fe or not ce:
                    continue
                fecha = parse_fecha_es(fe.get_text(" ", strip=True))
                if not fecha:
                    continue
                nums = []
                for sp in ce.select("span"):
                    t = sp.get_text(strip=True)
                    if re.fullmatch(r"\d{1,2}", t):
                        n = int(t)
                        if 1 <= n <= 49:
                            nums.append(n)
                if len(nums) < 6:
                    raw = ce.get_text(" ", strip=True)
                    nums = [
                        int(x) for x in re.findall(r"(?<!\d)\d{1,2}(?!\d)", raw)
                        if 1 <= int(x) <= 49
                    ][:6]
                nums = sorted(set(nums))
                if validate_draw(nums):
                    rows.append((fecha, nums))
            except Exception:
                continue
        return self._clean_data(rows)

    def _scrape_year_loteriadehoy(self, year):
        """Parser de loteriadehoy.

        CORRECCIÓN: en lugar de borrar día/mes/año de la lista de números
        (lo que borraría números ganadores legítimos que coincidan),
        eliminamos el PATRÓN de fecha del texto del bloque antes de
        extraer los números.
        """
        url = self.LOTERIADEHOY.format(year=int(year))
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text("\n", strip=True)
        lines = [re.sub(r"\s+", " ", x.strip()) for x in text.splitlines() if x.strip()]

        rows = []
        for i, line in enumerate(lines):
            fecha = parse_fecha_es(line)
            if not fecha or fecha.year != int(year):
                m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", line)
                if m:
                    try:
                        fecha = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                    except Exception:
                        fecha = None
            if not fecha or fecha.year != int(year):
                continue
            block = " ".join(lines[i:i + 4])
            # Eliminar patrones de fecha del bloque ANTES de extraer números
            block_no_date = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", " ", block)
            block_no_date = re.sub(
                r"\b\d{1,2}\s+de\s+[A-Za-zÁÉÍÓÚáéíóúñÑ]+(?:\s+de\s+\d{4})?\b",
                " ", block_no_date, flags=re.IGNORECASE,
            )
            nums = [
                int(x) for x in re.findall(r"(?<!\d)\d{1,2}(?!\d)", block_no_date)
                if 1 <= int(x) <= 49
            ]
            nums = list(dict.fromkeys(nums))[:6]
            if validate_draw(nums):
                rows.append((fecha, sorted(nums)))
        return self._clean_data(rows)

    def _scrape_year_resultadosbonoloto(self, year):
        url = self.RESULTADOSBONOLOTO.format(year=int(year))
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        date_re = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")

        for fila in soup.select("tr, div.row, li"):
            txt = fila.get_text(" ", strip=True)
            dm = date_re.search(txt)
            if not dm:
                continue
            try:
                day = int(dm.group(1))
                month = int(dm.group(2))
                y = int(dm.group(3))
                if y != int(year):
                    continue
                fecha = datetime(y, month, day)
            except Exception:
                continue

            # Elimina sólo la fecha completa; no borres de nuevo el día/mes,
            # porque esos valores también pueden ser números ganadores.
            txt_no_date = txt[:dm.start()] + " " + txt[dm.end():]

            nums = [
                int(x)
                for x in re.findall(r"(?<!\d)\d{1,2}(?!\d)", txt_no_date)
                if 1 <= int(x) <= 49
            ]
            nums = list(dict.fromkeys(nums))[:6]
            if validate_draw(nums):
                rows.append((fecha, sorted(nums)))

        return self._clean_data(rows)

    @staticmethod
    def _parse_any_date(text):
        """Parsea fechas ISO, DD/MM/YYYY, DD-MM-YYYY y fechas en español."""
        if not text:
            return None
        value = str(text).strip()
        patterns = (
            (r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)", "iso"),
            (r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)", "dmy"),
        )
        for pattern, kind in patterns:
            match = re.search(pattern, value)
            if not match:
                continue
            try:
                if kind == "iso":
                    return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                return datetime(int(match.group(3)), int(match.group(2)), int(match.group(1)))
            except ValueError:
                return None
        month_pattern = (
            r"(?<!\w)(\d{1,2})\s+(?:de\s+)?"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
            r"septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s+(?:de\s+)?(\d{4})(?!\d)"
        )
        match = re.search(month_pattern, value, flags=re.IGNORECASE)
        if match:
            return parse_fecha_es(" ".join(match.groups()))
        return parse_fecha_es(value)

    @staticmethod
    def _strip_date_from_text(text):
        value = str(text or "")
        value = re.sub(r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)", " ", value)
        value = re.sub(r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-]\d{4}(?!\d)", " ", value)
        value = re.sub(
            r"(?<!\w)\d{1,2}\s+(?:de\s+)?(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
            r"septiembre|setiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?\d{4}(?!\d)",
            " ", value, flags=re.IGNORECASE,
        )
        return value

    def _parse_csv_archive(self, content, source_name="archive"):
        """Parsea un histórico CSV/TSV con fecha y al menos seis números.

        No acepta HTML de error como CSV. El parser no confía en el encabezado:
        busca fecha y seis números válidos en cada fila, y deduplica por fecha.
        """
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig", errors="replace")
        content = str(content or "").lstrip("\ufeff")
        if (len(content.strip()) < 200 or
                re.search(r"<(?:!doctype\s+html|html|body)\b", content[:2000], re.I)):
            raise ValueError(f"{source_name}: la descarga no parece un archivo histórico CSV/TSV")

        best_rows = []
        best_delimiter = None
        for delimiter in (",", ";", "\t", "|"):
            try:
                parsed_rows = list(csv.reader(io.StringIO(content), delimiter=delimiter))
            except Exception:
                continue
            rows = []
            for cells in parsed_rows:
                cells = [str(c).strip().strip('"') for c in cells]
                if not cells:
                    continue
                joined = " ".join(cells)
                date_index = None
                fecha = None
                for j, cell in enumerate(cells):
                    dt = self._parse_any_date(cell)
                    if dt is not None:
                        date_index, fecha = j, dt
                        break
                if fecha is None:
                    fecha = self._parse_any_date(joined)
                if fecha is None or not (self.start_year <= fecha.year <= self.end_year):
                    continue

                candidate_cells = cells[date_index + 1:] if date_index is not None else []
                candidate_text = " ".join(candidate_cells)
                token_values = [
                    int(token) for token in re.findall(r"(?<!\d)\d{1,2}(?!\d)", candidate_text)
                    if 1 <= int(token) <= MAX_NUM
                ]
                if len(set(token_values)) < NUMBERS_PER_DRAW:
                    no_date = self._strip_date_from_text(joined)
                    token_values = [
                        int(token) for token in re.findall(r"(?<!\d)\d{1,2}(?!\d)", no_date)
                        if 1 <= int(token) <= MAX_NUM
                    ]
                numbers = []
                for value in token_values:
                    if value not in numbers:
                        numbers.append(value)
                    if len(numbers) == NUMBERS_PER_DRAW:
                        break
                if validate_draw(numbers):
                    rows.append((fecha, sorted(numbers)))
            if len(rows) > len(best_rows):
                best_rows = rows
                best_delimiter = delimiter

        cleaned = self._clean_data(best_rows)
        if len(cleaned) < 50:
            raise ValueError(
                f"{source_name}: sólo se pudieron interpretar {len(cleaned)} sorteos; "
                f"delimitador candidato={best_delimiter!r}"
            )
        log.log(f"Archivo histórico {source_name}: {len(cleaned):,} fechas interpretadas", "DATA")
        return cleaned

    def _scrape_full_archives(self):
        """Intenta recuperar dos archivos históricos y un CSV alternativo completos."""
        # La URL rotulada como CSV en Azar y Suerte actualmente resuelve a HTML
        # en la comprobación externa; no se trata como archivo CSV directo.
        sources = dict(self.LOTOIDEAS_ARCHIVES)
        results = {}
        for name, url in sources.items():
            try:
                response = self._request(url, attempts=2)
                rows = self._parse_csv_archive(response.content, source_name=name)
                results[name] = rows
                self.last_source_counts[name] = len(rows)
            except Exception as exc:
                self.last_source_counts[name] = 0
                log.warn(f"Archivo histórico {name} no disponible/parseable: {type(exc).__name__}: {exc}")
        return results

    def _fetch_official_total(self):
        """Lee el total histórico publicado por SELAE; None si no es verificable."""
        try:
            response = self.session.get(self.SELAE_STATS, timeout=20)
            if response.status_code != 200 or not response.text:
                raise RuntimeError(f"HTTP {response.status_code}")
            text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
            patterns = (
                r"Total histórico de los sorteos celebrados desde que se inicio la venta de este juego\s*,\s*([\d.]+)",
                r"TOTAL\s*:\s*Total histórico[^0-9]{0,250}([\d.]+)",
            )
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    count = int(match.group(1).replace(".", ""))
                    if count >= 1000:
                        log.log(f"Total oficial de sorteos publicado por SELAE: {count:,}", "AUDIT")
                        return count
            raise ValueError("La página oficial no contiene un total histórico reconocible")
        except Exception as exc:
            log.warn(f"No se pudo verificar el total histórico oficial de SELAE: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _selae_date_from_slug(value):
        """Extrae la fecha de una URL oficial tipo bonoloto-resultados-del-23-de-octubre-de-2026."""
        value = unquote(str(value or "")).lower()
        match = re.search(
            r"bonoloto-resultados-del-(\d{1,2})-de-([a-záéíóúñ]+)-de-(\d{4})",
            value, flags=re.IGNORECASE,
        )
        if not match:
            return None
        month = MESES.get(match.group(2).replace("é", "e").replace("á", "a"))
        if month is None:
            month = MESES.get(match.group(2))
        try:
            return datetime(int(match.group(3)), int(month), int(match.group(1))) if month else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _selae_date_appears(text, expected_date):
        """Comprueba que la página consultada menciona la fecha solicitada."""
        body = re.sub(r"\s+", " ", str(text or "").lower())
        month_name = next((name for name, number in MESES.items()
                           if number == expected_date.month and len(name) > 3), None)
        variants = [
            expected_date.strftime("%Y-%m-%d"),
            expected_date.strftime("%d/%m/%Y"),
            expected_date.strftime("%d-%m-%Y"),
        ]
        if month_name:
            variants.extend([
                f"{expected_date.day} de {month_name} de {expected_date.year}",
                f"{expected_date.day:02d} de {month_name} de {expected_date.year}",
            ])
        for variant in variants:
            if variant in body:
                return True
        # Algunas páginas usan el mes abreviado; no se acepta una coincidencia de
        # día aislada porque podría confundirse con números ganadores.
        return False

    @staticmethod
    def _extract_selae_combination(html_or_text):
        """Extrae seis números posteriores a la etiqueta oficial «combinación ganadora».

        No lee el complementario/reintegro como parte de la combinación. Devuelve
        una combinación ordenada, o None si el marcado no permite una extracción
        inequívoca de seis números distintos válidos.
        """
        if not html_or_text:
            return None
        if "<" in str(html_or_text) and ">" in str(html_or_text):
            text = BeautifulSoup(str(html_or_text), "html.parser").get_text(" ", strip=True)
        else:
            text = str(html_or_text)
        text = re.sub(r"\s+", " ", text)
        label = re.search(r"combinaci[oó]n\s+ganadora", text, flags=re.IGNORECASE)
        if not label:
            return None
        tail = text[label.end():label.end() + 700]
        # Forma habitual publicada por SELAE: 05 - 18 - 22 - 34 - 43 - 46.
        sequence = re.search(
            r"(?<!\d)(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*[-–—]\s*"
            r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*[-–—]\s*"
            r"(\d{1,2})\s*[-–—]\s*(\d{1,2})(?!\d)",
            tail,
        )
        if sequence:
            nums = [int(value) for value in sequence.groups()]
            if validate_draw(nums):
                return sorted(nums)
        # Fallback conservador para plantillas que separen los seis números con
        # espacios/etiquetas en vez de guiones. Limita la lectura al bloque anterior
        # al complementario y conserva sólo el primer conjunto inequívoco de seis.
        numbers_label = re.search(
            r"siguientes\s+n[uú]meros\s*:?\s*(.{1,250})",
            tail, flags=re.IGNORECASE,
        )
        if numbers_label:
            block = re.split(
                r"complementario|reintegro|categor[ií]as|otros resultados",
                numbers_label.group(1), maxsplit=1, flags=re.IGNORECASE,
            )[0]
            nums = []
            for token in re.findall(r"(?<!\d)\d{1,2}(?!\d)", block):
                number = int(token)
                if 1 <= number <= MAX_NUM and number not in nums:
                    nums.append(number)
                if len(nums) == NUMBERS_PER_DRAW:
                    break
            if validate_draw(nums):
                return sorted(nums)
        return None

    def _parse_selae_result_article(self, url, expected_date, html):
        """Valida fecha y combinación de una ficha oficial de resultados SELAE."""
        soup = BeautifulSoup(html or "", "html.parser")
        text = soup.get_text(" ", strip=True)
        # La fecha de la URL se toma como dato esperado, pero se exige también una
        # coincidencia explícita en el contenido antes de aceptar la combinación.
        if not self._selae_date_appears(text, expected_date):
            log.warn(
                f"SELAE: ficha no verificable para {expected_date:%Y-%m-%d}; "
                "el contenido no confirma la fecha solicitada."
            )
            return None
        combo = self._extract_selae_combination(text)
        if not validate_draw(combo):
            log.warn(f"SELAE: no se pudieron extraer seis números válidos para {expected_date:%Y-%m-%d}.")
            return None
        return expected_date, combo

    def _scrape_selae_recent_results(self):
        """Descarga resultados recientes desde SELAE y comprueba la fecha en cada ficha.

        No intenta descargar miles de fichas oficiales de 1988 en adelante. La fuente
        oficial se usa para confirmar los resultados recientes; el histórico largo se
        completa con los archivos y scrapers multifuente existentes y con el contador
        total oficial, cuando éste responde.
        """
        self.selae_results_page_ok = False
        self.selae_results_page_latest_date = None
        found_links = {}
        try:
            response = self.session.get(self.SELAE_RESULTS_LIST, timeout=18)
            if response.status_code != 200 or not response.text:
                raise RuntimeError(f"HTTP {response.status_code}")
            soup = BeautifulSoup(response.text, "html.parser")
            for anchor in soup.select("a[href]"):
                href = urljoin(self.SELAE_RESULTS_LIST, anchor.get("href", ""))
                if "bonoloto-resultados-del-" not in href.lower():
                    continue
                fecha = self._selae_date_from_slug(href)
                if fecha is None:
                    fecha = self._parse_any_date(anchor.get_text(" ", strip=True))
                if fecha is None or fecha.year < self.start_year or fecha.year > _madrid_now().year:
                    continue
                found_links[fecha.date().isoformat()] = (fecha, href)
            if not found_links:
                raise RuntimeError("la página oficial no contiene enlaces de resultados con fecha reconocible")
            self.selae_results_page_ok = True
            ordered = sorted(found_links.values(), key=lambda item: item[0], reverse=True)[:40]
            self.selae_results_page_latest_date = max(item[0].date() for item in ordered)
        except Exception as exc:
            log.warn(f"SELAE: no se pudo recuperar el índice de resultados recientes: {type(exc).__name__}: {exc}")
            self.last_source_counts["selae_results_index_error"] = f"{type(exc).__name__}: {exc}"
            return []

        rows = []
        errors = []
        # Limitar concurrencia con SELAE para evitar bombardear el sitio oficial.
        def fetch_one(item):
            fecha, url = item
            response = self.session.get(url, timeout=18)
            if response.status_code != 200 or not response.text:
                raise RuntimeError(f"HTTP {response.status_code}")
            row = self._parse_selae_result_article(url, fecha, response.text)
            if row is None:
                raise ValueError("ficha oficial sin combinación verificable")
            return row

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_one, item): item for item in ordered}
            for future in as_completed(futures):
                fecha, url = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    errors.append({"date": fecha.date().isoformat(), "url": url,
                                   "error": f"{type(exc).__name__}: {exc}"})
        rows = self._clean_data(rows)
        self.official_selae_results = {
            fecha.date().isoformat(): (fecha, sorted(draw)) for fecha, draw in rows
        }
        self.last_source_counts["selae_result_rows"] = len(rows)
        self.last_source_counts["selae_result_links_seen"] = len(ordered)
        self.last_source_counts["selae_result_page_errors"] = len(errors)
        if errors:
            log.warn(f"SELAE: {len(errors)} fichas oficiales recientes no se pudieron verificar.")
        log.log(f"SELAE: {len(rows)} resultados oficiales recientes comprobados por fecha", "DATA")
        return rows

    @staticmethod
    def _retry_delay_from_response(response, attempt):
        """Calcula un backoff acotado y respeta Retry-After si está presente."""
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return min(10.0, max(0.0, float(retry_after)))
            except (TypeError, ValueError):
                try:
                    when = parsedate_to_datetime(str(retry_after))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    seconds = (when.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
                    return min(10.0, max(0.0, seconds))
                except Exception:
                    pass
        return min(5.0, float(2 ** attempt))

    def _fetch_selae_result_for_date(self, expected_date):
        """Consulta el resultado oficial por fecha; reintenta errores transitorios."""
        params = {"game_id": "BONO", "fecha_sorteo": expected_date.strftime("%Y%m%d")}
        last_error = None
        # Tres intentos como máximo. Retry-After de SELAE se respeta con tope de 10 s;
        # los fallos de red aplican backoff exponencial acotado. No se reintenta una
        # respuesta 200 que no confirma fecha/combinación: eso sería un resultado ausente.
        for attempt in range(3):
            response = None
            try:
                response = self.session.get(self.SELAE_RESULT_WIDGET, params=params, timeout=12)
                status = int(getattr(response, "status_code", 0))
                if status in (429, 500, 502, 503, 504):
                    if attempt < 2:
                        delay = self._retry_delay_from_response(response, attempt)
                        log.warn(
                            f"SELAE: HTTP {status} para {expected_date:%Y-%m-%d}; "
                            f"reintento {attempt + 2}/3 en {delay:.1f}s."
                        )
                        time.sleep(delay)
                        continue
                    last_error = RuntimeError(f"HTTP {status}")
                    break
                if status != 200 or not getattr(response, "text", ""):
                    log.warn(
                        f"SELAE: HTTP {status} o respuesta vacía para {expected_date:%Y-%m-%d}; "
                        "no se aceptará como resultado oficial."
                    )
                    return None
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(" ", strip=True)
                # Una plantilla vacía/fecha distinta no valida un resultado. Puede ser
                # que el sorteo aún no se haya publicado; las fuentes secundarias lo
                # intentarán después, y el control de cobertura decidirá si abortar.
                if not self._selae_date_appears(text, expected_date):
                    return None
                combo = self._extract_selae_combination(text)
                if not validate_draw(combo):
                    return None
                log.ok(f"SELAE: recuperado sorteo faltante {expected_date:%d/%m/%Y} vía fichero oficial")
                return expected_date, combo
            except Exception as exc:
                last_error = exc
                if attempt >= 2:
                    break
                delay = self._retry_delay_from_response(response, attempt) if response is not None else min(5.0, float(2 ** attempt))
                log.warn(
                    f"SELAE: consulta puntual {expected_date:%Y-%m-%d} fallida "
                    f"({type(exc).__name__}: {exc}); reintento {attempt + 2}/3 en {delay:.1f}s."
                )
                time.sleep(delay)
        log.warn(
            f"SELAE: consulta puntual fallida para {expected_date:%Y-%m-%d} tras 3 intentos: "
            f"{type(last_error).__name__}: {last_error}"
        )
        return None

    def _apply_selae_overrides(self, data):
        """Hace prevalecer una ficha SELAE validada frente a resultados secundarios discrepantes."""
        by_date = {fecha.date().isoformat(): (fecha, sorted(draw)) for fecha, draw in data or []}
        override_count = 0
        conflicts = []
        for key, (fecha, draw) in self.official_selae_results.items():
            existing = by_date.get(key)
            if existing is not None and tuple(existing[1]) != tuple(draw):
                conflicts.append({"date": key, "previous_draw": list(existing[1]),
                                 "selae_draw": list(draw), "resolution": "SELAE official detail page"})
            if existing is None or tuple(existing[1]) != tuple(draw):
                override_count += 1
            by_date[key] = (fecha, list(draw))
        if override_count:
            log.log(f"SELAE: {override_count} fechas oficiales insertadas/corregidas en la fusión", "AUDIT")
        self.last_source_counts["selae_dates_overrode_or_added"] = override_count
        self.last_source_counts["selae_conflicts_resolved_by_official_page"] = len(conflicts)
        if conflicts:
            self.last_source_counts["selae_conflicts_sample"] = conflicts[:50]
        return sorted(by_date.values(), key=lambda item: item[0])

    @staticmethod
    def _recent_coverage_cutoff(selae_page_ok, selae_latest_date):
        """Última fecha cuyo sorteo debe constar según el momento de ejecución.

        SELAE celebra actualmente Bonoloto todos los días de la semana, incluidos
        sábados y domingos. Antes de las 21:30 (hora peninsular) se exige hasta ayer;
        desde la hora oficial del sorteo se exige el día actual. Si el índice SELAE
        trae una fecha aún más reciente, también se toma como ancla. Si el sorteo de
        hoy aún no se ha publicado ni puede recuperarse en fuentes secundarias, Fuxing
        no debe ocultar el retraso: lo notificará y detendrá el entrenamiento.
        """
        now = _madrid_now()
        clock_cutoff = now.date() if (now.hour, now.minute) >= (21, 30) else now.date() - timedelta(days=1)
        if selae_page_ok and selae_latest_date is not None:
            return max(clock_cutoff, selae_latest_date)
        return clock_cutoff

    def _ensure_recent_coverage(self, data):
        """Repara fechas recientes ausentes con concurrencia limitada y devuelve huecos.

        Se comprueban fechas naturales en la ventana actual porque SELAE celebra Bonoloto
        diariamente, también los domingos. Las consultas puntuales se agrupan en lotes de
        dos para limitar la carga, y el propio fetch aplica reintentos/backoff en respuestas
        transitorias. No se omite un día solo por ser domingo.
        """
        cutoff = self._recent_coverage_cutoff(
            self.selae_results_page_ok, self.selae_results_page_latest_date
        )
        start = cutoff - timedelta(days=29)
        by_date = {fecha.date().isoformat(): (fecha, sorted(draw)) for fecha, draw in data or []}
        expected = []
        cursor = start
        while cursor <= cutoff:
            expected.append(cursor)
            cursor += timedelta(days=1)
        missing = [day for day in expected if day.isoformat() not in by_date]
        if missing:
            log.warn(
                f"Histórico reciente: faltan {len(missing)} fechas en la ventana diaria "
                f"{start:%d/%m/%Y}–{cutoff:%d/%m/%Y}; se intentará SELAE y después las fuentes históricas."
            )
            # Primero se utiliza la vía oficial por fecha. Como máximo se consultan
            # las fechas realmente ausentes, no una petición por cada día desde 1988.
            # El límite de dos workers y una pausa entre lotes evitan una ráfaga larga.
            official_additions = []
            batch_size = 2
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                for batch_start in range(0, len(missing), batch_size):
                    batch = missing[batch_start:batch_start + batch_size]
                    futures = {
                        executor.submit(
                            self._fetch_selae_result_for_date,
                            datetime(day.year, day.month, day.day),
                        ): day
                        for day in batch
                    }
                    for future in as_completed(futures):
                        day = futures[future]
                        try:
                            row = future.result()
                        except Exception as exc:
                            log.warn(
                                f"SELAE: reparación de {day:%Y-%m-%d} fallida: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            row = None
                        if row is not None:
                            key = row[0].date().isoformat()
                            self.official_selae_results[key] = row
                            official_additions.append(row)
                    if batch_start + batch_size < len(missing):
                        time.sleep(0.5)
            if official_additions:
                data = self._merge(data, official_additions)
                data = self._apply_selae_overrides(data)
                by_date = {fecha.date().isoformat(): (fecha, sorted(draw)) for fecha, draw in data}
            missing = [day for day in expected if day.isoformat() not in by_date]

        if missing:
            # Reintento del año afectado en las fuentes históricas que ya forman parte
            # de Fuxing. Es una recuperación de datos; no modifica el motor predictivo.
            years = sorted({day.year for day in missing})
            secondary_rows = []
            for year in years:
                try:
                    secondary_rows.extend(self.scrape_year(year))
                except Exception as exc:
                    log.warn(f"Reintento histórico del año {year} fallido: {type(exc).__name__}: {exc}")
            if secondary_rows:
                data = self._merge(data, secondary_rows)
                data = self._apply_selae_overrides(data)
                by_date = {fecha.date().isoformat(): (fecha, sorted(draw)) for fecha, draw in data}
            missing = [day for day in expected if day.isoformat() not in by_date]

        self.last_source_counts["recent_coverage_window_start"] = start.isoformat()
        self.last_source_counts["recent_coverage_cutoff"] = cutoff.isoformat()
        self.last_source_counts["recent_expected_dates"] = len(expected)
        self.last_source_counts["recent_missing_dates"] = [day.isoformat() for day in missing]
        self.last_source_counts["recent_window_complete"] = not bool(missing)
        if missing:
            log.error(
                "No se puede garantizar que el histórico esté al día. Fechas recientes todavía ausentes: "
                + ", ".join(day.isoformat() for day in missing)
            )
        else:
            log.ok(f"Cobertura reciente comprobada: todos los sorteos diarios hasta {cutoff:%d/%m/%Y} están presentes.")
        return data, missing

    def _fallback_without_selae(self, force_refresh=False):
        """Recupera el histórico con fuentes secundarias si SELAE no aporta contador."""
        old = self._load_sqlite() if not force_refresh else []
        archive_sets = self._scrape_full_archives()
        archive_rows = [row for rows in archive_sets.values() for row in (rows or [])]
        recent_rows = []
        for year in sorted({_madrid_now().year, _madrid_now().year - 1}):
            if not self.start_year <= year <= self.end_year:
                continue
            try:
                rows = self.scrape_year(year)
                recent_rows.extend(rows)
                log.log(f"Respaldo histórico {year}: {len(rows)} sorteos recuperados", "DATA")
            except Exception as exc:
                log.warn(f"Respaldo histórico {year} fallido: {type(exc).__name__}: {exc}")

        source_sets = [old, archive_rows, recent_rows, list(self.official_selae_results.values())]
        data = self._merge(*source_sets)
        data = self._apply_selae_overrides(data)
        data, missing_recent = self._ensure_recent_coverage(data)

        structural = self._structural_checks(data) if data else {"all_pass": False, "reason": "empty_history"}
        if not structural.get("all_pass", False):
            # Si la caché/archivos no superan controles básicos, reintentar todas las
            # anualidades en fuentes secundarias en vez de confiar en datos parciales.
            log.warn("Respaldo sin SELAE no supera validación estructural; se reintentan todas las anualidades.")
            all_year_rows = []
            years = list(range(self.start_year, self.end_year + 1))
            with ThreadPoolExecutor(max_workers=min(8, len(years) or 1)) as executor:
                futures = {executor.submit(self.scrape_year, year): year for year in years}
                for future in as_completed(futures):
                    year = futures[future]
                    try:
                        all_year_rows.extend(future.result())
                    except Exception as exc:
                        log.warn(f"Reintento anual {year} fallido: {type(exc).__name__}: {exc}")
            data = self._merge(old, archive_rows, all_year_rows, list(self.official_selae_results.values()))
            data = self._apply_selae_overrides(data)
            data, missing_recent = self._ensure_recent_coverage(data)
            structural = self._structural_checks(data) if data else {"all_pass": False, "reason": "empty_history"}

        if not data or missing_recent or not structural.get("all_pass", False):
            if data:
                self._save_candidate(data)
            failed = [key for key, value in structural.items() if isinstance(value, bool) and value is False]
            raise RuntimeError(
                "SELAE no ha proporcionado un contador verificable y el histórico alternativo no supera "
                "todos los controles. No se ejecutan predicciones con datos posiblemente atrasados. "
                f"Fallo estructural={failed}; huecos recientes={[d.isoformat() for d in missing_recent]}; "
                f"detalles={structural}. Se conserva la caché operativa anterior y, si se pudo generar, "
                "la descarga candidata queda separada para auditoría."
            )

        self.last_source_counts.update({
            "fallback_mode": "without_selae_structurally_checked_with_recent_gap_repair",
            "operational_cache_rows": len(old),
            "archive_source_rows": len(archive_rows),
            "recent_annual_source_rows": len(recent_rows),
            "unique_dates_used": len(data),
            "annual_scrape_performed": True,
            "source_comparison_performed": True,
            "official_total_verification": False,
        })
        audit = {
            "generated": _madrid_now().isoformat(timespec="seconds"),
            "total_draws": len(data),
            "unique_dates": len({fecha.date().isoformat() for fecha, _ in data}),
            "duplicate_date_rows": len(data) - len({fecha.date().isoformat() for fecha, _ in data}),
            "min_date": data[0][0].strftime("%Y-%m-%d"),
            "max_date": data[-1][0].strftime("%Y-%m-%d"),
            "expected_start": self.start_year,
            "expected_end": self.end_year,
            "official_expected_total": None,
            "official_count_difference": None,
            "completeness_verified_by_official_count": False,
            "verification_mode": "fallback_structural_plus_recent_daily_coverage",
            "selae_available": bool(self.selae_results_page_ok or self.official_selae_results),
            "selae_total_available": False,
            "selae_recent_results_available": bool(self.selae_results_page_ok or self.official_selae_results),
            "selae_recent_results_rows": len(self.official_selae_results),
            "source_counts": dict(self.last_source_counts),
            "source_comparison_performed": True,
            "annual_scrape_performed": True,
            "recent_coverage_cutoff": self.last_source_counts.get("recent_coverage_cutoff"),
            "recent_window_complete": True,
            "structural_checks": structural,
            "warning": (
                "SELAE no ha facilitado el contador total. El histórico se ha aceptado tras controles "
                "estructurales y verificación de las fechas diarias recientes, pero no puede certificarse "
                "el recuento completo desde 1988 contra el contador oficial."
            ),
        }
        _safe_json_write(AUDIT_FILE, audit)
        self.last_history_audit = audit
        self._save_all(data)
        try:
            CANDIDATE_CACHE_FILE.unlink(missing_ok=True)
        except Exception as exc:
            log.warn(f"No se pudo limpiar caché candidata tras aceptar el respaldo: {exc}")
        log.warn(
            f"SELAE sin contador: se continúa con {len(data):,} fechas mediante histórico alternativo; "
            f"último sorteo={data[-1][0]:%d/%m/%Y}. La completitud global no está certificada por el contador oficial."
        )
        return data

    def _scrape_year_laweb(self, year):
        """Obtiene sorteos recientes de la tabla histórica de La Web de la Primitiva."""
        response = self._request(self.LAWEB_HISTORICO, attempts=2)
        soup = BeautifulSoup(response.text, "html.parser")
        lines = [re.sub(r"\s+", " ", x.strip()) for x in soup.get_text("\n", strip=True).splitlines() if x.strip()]
        rows = []
        date_re = re.compile(r"(?<!\d)(\d{1,2})[-/](\d{1,2})[-/](\d{4})(?!\d)")
        for i, line in enumerate(lines):
            match = date_re.search(line)
            if not match:
                continue
            try:
                fecha = datetime(int(match.group(3)), int(match.group(2)), int(match.group(1)))
            except ValueError:
                continue
            if fecha.year != int(year):
                continue
            block_lines = []
            for next_line in lines[i:i + 4]:
                if next_line != line and date_re.search(next_line):
                    break
                block_lines.append(next_line)
            block = self._strip_date_from_text(" ".join(block_lines))
            nums = []
            for token in re.findall(r"(?<!\d)\d{1,2}(?!\d)", block):
                value = int(token)
                if 1 <= value <= MAX_NUM and value not in nums:
                    nums.append(value)
                if len(nums) == NUMBERS_PER_DRAW:
                    break
            if validate_draw(nums):
                rows.append((fecha, sorted(nums)))
        return self._clean_data(rows)

    def scrape_year(self, year):
        year = int(year)
        results = {}

        def try_source(name, fn):
            try:
                rows = fn(year)
                results[name] = rows
                if rows:
                    log.log(f"Año {year} · {name}: {len(rows)} sorteos válidos", "DATA")
            except Exception as e:
                results[name] = []
                log.warn(f"Año {year} · {name}: {type(e).__name__}: {e}")

        source_functions = [
            ("lotocrack", self._scrape_year_lotocrack),
            ("elgordo", self._scrape_year_elgordo),
            ("loteriadehoy", self._scrape_year_loteriadehoy),
            ("resultadosbonoloto", self._scrape_year_resultadosbonoloto),
        ]
        if year >= _madrid_now().year - 1:
            source_functions.append(("laweb_primitiva", self._scrape_year_laweb))
        with ThreadPoolExecutor(max_workers=len(source_functions)) as ex:
            futs = [ex.submit(try_source, name, fn) for name, fn in source_functions]
            for f in as_completed(futs):
                f.result()

        valid_sources = {}
        for name, rows in results.items():
            # La tabla histórica de La Web muestra una ventana reciente (no un
            # año completo); se acepta desde una fila para contrastar el tramo
            # común con las fuentes anuales. Las demás fuentes deben cubrir ≥50.
            minimum_rows = 1 if name == "laweb_primitiva" else 50
            if len(rows) >= minimum_rows:
                valid_sources[name] = rows
            elif rows:
                log.warn(f"Año {year} · {name}: solo {len(rows)} sorteos, se descarta")

        if not valid_sources:
            log.warn(f"Año {year}: ninguna fuente fiable", "WARNING")
            return []

        # Audita por fecha todas las fuentes anuales que superaron el filtro mínimo.
        # Mayoría estricta resuelve una discrepancia; un empate queda sin resolver.
        source_maps = {
            name: {fecha.date().isoformat(): tuple(draw) for fecha, draw in rows}
            for name, rows in valid_sources.items()
        }
        all_dates = sorted(set().union(*(set(v.keys()) for v in source_maps.values())))
        for date_key in all_dates:
            votes = Counter(source_map[date_key] for source_map in source_maps.values() if date_key in source_map)
            if len(votes) <= 1:
                continue
            ranked = votes.most_common()
            unresolved = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
            detail = {
                "year": year,
                "date": date_key,
                "source_votes": {" ".join(map(str, draw)): count for draw, count in votes.items()},
                "status": "unresolved_tie" if unresolved else "resolved_by_majority",
                "winning_draw": None if unresolved else list(ranked[0][0]),
            }
            with self._source_audit_lock:
                self.last_annual_source_conflicts.append(detail)
                if unresolved:
                    self.last_annual_unresolved_dates.append(date_key)
            if unresolved:
                log.warn(f"Año {year}: discrepancia anual sin mayoría en {date_key}; no se decide esa combinación.")

        tagged_rows = []
        for name, rows in valid_sources.items():
            for fecha, draw in rows:
                tagged_rows.append((fecha, draw))
        merged = self._clean_data(tagged_rows)

        log.log(
            f"Año {year}: fusión final {len(merged)} sorteos "
            f"(fuentes usadas: {list(valid_sources.keys())})",
            "DATA",
        )
        return merged

    def _load_candidate(self):
        """Carga datos de una descarga incompleta sin hacerlos pasar por histórico validado."""
        if not CANDIDATE_CACHE_FILE.exists():
            return []
        try:
            with open(CANDIDATE_CACHE_FILE, "rb") as handle:
                rows = pickle.load(handle)
            cleaned = self._clean_data(rows)
            log.log(f"Caché candidata (NO validada): {len(cleaned):,} fechas", "DATA")
            return cleaned
        except Exception as exc:
            log.warn(f"No se pudo reutilizar la caché candidata: {type(exc).__name__}: {exc}")
            return []

    def _save_candidate(self, data):
        """Guarda una descarga parcial en staging; nunca sustituye el histórico operativo."""
        path = CANDIDATE_CACHE_FILE
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            self._backup_file(path)
            os.replace(tmp, path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def _structural_checks(self, merged):
        """Comprueba coherencia interna; no acredita completitud ni veracidad oficial."""
        checks = {}
        if not merged:
            return {"all_pass": False, "reason": "empty_history", "checks": checks}

        try:
            dates_are_datetime = all(isinstance(fecha, datetime) for fecha, _ in merged)
        except Exception:
            dates_are_datetime = False
        checks["all_dates_are_datetime"] = dates_are_datetime
        checks["no_unresolved_source_conflicts"] = not bool(self.last_source_conflicts)

        try:
            checks["all_draws_valid_6_unique_in_range"] = all(
                validate_draw(list(draw)) for _, draw in merged
            )
        except Exception:
            checks["all_draws_valid_6_unique_in_range"] = False

        if not dates_are_datetime:
            checks.update({
                "no_duplicate_dates": False,
                "duplicate_dates_count": None,
                "chronological_order": False,
                "first_date_correct": False,
                "first_date_observed": None,
                "last_date_recent": False,
                "last_date_observed": None,
                "days_since_last_draw": None,
                "count_plausible": 8500 <= len(merged) <= 9800,
                "count_observed": len(merged),
                "max_gap_days": None,
                "no_huge_gaps": False,
            })
        else:
            dates = [fecha.date() for fecha, _ in merged]
            date_keys = [fecha.isoformat() for fecha in dates]
            duplicate_count = len(date_keys) - len(set(date_keys))
            checks["no_duplicate_dates"] = duplicate_count == 0
            checks["duplicate_dates_count"] = duplicate_count
            checks["chronological_order"] = dates == sorted(dates)
            first_date, last_date = dates[0], dates[-1]
            checks["first_date_correct"] = first_date == datetime(1988, 2, 28).date()
            checks["first_date_observed"] = first_date.isoformat()
            days_since_last = (_madrid_now().date() - last_date).days
            checks["last_date_recent"] = 0 <= days_since_last <= 3
            checks["last_date_observed"] = last_date.isoformat()
            checks["days_since_last_draw"] = days_since_last
            checks["count_plausible"] = 8500 <= len(merged) <= 9800
            checks["count_observed"] = len(merged)
            ordered_dates = sorted(dates)
            biggest_gap = max(
                ((ordered_dates[i] - ordered_dates[i - 1]).days for i in range(1, len(ordered_dates))),
                default=0,
            )
            checks["max_gap_days"] = biggest_gap

            # Excepción histórica: suspensión de sorteos por COVID-19.
            huecos = [
                (
                    ordered_dates[i - 1],
                    ordered_dates[i],
                    (ordered_dates[i] - ordered_dates[i - 1]).days,
                )
                for i in range(1, len(ordered_dates))
            ]

            hueco_covid_valido = (
                any(
                    anterior == __import__("datetime").date(2020, 3, 14)
                    and siguiente == __import__("datetime").date(2020, 5, 25)
                    and dias == 72
                    for anterior, siguiente, dias in huecos
                )
            )

            otros_huecos_grandes = [
                (anterior, siguiente, dias)
                for anterior, siguiente, dias in huecos
                if dias > 30
                and not (
                    anterior == __import__("datetime").date(2020, 3, 14)
                    and siguiente == __import__("datetime").date(2020, 5, 25)
                    and dias == 72
                )
            ]

            checks["no_huge_gaps"] = (
                not otros_huecos_grandes
                and (
                    biggest_gap <= 30
                    or hueco_covid_valido
                )
            )

        required = (
            "all_dates_are_datetime", "no_unresolved_source_conflicts",
            "all_draws_valid_6_unique_in_range", "no_duplicate_dates",
            "chronological_order", "first_date_correct",
            "last_date_recent", "count_plausible", "no_huge_gaps",
        )
        checks["all_pass"] = all(checks.get(key) is True for key in required)
        return checks

    def scrape_with_cache(self, force_refresh=False):
        """Actualiza el histórico y aborta antes de predecir si no cuadra con SELAE.

        No usa días naturales como estimación de sorteos históricos. Primero combina
        caché y archivos completos, luego refresca los dos últimos años y, si persiste
        una diferencia importante respecto de SELAE, consulta todos los años.
        """
        self.last_source_counts = {}
        self.last_source_conflicts = []
        self.last_annual_source_conflicts = []
        self.last_annual_unresolved_dates = []
        self.last_candidate_cache_conflicts = []

        # SELAE se consulta primero tanto para el contador global como para las
        # combinaciones oficiales recientes. El índice puede funcionar aunque falle
        # la página de estadísticas, y viceversa.
        self.official_selae_results = {}
        self.selae_results_page_ok = False
        self.selae_results_page_latest_date = None
        official_recent_rows = self._scrape_selae_recent_results()
        official_total = self._fetch_official_total()
        self.official_expected_total = official_total
        self.last_source_counts["selae_total_available"] = official_total is not None
        self.last_source_counts["selae_recent_index_available"] = self.selae_results_page_ok
        if official_total is None:
            log.warn(
                "SELAE no ha proporcionado un contador verificable. Se activa la recuperación "
                "multifuente; solo se continuará si el histórico supera los controles estructurales "
                "y no quedan fechas recientes ausentes."
            )
            return self._fallback_without_selae(force_refresh=force_refresh)

        old = []
        had_operational_cache = False
        operational_cache_unique_dates = 0
        if not force_refresh:
            old = self._load_sqlite()
            had_operational_cache = bool(old)
            operational_cache_unique_dates = len({d.date() for d, _ in old})
            if old:
                log.log(f"Caché SQLite: {len(old):,} sorteos válidos", "DATA")
            candidate = self._load_candidate()
            if candidate:
                # Never let an unvalidated staging file outvote an operational row.
                # It may fill a date absent from SQLite, but a different combination
                # for an already-present date is logged and excluded from the merge.
                old_by_date = {d.date().isoformat(): (d, tuple(draw)) for d, draw in old}
                candidate_additions = []
                for fecha, draw in candidate:
                    key = fecha.date().isoformat()
                    existing = old_by_date.get(key)
                    if existing is None:
                        candidate_additions.append((fecha, list(draw)))
                    elif existing[1] != tuple(draw):
                        self.last_candidate_cache_conflicts.append({
                            "date": key,
                            "operational_draw": list(existing[1]),
                            "candidate_draw": list(draw),
                            "handling": "operational cache retained; candidate row excluded pending cross-check",
                        })
                if candidate_additions:
                    old = sorted(old + candidate_additions, key=lambda x: x[0])
                if self.last_candidate_cache_conflicts:
                    log.warn(
                        f"Caché candidata: {len(self.last_candidate_cache_conflicts)} conflictos con SQLite; "
                        "se conserva la combinación operativa hasta contrastarla con otras fuentes."
                    )

        # official_total ya se ha consultado al inicio. Si llegamos aquí, SELAE
        # proporcionó un contador y se conserva el flujo normal de verificación.
        archive_sets = self._scrape_full_archives()
        archive_rows = []
        for source_rows in archive_sets.values():
            archive_rows.extend(source_rows)
        current_year = min(self.end_year, _madrid_now().year)
        recent_years = {y for y in (current_year, current_year - 1)
                        if self.start_year <= y <= self.end_year}
        # Use the count of the operational SQLite cache BEFORE adding unvalidated
        # staging rows; otherwise a partial candidate could falsely satisfy the
        # official count and suppress the full initial audit.
        cached_unique_dates = operational_cache_unique_dates
        # A first install, or when the operational cache's count fails the official
        # total, audit every year through the annual sources rather than trusting a
        # full-archive file merely because its total happens to match. Subsequent
        # validated daily runs only refresh current/previous year unless forced.
        first_or_incomplete_cache = (not had_operational_cache or
                                     (official_total is not None and cached_unique_dates != official_total))
        years_to_fetch = (
            set(range(self.start_year, self.end_year + 1))
            if force_refresh or first_or_incomplete_cache else set(recent_years)
        )

        # Una primera instalación exige contrastar todo el intervalo con las fuentes
        # anuales; las ejecuciones siguientes se centran en los años recientes salvo
        # que el contador oficial revele que la caché está incompleta.
        def fetch_years(year_set, label):
            result = []
            ordered = sorted(year_set)
            if not ordered:
                return result
            log.log(f"{label}: {len(ordered)} años ({ordered[0]}-{ordered[-1]})", "NETWORK")
            done = 0
            with ThreadPoolExecutor(max_workers=min(8, len(ordered))) as ex:
                futs = {ex.submit(self.scrape_year, year): year for year in ordered}
                for future in as_completed(futs):
                    year = futs[future]
                    done += 1
                    try:
                        result.extend(future.result())
                    except Exception as exc:
                        log.warn(f"Año {year}: error final {type(exc).__name__}: {exc}")
                    log.progress(2.0 + 18.0 * done / max(1, len(ordered)),
                                 f"{label}: {done}/{len(ordered)} años · último={year}")
            return result

        fresh = fetch_years(
            years_to_fetch,
            "Auditoría anual completa" if (force_refresh or first_or_incomplete_cache) else "Actualización reciente",
        )
        merged = self._merge(old, archive_rows, fresh, official_recent_rows)
        merged = self._apply_selae_overrides(merged)

        # Si no cuadra el contador, reintenta los años históricos que aún no se han
        # consultado. La comprobación exacta se hace al final, no por días naturales.
        if not force_refresh and official_total is not None and len(merged) != official_total:
            all_years = set(range(self.start_year, self.end_year + 1))
            remaining_years = all_years - years_to_fetch
            log.warn(
                f"Preauditoría tras archivos y años recientes: {len(merged):,} fechas únicas; "
                f"SELAE publica {official_total:,}. Se consultan los años restantes."
            )
            fresh.extend(fetch_years(remaining_years, "Recuperación histórica"))
            merged = self._merge(old, archive_rows, fresh, official_recent_rows)
            merged = self._apply_selae_overrides(merged)

        if len(merged) < len(old):
            log.warn("La fusión reduciría el histórico local; se conserva la caché anterior.")
            merged = old

        # Contrasta discrepancias entre cada par de archivos históricos por fecha.
        # Una discrepancia no se oculta: se conserva para la auditoría del conjunto.
        source_date_draws = {
            name: {d.date().isoformat(): tuple(draw) for d, draw in rows}
            for name, rows in archive_sets.items()
        }
        conflict_dates = set()
        source_conflict_details = []
        names = list(source_date_draws)
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                map_a, map_b = source_date_draws[name_a], source_date_draws[name_b]
                for date_key in sorted(map_a.keys() & map_b.keys()):
                    if map_a[date_key] != map_b[date_key]:
                        conflict_dates.add(date_key)
                        source_conflict_details.append({
                            "date": date_key, "source_a": name_a, "draw_a": list(map_a[date_key]),
                            "source_b": name_b, "draw_b": list(map_b[date_key]),
                        })
        # Una discrepancia entre archivos se considera resuelta sólo si el resultado
        # de la consulta anual (que combina páginas por año) coincide con uno de los
        # dos candidatos. Si no existe ese desempate, no se sustituye el histórico.
        annual_by_date = {d.date().isoformat(): tuple(draw) for d, draw in fresh}
        unresolved_details = []
        resolved_dates = set()
        for detail in source_conflict_details:
            a = tuple(detail["draw_a"])
            b = tuple(detail["draw_b"])
            annual = annual_by_date.get(detail["date"])
            if annual is not None and annual in (a, b):
                detail["resolution_status"] = "corroborated_by_annual_scrape"
                detail["annual_scrape_draw"] = list(annual)
                resolved_dates.add(detail["date"])
            else:
                detail["resolution_status"] = "unresolved"
                unresolved_details.append(detail)
        unresolved_archive_dates = {detail["date"] for detail in unresolved_details}
        unresolved_annual_dates = set(self.last_annual_unresolved_dates)
        self.last_source_conflicts = sorted(unresolved_archive_dates | unresolved_annual_dates)
        self.last_source_counts["archive_conflict_dates_total"] = len(conflict_dates)
        self.last_source_counts["archive_conflict_dates_resolved_by_annual_scrape"] = len(resolved_dates)
        self.last_source_counts["archive_conflict_dates_unresolved"] = len(unresolved_archive_dates)
        self.last_source_counts["annual_source_conflicts_total"] = len(self.last_annual_source_conflicts)
        self.last_source_counts["annual_source_conflict_dates_unresolved"] = len(unresolved_annual_dates)
        self.last_source_counts["merged_unique_dates"] = len(merged)
        self.last_source_counts["old_cache_dates"] = len(old)
        self.last_source_counts["annual_fetch_dates"] = len(fresh)
        self.last_source_counts["candidate_cache_conflicts"] = len(self.last_candidate_cache_conflicts)
        self.last_source_counts["full_annual_audit_requested"] = bool(
            force_refresh or first_or_incomplete_cache
        )

        # Comprueba el tramo reciente sorteo a sorteo antes de tocar la caché operativa.
        # Si SELAE y las fuentes secundarias dejan fechas ausentes, se aborta antes del
        # entrenamiento para no producir predicciones con un histórico atrasado.
        merged, missing_recent = self._ensure_recent_coverage(merged)
        merged = self._apply_selae_overrides(merged)
        if missing_recent:
            self._save_candidate(merged)
            raise RuntimeError(
                "HISTÓRICO ATRASADO: no se pudieron recuperar todos los sorteos diarios "
                f"esperados en la ventana reciente. Fechas ausentes={[d.isoformat() for d in missing_recent]}. "
                "No se sustituye el histórico operativo ni se ejecuta el motor predictivo."
            )

        # Valida antes de reemplazar los ficheros operativos. Si el contador no coincide,
        # el archivo candidato queda separado y se conservan las copias operativas.
        self._audit_coverage(
            merged, official_total=official_total, source_counts=self.last_source_counts,
            source_conflicts=self.last_source_conflicts,
        )
        all_conflict_details = (
            source_conflict_details + self.last_annual_source_conflicts
            + [{**item, "source": "unvalidated_candidate_cache"}
               for item in self.last_candidate_cache_conflicts]
        )
        if all_conflict_details:
            audit = dict(self.last_history_audit)
            audit["source_conflict_details_sample"] = all_conflict_details[:200]
            _safe_json_write(AUDIT_FILE, audit)
            self.last_history_audit = audit

        # official_total no puede ser None aquí: el caso sin contador retorna por la
        # ruta de respaldo estricta al principio de la función. Se elimina el bloque
        # histórico inalcanzable para no conservar una opción CLI aparentemente activa.
        expected_first_date = (datetime(1988, 2, 28).date() if self.start_year <= 1988
                               else datetime(self.start_year, 1, 1).date())
        first_date = merged[0][0].date() if merged else None
        last_date = merged[-1][0].date() if merged else None
        first_ok = (first_date == expected_first_date) if self.start_year <= 1988 else (
            first_date is not None and first_date.year == self.start_year
        )
        problems = []
        if not first_ok:
            problems.append(f"inicio esperado={expected_first_date}, observado={first_date}")
        if last_date is None or (_madrid_now().date() - last_date).days > 3:
            problems.append(f"histórico desactualizado, última fecha={last_date}")
        if len(merged) != official_total:
            problems.append(
                f"recuento={len(merged):,}, SELAE={official_total:,}, diferencia={official_total-len(merged):+,}"
            )
        if len({d.date() for d, _ in merged}) != len(merged):
            problems.append("hay fechas duplicadas")
        if self.last_source_conflicts:
            problems.append(
                f"{len(self.last_source_conflicts)} fechas con discrepancias entre archivos sin desempate anual"
            )

        if problems:
            self._save_candidate(merged)
            raise RuntimeError(
                "HISTÓRICO NO VALIDADO: " + "; ".join(problems) + ". "
                "La candidata queda en bonoloto_history_candidate.pkl; no se han reemplazado "
                "SQLite/CSV/PKL operativos ni se ejecutarán modelos hasta resolverlo."
            )

        self._save_all(merged)
        try:
            CANDIDATE_CACHE_FILE.unlink(missing_ok=True)
        except Exception as exc:
            log.warn(f"No se pudo eliminar la caché candidata ya validada: {exc}")
        log.ok(f"Recuento histórico coincide con SELAE: {len(merged):,}/{official_total:,} sorteos.")
        if self.last_source_conflicts:
            log.warn(
                f"Hay {len(self.last_source_conflicts)} fechas con discrepancias entre los dos archivos históricos. "
                "Se registraron para revisión; el recuento total por sí solo no demuestra que cada combinación sea correcta."
            )
        return merged

    def scrape_all_years(self):
        return self.scrape_with_cache(force_refresh=True)

    @staticmethod
    def _atomic_text_write(path, writer):
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            DataScraper._backup_file(path)
            os.replace(tmp, path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def _save_all(self, data):
        # Cada formato conserva un .bak previo y se reemplaza de forma atómica.
        self._save_sqlite(data)
        pickle_tmp = CACHE_FILE.with_suffix(CACHE_FILE.suffix + ".tmp")
        try:
            with open(pickle_tmp, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            self._backup_file(CACHE_FILE)
            os.replace(pickle_tmp, CACHE_FILE)
        finally:
            try:
                if pickle_tmp.exists():
                    pickle_tmp.unlink()
            except Exception:
                pass

        def write_csv(handle):
            writer = csv.writer(handle)
            writer.writerow(["fecha", "n1", "n2", "n3", "n4", "n5", "n6"])
            for fecha, draw in data:
                writer.writerow([fecha.strftime("%Y-%m-%d"), *draw])
        self._atomic_text_write(CSV_FILE, write_csv)

    def _audit_coverage(self, data, official_total=None, source_counts=None, source_conflicts=None):
        if not data:
            return
        dates = [d.date().isoformat() for d, _ in data]
        by_year = Counter(d.year for d, _ in data)
        unique_dates = set(dates)
        audit = {
            "generated": _madrid_now().isoformat(timespec="seconds"),
            "total_draws": len(data),
            "unique_dates": len(unique_dates),
            "duplicate_date_rows": len(dates) - len(unique_dates),
            "min_date": data[0][0].strftime("%Y-%m-%d"),
            "max_date": data[-1][0].strftime("%Y-%m-%d"),
            "years": dict(sorted(by_year.items())),
            "expected_start": self.start_year,
            "expected_end": self.end_year,
            "official_expected_total": official_total,
            "official_count_difference": (official_total - len(unique_dates)) if official_total is not None else None,
            "completeness_verified_by_official_count": bool(official_total is not None and len(unique_dates) == official_total),
            "source_counts": dict(source_counts or {}),
            "source_conflict_dates_count": len(source_conflicts or []),
            "source_conflict_dates_sample": list((source_conflicts or [])[:100]),
            "warning": None if official_total is not None and len(unique_dates) == official_total else
                "El histórico no debe considerarse completo: el recuento no se ha podido verificar o no coincide con SELAE.",
        }
        _safe_json_write(AUDIT_FILE, audit)
        self.last_history_audit = audit
        log.log(
            f"Histórico auditado: {len(unique_dates):,} fechas únicas | "
            f"{data[0][0]:%d/%m/%Y} -> {data[-1][0]:%d/%m/%Y} | "
            f"SELAE={official_total if official_total is not None else 'no disponible'}",
            "AUDIT",
        )


# ================================================================
# AUDITORÍA DEL DATASET
# ================================================================

def dataset_audit(data):
    errors = []
    seen_rows = set()
    rows_by_date = defaultdict(set)
    for fecha, draw in data:
        date_key = fecha.date().isoformat() if hasattr(fecha, "date") else str(fecha)[:10]
        normalized = tuple(sorted(int(x) for x in draw)) if draw is not None else tuple()
        if not validate_draw(normalized):
            errors.append((str(fecha), list(draw) if draw is not None else None, "invalid_draw"))
        key = (date_key, normalized)
        if key in seen_rows:
            errors.append((str(fecha), list(normalized), "duplicate_row"))
        seen_rows.add(key)
        rows_by_date[date_key].add(normalized)
    date_row_counts = Counter(
        f.date().isoformat() if hasattr(f, "date") else str(f)[:10]
        for f, _ in data
    )
    duplicate_dates = sorted(date for date, count in date_row_counts.items() if count > 1)
    conflicting_combination_dates = sorted(date for date, draws_on_date in rows_by_date.items()
                                           if len(draws_on_date) > 1)
    dates = [x[0] for x in data]
    existing = {}
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as handle:
            candidate = json.load(handle)
            if isinstance(candidate, dict):
                existing = candidate
    except Exception:
        pass
    report = dict(existing)
    report.update({
        "dataset_audit_generated": datetime.now().isoformat(),
        "records": len(data),
        "unique_dates_in_dataset": len(rows_by_date),
        "errors": len(errors),
        "duplicates_or_invalid": errors[:100],
        "duplicate_dates_count": len(duplicate_dates),
        "duplicate_dates_sample": duplicate_dates[:100],
        "conflicting_combination_dates_count": len(conflicting_combination_dates),
        "conflicting_combination_dates_sample": conflicting_combination_dates[:100],
        "min_date": min(dates).isoformat() if dates else None,
        "max_date": max(dates).isoformat() if dates else None,
    })
    _safe_json_write(AUDIT_FILE, report)
    log.log(f"Auditoría dataset: {len(data):,} registros, {len(errors)} incidencias y {len(duplicate_dates)} fechas duplicadas", "AUDIT")
    return report


def randomness_audit(data):
    freq = np.zeros(MAX_NUM, dtype=float)
    for _, draw in data:
        for n in draw:
            freq[n - 1] += 1
    chi2 = chi_square(freq)
    ent = entropy(freq / (freq.sum() + 1e-12))
    z = safe_zscore(freq)
    outliers = [i + 1 for i, x in enumerate(z) if abs(x) >= 2.5]
    log.log(
        f"Auditoría aleatoriedad: chi2={chi2:.3f}  entropia={ent:.5f}  "
        f"outliers(|z|>=2.5)={len(outliers)}",
        "AUDIT",
    )
    return {
        "frequency": freq.tolist(),
        "chi_square": chi2,
        "entropy": ent,
        "outliers": outliers,
    }
# ================================================================
# PARTE 3 — FEATURE ENGINEERING CORREGIDO
# ================================================================
# FIX aplicado:
#   - _exponential_frequency_from_counts: pesos CORREGIDOS. Antes daba
#     más peso a los sorteos ANTIGUOS y excluía el más reciente. Ahora
#     da más peso a los recientes e INCLUYE el sorteo i.
#   - number_features / build_dataset: cum y draws se precomputan una
#     vez y se pasan como argumentos. Mismo resultado, menos tiempo.
#   - Eliminadas _pair_excess_from_counts y _window_draws (nunca se
#     llamaban y contenían errores conceptuales).
# ================================================================

from math import comb as _math_comb

PROB_PAIR = _math_comb(47, 4) / _math_comb(49, 6)      # ≈ 0.012755
PROB_TRIPLE = _math_comb(46, 3) / _math_comb(49, 6)    # ≈ 0.0010855


def ljung_box(series, lags=20):
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < lags + 5:
        return 0.0, 1.0
    x = x - x.mean()
    denom = np.sum(x * x) + 1e-12
    q = 0.0
    for k in range(1, lags + 1):
        rk = np.sum(x[k:] * x[:-k]) / denom
        q += (rk * rk) / (n - k)
    q *= n * (n + 2)
    try:
        from math import erf
        df = lags
        z = ((q / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
        p = 0.5 * (1 - erf(z / math.sqrt(2)))
    except Exception:
        p = 1.0
    return float(q), float(p)


# ----------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------

class FeatureEngineeringV3:
    """Feature engineering sin fuga de información.

    Todas las features de un índice i usan SOLO sorteos hasta i (incluido).
    El target es el sorteo i+1.
    """

    HALF_LIFE = FEATURE_HALF_LIFE

    # ------------------------------------------------------------
    @staticmethod
    def _build_prefix_counts(draws):
        """Matriz acumulativa (n+1, 49). cum[k, j] = veces que el número
        j+1 apareció en los primeros k sorteos."""
        n = len(draws)
        cum = np.zeros((n + 1, MAX_NUM), dtype=np.float64)
        for i, d in enumerate(draws):
            cum[i + 1] = cum[i]
            for num in d:
                cum[i + 1, num - 1] += 1.0
        return cum

    # ------------------------------------------------------------
    @staticmethod
    def _last_seen_index(draws, end):
        """Para cada número, índice del último sorteo ≤ end en que apareció
        (o -1 si nunca)."""
        last = np.full(MAX_NUM, -1, dtype=np.int64)
        for i in range(end):
            for num in draws[i]:
                last[num - 1] = i
        return last

    # ------------------------------------------------------------
    # FIX CRÍTICO: pesos corregidos + incluye sorteo más reciente
    # ------------------------------------------------------------
    @staticmethod
    def _exponential_frequency_from_counts(cum, index_p, half_life):
        """Frecuencia exponencial hasta el sorteo index_p - 1 (inclusive).

        index_p = número de sorteos a considerar (usualmente index + 1).

        Pesos: w = exp(-decay * back), donde back = 0 para el sorteo
        más reciente y back = N-1 para el más antiguo. Así el peso
        decrece al alejarnos del presente.
        """
        N = index_p
        if N <= 0:
            return np.full(MAX_NUM, 1.0 / MAX_NUM, dtype=np.float64)
        decay = math.log(2.0) / max(half_life, 1e-6)
        # diffs[j] = one-hot del sorteo j (j = 0..N-1)
        diffs = cum[1:N + 1] - cum[:N]
        # backs[j] = 0 para el más reciente, N-1 para el más antiguo
        backs = np.arange(N - 1, -1, -1, dtype=np.float64)
        w = np.exp(-decay * backs)
        weighted = np.sum(diffs * w[:, None], axis=0)
        s = weighted.sum()
        return weighted / s if s > 0 else np.full(MAX_NUM, 1.0 / MAX_NUM)

    @staticmethod
    def _recency_from_last_seen(last, index):
        """Recencia a partir del último índice visto."""
        gaps = (index - last).astype(np.float64)
        gaps[last < 0] = index + 1.0
        return np.log1p(gaps)

    @staticmethod
    def _pair_excess_direct(rows, n_draws):
        if n_draws == 0:
            return np.zeros(MAX_NUM, dtype=np.float64)
        pair = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
        for d in rows:
            for a, b in combinations(d, 2):
                pair[a - 1, b - 1] += 1.0
                pair[b - 1, a - 1] += 1.0
        expected = PROB_PAIR * n_draws
        excess = (pair - expected) / math.sqrt(expected + 1e-9)
        np.fill_diagonal(excess, 0.0)
        return np.sum(excess, axis=1) / (MAX_NUM - 1)

    @staticmethod
    def _triple_excess_direct(rows, n_draws):
        if n_draws == 0:
            return np.zeros(MAX_NUM, dtype=np.float64)
        triple = defaultdict(int)
        for d in rows:
            for tri in combinations(d, 3):
                triple[tri] += 1
        expected = PROB_TRIPLE * n_draws
        score = np.zeros(MAX_NUM, dtype=np.float64)
        for tri, c in triple.items():
            a, b, cc = tri
            excess = (c - expected) / math.sqrt(expected + 1e-9)
            score[a - 1] += excess
            score[b - 1] += excess
            score[cc - 1] += excess
        return score / 1128.0

    @staticmethod
    def _precompute_pair_triple_features(draws, window=500):
        """Precomputa features móviles manteniendo la fórmula histórica exacta.

        La versión anterior reconstruía la matriz de pares y el contador de ternas
        sobre hasta 500 sorteos para cada muestra (miles de veces). Esta versión
        actualiza ambos contadores al entrar/salir un sorteo. Conserva incluso la
        definición heredada de _triple_excess_direct, que sólo suma ternas observadas;
        cambiar esa definición sería una modificación estadística separada, no una
        optimización equivalente.
        """
        rows = [tuple(int(x) for x in d) for d in draws]
        n = len(rows)
        pair_features = np.zeros((n, MAX_NUM), dtype=np.float64)
        triple_features = np.zeros((n, MAX_NUM), dtype=np.float64)
        pair_counts = np.zeros((MAX_NUM, MAX_NUM), dtype=np.int32)
        triple_counts = Counter()
        triple_count_sum_by_number = np.zeros(MAX_NUM, dtype=np.int64)
        triple_unique_by_number = np.zeros(MAX_NUM, dtype=np.int32)

        def update_draw(draw, sign):
            for a, b in combinations(draw, 2):
                pair_counts[a - 1, b - 1] += sign
                pair_counts[b - 1, a - 1] += sign
            for tri in combinations(draw, 3):
                old_count = int(triple_counts.get(tri, 0))
                new_count = old_count + sign
                a, b, c = tri
                if sign > 0:
                    if old_count == 0:
                        triple_unique_by_number[a - 1] += 1
                        triple_unique_by_number[b - 1] += 1
                        triple_unique_by_number[c - 1] += 1
                    triple_counts[tri] = new_count
                    triple_count_sum_by_number[a - 1] += 1
                    triple_count_sum_by_number[b - 1] += 1
                    triple_count_sum_by_number[c - 1] += 1
                else:
                    if old_count <= 0:
                        raise RuntimeError(f"Contador de ternas inconsistente para {tri}")
                    triple_count_sum_by_number[a - 1] -= 1
                    triple_count_sum_by_number[b - 1] -= 1
                    triple_count_sum_by_number[c - 1] -= 1
                    if new_count == 0:
                        del triple_counts[tri]
                        triple_unique_by_number[a - 1] -= 1
                        triple_unique_by_number[b - 1] -= 1
                        triple_unique_by_number[c - 1] -= 1
                    else:
                        triple_counts[tri] = new_count

        for i, draw in enumerate(rows):
            if i >= window:
                update_draw(rows[i - window], -1)
            update_draw(draw, +1)
            n_window = min(i + 1, window)
            expected_pair = PROB_PAIR * n_window
            pair_denominator = math.sqrt(expected_pair + 1e-9)
            pair_excess_matrix = (pair_counts.astype(np.float64) - expected_pair) / pair_denominator
            np.fill_diagonal(pair_excess_matrix, 0.0)
            pair_features[i] = np.sum(pair_excess_matrix, axis=1) / (MAX_NUM - 1)

            expected_triple = PROB_TRIPLE * n_window
            triple_denominator = math.sqrt(expected_triple + 1e-9)
            triple_features[i] = (
                triple_count_sum_by_number.astype(np.float64)
                - expected_triple * triple_unique_by_number
            ) / triple_denominator / 1128.0

        return pair_features, triple_features

    @staticmethod
    def structural_features(draws, end, window=80):
        """Estadísticas estructurales de los `window` sorteos hasta `end`."""
        start = max(0, end - window)
        rows = draws[start:end]
        if not rows:
            return np.zeros(14, dtype=np.float64)

        sums = np.array([sum(d) for d in rows], dtype=float)
        ranges = np.array([draw_range(d) for d in rows], dtype=float)
        evens = np.array([even_count(d) for d in rows], dtype=float)
        lows = np.array([low_count(d) for d in rows], dtype=float)
        consec = np.array([
            sum(d[i + 1] == d[i] + 1 for i in range(5)) for d in rows
        ], dtype=float)

        return np.array([
            float(np.mean(sums)), float(np.std(sums)),
            float(np.mean(ranges)), float(np.std(ranges)),
            float(np.mean(evens)), float(np.std(evens)),
            float(np.mean(lows)), float(np.std(lows)),
            float(np.mean(consec)), float(np.std(consec)),
            float(np.min(sums)), float(np.max(sums)),
            float(np.min(ranges)), float(np.max(ranges)),
        ], dtype=np.float64)

    @staticmethod
    def cyclical_features(fecha):
        doy = fecha.timetuple().tm_yday
        dow = fecha.weekday()
        dom = fecha.day
        month = fecha.month
        return np.array([
            math.sin(2 * math.pi * doy / 365.25),
            math.cos(2 * math.pi * doy / 365.25),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            math.sin(2 * math.pi * dom / 31),
            math.cos(2 * math.pi * dom / 31),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
        ], dtype=np.float64)

    @staticmethod
    def number_features(
        data, index, cum=None, draws=None, last_seen_prefix=None,
        pair_excess=None, triple_excess=None,
    ):
        """Construye el vector de features para el índice dado.

        Usa los sorteos HASTA index (incluido). El target asociado
        es data[index + 1].

        cum y draws son opcionales: si se pasan (precomputados), se
        evita reconstruirlos en cada llamada.
        """
        if draws is None:
            draws = [tuple(d) for _, d in data]
        if cum is None:
            cum = FeatureEngineeringV3._build_prefix_counts(draws)

        fecha = data[index][0]

        exp_freq = FeatureEngineeringV3._exponential_frequency_from_counts(
            cum, index + 1, FeatureEngineeringV3.HALF_LIFE
        )

        hard_freqs = []
        for w in FEATURE_WINDOWS:
            start = max(0, index + 1 - w)
            n_rows = index + 1 - start
            if n_rows <= 0:
                hard_freqs.append(np.full(MAX_NUM, 1.0 / MAX_NUM))
            else:
                counts = cum[index + 1] - cum[start]
                hard_freqs.append(counts / n_rows)

        if last_seen_prefix is not None:
            last = last_seen_prefix[index + 1]
        else:
            last = FeatureEngineeringV3._last_seen_index(draws, index + 1)
        rec = FeatureEngineeringV3._recency_from_last_seen(last, index)

        if pair_excess is None or triple_excess is None:
            start_pair = max(0, index + 1 - 500)
            pair_rows = draws[start_pair:index + 1]
            if pair_excess is None:
                pair_excess = FeatureEngineeringV3._pair_excess_direct(pair_rows, len(pair_rows))
            if triple_excess is None:
                triple_excess = FeatureEngineeringV3._triple_excess_direct(pair_rows, len(pair_rows))
        pair_ex = np.asarray(pair_excess, dtype=np.float64)
        triple_ex = np.asarray(triple_excess, dtype=np.float64)
        if pair_ex.shape != (MAX_NUM,) or triple_ex.shape != (MAX_NUM,):
            raise ValueError("Features pareadas/triples precomputadas deben tener 49 valores")

        per_number = np.stack(
            [exp_freq] + hard_freqs + [rec, pair_ex, triple_ex],
            axis=1,
        )
        flat = per_number.flatten().astype(np.float32)

        struct = FeatureEngineeringV3.structural_features(draws, index + 1)
        cyc = FeatureEngineeringV3.cyclical_features(fecha)

        global_freq = cum[index + 1].copy()
        ent = entropy(global_freq / (global_freq.sum() + 1e-12))
        chi2 = chi_square(global_freq)

        extra = np.array([ent, chi2], dtype=np.float64)

        return np.concatenate([flat, struct, cyc, extra]).astype(np.float32)

    @staticmethod
    def feature_dim():
        per_number = MAX_NUM * (1 + len(FEATURE_WINDOWS) + 1 + 1 + 1)
        global_feats = 14 + 8 + 2
        return per_number + global_feats


# ----------------------------------------------------------------
# Construcción del dataset supervisado
# ----------------------------------------------------------------

def build_dataset(data, seq_len=SEQ_LEN):
    if len(data) <= seq_len + 30:
        raise ValueError(
            f"No hay suficientes sorteos para construir el dataset: {len(data)}"
        )

    draws = [tuple(d) for _, d in data]
    # CORRECCIÓN: cum se precomputa UNA VEZ y se pasa a number_features.
    cum = FeatureEngineeringV3._build_prefix_counts(draws)
    # Última aparición por prefijo: evita recorrer todo el histórico para cada muestra.
    last_seen_prefix = np.full((len(draws) + 1, MAX_NUM), -1, dtype=np.int32)
    for i, draw in enumerate(draws):
        last_seen_prefix[i + 1] = last_seen_prefix[i]
        for num in draw:
            last_seen_prefix[i + 1, num - 1] = i

    # Optimización exacta: el código original recalculaba pares y ternas hasta
    # en 500 sorteos para cada muestra. Se actualizan los contadores por ventana
    # una sola vez y cada fila feature reutiliza su vector ya calculado.
    log.log("Precomputando features móviles de pares y ternas (ventana=500)...", "MODEL")
    pair_feature_cache, triple_feature_cache = FeatureEngineeringV3._precompute_pair_triple_features(draws, window=500)

    X_seq, X_feat, Y = [], [], []

    total = len(draws) - seq_len
    log.log(f"Construyendo dataset: {total:,} muestras a procesar...", "MODEL")

    for i in range(seq_len - 1, len(draws) - 1):
        if i % 100 == 0:
            pause_checkpoint(f"construcción del dataset, muestra {i}/{len(draws)}")
        seq = [one_hot_draw(draws[k]) for k in range(i - seq_len + 1, i + 1)]
        feat = FeatureEngineeringV3.number_features(
            data, i, cum=cum, draws=draws, last_seen_prefix=last_seen_prefix,
            pair_excess=pair_feature_cache[i],
            triple_excess=triple_feature_cache[i],
        )
        X_seq.append(seq)
        X_feat.append(feat)
        Y.append(one_hot_draw(draws[i + 1]))

        done = i - (seq_len - 1) + 1
        if done % 200 == 0 or done == total:
            pct = 20.0 + 12.0 * (done / max(1, total))
            log.progress(pct, f"Dataset: {done:,}/{total:,} muestras")

    X_seq = np.asarray(X_seq, dtype=np.float32)
    X_feat = np.asarray(X_feat, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    log.log(
        f"Dataset construido: X_seq={X_seq.shape}, "
        f"X_feat={X_feat.shape}, Y={Y.shape}",
        "MODEL",
    )
    return X_seq, X_feat, Y


# ----------------------------------------------------------------
# Auditoría de features
# ----------------------------------------------------------------

def feature_audit(X_feat, Y):
    """Tests de aleatoriedad sobre cada feature respecto al target."""
    n_features = X_feat.shape[1]
    results = []
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or len(Y) != len(X_feat):
        raise ValueError(
            f"Dimensiones incompatibles en feature_audit: X_feat={X_feat.shape}, Y={Y.shape}"
        )

    log.log(f"Auditoría de {n_features} features contra el target...", "AUDIT")

    for j in range(n_features):
        col = np.asarray(X_feat[:, j], dtype=np.float64)
        corrs = []
        if np.std(col) >= 1e-12:
            for target_idx in range(Y.shape[1]):
                target_col = Y[:, target_idx]
                if np.std(target_col) < 1e-12:
                    continue
                value = np.corrcoef(col, target_col)[0, 1]
                if np.isfinite(value):
                    corrs.append(float(value))
        corr = float(np.mean(np.abs(corrs))) if corrs else 0.0

        q, p = ljung_box(col, lags=10)

        results.append({
            "index": j,
            "corr_target": corr,
            "ljung_box_q": q,
            "ljung_box_p": p,
        })

        if j % 50 == 0:
            pct = 45.0 + 8.0 * (j / n_features)
            log.progress(pct, f"Auditoría features: {j}/{n_features}")

    results.sort(key=lambda r: abs(r["corr_target"]), reverse=True)
    top = results[:30]
    significant = [r for r in results if r["ljung_box_p"] < 0.05]

    report = {
        "n_features": n_features,
        "significant_ljung_box": len(significant),
        "top_correlations": top,
        "note": "Ljung-Box detecta autocorrelación; no implica capacidad predictiva.",
    }
    with open(FEATURE_AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.log(
        f"Features con Ljung-Box p<0.05: {len(significant)}/{n_features}",
        "AUDIT",
    )
    if top:
        best = top[0]
        log.log(
            f"Feature más correlacionada con el target: "
            f"idx={best['index']} corr={best['corr_target']:+.4f} "
            f"p={best['ljung_box_p']:.4f}",
            "AUDIT",
        )
    return report

# ================================================================
# PARTE 4 — MODELOS
# ================================================================

class StatisticalEngine:
    """Motor estadístico multiescala con coocurrencia corregida."""

    WINDOWS = (5, 10, 20, 40, 80, 160, 320, 640)

    def __init__(self, data):
        self.data = sorted(data, key=lambda x: x[0])
        self.draws = [tuple(d) for _, d in self.data]

    def _freq(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        f = np.zeros(MAX_NUM, dtype=np.float64)
        for d in rows:
            for n in d:
                f[n - 1] += 1.0
        s = f.sum()
        return f / s if s > 0 else np.full(MAX_NUM, 1.0 / MAX_NUM)

    def _absence(self):
        gaps = np.zeros(MAX_NUM, dtype=np.float64)
        for n in range(1, MAX_NUM + 1):
            gap = 0
            for d in reversed(self.draws):
                if n in d:
                    break
                gap += 1
            gaps[n - 1] = gap
        return gaps

    def _pair_excess(self, window=500):
        rows = self.draws[-window:] if window else self.draws
        n_draws = len(rows)
        if n_draws == 0:
            return np.zeros(MAX_NUM, dtype=np.float64)

        pair = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
        for d in rows:
            for a, b in combinations(d, 2):
                pair[a - 1, b - 1] += 1.0
                pair[b - 1, a - 1] += 1.0

        expected = PROB_PAIR * n_draws
        excess = (pair - expected) / math.sqrt(expected + 1e-9)
        np.fill_diagonal(excess, 0.0)
        return np.sum(excess, axis=1) / (MAX_NUM - 1)

    def score_numbers(self):
        f_full = self._freq()
        f_wins = {w: self._freq(w) for w in self.WINDOWS}
        absence = self._absence()
        absence_z = safe_zscore(absence)
        pair_score = self._pair_excess(500)
        pair_z = safe_zscore(pair_score)

        weights = {5: 0.22, 10: 0.20, 20: 0.18, 40: 0.14,
                   80: 0.10, 160: 0.07, 320: 0.05, 640: 0.04}
        trend = np.zeros(MAX_NUM, dtype=np.float64)
        for w, wt in weights.items():
            trend += wt * safe_zscore(f_wins[w])

        recency_signal = np.tanh(absence_z / 3.0) * 0.04
        pair_signal = pair_z * 0.06

        raw = trend + recency_signal + pair_signal

        return raw, {
            "full": f_full,
            "windows": f_wins,
            "absence": absence,
            "trend": trend,
            "pair_excess": pair_score,
        }

    def probability_vector(self):
        raw, details = self.score_numbers()
        p = softmax(raw, temperature=1.25)
        return p, details

    def summary(self):
        f = self._freq()
        return {
            "total_draws": len(self.draws),
            "min_date": self.data[0][0].isoformat() if self.data else None,
            "max_date": self.data[-1][0].isoformat() if self.data else None,
            "chi_square": chi_square(f * len(self.draws) * NUMBERS_PER_DRAW),
            "entropy": entropy(f),
        }


# ----------------------------------------------------------------
# 4.2 — ENSEMBLE DE ÁRBOLES
# ----------------------------------------------------------------

class TreeEnsemble:
    """Ensemble de clasificadores por número."""

    def __init__(self):
        self.scaler = None
        self.logistic = [None] * MAX_NUM
        self.hist_gb = [None] * MAX_NUM
        self.extra = [None] * MAX_NUM
        self.forest = [None] * MAX_NUM
        self.xgb = [None] * MAX_NUM
        self.lgbm = [None] * MAX_NUM
        self.fitted = False
        self.feature_count = None
        self.training_rows = 0
        self.report = {}

    @staticmethod
    def _safe_proba(model, X):
        if model is None:
            return None
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, progress_callback=None):
        if not SKLEARN_AVAILABLE:
            return {"available": False}

        X_train = np.asarray(X_train, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.float32)
        self.feature_count = X_train.shape[1]
        self.training_rows = len(X_train)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train)

        for n in range(MAX_NUM):
            pause_checkpoint(f"entrenamiento de modelos para el número {n + 1}/{MAX_NUM}")
            if progress_callback:
                pct = 55.0 + 12.0 * (n / MAX_NUM)
                progress_callback(pct, f"Árboles: número {n+1}/{MAX_NUM}")

            y = Y_train[:, n].astype(np.int8)
            if len(np.unique(y)) < 2:
                continue

            try:
                lr = LogisticRegression(
                    C=0.08, max_iter=2500,
                    class_weight="balanced", solver="lbfgs",
                    random_state=ULTIMATE_SEED, n_jobs=None,
                )
                pause_checkpoint(f"antes de ajustar LogisticRegression, número {n + 1}")
                lr.fit(Xs, y)
                self.logistic[n] = lr
            except Exception as e:
                log.warn(f"Logistic n={n+1}: {e}")

            try:
                gb = HistGradientBoostingClassifier(
                    learning_rate=0.035, max_iter=400,
                    max_leaf_nodes=31, max_depth=7,
                    min_samples_leaf=25, l2_regularization=2.0,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=ULTIMATE_SEED + n,
                )
                pause_checkpoint(f"antes de ajustar HistGradientBoosting, número {n + 1}")
                gb.fit(Xs, y)
                self.hist_gb[n] = gb
            except Exception as e:
                log.warn(f"HistGB n={n+1}: {e}")

            try:
                et = ExtraTreesClassifier(
                    n_estimators=TREE_ESTIMATORS,
                    max_features="sqrt", min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    bootstrap=False, n_jobs=MAX_CPU_WORKERS,
                    random_state=ULTIMATE_SEED + 1000 + n,
                )
                pause_checkpoint(f"antes de ajustar ExtraTrees, número {n + 1}")
                et.fit(X_train, y)
                self.extra[n] = et
            except Exception as e:
                log.warn(f"ExtraTrees n={n+1}: {e}")

            try:
                rf = RandomForestClassifier(
                    n_estimators=TREE_ESTIMATORS,
                    max_features="sqrt", min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    bootstrap=True, n_jobs=MAX_CPU_WORKERS,
                    random_state=ULTIMATE_SEED + 2000 + n,
                )
                pause_checkpoint(f"antes de ajustar RandomForest, número {n + 1}")
                rf.fit(X_train, y)
                self.forest[n] = rf
            except Exception as e:
                log.warn(f"RandomForest n={n+1}: {e}")

            if XGB_AVAILABLE:
                try:
                    xgb = XGBClassifier(
                        n_estimators=XGB_ESTIMATORS,
                        max_depth=5, learning_rate=0.035,
                        subsample=0.85, colsample_bytree=0.80,
                        min_child_weight=8, reg_alpha=0.05,
                        reg_lambda=2.0, objective="binary:logistic",
                        eval_metric="logloss", tree_method="hist",
                        n_jobs=MAX_CPU_WORKERS,
                        random_state=ULTIMATE_SEED + 3000 + n,
                        verbosity=0,
                    )
                    pause_checkpoint(f"antes de ajustar XGBoost, número {n + 1}")
                    xgb.fit(X_train, y)
                    self.xgb[n] = xgb
                except Exception as e:
                    log.warn(f"XGB n={n+1}: {e}")

            if LGBM_AVAILABLE:
                try:
                    lg = lgb.LGBMClassifier(
                        n_estimators=LGBM_ESTIMATORS,
                        max_depth=-1, num_leaves=31,
                        learning_rate=0.03,
                        subsample=0.85, colsample_bytree=0.80,
                        min_child_samples=20,
                        reg_alpha=0.05, reg_lambda=2.0,
                        objective="binary", n_jobs=MAX_CPU_WORKERS,
                        random_state=ULTIMATE_SEED + 4000 + n,
                        verbosity=-1,
                    )
                    pause_checkpoint(f"antes de ajustar LightGBM, número {n + 1}")
                    lg.fit(X_train, y)
                    self.lgbm[n] = lg
                except Exception as e:
                    log.warn(f"LightGBM n={n+1}: {e}")

        self.fitted = True

        metrics = {
            "available": True,
            "rows": len(X_train),
            "features": self.feature_count,
            "xgboost": XGB_AVAILABLE,
            "lightgbm": LGBM_AVAILABLE,
        }
        if X_val is not None and Y_val is not None and len(X_val):
            try:
                pred = self.predict(X_val)
                metrics["val_brier"] = float(np.mean((Y_val - pred) ** 2))
                metrics["val_hits_top12"] = float(self._topk_hits(Y_val, pred, 12))
            except Exception as e:
                log.warn(f"Métricas validación: {e}")

        self.report = metrics
        return metrics

    @staticmethod
    def _topk_hits(y, p, k=12):
        y = np.asarray(y)
        p = np.asarray(p)
        total = 0.0
        for i in range(len(y)):
            if i % 250 == 0:
                pause_checkpoint(f"métricas top-k, fila {i}/{len(y)}")
            pred = set(np.argsort(p[i])[-k:])
            actual = set(np.where(y[i] > 0.5)[0])
            total += len(pred & actual)
        return total / max(1, len(y))

    def predict_components(self, X):
        if not self.fitted or self.scaler is None:
            return None, None

        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)

        logistic = np.full((len(X), MAX_NUM), 1.0 / MAX_NUM, dtype=float)
        nonlinear = [np.full((len(X), MAX_NUM), 1.0 / MAX_NUM) for _ in range(5)]

        for n in range(MAX_NUM):
            pause_checkpoint(f"inferencia de modelos, número {n + 1}/{MAX_NUM}")
            pred = self._safe_proba(self.logistic[n], Xs)
            if pred is not None:
                logistic[:, n] = pred

            for model, idx, use_scaled in (
                (self.hist_gb[n], 0, True),
                (self.extra[n], 1, False),
                (self.forest[n], 2, False),
                (self.xgb[n], 3, False),
                (self.lgbm[n], 4, False),
            ):
                pred = self._safe_proba(model, Xs if use_scaled else X)
                if pred is not None:
                    nonlinear[idx][:, n] = pred

        logistic = normalize_rows(logistic)

        present = []
        if any(x is not None for x in self.hist_gb):
            present.append(nonlinear[0])
        if any(x is not None for x in self.extra):
            present.append(nonlinear[1])
        if any(x is not None for x in self.forest):
            present.append(nonlinear[2])
        if any(x is not None for x in self.xgb):
            present.append(nonlinear[3])
        if any(x is not None for x in self.lgbm):
            present.append(nonlinear[4])

        if present:
            blend = np.mean(present, axis=0)
            blend = normalize_rows(blend)
        else:
            blend = np.full_like(logistic, 1.0 / MAX_NUM)

        return logistic, blend

    def predict(self, X):
        a, b = self.predict_components(X)
        if a is None:
            return None
        return normalize_rows(0.40 * a + 0.60 * b)


# ----------------------------------------------------------------
# 4.3 — BAYESIANO EMPÍRICO
# ----------------------------------------------------------------

class EmpiricalBayes:
    """Intervalos de credibilidad por número con bootstrap vectorizado."""

    def __init__(self, data, n_bootstrap=2000, n_dirichlet=5000):
        self.data = sorted(data, key=lambda x: x[0])
        self.draws = [tuple(d) for _, d in self.data]
        self.n_bootstrap = n_bootstrap
        self.n_dirichlet = n_dirichlet

    def fit(self, progress_callback=None):
        n = len(self.draws)
        if n < 100:
            self.mean = np.full(MAX_NUM, 1.0 / MAX_NUM)
            self.q05 = self.mean.copy()
            self.q95 = self.mean.copy()
            self.p_top6 = self.mean.copy() * 6
            return self

        rng = np.random.default_rng(ULTIMATE_SEED + 500)

        one_hot = np.zeros((n, MAX_NUM), dtype=np.float32)
        for i, d in enumerate(self.draws):
            for num in d:
                one_hot[i, num - 1] = 1.0

        boot_freqs = np.zeros((self.n_bootstrap, MAX_NUM), dtype=np.float64)
        for b in range(self.n_bootstrap):
            if b % 50 == 0:
                pause_checkpoint(f"bootstrap bayesiano {b}/{self.n_bootstrap}")
            idx = rng.integers(0, n, size=n)
            boot_freqs[b] = one_hot[idx].sum(axis=0)
            boot_freqs[b] /= (boot_freqs[b].sum() + 1e-12)
            if progress_callback and b % 200 == 0:
                progress_callback(90.0, f"Bayesiano: bootstrap {b}/{self.n_bootstrap}")

        mean = boot_freqs.mean(axis=0)
        var = boot_freqs.var(axis=0)
        s = mean * (1 - mean) / (var + 1e-12) - 1.0
        concentration = float(np.clip(np.median(s), 5.0, 500.0))
        alpha = np.clip(mean * concentration, 1e-3, None)

        samples = rng.dirichlet(alpha, size=self.n_dirichlet)

        self.mean = samples.mean(axis=0)
        self.q05 = np.quantile(samples, 0.05, axis=0)
        self.q95 = np.quantile(samples, 0.95, axis=0)

        top6 = np.argpartition(-samples, 6, axis=1)[:, :6]
        flat = top6.flatten()
        counts = np.bincount(flat, minlength=MAX_NUM).astype(np.float64)
        self.p_top6 = counts / self.n_dirichlet

        log.log(
            f"Bayesiano: concentración={concentration:.1f}, "
            f"p_top6 max={self.p_top6.max():.3f} min={self.p_top6.min():.3f}",
            "MODEL",
        )
        return self


# ----------------------------------------------------------------
# 4.4 — TRANSFORMER + ENSEMBLE
# ----------------------------------------------------------------

if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]


    class FuxingTransformer(nn.Module):
        def __init__(self, feature_size, d_model=256, heads=8, layers=6, dropout=0.15):
            super().__init__()
            self.proj = nn.Linear(MAX_NUM, d_model)
            self.pos = PositionalEncoding(d_model, max_len=SEQ_LEN + 8)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads,
                dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.seq_norm = nn.LayerNorm(d_model)

            self.feat = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128), nn.GELU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(d_model + 128, 512),
                nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout + 0.05),
                nn.Linear(512, 256),
                nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, MAX_NUM),
            )

        def forward(self, seq, feat):
            x = self.proj(seq)
            x = self.pos(x)
            x = self.encoder(x)
            x = self.seq_norm(x[:, -1])
            f = self.feat(feat)
            return self.fusion(torch.cat([x, f], dim=1))


class NeuralEnsemble:
    def __init__(self, n_members=NEURAL_ENSEMBLE_SIZE):
        self.n_members = max(1, int(n_members))
        self.models = []
        self.scaler = None
        self.fitted = False
        self.reports = []

    def fit(self, X_seq, X_feat, Y, progress_callback=None):
        if not TORCH_AVAILABLE:
            return {"available": False}

        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_feat = np.asarray(X_feat, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        n = len(X_feat)
        if n < 300:
            log.warn("Red: muestras insuficientes, se omite.")
            return {"available": True, "trained": False}

        split = int(n * 0.70)
        val_cut = int(n * 0.75)
        split = max(200, min(split, n - 100))
        val_cut = max(split + 30, min(val_cut, n))

        self.scaler = StandardScaler()
        self.scaler.fit(X_feat[:split])
        feat_scaled = self.scaler.transform(X_feat).astype(np.float32)

        device = torch.device(DEVICE)

        for member in range(self.n_members):
            pause_checkpoint(f"preparando red neuronal {member + 1}/{self.n_members}")
            seed = ULTIMATE_SEED + 101 * member
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            log.log(f"Red Transformer {member+1}/{self.n_members} (seed={seed})", "MODEL")

            model = FuxingTransformer(
                feature_size=X_feat.shape[1],
                d_model=256, heads=8, layers=6, dropout=0.15,
            ).to(device)

            tx = torch.tensor(X_seq[:split], dtype=torch.float32, device=device)
            tf = torch.tensor(feat_scaled[:split], dtype=torch.float32, device=device)
            ty = torch.tensor(Y[:split], dtype=torch.float32, device=device)
            vx = torch.tensor(X_seq[split:val_cut], dtype=torch.float32, device=device)
            vf = torch.tensor(feat_scaled[split:val_cut], dtype=torch.float32, device=device)
            vy = torch.tensor(Y[split:val_cut], dtype=torch.float32, device=device)

            pos = ty.mean(dim=0)
            neg = 1.0 - pos
            pos_weight = torch.clamp(neg / torch.clamp(pos, min=1e-4), 1.0, 10.0)

            ds = TensorDataset(tx.detach().cpu(), tf.detach().cpu(), ty.detach().cpu())
            batch_size = min(256, max(32, len(ds) // 12))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=(device.type == "cuda"))

            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best = float("inf"); best_state = None; bad = 0
            epochs = 80
            for epoch in range(epochs):
                pause_checkpoint(f"red neuronal {member + 1}/{self.n_members}, época {epoch + 1}/{epochs}")
                if progress_callback:
                    base = 68.0 + (member / self.n_members) * 20.0
                    span = 20.0 / self.n_members
                    progress_callback(
                        base + ((epoch + 1) / epochs) * span,
                        f"Red {member+1}/{self.n_members}: época {epoch+1}/{epochs}",
                    )
                model.train()
                for xb, fb, yb in loader:
                    pause_checkpoint(f"red neuronal {member + 1}/{self.n_members}, lote de época {epoch + 1}")
                    xb = xb.to(device, non_blocking=True)
                    fb = fb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb, fb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step(epoch + 1)

                model.eval()
                with torch.no_grad():
                    vlogits = model(vx, vf)
                    vloss = criterion(vlogits, vy).item()
                if vloss + 1e-7 < best:
                    best = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= 15:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            self.reports.append({"seed": seed, "epochs": epoch + 1, "val_loss": float(best)})
            log.log(f"Red {member+1}: épocas={epoch+1} val_loss={best:.6f}", "MODEL")

        self.fitted = bool(self.models)
        return {
            "available": True, "trained": self.fitted,
            "ensemble_size": len(self.models),
            "members": self.reports,
        }

    def predict(self, X_seq_last, X_feat_last):
        if not self.fitted or self.scaler is None:
            return None
        device = torch.device(DEVICE)
        feat = self.scaler.transform(np.asarray(X_feat_last, dtype=np.float32)).astype(np.float32)
        xs = torch.tensor(X_seq_last, dtype=torch.float32, device=device)
        xf = torch.tensor(feat, dtype=torch.float32, device=device)
        preds = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                logits = m(xs, xf)
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return normalize_rows(np.mean(preds, axis=0))
    
# ================================================================
# PARTE 5 — CALIBRACIÓN + CONFORMAL PREDICTION
# ================================================================

class IsotonicCalibrator:
    """Calibra las probabilidades del ensemble por número."""

    def __init__(self, min_samples=200):
        self.calibrators = [None] * MAX_NUM
        self.fitted = False
        self.min_samples = min_samples
        self.report = {}

    def fit(self, P_val, Y_val):
        if not SKLEARN_AVAILABLE:
            return {"available": False}

        P_val = np.asarray(P_val, dtype=float)
        Y_val = np.asarray(Y_val, dtype=float)
        if len(P_val) < self.min_samples:
            log.warn(
                f"Calibración: sólo {len(P_val)} muestras; "
                f"se omite (mínimo {self.min_samples})."
            )
            return {"available": True, "fitted": False}

        fitted_count = 0
        brier_before, brier_after = [], []

        for n in range(MAX_NUM):
            pause_checkpoint(f"calibración del número {n + 1}/{MAX_NUM}")
            p = P_val[:, n]
            y = Y_val[:, n]
            if len(np.unique(y)) < 2 or np.std(p) < 1e-9:
                continue

            try:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(p, y)
                self.calibrators[n] = iso

                p_cal = iso.predict(p)
                brier_before.append(float(np.mean((p - y) ** 2)))
                brier_after.append(float(np.mean((p_cal - y) ** 2)))
                fitted_count += 1
            except Exception as e:
                log.warn(f"Calibración n={n+1}: {e}")

        self.fitted = fitted_count > 0
        self.report = {
            "fitted_numbers": fitted_count,
            "brier_before": float(np.mean(brier_before)) if brier_before else None,
            "brier_after": float(np.mean(brier_after)) if brier_after else None,
            "brier_evaluation": "in-sample; indicador descriptivo, no estimación independiente",
        }
        if self.report["brier_before"] is not None:
            improvement = (
                (self.report["brier_before"] - self.report["brier_after"])
                / self.report["brier_before"] * 100
            )
            log.log(
                f"Calibración isotónica: {fitted_count}/49 números · "
                f"Brier {self.report['brier_before']:.6f} -> "
                f"{self.report['brier_after']:.6f} "
                f"({improvement:+.2f}%)",
                "MODEL",
            )
        return self.report

    def transform(self, P):
        P = np.asarray(P, dtype=float)
        single = P.ndim == 1
        if single:
            P = P[None, :]
        out = P.copy()
        for n, iso in enumerate(self.calibrators):
            if iso is None:
                continue
            try:
                out[:, n] = iso.predict(P[:, n])
            except Exception:
                pass
        out = normalize_rows(out)
        return out[0] if single else out


# ----------------------------------------------------------------
# 5.2 — CONFORMAL PREDICTION
# ----------------------------------------------------------------

class ConformalPredictor:
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.threshold = None
        self.fitted = False
        self.coverage_val = None
        self.diagnostics = {}

    def fit(self, P_cal, Y_cal):
        """P_cal debe estar en la misma escala que predict_set (normalize_rows)."""
        P_cal = np.asarray(P_cal, dtype=float)
        Y_cal = np.asarray(Y_cal, dtype=float)
        if (P_cal.ndim != 2 or Y_cal.ndim != 2 or P_cal.shape != Y_cal.shape
                or P_cal.shape[1] != MAX_NUM or not np.isfinite(P_cal).all()
                or not np.isfinite(Y_cal).all()):
            return {"available": True, "fitted": False, "error": "Dimensiones o valores no válidos"}
        n = len(P_cal)
        if n < 100:
            return {"available": True, "fitted": False}

        scores = []
        valid_calibration_rows = []
        for i in range(n):
            if i % 500 == 0:
                pause_checkpoint(f"calibración conformal, fila {i}/{n}")
            real_idx = np.where(Y_cal[i] > 0.5)[0]
            if len(real_idx) != NUMBERS_PER_DRAW:
                continue
            s = 1.0 - float(np.min(P_cal[i, real_idx]))
            if np.isfinite(s):
                scores.append(s)
                valid_calibration_rows.append(i)

        if len(scores) < 100:
            return {"available": True, "fitted": False, "error": "Menos de 100 filas de calibración válidas"}

        scores = np.asarray(scores, dtype=float)
        # Split-conformal: usa el k-ésimo estadístico de orden, no el cuantil
        # interpolado por defecto de NumPy. Si k>n, el umbral correcto es +inf.
        n_scores = len(scores)
        k = int(math.ceil((n_scores + 1) * (1 - self.alpha)))
        if k > n_scores:
            self.threshold = float("inf")
        else:
            self.threshold = float(np.sort(scores)[max(0, k - 1)])
        self.fitted = True

        covered = 0
        for i in valid_calibration_rows:
            real_idx = np.where(Y_cal[i] > 0.5)[0]
            mask = (1.0 - P_cal[i]) <= self.threshold
            if all(mask[idx] for idx in real_idx):
                covered += 1
        self.coverage_val = covered / max(1, len(valid_calibration_rows))
        calibration_set_sizes = []
        for i in valid_calibration_rows:
            mask = (1.0 - P_cal[i]) <= self.threshold
            calibration_set_sizes.append(int(np.count_nonzero(mask)))
        size_array = np.asarray(calibration_set_sizes, dtype=int)
        fraction_all = float(np.mean(size_array == MAX_NUM)) if len(size_array) else None
        fraction_large = float(np.mean(size_array >= int(0.9 * MAX_NUM))) if len(size_array) else None
        warning = None
        if self.threshold == float("inf") or (len(size_array) and np.median(size_array) == MAX_NUM):
            warning = "El conjunto conformal suele incluir los 49 números y no reduce candidatos."
        elif len(size_array) and fraction_large >= 0.5:
            warning = "Al menos la mitad de las filas de calibración generan conjuntos con 90% o más de los números."

        self.diagnostics = {
            "available": True, "fitted": True,
            "threshold": self.threshold,
            "coverage_val": self.coverage_val,
            "calibration_rows": int(n_scores),
            "valid_calibration_rows": int(len(valid_calibration_rows)),
            "finite_sample_order_k": int(k),
            "mean_calibration_set_size": float(np.mean(size_array)) if len(size_array) else None,
            "median_calibration_set_size": float(np.median(size_array)) if len(size_array) else None,
            "fraction_all_49": fraction_all,
            "fraction_size_ge_90pct": fraction_large,
            "set_saturation_warning": warning,
            "diagnostic_scope": "Tamaño y cobertura medidos en el bloque usado para ajustar el umbral; no son una estimación independiente de cobertura futura.",
        }

        log.log(
            f"Conformal alpha={self.alpha:.2f}: umbral={self.threshold:.4f} · "
            f"cobertura de calibración={self.coverage_val*100:.2f}% · "
            f"tamaño mediano conjunto={np.median(size_array):.0f}/49",
            "MODEL",
        )
        if warning:
            log.warn(warning)
        return dict(self.diagnostics)

    def predict_set(self, p):
        if not self.fitted:
            k = max(6, int(round(MAX_NUM * (1 - self.alpha))))
            return set(np.argsort(p)[-k:] + 1)
        p = np.asarray(p, dtype=float)
        mask = (1.0 - p) <= self.threshold
        s = set((np.where(mask)[0] + 1).tolist())
        if len(s) < 6:
            top = np.argsort(p)[-6:] + 1
            s.update(top.tolist())
        return s


# ----------------------------------------------------------------
# 5.3 — OPTIMIZADOR DE PESOS DEL ENSEMBLE
# ----------------------------------------------------------------

class WeightOptimizer:
    """Busca los pesos del ensemble minimizando Brier y maximizando
    hits@12/18 sobre un conjunto de validación temporal."""

    def __init__(self, n_dirichlet=30000, n_refine=10000):
        self.n_dirichlet = n_dirichlet
        self.n_refine = n_refine
        self.weights = None
        self.report = {}

    def optimize(self, y_val, components, progress_callback=None):
        valid = [(i, normalize_rows(p)) for i, p in enumerate(components) if p is not None]
        if not valid or len(y_val) == 0:
            self.weights = np.zeros(len(components))
            if valid:
                self.weights[[i for i, _ in valid]] = 1.0 / len(valid)
            return self.weights

        y_val = np.asarray(y_val, dtype=float)
        n_comp = len(valid)
        stack = np.stack([p for _, p in valid], axis=0)

        rng = np.random.default_rng(ULTIMATE_SEED + 777)

        candidates = []
        for i in range(n_comp):
            w = np.zeros(n_comp); w[i] = 1.0
            candidates.append(w)
        candidates.append(np.ones(n_comp) / n_comp)

        if progress_callback:
            progress_callback(88.0, "Pesos: generando combinaciones Dirichlet")
        alphas = (0.25, 0.5, 1.0, 2.0)
        per = self.n_dirichlet // len(alphas)
        for a in alphas:
            candidates.extend(rng.dirichlet(np.full(n_comp, a), size=per))

        candidates = np.asarray(candidates, dtype=float)
        best_obj = float("inf")
        best_w = candidates[0].copy()
        best_metrics = {}

        batch = 256
        total = len(candidates)
        for start in range(0, total, batch):
            pause_checkpoint(f"optimización de pesos, candidatos {start}/{total}")
            end = min(total, start + batch)
            if progress_callback:
                progress_callback(
                    88.0 + 4.0 * (end / total),
                    f"Pesos: evaluando {end:,}/{total:,}",
                )
            w_batch = candidates[start:end]
            mix = np.einsum("bc,cnk->bnk", w_batch, stack)
            # Cada componente representa probabilidades marginales de inclusión
            # que suman 6 por sorteo, no una distribución categórica que suma 1.
            # Los pesos suman 1, por tanto el promedio ponderado ya suma 6.
            # Dividir por mix.sum(axis=2) era un error: optimizaba contra Y a escala
            # de suma 1, distinta de la utilizada por FuxingPredictor._combine.

            yb = y_val[None, :, :]
            briers = np.mean((yb - mix) ** 2, axis=(1, 2))

            top12 = np.argpartition(mix, -12, axis=2)[:, :, -12:]
            top18 = np.argpartition(mix, -18, axis=2)[:, :, -18:]
            actual = y_val[None, :, :] > 0.5
            h12 = np.mean(np.sum(np.take_along_axis(actual, top12, axis=2), axis=2), axis=1)
            h18 = np.mean(np.sum(np.take_along_axis(actual, top18, axis=2), axis=2), axis=1)

            objs = briers - 0.010 * h12 - 0.004 * h18
            local = int(np.argmin(objs))
            if objs[local] < best_obj:
                best_obj = float(objs[local])
                best_w = w_batch[local].copy()
                best_metrics = {
                    "brier": float(briers[local]),
                    "hits12": float(h12[local]),
                    "hits18": float(h18[local]),
                }

        if progress_callback:
            progress_callback(92.0, f"Pesos: refinamiento local ({self.n_refine:,} iter)")
        local_w = best_w.copy()
        for radius in (0.10, 0.05, 0.02, 0.01):
            pause_checkpoint("refinamiento de pesos")
            for k in range(self.n_refine // 4):
                if k % 50 == 0:
                    pause_checkpoint(f"refinamiento de pesos, iteración {k}/{self.n_refine // 4}")
                trial = np.clip(local_w + rng.normal(0, radius, size=n_comp), 0.0, None)
                if trial.sum() <= 0:
                    continue
                trial /= trial.sum()

                mix = np.einsum("c,cnk->nk", trial, stack)
                # Mantener la escala de inclusión marginal: suma 6 por fila.
                brier = float(np.mean((y_val - mix) ** 2))

                top12 = np.argpartition(mix, -12, axis=1)[:, -12:]
                top18 = np.argpartition(mix, -18, axis=1)[:, -18:]
                actual = y_val > 0.5
                h12 = float(np.mean(np.sum(np.take_along_axis(actual, top12, axis=1), axis=1)))
                h18 = float(np.mean(np.sum(np.take_along_axis(actual, top18, axis=1), axis=1)))
                obj = brier - 0.010 * h12 - 0.004 * h18

                if obj < best_obj:
                    best_obj = obj
                    best_w = trial.copy()
                    local_w = trial.copy()
                    best_metrics = {"brier": brier, "hits12": h12, "hits18": h18}

        full = np.zeros(len(components), dtype=float)
        for w, (idx, _) in zip(best_w, valid):
            full[idx] = w
        self.weights = full
        self.report = {
            "weights": full.tolist(),
            "components_present": [i for i, _ in valid],
            "probability_scale": "marginal_inclusion_probabilities_row_sum_6",
            "objective": "brier - 0.010*hits@12 - 0.004*hits@18",
            **best_metrics,
        }
        log.log(
            "Pesos óptimos: " + ", ".join(f"{x:.3f}" for x in full),
            "OPTIM",
        )
        log.log(
            f"Validación: Brier={best_metrics.get('brier', 0):.6f} · "
            f"hits@12={best_metrics.get('hits12', 0):.3f} · "
            f"hits@18={best_metrics.get('hits18', 0):.3f}",
            "OPTIM",
        )
        return full
    
# ================================================================
# PARTE 6 — VALIDACIÓN RIGUROSA
# ================================================================

class NestedWalkForward:
    def __init__(self, n_splits=6, purge=SEQ_LEN, embargo=7):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.results = []

    def run(self, X_seq, X_feat, Y, train_fn, predict_fn, progress_callback=None):
        n = len(Y)
        if n < self.n_splits * 100:
            log.warn(
                f"Walk-forward: pocas muestras ({n}), se reduce a 3 splits."
            )
            self.n_splits = 3

        block = n // (self.n_splits + 1)
        results = []

        for k in range(self.n_splits):
            pause_checkpoint(f"validación walk-forward, split {k + 1}/{self.n_splits}")
            test_start = (k + 1) * block
            test_end = min(n, test_start + block)
            train_end = max(0, test_start - self.purge)
            if train_end < 200 or test_end <= test_start:
                continue

            Xtr_seq = X_seq[:train_end]
            Xtr_feat = X_feat[:train_end]
            Ytr = Y[:train_end]
            Xte_seq = X_seq[test_start:test_end]
            Xte_feat = X_feat[test_start:test_end]
            Yte = Y[test_start:test_end]

            if progress_callback:
                progress_callback(
                    92.0 + 2.0 * (k / self.n_splits),
                    f"Walk-forward: split {k+1}/{self.n_splits} "
                    f"(train={train_end}, test={test_end - test_start})",
                )

            try:
                model = train_fn(Xtr_seq, Xtr_feat, Ytr)
                Pte = predict_fn(model, Xte_seq, Xte_feat, test_start)
            except Exception as e:
                log.warn(f"Walk-forward split {k+1}: {e}")
                continue

            if Pte is None:
                continue

            brier = float(np.mean((Yte - Pte) ** 2))
            h12 = self._hits(Pte, Yte, 12)
            h18 = self._hits(Pte, Yte, 18)

            results.append({
                "split": k + 1,
                "train_size": int(train_end),
                "test_size": int(test_end - test_start),
                "brier": brier,
                "hits_top12": h12,
                "hits_top18": h18,
            })
            log.log(
                f"Walk-forward {k+1}/{self.n_splits}: "
                f"Brier={brier:.6f} hits@12={h12:.3f} hits@18={h18:.3f}",
                "AUDIT",
            )

        self.results = results
        if not results:
            return {"tests": 0}

        summary = {
            "tests": len(results),
            "mean_brier": float(np.mean([r["brier"] for r in results])),
            "mean_hits_top12": float(np.mean([r["hits_top12"] for r in results])),
            "mean_hits_top18": float(np.mean([r["hits_top18"] for r in results])),
            "splits": results,
        }
        log.log(
            f"Walk-forward resumen: {summary['tests']} splits · "
            f"Brier={summary['mean_brier']:.6f} · "
            f"hits@12={summary['mean_hits_top12']:.3f} · "
            f"hits@18={summary['mean_hits_top18']:.3f}",
            "AUDIT",
        )
        return summary

    @staticmethod
    def _hits(P, Y, k):
        total = 0.0
        for i in range(len(Y)):
            pred = set(np.argsort(P[i])[-k:])
            actual = set(np.where(Y[i] > 0.5)[0])
            total += len(pred & actual)
        return total / max(1, len(Y))


# ----------------------------------------------------------------
# 6.2 — DIEBOLD-MARIANO CONTRA BASELINE ALEATORIO
# ----------------------------------------------------------------

def diebold_mariano(model_probs, Y, baseline_probs=None, max_lag=None):
    """Comparación pareada de Brier frente a una referencia, con varianza HAC.

    mean_diff = Brier_referencia - Brier_modelo; por tanto, positivo favorece
    al modelo y negativo favorece a la referencia. Es una prueba aproximada,
    interpretable sólo si las predicciones son genuinamente fuera de muestra.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != MAX_NUM:
        return {"n": 0, "error": "Y debe tener forma (n, 49)"}
    try:
        model_probs = normalize_rows(np.asarray(model_probs, dtype=float))
        if baseline_probs is None:
            baseline_probs = np.full_like(Y, NUMBERS_PER_DRAW / MAX_NUM)
        else:
            baseline_probs = normalize_rows(np.asarray(baseline_probs, dtype=float))
    except Exception as exc:
        return {"n": 0, "error": f"No se pudieron normalizar probabilidades: {exc}"}
    if model_probs.shape != Y.shape or baseline_probs.shape != Y.shape:
        return {"n": 0, "error": "Dimensiones incompatibles"}
    if not (np.isfinite(Y).all() and np.isfinite(model_probs).all() and np.isfinite(baseline_probs).all()):
        return {"n": 0, "error": "Datos no finitos en predicciones o etiquetas"}

    loss_model = np.mean((Y - model_probs) ** 2, axis=1)
    loss_base = np.mean((Y - baseline_probs) ** 2, axis=1)
    d = np.asarray(loss_base - loss_model, dtype=float)
    n = len(d)
    if n < 10:
        return {
            "n": n, "dm_stat": None, "p_value": None,
            "mean_diff": float(d.mean()) if n else None,
            "model_brier": float(loss_model.mean()) if n else None,
            "baseline_brier": float(loss_base.mean()) if n else None,
            "direction": "muestra_insuficiente",
            "method": "muestra insuficiente",
        }

    mean_d = float(d.mean())
    centered = d - mean_d
    lag = int(max_lag if max_lag is not None else min(7, max(1, int(n ** (1 / 3)))))
    lag = max(0, min(lag, n - 1))
    long_var = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        long_var += 2.0 * weight * gamma
    long_var = max(long_var, 1e-18)
    dm_stat = mean_d / math.sqrt(long_var / n)
    # erfc evita la cancelación numérica de 1-Phi(z), que convertía p en 0.0.
    p_value = math.erfc(abs(dm_stat) / math.sqrt(2.0))
    significant = bool(p_value < 0.05)
    improvement = bool(significant and mean_d > 0)
    deterioration = bool(significant and mean_d < 0)
    if mean_d > 0:
        direction = "modelo_mejor_en_Brier"
    elif mean_d < 0:
        direction = "referencia_mejor_en_Brier"
    else:
        direction = "empate_en_Brier"

    result = {
        "n": n,
        "mean_diff": mean_d,
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "model_brier": float(loss_model.mean()),
        "baseline_brier": float(loss_base.mean()),
        "model_better": bool(mean_d > 0),
        "direction": direction,
        "significant_difference_5pct": significant,
        "significant_improvement_5pct": improvement,
        "significant_deterioration_5pct": deterioration,
        # Compatibilidad con informes anteriores: significa mejora significativa, no cualquier diferencia.
        "significant_5pct": improvement,
        "method": "Brier pareado con varianza HAC/Newey-West; aproximación normal bilateral",
        "hac_max_lag": lag,
    }
    verdict = (
        "mejora significativa del modelo" if improvement else
        "deterioro significativo del modelo" if deterioration else
        "diferencia no significativa al 5%"
    )
    log.log(
        f"DM-HAC: {verdict}; diff={mean_d:+.6f}; "
        f"Brier modelo={result['model_brier']:.6f}; referencia={result['baseline_brier']:.6f}; "
        f"DM={dm_stat:+.3f}; p={p_value:.3g}",
        "AUDIT",
    )
    return result


# ----------------------------------------------------------------
# 6.3 — BACKTESTING DIARIO
# ----------------------------------------------------------------

class DailyBacktester:
    """Walk-forward: en cada punto, top-K números del motor
    estadístico vs sorteo real. Mide hits."""

    def __init__(self, data, top_k=12, start=800):
        self.data = sorted(data, key=lambda x: x[0])
        self.top_k = top_k
        self.start = start

    def run(self, step=1, progress_callback=None):
        n = len(self.data)
        if n <= self.start + 1:
            return {"tests": 0}

        hits = []
        six_hit = 0
        total = (n - 1 - self.start) // step
        done = 0

        for i in range(self.start, n - 1, step):
            if (i - self.start) % 5 == 0:
                pause_checkpoint(f"backtest diario, sorteo {i}/{n - 1}")
            train = self.data[:i]
            actual = set(self.data[i][1])
            engine = StatisticalEngine(train)
            p, _ = engine.probability_vector()
            top = set(np.argsort(p)[-self.top_k:] + 1)
            h = len(actual & top)
            hits.append(h)
            if h == 6:
                six_hit += 1
            done += 1
            if progress_callback and done % 25 == 0:
                progress_callback(
                    30.0 + 10.0 * (done / max(1, total)),
                    f"Backtest diario: {done}/{total}",
                )

        arr = np.asarray(hits, dtype=float)
        expected_random = NUMBERS_PER_DRAW * self.top_k / MAX_NUM
        result = {
            "tests": int(len(arr)),
            "mean_hits_top12": float(arr.mean()) if len(arr) else 0.0,
            "median_hits_top12": float(np.median(arr)) if len(arr) else 0.0,
            "p_ge_3": float(np.mean(arr >= 3)) if len(arr) else 0.0,
            "p_ge_4": float(np.mean(arr >= 4)) if len(arr) else 0.0,
            "p_ge_5": float(np.mean(arr >= 5)) if len(arr) else 0.0,
            "p_6": float(np.mean(arr == 6)) if len(arr) else 0.0,
            "six_hit_count": int(six_hit),
            "random_expected_hits_top12": float(expected_random),
            "lift_over_random": float(arr.mean() / expected_random) if expected_random else 0.0,
        }
        with open(BACKTEST_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.log(
            f"Backtest diario: {result['tests']} sorteos · "
            f"media={result['mean_hits_top12']:.3f} · "
            f"lift={result['lift_over_random']:.3f}",
            "AUDIT",
        )
        return result
    
# ================================================================
# PARTE 7 — GENERACIÓN DE APUESTAS
# ================================================================

class VectorizedMonteCarlo:
    def __init__(self, n_iterations=1_000_000, seed=ULTIMATE_SEED):
        self.n_iterations = int(n_iterations)
        self.seed = int(seed)

    def sample(self, probs, constraints=True, progress_callback=None, batch_size=50000):
        probs = normalize_mass(probs)
        rng = np.random.default_rng(self.seed)

        freq = np.zeros(MAX_NUM, dtype=np.float64)
        valid = 0
        done = 0
        total = self.n_iterations

        while done < total:
            pause_checkpoint(f"Monte Carlo, iteración {done:,}/{total:,}")
            b = min(batch_size, total - done)
            u = rng.random((b, MAX_NUM))
            logits = np.log(np.clip(probs, 1e-12, None)) - np.log(
                -np.log(np.clip(u, 1e-12, 1 - 1e-12))
            )
            top6 = np.argpartition(-logits, 6, axis=1)[:, :6]

            if constraints:
                sorted_top = np.sort(top6 + 1, axis=1)
                evens = (sorted_top % 2 == 0).sum(axis=1)
                lows = (sorted_top <= 24).sum(axis=1)
                sums = sorted_top.sum(axis=1)
                spread = sorted_top[:, -1] - sorted_top[:, 0]
                diffs = np.diff(sorted_top, axis=1) == 1
                cons = diffs.sum(axis=1)

                mask = (
                    (evens >= 1) & (evens <= 5) &
                    (lows >= 1) & (lows <= 5) &
                    (sums >= 45) & (sums <= 255) &
                    (spread >= 10) &
                    (cons <= 3)
                )
                kept = sorted_top[mask]
            else:
                kept = np.sort(top6 + 1, axis=1)

            if len(kept):
                valid += len(kept)
                flat = kept.flatten() - 1
                np.add.at(freq, flat, 1.0)

            done += b
            if progress_callback and (done % batch_size == 0 or done == total):
                progress_callback(
                    55.0 + 8.0 * (done / total),
                    f"Monte Carlo: {done:,}/{total:,}",
                )

        if valid == 0:
            return probs
        return normalize_mass(freq / valid)

    def top_combinations(self, probs, n=15, n_candidates=None, progress_callback=None):
        if n_candidates is None:
            n_candidates = min(2_000_000, max(200_000, n * 100_000))

        probs = normalize_mass(probs)
        rng = np.random.default_rng(self.seed + 7)

        pool_chunks = []
        total_unique = 0
        done = 0
        batch_size = 50000

        while done < n_candidates and total_unique < n_candidates:
            pause_checkpoint(f"generación de candidatos, {done:,}/{n_candidates:,}")
            b = min(batch_size, n_candidates - done)
            u = rng.random((b, MAX_NUM))
            logits = np.log(np.clip(probs, 1e-12, None)) - np.log(
                -np.log(np.clip(u, 1e-12, 1 - 1e-12))
            )
            top6 = np.argpartition(-logits, 6, axis=1)[:, :6]
            top6 = np.sort(top6 + 1, axis=1)

            evens = (top6 % 2 == 0).sum(axis=1)
            lows = (top6 <= 24).sum(axis=1)
            sums = top6.sum(axis=1)
            spread = top6[:, -1] - top6[:, 0]
            cons = (np.diff(top6, axis=1) == 1).sum(axis=1)
            mask = (
                (evens >= 1) & (evens <= 5) &
                (lows >= 1) & (lows <= 5) &
                (sums >= 45) & (sums <= 255) &
                (spread >= 10) &
                (cons <= 3)
            )
            kept = top6[mask]

            if len(kept):
                unique_batch = np.unique(kept, axis=0)
                pool_chunks.append(unique_batch)
                if len(pool_chunks) > 1:
                    merged = np.unique(np.vstack(pool_chunks), axis=0)
                    pool_chunks = [merged]
                total_unique = sum(len(c) for c in pool_chunks)

            done += b
            if progress_callback and (done % batch_size == 0 or done >= n_candidates):
                progress_callback(
                    60.0 + 8.0 * (done / max(1, n_candidates)),
                    f"Candidatos únicos: {total_unique:,} / {done:,} generados",
                )
            if total_unique >= n_candidates:
                break

        if not pool_chunks:
            return []

        final = pool_chunks[0]
        if len(final) > n_candidates:
            idx = rng.permutation(len(final))[:n_candidates]
            final = final[idx]

        return [tuple(int(x) for x in row) for row in final]


# ----------------------------------------------------------------
# 7.2 — MODELO DE POPULARIDAD
# ----------------------------------------------------------------

class PopularityModel:
    def __init__(self):
        self.birthday_penalty = 0.15
        self.sequence_penalty = 0.30
        self.low_cluster_penalty = 0.10
        self.repeated_decade_penalty = 0.08
        self.diagonal_penalty = 0.12
        self.mod5_penalty = 0.06

    def score(self, draw):
        draw = tuple(sorted(draw))
        s = 0.0

        n31 = sum(n <= 31 for n in draw)
        if n31 >= 5:
            s -= self.birthday_penalty
        if n31 == 6:
            s -= self.birthday_penalty * 0.7

        n12 = sum(n <= 12 for n in draw)
        if n12 >= 4:
            s -= self.low_cluster_penalty

        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        if cons >= 4:
            s -= self.sequence_penalty
        elif cons >= 2:
            s -= self.sequence_penalty * 0.3

        decades = set((n - 1) // 10 for n in draw)
        if len(decades) <= 2:
            s -= self.repeated_decade_penalty

        if len({n % 5 for n in draw}) <= 2:
            s -= self.mod5_penalty

        diag1 = {1, 8, 15, 22, 29, 36, 43}
        diag2 = {7, 14, 21, 28, 35, 42, 49}
        if len(set(draw) & diag1) >= 4 or len(set(draw) & diag2) >= 4:
            s -= self.diagonal_penalty

        return s


# ----------------------------------------------------------------
# 7.3 — PUNTUACIÓN ESTRUCTURAL
# ----------------------------------------------------------------

class StructuralScorer:
    def __init__(self, data):
        self.data = sorted(data, key=lambda x: x[0])
        self.draws = [tuple(d) for _, d in self.data]
        self._build_reference_stats()

    def _build_reference_stats(self):
        recent = self.draws[-500:]
        self.sums = np.array([sum(d) for d in recent], dtype=float)
        self.spreads = np.array([draw_range(d) for d in recent], dtype=float)
        self.parities = Counter(even_count(d) for d in recent)
        self.lows = Counter(low_count(d) for d in recent)
        self.historical = set(tuple(sorted(d)) for d in self.draws)

    def score(self, draw):
        draw = tuple(sorted(draw))
        s = 0.0
        total = sum(draw)
        even = even_count(draw)
        low = low_count(draw)
        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        spread = draw_range(draw)

        s += math.log1p(self.parities.get(even, 0))
        s += math.log1p(self.lows.get(low, 0))

        if len(self.sums):
            z = abs((total - self.sums.mean()) / (self.sums.std() + 1e-9))
            s -= min(z, 5.0) * 0.12

        if cons > 2:
            s -= 0.25 * (cons - 2)
        if spread < 15:
            s -= 0.20
        if spread > 48:
            s -= 0.10

        return s


# ----------------------------------------------------------------
# 7.4 — GENERADOR DE CARTERA DIVERSIFICADA
# ----------------------------------------------------------------

class BetPortfolio:
    """Selecciona N combinaciones que maximicen score global y
    diversidad (poca superposición entre ellas)."""

    def __init__(self, data, number_probs):
        self.data = sorted(data, key=lambda x: x[0])
        self.number_probs = normalize_mass(number_probs)
        self.structural = StructuralScorer(self.data)
        self.popularity = PopularityModel()

    def _combo_score(self, draw):
        s = 0.0
        for n in draw:
            s += math.log(self.number_probs[n - 1] + 1e-12)
        s += self.structural.score(draw)
        s += self.popularity.score(draw)
        if tuple(sorted(draw)) in self.structural.historical:
            s -= 0.04
        return float(s)

    def select(self, candidates, n=15, max_overlap=4, progress_callback=None):
        if not candidates:
            return []

        scored = []
        total = len(candidates)
        for i, c in enumerate(candidates):
            scored.append((self._combo_score(c), c))
            if progress_callback and i % 5000 == 0:
                progress_callback(
                    70.0 + 6.0 * (i / max(1, total)),
                    f"Cartera: puntuando {i:,}/{total:,}",
                )

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for sc, draw in scored:
            if all(len(set(draw) & set(prev)) <= max_overlap for prev in selected):
                selected.append(draw)
            if len(selected) >= n:
                break

        return selected
    
# ================================================================
# PARTE 8 — MOTOR SEMANAL ROBUSTO
# ================================================================

class WeeklyEngine:
    WINDOWS = (10, 25, 50, 100, 250, 500, 1000, 2000)
    WINDOW_WEIGHTS = {
        10: 0.10, 25: 0.13, 50: 0.15, 100: 0.15,
        250: 0.14, 500: 0.12, 1000: 0.11, 2000: 0.10,
    }

    def __init__(self, data, progress_callback=None, model_probabilities=None):
        self.data = sorted(data, key=lambda x: x[0])
        self.draws = [tuple(d) for _, d in self.data]
        self.progress = progress_callback or (lambda p, m: None)
        self.model_probs = (
            normalize_mass(model_probabilities)
            if model_probabilities is not None else None
        )
        self._pair_cache = None

    @staticmethod
    def _freq(rows):
        f = np.zeros(MAX_NUM, dtype=np.float64)
        for _, d in rows:
            for n in d:
                f[n - 1] += 1.0
        s = f.sum()
        return f / s if s > 0 else np.full(MAX_NUM, 1.0 / MAX_NUM)

    def _multi_scale(self):
        components = []
        global_p = self._freq(self.data)
        for w, wt in self.WINDOW_WEIGHTS.items():
            rows = self.data[-min(w, len(self.data)):]
            p = self._freq(rows)
            p = 0.85 * p + 0.15 * global_p
            components.append((wt, p))
        total = sum(w for w, _ in components)
        base = sum((w / total) * p for w, p in components)
        return normalize_mass(base), components

    def _stability(self, components):
        matrix = np.vstack([p for _, p in components])
        mean_p = matrix.mean(axis=0)
        var = matrix.std(axis=0)
        stab = mean_p / (1.0 + 8.0 * var)
        return normalize_mass(stab)

    def _analog(self, n_anchors=500):
        n = len(self.draws)
        if n < 1000:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0

        def signature(end):
            rows = self.data[:end]
            parts = []
            for w in (10, 25, 50, 100, 250):
                parts.append(self._freq(rows[-w:]))
            recent = self.draws[max(0, end - 50):end]
            if recent:
                sums = np.array([sum(d) for d in recent], dtype=float)
                evens = np.array([even_count(d) for d in recent], dtype=float)
                extra = np.array([sums.mean(), sums.std(), evens.mean(), evens.std()])
            else:
                extra = np.zeros(4)
            return np.concatenate(parts + [extra])

        current = signature(n)
        lo, hi = 700, n - 8
        if hi <= lo:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0
        step = max(1, (hi - lo) // n_anchors)
        candidates = list(range(lo, hi, step))[-n_anchors:]

        scale = np.std(current) + 1e-9
        scored = []
        for anchor in candidates:
            sig = signature(anchor)
            dist = float(np.mean(((sig - current) / scale) ** 2))
            scored.append((dist, anchor))
        scored.sort(key=lambda x: x[0])
        chosen = scored[:min(60, len(scored))]

        out = np.zeros(MAX_NUM, dtype=np.float64)
        total_w = 0.0
        for rank, (dist, anchor) in enumerate(chosen):
            w = math.exp(-min(dist, 25.0) / 4.0) / (1.0 + 0.015 * rank)
            for _, draw in self.data[anchor:anchor + 7]:
                for nmb in draw:
                    out[nmb - 1] += w
            total_w += 7.0 * w
        if total_w <= 0:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0
        out /= total_w

        best_dist = float(chosen[0][0]) if chosen else 0.0
        reliability = math.exp(-min(best_dist, 20.0) / 5.0)
        return normalize_mass(out), reliability, len(chosen)

    def _pair_triple_scores(self):
        if self._pair_cache is not None:
            return self._pair_cache
        rows = self.data[-1000:]
        pair = Counter()
        triple = Counter()
        for _, draw in rows:
            for p in combinations(draw, 2):
                pair[p] += 1
            for tri in combinations(draw, 3):
                triple[tri] += 1

        pair_score = np.zeros(MAX_NUM, dtype=np.float64)
        for (a, b), c in pair.items():
            pair_score[a - 1] += c
            pair_score[b - 1] += c
        pair_score = normalize_mass(pair_score)

        triple_score = np.zeros(MAX_NUM, dtype=np.float64)
        for tri, c in triple.items():
            for n in tri:
                triple_score[n - 1] += c
        triple_score = normalize_mass(triple_score)

        self._pair_cache = (pair_score, triple_score, pair, triple)
        return self._pair_cache

    def robust_vector(self):
        self.progress(6, "Semanal: calculando multiescala...")
        base, comps = self._multi_scale()

        self.progress(20, "Semanal: evaluando analogías históricas...")
        analog, analog_rel, analog_n = self._analog()

        self.progress(38, "Semanal: midiendo estabilidad entre ventanas...")
        stab = self._stability(comps)

        self.progress(48, "Semanal: coocurrencias de pares y triples...")
        pair_s, triple_s, pair, triple = self._pair_triple_scores()

        global_p = self._freq(self.data)

        wa = 0.22 * analog_rel
        wb = 0.28
        ws = 0.22
        wp = 0.12
        wt = 0.06
        wg = max(0.0, 1.0 - (wa + wb + ws + wp + wt))

        robust = (
            wg * global_p +
            wb * base +
            wa * analog +
            ws * stab +
            wp * pair_s +
            wt * triple_s
        )
        robust = normalize_mass(robust)

        if self.model_probs is not None:
            robust = normalize_mass(0.85 * robust + 0.15 * self.model_probs)
            self.progress(56, "Semanal: integrando señal del ensemble Fuxing...")

        details = {
            "base": base,
            "analog": analog,
            "stability": stab,
            "pair": pair_s,
            "triple": triple_s,
            "analog_reliability": analog_rel,
            "analog_count": analog_n,
            "pair_count": len(pair),
            "triple_count": len(triple),
            "weights": {"global": wg, "base": wb, "analog": wa,
                        "stability": ws, "pair": wp, "triple": wt},
        }
        return robust, details

    def combo_score(self, draw, probs, details):
        pair_s, triple_s, pair, triple = self._pair_triple_scores()
        stab = details["stability"]
        analog = details["analog"]

        s = 0.0
        for n in draw:
            s += math.log(probs[n - 1] + 1e-12)
            s += 0.55 * math.log(stab[n - 1] + 1e-12)
            s += 0.35 * math.log(analog[n - 1] + 1e-12)

        for a, b in combinations(draw, 2):
            s += 0.012 * math.log1p(pair.get((a, b), 0))
        for tri in combinations(draw, 3):
            s += 0.006 * math.log1p(triple.get(tri, 0))

        even = even_count(draw)
        low = low_count(draw)
        total = sum(draw)
        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        spread = draw_range(draw)

        if even in (0, 6): s -= 0.35
        if low in (0, 6): s -= 0.30
        if total < 75 or total > 225: s -= 0.10
        if cons > 2: s -= 0.18 * (cons - 2)
        if spread < 15: s -= 0.18
        if sum(n <= 31 for n in draw) >= 5: s -= 0.08

        return float(s)

    def deterministic_best(self, probs, details):
        top = sorted((np.argsort(probs)[::-1][:12] + 1).tolist())
        best, best_s = None, -float("inf")
        for combo in combinations(top, 6):
            if not self._valid(combo):
                continue
            sc = self.combo_score(combo, probs, details)
            if sc > best_s:
                best_s, best = sc, combo
        return tuple(best) if best else tuple(sorted(top[:6]))

    @staticmethod
    def _valid(draw):
        even = even_count(draw)
        low = low_count(draw)
        total = sum(draw)
        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        spread = draw_range(draw)
        return (
            1 <= even <= 5 and
            1 <= low <= 5 and
            45 <= total <= 255 and
            cons <= 3 and
            spread >= 10
        )

    def best_combination(self, probs, details, iterations=MC_ITERATIONS_SEMANAL):
        self.progress(62, f"Semanal: Monte Carlo {iterations:,} iteraciones...")
        probs = normalize_mass(probs)
        rng = np.random.default_rng(ULTIMATE_SEED + 9001)
        pair_s, triple_s, pair_counts, triple_counts = self._pair_triple_scores()
        stab = np.asarray(details["stability"], dtype=float)
        analog = np.asarray(details["analog"], dtype=float)

        pair_lookup = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float32)
        for (a, b), count in pair_counts.items():
            pair_lookup[a - 1, b - 1] = pair_lookup[b - 1, a - 1] = count
        triple_lookup = np.zeros((MAX_NUM, MAX_NUM, MAX_NUM), dtype=np.float32)
        for (a, b, c), count in triple_counts.items():
            triple_lookup[a - 1, b - 1, c - 1] = count

        def score_batch(rows):
            idx = rows - 1
            scores = np.log(probs[idx] + 1e-12).sum(axis=1)
            scores += 0.55 * np.log(stab[idx] + 1e-12).sum(axis=1)
            scores += 0.35 * np.log(analog[idx] + 1e-12).sum(axis=1)
            for i, j in combinations(range(6), 2):
                scores += 0.012 * np.log1p(pair_lookup[idx[:, i], idx[:, j]])
            for i, j, k in combinations(range(6), 3):
                scores += 0.006 * np.log1p(
                    triple_lookup[idx[:, i], idx[:, j], idx[:, k]]
                )

            evens = (rows % 2 == 0).sum(axis=1)
            lows = (rows <= 24).sum(axis=1)
            totals = rows.sum(axis=1)
            consecutive = (np.diff(rows, axis=1) == 1).sum(axis=1)
            spreads = rows[:, -1] - rows[:, 0]
            scores -= np.where((evens == 0) | (evens == 6), 0.35, 0.0)
            scores -= np.where((lows == 0) | (lows == 6), 0.30, 0.0)
            scores -= np.where((totals < 75) | (totals > 225), 0.10, 0.0)
            scores -= 0.18 * np.maximum(0, consecutive - 2)
            scores -= np.where(spreads < 15, 0.18, 0.0)
            scores -= np.where((rows <= 31).sum(axis=1) >= 5, 0.08, 0.0)

            valid = (
                (evens >= 1) & (evens <= 5) &
                (lows >= 1) & (lows <= 5) &
                (totals >= 45) & (totals <= 255) &
                (consecutive <= 3) & (spreads >= 10)
            )
            return scores, valid

        best, best_s = None, -float("inf")
        done = 0
        batch = 50000
        while done < iterations:
            pause_checkpoint(f"Monte Carlo semanal, {done:,}/{iterations:,}")
            b = min(batch, iterations - done)
            u = rng.random((b, MAX_NUM))
            logits = np.log(np.clip(probs, 1e-12, None)) - np.log(
                -np.log(np.clip(u, 1e-12, 1 - 1e-12))
            )
            top6 = np.argpartition(-logits, 6, axis=1)[:, :6]
            top6 = np.sort(top6 + 1, axis=1)
            scores, valid = score_batch(top6)
            if np.any(valid):
                valid_indices = np.flatnonzero(valid)
                local = valid_indices[int(np.argmax(scores[valid]))]
                if float(scores[local]) > best_s:
                    best_s = float(scores[local])
                    best = tuple(int(x) for x in top6[local])

            done += b
            self.progress(
                62 + 16 * (done / max(1, iterations)),
                f"Semanal: Monte Carlo {done:,}/{iterations:,}",
            )

        det = self.deterministic_best(probs, details)
        s_det = self.combo_score(det, probs, details)
        if s_det > best_s:
            best, best_s = det, s_det
        return best, best_s

    def backtest(self, max_weeks=80):
        n = len(self.draws)
        if n < 1100:
            return {"tests": 0, "reason": "histórico insuficiente"}

        end = n - 8
        start = max(1000, end - max_weeks * 7)
        anchors = list(range(start, end, 7))[-max_weeks:]
        if not anchors:
            return {"tests": 0}

        self.progress(80, f"Semanal: backtesting {len(anchors)} semanas...")
        hits = []
        best_hits = []
        exact = 0
        for pos, anchor in enumerate(anchors):
            train = self.data[:anchor]
            eng = WeeklyEngine(train)
            p, det = eng.robust_vector()
            combo, _ = eng.best_combination(p, det, iterations=200_000)
            future = self.data[anchor:anchor + 7]
            week_hits = [len(set(combo) & set(d)) for _, d in future]
            hits.extend(week_hits)
            best_hits.append(max(week_hits))
            if max(week_hits) == 6:
                exact += 1
            if pos % max(1, len(anchors) // 8) == 0:
                self.progress(
                    80 + 12 * (pos / max(1, len(anchors))),
                    f"Semanal: backtest {pos+1}/{len(anchors)}",
                )

        arr = np.asarray(hits, dtype=float)
        bh = np.asarray(best_hits, dtype=float)
        exp_rand = NUMBERS_PER_DRAW * 6 / MAX_NUM
        result = {
            "tests": int(len(anchors)),
            "draws_evaluated": int(len(arr)),
            "mean_hits_per_draw": float(arr.mean()) if len(arr) else 0.0,
            "median_hits_per_draw": float(np.median(arr)) if len(arr) else 0.0,
            "p_ge_2": float(np.mean(arr >= 2)) if len(arr) else 0.0,
            "p_ge_3": float(np.mean(arr >= 3)) if len(arr) else 0.0,
            "p_ge_4": float(np.mean(arr >= 4)) if len(arr) else 0.0,
            "mean_best_hits_week": float(bh.mean()) if len(bh) else 0.0,
            "p_best_ge_3": float(np.mean(bh >= 3)) if len(bh) else 0.0,
            "p_best_ge_4": float(np.mean(bh >= 4)) if len(bh) else 0.0,
            "six_hit_weeks": int(exact),
            "random_expected_hits_per_draw": float(exp_rand),
            "lift_over_random": float(arr.mean() / exp_rand) if exp_rand else 0.0,
        }
        log.log(
            f"Backtest semanal: {result['tests']} semanas · "
            f"media={result['mean_hits_per_draw']:.3f} · "
            f"mejor_semana={result['mean_best_hits_week']:.3f} · "
            f"lift={result['lift_over_random']:.3f}",
            "AUDIT",
        )
        return result

    def predict(self, mc_iterations=MC_ITERATIONS_SEMANAL, run_backtest=True):
        if len(self.draws) < 1000:
            raise ValueError(
                f"Semanal requiere >=1000 sorteos; tienes {len(self.draws)}."
            )
        self.progress(3, "Semanal: iniciando análisis robusto...")
        probs, details = self.robust_vector()
        self.progress(60, "Semanal: vector robusto consolidado")
        combo, score = self.best_combination(probs, details, iterations=mc_iterations)
        self.progress(90, "Semanal: validando estabilidad de la combinación")
        backtest = self.backtest(max_weeks=60) if run_backtest else {"tests": 0}

        stab_vals = [details["stability"][n - 1] for n in combo]
        stab_score = float(np.mean(stab_vals) / (np.max(details["stability"]) + 1e-12))
        conc = float(np.mean([probs[n - 1] for n in combo]) /
                     (np.mean(np.sort(probs)[-6:]) + 1e-12))

        self.progress(99, "Semanal: análisis terminado")

        return {
            "combination": tuple(sorted(int(x) for x in combo)),
            "score": float(score),
            "probabilities": probs,
            "details_keys": list(details.keys()),
            "stability_score": max(0.0, min(1.0, stab_score)),
            "concentration_score": max(0.0, min(1.0, conc)),
            "analog_reliability": float(details.get("analog_reliability", 0.0)),
            "analog_count": int(details.get("analog_count", 0)),
            "pair_count": int(details.get("pair_count", 0)),
            "triple_count": int(details.get("triple_count", 0)),
            "backtest": backtest,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "period_days": 7,
        }
    
# ================================================================
# PARTE 9 — ORQUESTADOR
# ================================================================

class FuxingPredictor:
    """Predictor principal. Mantiene el estado entre fases."""

    def __init__(self):
        self.data = None
        self.X_seq = None
        self.X_feat = None
        self.Y = None

        self.stat_engine = None
        self.tree_ensemble = None
        self.neural = None
        self.bayes = None

        self.calibrator = None
        self.conformal = None
        self.weights = None

        self.training_report = {}
        self.validation_report = {}
        self.backtest_report = {}

        # FIX: cache de componentes de calibración (evita recalcularlos
        # dos veces sobre el mismo rango en calibrate_and_optimize+validate).
        self._cached_cal_components = None

    def prepare(self, data):
        log.banner("FASE 1 — PREPARACIÓN DE DATOS Y FEATURES", "=")
        self.data = sorted(data, key=lambda x: x[0])
        log.log(f"Sorteos cargados: {len(self.data):,}", "DATA")
        log.log(
            f"Rango: {self.data[0][0]:%d/%m/%Y} -> {self.data[-1][0]:%d/%m/%Y}",
            "DATA",
        )

        dataset_audit(self.data)
        randomness_audit(self.data)

        self.X_seq, self.X_feat, self.Y = build_dataset(self.data, SEQ_LEN)

        feature_audit(self.X_feat, self.Y)

        self.stat_engine = StatisticalEngine(self.data)

        log.ok(
            f"Preparación completa: {len(self.X_seq):,} muestras, "
            f"{self.X_feat.shape[1]} features por muestra"
        )

    def train_models(self):
        log.banner("FASE 2 — ENTRENAMIENTO DE MODELOS", "=")

        n = len(self.X_feat)
        if n < 500:
            raise ValueError(f"Muestras insuficientes para entrenar: {n}")

        split = int(n * 0.70)
        val_end = int(n * 0.75)
        split = max(300, min(split, n - 200))
        val_end = max(split + 30, min(val_end, n))

        Xtr, Xv = self.X_feat[:split], self.X_feat[split:val_end]
        Ytr, Yv = self.Y[:split], self.Y[split:val_end]

        log.log(
            f"Split temporal: train={len(Xtr):,} val={len(Xv):,} "
            f"(sin mezcla aleatoria)",
            "MODEL",
        )

        log.log("Entrenando ensemble de árboles...", "MODEL")
        self.tree_ensemble = TreeEnsemble()
        tree_report = self.tree_ensemble.fit(
            Xtr, Ytr, Xv, Yv,
            progress_callback=lambda p, m: log.progress(p, m),
        )
        log.ok(
            f"Árboles: {tree_report.get('rows', 0):,} filas · "
            f"XGB={tree_report.get('xgboost')} "
            f"LGBM={tree_report.get('lightgbm')}"
        )

        log.log("Entrenando ensemble de Transformers...", "MODEL")
        self.neural = NeuralEnsemble(NEURAL_ENSEMBLE_SIZE)
        nn_report = self.neural.fit(
            self.X_seq, self.X_feat, self.Y,
            progress_callback=lambda p, m: log.progress(p, m),
        )
        if nn_report.get("trained"):
            log.ok(
                f"Red: {nn_report.get('ensemble_size')} modelos · "
                f"val_loss media="
                f"{np.mean([m['val_loss'] for m in nn_report.get('members', [])]):.6f}"
            )
        else:
            log.warn("Red neuronal no entrenada (muestras insuficientes).")

        self.training_report = {
            "tree": tree_report,
            "neural": nn_report,
        }

    def calibrate_and_optimize(self):
        """Optimiza pesos y calibra en bloques cronológicos separados del entrenamiento."""
        log.banner("FASE 3 — CALIBRACIÓN, CONFORMAL Y PESOS", "=")
        n = len(self.X_feat)
        if n < 500:
            self.weights = np.array([0.40, 0.15, 0.30, 0.15], dtype=float)
            log.warn("Muestras insuficientes para optimización temporal; se usan pesos base.")
            return

        opt_start, opt_end = int(n * 0.75), int(n * 0.80)
        cal_start, cal_end = int(n * 0.80), int(n * 0.85)
        conf_start, conf_end = int(n * 0.85), int(n * 0.90)

        log.log(f"Bloques temporales: optimización={opt_start}:{opt_end}, "
                f"calibración={cal_start}:{cal_end}, conformal={conf_start}:{conf_end}",
                "MODEL")

        stat_opt, logistic_opt, tree_opt, nn_opt = self._compute_components(
            range(opt_start, opt_end), "optimización de pesos"
        )
        Y_opt = self.Y[opt_start:opt_end]
        if stat_opt is not None and len(Y_opt):
            optimizer = WeightOptimizer()
            self.weights = optimizer.optimize(
                Y_opt, [stat_opt, logistic_opt, tree_opt, nn_opt],
                progress_callback=lambda p, m: log.progress(p, m),
            )
            self.training_report["weight_optimization"] = optimizer.report
        else:
            self.weights = np.array([0.40, 0.15, 0.30, 0.15], dtype=float)

        log.log("Ajustando calibración en un bloque posterior e independiente...", "MODEL")
        stat_cal, logistic_cal, tree_cal, nn_cal = self._compute_components(
            range(cal_start, cal_end), "calibración"
        )
        Y_cal = self.Y[cal_start:cal_end]
        P_cal = self._combine([stat_cal, logistic_cal, tree_cal, nn_cal], weights=self.weights)
        self.calibrator = IsotonicCalibrator()
        if P_cal is not None and len(Y_cal):
            self.calibrator.fit(P_cal, Y_cal)

        log.log("Ajustando conformal en un bloque distinto de calibración...", "MODEL")
        stat_conf, logistic_conf, tree_conf, nn_conf = self._compute_components(
            range(conf_start, conf_end), "conformal"
        )
        Y_conf = self.Y[conf_start:conf_end]
        P_conf = self._combine(
            [stat_conf, logistic_conf, tree_conf, nn_conf], weights=self.weights
        )
        if P_conf is not None and len(Y_conf):
            if self.calibrator is not None and self.calibrator.fitted:
                P_conf = self.calibrator.transform(P_conf)
            self.conformal = ConformalPredictor(alpha=0.10)
            conformal_report = self.conformal.fit(normalize_rows(P_conf), Y_conf)
            self.conformal.diagnostics = dict(conformal_report)
            self.training_report["conformal"] = conformal_report

        log.ok("Pesos finales: " + ", ".join(f"{x:.3f}" for x in self.weights))

    def _compute_components(self, index_range, label):
        idx = list(index_range)
        if not idx:
            return None, None, None, None

        stat_rows = []
        for j in idx:
            data_index = SEQ_LEN - 1 + j
            if data_index >= len(self.data) - 1:
                break
            train = self.data[:data_index + 1]
            p, _ = StatisticalEngine(train).probability_vector()
            stat_rows.append(normalize_mass(p))
            if len(stat_rows) % 50 == 0:
                log.progress(
                    45.0 + 8.0 * (len(stat_rows) / len(idx)),
                    f"{label}: estadístico {len(stat_rows)}/{len(idx)}",
                )
        stat = np.asarray(stat_rows, dtype=float) if stat_rows else None

        Xf = self.X_feat[idx[:len(stat_rows)]]
        Xs = self.X_seq[idx[:len(stat_rows)]]

        logistic = tree = nn = None
        if self.tree_ensemble is not None and self.tree_ensemble.fitted:
            # predict_proba es independiente por fila para estos estimadores: el
            # lote elimina miles de llamadas Python sin cambiar la fórmula.
            # Si un estimador no tolera el lote, se vuelve al camino original fila a fila.
            try:
                a_batch, b_batch = self.tree_ensemble.predict_components(Xf)
                if (a_batch is not None and b_batch is not None
                        and a_batch.shape == (len(Xf), MAX_NUM)
                        and b_batch.shape == (len(Xf), MAX_NUM)):
                    logistic, tree = a_batch, b_batch
                else:
                    raise ValueError("Salida por lotes con forma inesperada")
            except Exception as batch_exc:
                log.warn(f"Inferencia de árboles por lotes falló; se usa modo fila a fila: {batch_exc}")
                log_rows = []
                for i in range(len(Xf)):
                    a, b = self.tree_ensemble.predict_components(Xf[i:i + 1])
                    if a is not None and b is not None:
                        log_rows.append((a[0], b[0]))
                    else:
                        log_rows.append((np.full(MAX_NUM, 1.0 / MAX_NUM),
                                         np.full(MAX_NUM, 1.0 / MAX_NUM)))
                logistic = np.asarray([x[0] for x in log_rows])
                tree = np.asarray([x[1] for x in log_rows])

        if self.neural is not None and self.neural.fitted:
            nn = self.neural.predict(Xs, Xf)

        return stat, logistic, tree, nn

    @staticmethod
    def _combine(components, weights):
        w = np.asarray(weights, dtype=float)
        valid = [(i, normalize_rows(p)) for i, p in enumerate(components) if p is not None]
        if not valid:
            return None
        w_valid = np.array([w[i] for i, _ in valid])
        w_valid = w_valid / (w_valid.sum() + 1e-12)
        out = np.zeros_like(valid[0][1])
        for ww, (_, p) in zip(w_valid, valid):
            out += ww * p
        return normalize_rows(out)

    def validate(self, run_walk_forward=True, run_backtest=True):
        log.banner("FASE 4 — VALIDACIÓN", "=")
        n = len(self.X_feat)
        test_start = int(n * 0.90)
        test_end = int(n * 0.95)
        if test_end <= test_start:
            log.warn("No hay bloque temporal suficiente para la auditoría final.")
        else:
            stat_test, logistic_test, tree_test, nn_test = self._compute_components(
                range(test_start, test_end), "auditoría final no usada para ajuste"
            )
            Y_test = self.Y[test_start:test_end]
            P_test = self._combine(
                [stat_test, logistic_test, tree_test, nn_test],
                weights=self.weights if self.weights is not None
                        else [0.40, 0.15, 0.30, 0.15],
            )
            if P_test is not None and len(Y_test):
                P_eval = self.calibrator.transform(P_test) if (
                    self.calibrator is not None and self.calibrator.fitted
                ) else P_test
                self.validation_report["held_out_test"] = {
                    "start_index": test_start,
                    "end_index": test_end,
                    "samples": int(len(Y_test)),
                    "brier": float(np.mean((Y_test - P_eval) ** 2)),
                    "note": "Bloque cronológico reservado; no usado para entrenar, optimizar pesos ni calibrar.",
                }
                self.validation_report["diebold_mariano"] = diebold_mariano(P_eval, Y_test)

        if run_walk_forward:
            # Este backtest usa exclusivamente el motor estadístico. No se presenta
            # como validación del ensemble de árboles/Transformer.
            def train_fn(Xs, Xf, Y):
                return {"train_size": len(Y), "model": "statistical_only"}

            def predict_fn(model, Xs, Xf, test_start_idx):
                rows = []
                for j in range(len(Xf)):
                    if j % 10 == 0:
                        pause_checkpoint(f"walk-forward, muestra {j}/{len(Xf)}")
                    data_index = min(SEQ_LEN - 1 + int(test_start_idx) + j,
                                     len(self.data) - 2)
                    prefix = self.data[:data_index + 1]
                    if len(prefix) < 30:
                        return None
                    p, _ = StatisticalEngine(prefix).probability_vector()
                    # Evaluation target has six positive labels. Use the same 0..1
                    # marginal-inclusion scale (row sum 6) as production combination;
                    # a sum-1 categorical vector makes the Brier score incomparable.
                    rows.append(normalize_rows(p))
                return np.asarray(rows, dtype=float) if rows else None

            wf = NestedWalkForward(n_splits=4, purge=SEQ_LEN, embargo=7)
            wf_report = wf.run(
                self.X_seq, self.X_feat, self.Y, train_fn, predict_fn,
                progress_callback=lambda p, m: log.progress(p, m),
            )
            wf_report["scope"] = "motor estadístico únicamente; no valida el ensemble completo"
            self.validation_report["walk_forward_statistical_only"] = wf_report

        if run_backtest:
            bt = DailyBacktester(self.data, top_k=12, start=800)
            self.backtest_report["daily_statistical_only"] = bt.run(
                progress_callback=lambda p, m: log.progress(p, m)
            )

    def predict(self, mc_iterations=MC_ITERATIONS, n_combinations=15):
        log.banner("FASE 5 — PREDICCIÓN FINAL", "=")

        log.log("Calculando componentes para el próximo sorteo...", "PRED")
        seq_now, feat_now = self._future_sample()

        stat_now = normalize_mass(self.stat_engine.probability_vector()[0])[None, :]

        logistic_now = tree_now = nn_now = None
        if self.tree_ensemble is not None and self.tree_ensemble.fitted:
            a, b = self.tree_ensemble.predict_components(feat_now)
            if a is not None:
                logistic_now = a
                tree_now = b
        if self.neural is not None and self.neural.fitted:
            nn_now = self.neural.predict(seq_now, feat_now)

        combined = self._combine(
            [stat_now, logistic_now, tree_now, nn_now],
            weights=self.weights if self.weights is not None
                    else [0.40, 0.15, 0.30, 0.15],
        )[0]

        if self.calibrator is not None and self.calibrator.fitted:
            combined = self.calibrator.transform(combined)
        combined = normalize_mass(combined)

        log.log("Calculando intervalos de credibilidad bayesianos...", "PRED")
        self.bayes = EmpiricalBayes(self.data, n_bootstrap=1000, n_dirichlet=3000)
        self.bayes.fit(progress_callback=lambda p, m: log.progress(p, m))

        combined = normalize_mass(0.85 * combined + 0.15 * self.bayes.mean)

        log.log(
            f"Monte Carlo vectorizado ({mc_iterations:,} iteraciones)...",
            "PRED",
        )
        mc = VectorizedMonteCarlo(n_iterations=mc_iterations, seed=ULTIMATE_SEED)
        mc_freq = mc.sample(
            combined,
            constraints=True,
            progress_callback=lambda p, m: log.progress(p, m),
        )

        final_p = normalize_mass(0.85 * combined + 0.15 * mc_freq)

        log.log("Generando pool de combinaciones candidatas...", "PRED")
        mc_gen = VectorizedMonteCarlo(n_iterations=200_000, seed=ULTIMATE_SEED + 11)
        pool = mc_gen.top_combinations(
            final_p,
            n=n_combinations,
            n_candidates=300_000,
            progress_callback=lambda p, m: log.progress(p, m),
        )

        log.log("Seleccionando cartera diversificada...", "PRED")
        portfolio = BetPortfolio(self.data, final_p)
        selected = portfolio.select(
            pool,
            n=n_combinations,
            max_overlap=4,
            progress_callback=lambda p, m: log.progress(p, m),
        )

        conformal_set = None
        if self.conformal is not None and self.conformal.fitted:
            conformal_set = self.conformal.predict_set(normalize_rows(final_p))

        top_numbers = (np.argsort(final_p)[::-1] + 1)[:20].tolist()

        result = {
            "combinations": [tuple(sorted(c)) for c in selected],
            "primary_combination": tuple(sorted(selected[0])) if selected else None,
            "probabilities": final_p,
            "top_numbers": top_numbers,
            "bayes_mean": self.bayes.mean if self.bayes is not None else None,
            "bayes_q05": self.bayes.q05 if self.bayes is not None else None,
            "bayes_q95": self.bayes.q95 if self.bayes is not None else None,
            "bayes_p_top6": self.bayes.p_top6 if self.bayes is not None else None,
            "conformal_set": sorted(conformal_set) if conformal_set else None,
            "conformal_diagnostics": self.conformal.diagnostics if self.conformal is not None else None,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "training_report": self.training_report,
            "validation_report": self.validation_report,
            "backtest_report": self.backtest_report,
            "calibration_report": (
                self.calibrator.report if self.calibrator is not None else None
            ),
        }
        return result

    def _future_sample(self):
        """Construye la muestra que predice el sorteo SIGUIENTE."""
        draws = [tuple(d) for _, d in self.data]
        cum = FeatureEngineeringV3._build_prefix_counts(draws)
        i = len(draws) - 1
        seq = np.asarray(
            [one_hot_draw(draws[k]) for k in range(i - SEQ_LEN + 1, i + 1)],
            dtype=np.float32,
        )[None, :, :]
        feat = FeatureEngineeringV3.number_features(
            self.data, i, cum=cum, draws=draws,
        )
        return seq, np.asarray(feat, dtype=np.float32)[None, :]


# ================================================================
# PARTE 10 — CLI ORQUESTADOR + MAIN
# ================================================================

def _json_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):
        # Recorre también los elementos de arrays: ndarray.tolist() por sí solo
        # puede dejar float('nan')/float('inf') que json.dump serializa como
        # NaN/Infinity (extensiones no estándar de JSON).
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj


def _safe_json_write(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        # JSON estricto: si algún tipo no finito escapara al saneamiento,
        # fallar explícitamente en vez de escribir NaN/Infinity no estándar.
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _fmt_combo(combo):
    return " - ".join(f"{int(n):02d}" for n in combo)


def _madrid_now():
    """Devuelve la hora de Europe/Madrid o falla explícitamente si falta tzdata.

    No se permite degradar silenciosamente a UTC/zona local porque las fechas de sorteo,
    la hora de generación y la exportación se etiquetan como hora peninsular española.
    Instalar el requisito `tzdata` permite resolver Europe/Madrid también en Windows y
    en entornos Linux/Colab sin base IANA de zonas horarias instalada en el sistema.
    """
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "Fuxing necesita Python 3.9+ para gestionar Europe/Madrid. "
            "Actualiza Python e instala los requisitos extra."
        ) from exc
    try:
        return datetime.now(ZoneInfo("Europe/Madrid"))
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            "No está disponible la base de zonas horarias Europe/Madrid. "
            "Instala los requisitos con `python -m pip install -r requirements_extra.txt` "
            "(incluye tzdata) y vuelve a ejecutar. No se usará UTC silenciosamente."
        ) from exc


def _first_prediction_target_date(latest_actual_date, generated_at=None):
    """Fecha del siguiente sorteo aplicable a una predicción recién generada.

    Bonoloto celebra sorteos todos los días a las 21:30 hora peninsular. Si la
    predicción se genera antes de esa hora, puede aplicarse al sorteo de hoy; si se
    genera desde las 21:30, se asigna al día siguiente. Si el histórico ya contiene
    el resultado de hoy, el objetivo también pasa al día siguiente. Esta función solo
    afecta al registro histórico, no a los cálculos del motor predictivo.
    """
    generated_at = generated_at or _madrid_now()
    today = generated_at.date()
    if (generated_at.hour, generated_at.minute) >= (21, 30):
        earliest_target = today + timedelta(days=1)
    else:
        earliest_target = today
    if latest_actual_date is not None:
        if isinstance(latest_actual_date, datetime):
            latest_date = latest_actual_date.date()
        else:
            from datetime import date as _date
            latest_date = _date.fromisoformat(str(latest_actual_date)[:10])
        earliest_target = max(earliest_target, latest_date + timedelta(days=1))
    return earliest_target


def _assert_history_current_for_prediction(latest_actual_date, generated_at=None):
    """Impide etiquetar como predicción para mañana una salida con datos de ayer.

    La descarga histórica ocurre al comienzo de la ejecución. Si el entrenamiento
    atraviesa las 21:30 y termina después del sorteo diario, el código no puede
    reutilizar a ciegas el histórico descargado antes del sorteo: sin los seis números
    ganadores de hoy, cualquier predicción para mañana quedaría basada en datos obsoletos.
    En ese caso se aborta el registro final con un mensaje accionable; una nueva
    ejecución actualizará primero el histórico y entrenará con el dato disponible.
    No modifica los modelos ni sus cálculos.
    """
    generated_at = generated_at or _madrid_now()
    if (generated_at.hour, generated_at.minute) < (21, 30):
        return True
    if latest_actual_date is None:
        raise RuntimeError(
            "No se puede confirmar la actualidad del histórico tras las 21:30; "
            "vuelve a ejecutar Fuxing cuando SELAE o las fuentes históricas hayan publicado el resultado."
        )
    if isinstance(latest_actual_date, datetime):
        latest_date = latest_actual_date.date()
    else:
        from datetime import date as _date
        latest_date = _date.fromisoformat(str(latest_actual_date)[:10])
    if latest_date < generated_at.date():
        raise RuntimeError(
            "La ejecución ha terminado después de las 21:30, pero el histórico usado para entrenar "
            f"llega solo hasta {latest_date.isoformat()}. Para no generar una predicción para mañana "
            "con el sorteo de hoy ausente, se cancela el registro final. Vuelve a ejecutar Fuxing "
            "cuando el resultado de hoy esté publicado y pueda incorporarse al histórico."
        )
    return True


def _date_iso(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)[:10]


class PredictionHistoryStore:
    """Registro append-only de predicciones y actualización de aciertos, separado del motor."""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir else BASE_DIR
        self.data_dir = self.base_dir / "bonoloto_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.base_dir / "Predicciones y aciertos del programa"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.run_results_dir = self.base_dir / "Resultados"
        self.run_results_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "fuxing_prediction_history.sqlite3"
        self.xlsx_path = self.results_dir / "Historial_Bonoloto_Fuxing.xlsx"
        self._init_db()

    def _connect(self):
        con = sqlite3.connect(str(self.db_path), timeout=30)
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @contextmanager
    def _db_session(self):
        """Contexto de conexión SQLite que siempre confirma/revierte y cierra el archivo."""
        con = self._connect()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self):
        with self._db_session() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS tracking_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS actual_draws (
                    draw_date TEXT PRIMARY KEY,
                    winning_combo TEXT NOT NULL,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    prediction_type TEXT NOT NULL CHECK(prediction_type IN ('daily','weekly')),
                    combination_json TEXT NOT NULL,
                    target_start_date TEXT NOT NULL,
                    target_end_date TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    payload_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS prediction_targets (
                    prediction_id TEXT NOT NULL,
                    draw_date TEXT NOT NULL,
                    target_index INTEGER NOT NULL,
                    winning_combo TEXT,
                    hits INTEGER,
                    result_source TEXT,
                    evaluated_at TEXT,
                    PRIMARY KEY(prediction_id, draw_date),
                    FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_actual_draws_date ON actual_draws(draw_date);
                CREATE INDEX IF NOT EXISTS idx_predictions_type_date ON predictions(prediction_type, generated_at);
                CREATE INDEX IF NOT EXISTS idx_targets_date ON prediction_targets(draw_date);
            """)
            con.execute(
                "INSERT OR IGNORE INTO tracking_meta(key,value) VALUES('tracking_started_date',?)",
                (_madrid_now().date().isoformat(),),
            )
            con.execute(
                "INSERT OR IGNORE INTO tracking_meta(key,value) VALUES('schema_version','1')"
            )

    def start_run(self, mode):
        run_id = uuid.uuid4().hex
        now = _madrid_now().isoformat(timespec="seconds")
        with self._db_session() as con:
            con.execute(
                "INSERT INTO runs(run_id,started_at,mode,status) VALUES(?,?,?,'RUNNING')",
                (run_id, now, str(mode)),
            )
        return run_id

    def finish_run(self, run_id, status, message=""):
        if not run_id:
            return
        now = _madrid_now().isoformat(timespec="seconds")
        with self._db_session() as con:
            con.execute(
                "UPDATE runs SET finished_at=?,status=?,message=? WHERE run_id=?",
                (now, str(status), str(message)[:2000], run_id),
            )

    def sync_draws(self, data, official_dates=None):
        """Sincroniza resultados reales; una fila de SELAE validada prevalece sobre históricos secundarios."""
        official_dates = set(official_dates or ())
        now = _madrid_now().isoformat(timespec="seconds")
        with self._db_session() as con:
            for fecha, draw in data or []:
                key = _date_iso(fecha)
                combo = sorted(int(x) for x in draw)
                if not validate_draw(combo):
                    continue
                combo_text = json.dumps(combo, separators=(",", ":"))
                is_official = key in official_dates
                source = "SELAE (ficha oficial validada)" if is_official else "Histórico consolidado multifuente"
                existing = con.execute(
                    "SELECT winning_combo,source FROM actual_draws WHERE draw_date=?", (key,)
                ).fetchone()
                if existing and "SELAE" in str(existing[1]) and not is_official:
                    # Una caché secundaria posterior no puede pisar un resultado oficial previamente validado.
                    continue
                con.execute("""
                    INSERT INTO actual_draws(draw_date,winning_combo,source,updated_at)
                    VALUES(?,?,?,?)
                    ON CONFLICT(draw_date) DO UPDATE SET
                        winning_combo=excluded.winning_combo,
                        source=excluded.source,
                        updated_at=excluded.updated_at
                """, (key, combo_text, source, now))
            self._reconcile_targets(con)

    def add_prediction(self, run_id, prediction_type, combination, target_dates, mode, payload=None, generated_at=None):
        combo = sorted(int(x) for x in combination)
        if not validate_draw(combo):
            raise ValueError(f"Predicción {prediction_type} inválida: {combination!r}")
        targets = [_date_iso(x) for x in target_dates]
        if not targets:
            raise ValueError("Una predicción debe tener al menos una fecha objetivo")
        prediction_id = uuid.uuid4().hex
        generated_at = (generated_at or _madrid_now()).isoformat(timespec="seconds")
        payload_json = json.dumps(_json_safe(payload or {}), ensure_ascii=False, separators=(",", ":"))
        with self._db_session() as con:
            con.execute("""
                INSERT INTO predictions(
                    prediction_id,run_id,generated_at,prediction_type,combination_json,
                    target_start_date,target_end_date,mode,payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?)
            """, (
                prediction_id, run_id, generated_at, prediction_type,
                json.dumps(combo, separators=(",", ":")), min(targets), max(targets), str(mode), payload_json,
            ))
            for idx, target_date in enumerate(targets, start=1):
                con.execute("""
                    INSERT INTO prediction_targets(prediction_id,draw_date,target_index)
                    VALUES(?,?,?)
                """, (prediction_id, target_date, idx))
            self._reconcile_targets(con)
        return prediction_id

    @staticmethod
    def _reconcile_targets(con):
        """Recalcula aciertos sin borrar ni regenerar ninguna predicción guardada."""
        rows = con.execute("""
            SELECT t.prediction_id,t.draw_date,p.combination_json,a.winning_combo,a.source
            FROM prediction_targets t
            JOIN predictions p ON p.prediction_id=t.prediction_id
            LEFT JOIN actual_draws a ON a.draw_date=t.draw_date
        """).fetchall()
        evaluated_at = _madrid_now().isoformat(timespec="seconds")
        for prediction_id, draw_date, prediction_json, actual_json, source in rows:
            if not actual_json:
                con.execute("""
                    UPDATE prediction_targets SET winning_combo=NULL,hits=NULL,
                        result_source=NULL,evaluated_at=NULL
                    WHERE prediction_id=? AND draw_date=?
                """, (prediction_id, draw_date))
                continue
            predicted = set(json.loads(prediction_json))
            actual = set(json.loads(actual_json))
            hits = len(predicted & actual)
            con.execute("""
                UPDATE prediction_targets SET winning_combo=?,hits=?,result_source=?,evaluated_at=?
                WHERE prediction_id=? AND draw_date=?
            """, (actual_json, hits, source, evaluated_at, prediction_id, draw_date))

    def _snapshot(self):
        with self._db_session() as con:
            meta = dict(con.execute("SELECT key,value FROM tracking_meta").fetchall())
            draws = con.execute("""
                SELECT draw_date,winning_combo,source,updated_at FROM actual_draws ORDER BY draw_date
            """).fetchall()
            predictions = con.execute("""
                SELECT p.prediction_id,p.run_id,p.generated_at,p.prediction_type,p.combination_json,
                       p.target_start_date,p.target_end_date,p.mode,p.payload_json,
                       t.draw_date,t.target_index,t.winning_combo,t.hits,t.result_source,t.evaluated_at,
                       r.started_at,r.finished_at,r.status,r.message
                FROM predictions p
                JOIN prediction_targets t ON t.prediction_id=p.prediction_id
                JOIN runs r ON r.run_id=p.run_id
                ORDER BY p.generated_at,p.prediction_type,t.draw_date
            """).fetchall()
            runs = con.execute("""
                SELECT run_id,started_at,finished_at,mode,status,message FROM runs ORDER BY started_at
            """).fetchall()
        return meta, draws, predictions, runs

    def export_run_reports(self):
        """Genera informes TXT/JSON por ejecución y los actualiza cuando llegan resultados reales.

        Los registros de predicción originales permanecen append-only en SQLite. Los ficheros
        de informe se reescriben atómicamente solo para reflejar el estado más reciente de sus
        sorteos objetivo (pendiente/resultado real/aciertos); nunca recalculan ni sustituyen
        las combinaciones que generó el motor.
        """
        try:
            self.run_results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.warn(f"No se pudo crear la carpeta de informes {self.run_results_dir}: {exc}")
            return []
        try:
            with self._db_session() as con:
                runs = con.execute(
                    "SELECT run_id,started_at,finished_at,mode,status,message FROM runs ORDER BY started_at"
                ).fetchall()
                all_run_payloads = []
                for run_id, started_at, finished_at, mode, status, message in runs:
                    prediction_rows = con.execute("""
                        SELECT p.prediction_id,p.generated_at,p.prediction_type,p.combination_json,
                               t.draw_date,t.target_index,t.winning_combo,t.hits,t.result_source,t.evaluated_at
                        FROM predictions p
                        LEFT JOIN prediction_targets t ON t.prediction_id=p.prediction_id
                        WHERE p.run_id=?
                        ORDER BY p.generated_at,p.prediction_type,t.draw_date
                    """, (run_id,)).fetchall()
                    # No crear un informe provisional vacío para una ejecución que sigue
                    # trabajando; se crea al finalizar o cuando ya contiene predicciones.
                    if str(status) == "RUNNING" and not prediction_rows:
                        continue
                    predictions_by_id = {}
                    for row in prediction_rows:
                        (prediction_id, generated_at, prediction_type, combo_json, draw_date,
                         target_index, winning_json, hits, result_source, evaluated_at) = row
                        item = predictions_by_id.setdefault(prediction_id, {
                            "prediction_id": prediction_id,
                            "tipo": "diaria" if prediction_type == "daily" else "semanal",
                            "tipo_codigo": prediction_type,
                            "generada_en_madrid": generated_at,
                            "combinacion": json.loads(combo_json),
                            "combinacion_texto": " - ".join(f"{n:02d}" for n in json.loads(combo_json)),
                            "objetivos": [],
                        })
                        if draw_date is not None:
                            winning = json.loads(winning_json) if winning_json else None
                            item["objetivos"].append({
                                "fecha_sorteo_objetivo": draw_date,
                                "posicion_semanal": target_index,
                                "combinacion_ganadora": winning,
                                "combinacion_ganadora_texto": (
                                    " - ".join(f"{n:02d}" for n in winning) if winning else None
                                ),
                                "aciertos": hits,
                                "estado": "Evaluada" if hits is not None else "Pendiente de resultado",
                                "fuente_resultado": result_source,
                                "evaluada_en_madrid": evaluated_at,
                            })
                    predictions = list(predictions_by_id.values())
                    all_run_payloads.append({
                        "run_id": run_id,
                        "inicio_en_madrid": started_at,
                        "fin_en_madrid": finished_at,
                        "modo": mode,
                        "estado_ejecucion": status,
                        "mensaje": message or "",
                        "predicciones": predictions,
                        "nota": (
                            "El informe se actualiza en ejecuciones posteriores cuando se recupera el resultado real. "
                            "Si aún no está disponible, la combinación ganadora y los aciertos aparecen como pendientes."
                        ),
                    })
        except Exception as exc:
            log.warn(f"No se pudieron preparar los informes por ejecución: {type(exc).__name__}: {exc}")
            return []

        exported = []
        latest_text = None
        latest_json = None
        for payload in all_run_payloads:
            try:
                run_id = payload["run_id"]
                started = str(payload.get("inicio_en_madrid") or "")
                try:
                    stamp = datetime.fromisoformat(started).strftime("%Y%m%d_%H%M%S")
                except Exception:
                    stamp = re.sub(r"[^0-9A-Za-z_-]", "", started)[:24] or "fecha_desconocida"
                stem = f"Ejecucion_Fuxing_{stamp}_{run_id[:8]}"
                json_path = self.run_results_dir / f"{stem}.json"
                text_path = self.run_results_dir / f"{stem}.txt"

                lines = [
                    "FUXING — INFORME DE EJECUCIÓN",
                    "=" * 72,
                    f"ID de ejecución: {run_id}",
                    f"Inicio (Europe/Madrid): {payload.get('inicio_en_madrid') or 'No disponible'}",
                    f"Fin (Europe/Madrid): {payload.get('fin_en_madrid') or 'En curso/no disponible'}",
                    f"Modo solicitado: {payload.get('modo')}",
                    f"Estado: {payload.get('estado_ejecucion')}",
                    f"Mensaje: {payload.get('mensaje') or 'Sin mensaje adicional'}",
                    "",
                ]
                predictions = payload.get("predicciones") or []
                if not predictions:
                    lines.append("No se registró ninguna predicción en esta ejecución.")
                else:
                    for pred in predictions:
                        title = "PREDICCIÓN DIARIA" if pred["tipo_codigo"] == "daily" else "PREDICCIÓN SEMANAL"
                        lines.extend([
                            title,
                            "-" * 72,
                            f"Fecha/hora de generación (Europe/Madrid): {pred['generada_en_madrid']}",
                            f"Combinación predicha: {pred['combinacion_texto']}",
                        ])
                        for target in pred["objetivos"]:
                            pos = target.get("posicion_semanal")
                            position_text = f" (sorteo semanal {pos}/7)" if pred["tipo_codigo"] == "weekly" and pos else ""
                            lines.append(f"Sorteo objetivo: {target['fecha_sorteo_objetivo']}{position_text}")
                            lines.append(
                                f"Combinación ganadora: {target['combinacion_ganadora_texto']}"
                                if target["combinacion_ganadora_texto"] else
                                "Combinación ganadora: PENDIENTE DE RESULTADO"
                            )
                            lines.append(
                                f"Aciertos: {target['aciertos']}/6"
                                if target["aciertos"] is not None else "Aciertos: PENDIENTES"
                            )
                            lines.append(f"Fuente del resultado: {target['fuente_resultado'] or 'Todavía no disponible'}")
                            lines.append("")
                lines.extend([
                    "NOTA",
                    payload["nota"],
                    "Los aciertos se calculan como la intersección de los seis números predichos con los seis ganadores.",
                    "La Bonoloto es aleatoria; los resultados históricos no garantizan predicciones futuras.",
                    "",
                ])
                text_content = "\n".join(lines)
                json_content = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"

                for path, content in ((text_path, text_content), (json_path, json_content)):
                    tmp_path = path.with_suffix(path.suffix + ".tmp")
                    try:
                        with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
                            handle.write(content)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(tmp_path, path)
                    finally:
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                exported.append((text_path, json_path))
                latest_text, latest_json = text_content, json_content
            except Exception as exc:
                log.warn(
                    f"No se pudo actualizar el informe de la ejecución {payload.get('run_id', '?')[:8]}: "
                    f"{type(exc).__name__}: {exc}"
                )

        # Copias de acceso rápido, siempre correspondientes a la ejecución más reciente
        # con informe. Los informes individuales de ejecuciones anteriores se conservan.
        if latest_text is not None and latest_json is not None:
            for path, content in (
                (self.run_results_dir / "ULTIMO_RESULTADO_FUXING.txt", latest_text),
                (self.run_results_dir / "ultimo_resultado_fuxing.json", latest_json),
            ):
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                try:
                    with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(tmp_path, path)
                except Exception as exc:
                    log.warn(f"No se pudo actualizar el acceso rápido {path.name}: {exc}")
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        return exported

    def export_excel(self):
        """Escribe un libro Excel formateado, reemplazándolo de forma atómica."""
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
            from openpyxl.worksheet.table import Table, TableStyleInfo
        except Exception as exc:
            log.warn(
                "No se pudo crear el Excel porque falta openpyxl. Instala con "
                f"`pip install openpyxl`. La base SQLite conserva los datos. Detalle: {exc}"
            )
            return None

        meta, draws, predictions, runs = self._snapshot()
        wb = Workbook()
        ws_summary = wb.active
        ws_summary.title = "Resumen"
        ws_history = wb.create_sheet("Historial por sorteo")
        ws_detail = wb.create_sheet("Detalle predicciones")
        ws_runs = wb.create_sheet("Ejecuciones")

        navy = "17365D"
        blue = "1F4E78"
        pale_blue = "D9EAF7"
        pale_green = "E2F0D9"
        pale_yellow = "FFF2CC"
        pale_red = "FCE4D6"
        white = "FFFFFF"
        thin_gray = Side(style="thin", color="D9E1F2")

        def style_title(sheet, title, subtitle=None):
            sheet.sheet_view.showGridLines = False
            merge_columns = {"Resumen": 2, "Historial por sorteo": 12,
                             "Detalle predicciones": 13, "Ejecuciones": 6}.get(
                                 sheet.title, max(4, sheet.max_column)
                             )
            sheet.append([title])
            sheet.cell(1, 1).font = Font(name="Aptos Display", size=16, bold=True, color=white)
            sheet.cell(1, 1).fill = PatternFill("solid", fgColor=navy)
            sheet.cell(1, 1).alignment = Alignment(vertical="center")
            sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=merge_columns)
            sheet.row_dimensions[1].height = 28
            if subtitle:
                sheet.append([subtitle])
                sheet.cell(2, 1).font = Font(italic=True, color="595959")
                sheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=merge_columns)
            sheet.append([])  # reserva la fila 3; encabezados siempre en la fila 4

        # Resumen general
        style_title(ws_summary, "FUXING · HISTORIAL DE BONOLOTO", "Registro persistente de sorteos, predicciones originales y aciertos calculados")
        eval_rows = [row for row in predictions if row[12] is not None]
        daily_eval = [row[12] for row in eval_rows if row[3] == "daily"]
        weekly_eval = [row[12] for row in eval_rows if row[3] == "weekly"]
        latest_draw = draws[-1][0] if draws else "Sin datos"
        tracking_start = meta.get("tracking_started_date", "No disponible")
        summary_items = [
            ("Actualizado (Europe/Madrid)", _madrid_now().strftime("%d/%m/%Y %H:%M:%S")),
            ("Inicio del seguimiento nuevo", tracking_start),
            ("Sorteos reales únicos registrados", len(draws)),
            ("Último sorteo real registrado", latest_draw),
            ("Predicciones individuales conservadas", len({row[0] for row in predictions})),
            ("Objetivos evaluados (diarios y semanales)", len(eval_rows)),
            ("Media de aciertos · predicciones diarias evaluadas", round(sum(daily_eval)/len(daily_eval), 3) if daily_eval else "Pendiente de datos"),
            ("Media de aciertos · objetivos semanales evaluados", round(sum(weekly_eval)/len(weekly_eval), 3) if weekly_eval else "Pendiente de datos"),
            ("Base permanente", str(self.db_path)),
            ("Excel", str(self.xlsx_path)),
            ("Nota de interpretación", "La Bonoloto es aleatoria; el historial mide resultados observados y no garantiza capacidad predictiva."),
        ]
        ws_summary.append(["Indicador", "Valor"])
        for cell in ws_summary[4]:
            cell.font = Font(bold=True, color=white)
            cell.fill = PatternFill("solid", fgColor=blue)
        for item in summary_items:
            ws_summary.append(list(item))
        ws_summary.column_dimensions["A"].width = 48
        ws_summary.column_dimensions["B"].width = 95
        ws_summary.freeze_panes = "A5"

        # Agrupa todas las predicciones aplicables a una fecha; se muestran todas en
        # el resumen, y cada relación queda además como fila propia en el detalle.
        target_by_date = {}
        for row in predictions:
            (pred_id, run_id, generated_at, ptype, combo_json, start_date, end_date, mode,
             payload_json, draw_date, target_index, winning_combo, hits, result_source,
             evaluated_at, run_started, run_finished, run_status, run_message) = row
            item = {
                "prediction_id": pred_id, "run_id": run_id, "generated_at": generated_at,
                "type": ptype, "combo": json.loads(combo_json), "target_index": target_index,
                "hits": hits, "winning_combo": json.loads(winning_combo) if winning_combo else None,
                "result_source": result_source, "evaluated_at": evaluated_at,
                "run_status": run_status,
            }
            target_by_date.setdefault(draw_date, []).append(item)

        all_dates = sorted(set([row[0] for row in draws]) | set(target_by_date))
        draw_map = {date: (combo_json, source, updated) for date, combo_json, source, updated in draws}
        run_date_status = {}
        for run_id, started_at, finished_at, mode, status, message in runs:
            run_date_status.setdefault(started_at[:10], []).append((mode, status))

        history_headers = [
            "Fecha del sorteo", "Combinación ganadora", "Fuente del resultado",
            "Predicción diaria", "Fecha/hora predicción diaria", "Aciertos diaria", "Estado diaria",
            "Predicción semanal", "Fecha/hora predicción semanal", "Posición semanal", "Aciertos semanal", "Estado semanal",
        ]
        style_title(ws_history, "HISTORIAL POR SORTEO", "Una fila por fecha; si existen varias predicciones para un objetivo, todas se conservan separadas por línea.")
        ws_history.append(history_headers)
        header_row = 4
        for cell in ws_history[header_row]:
            cell.font = Font(bold=True, color=white)
            cell.fill = PatternFill("solid", fgColor=blue)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        tracking_start = meta.get("tracking_started_date", "9999-99-99")
        for draw_date in all_dates:
            actual = draw_map.get(draw_date)
            if actual:
                actual_combo = " - ".join(f"{n:02d}" for n in json.loads(actual[0]))
                actual_source = actual[1]
            else:
                actual_combo = "Pendiente de resultado"
                actual_source = "No disponible todavía"
            related = target_by_date.get(draw_date, [])
            daily = [x for x in related if x["type"] == "daily"]
            weekly = [x for x in related if x["type"] == "weekly"]

            def render(items, value_func):
                return "\n".join(value_func(item) for item in items) if items else ""

            daily_combo = render(daily, lambda x: " - ".join(f"{n:02d}" for n in x["combo"]) + f" [ID {x['prediction_id'][:8]}]")
            daily_time = render(daily, lambda x: x["generated_at"])
            daily_hits = render(daily, lambda x: "Pendiente" if x["hits"] is None else f"{x['hits']}/6")
            daily_status = render(daily, lambda x: "Pendiente de resultado" if x["hits"] is None else "Evaluada")
            weekly_combo = render(weekly, lambda x: " - ".join(f"{n:02d}" for n in x["combo"]) + f" [ID {x['prediction_id'][:8]}]")
            weekly_time = render(weekly, lambda x: x["generated_at"])
            weekly_index = render(weekly, lambda x: f"{x['target_index']}/7")
            weekly_hits = render(weekly, lambda x: "Pendiente" if x["hits"] is None else f"{x['hits']}/6")
            weekly_status = render(weekly, lambda x: "Pendiente de resultado" if x["hits"] is None else "Evaluada")
            if not daily and not weekly:
                if draw_date < tracking_start:
                    not_run_status = "Sin registro histórico anterior al seguimiento"
                elif draw_date in run_date_status:
                    not_run_status = "Fuxing ejecutado, sin predicción aplicable registrada"
                else:
                    not_run_status = "No ejecutado Fuxing"
                daily_status = not_run_status
                weekly_status = not_run_status
            ws_history.append([
                draw_date, actual_combo, actual_source,
                daily_combo, daily_time, daily_hits, daily_status,
                weekly_combo, weekly_time, weekly_index, weekly_hits, weekly_status,
            ])
        ws_history.freeze_panes = "A5"
        ws_history.auto_filter.ref = f"A4:L{ws_history.max_row}"
        widths = [15, 27, 38, 34, 27, 18, 36, 34, 27, 15, 18, 36]
        for idx, width in enumerate(widths, start=1):
            ws_history.column_dimensions[__import__('openpyxl').utils.get_column_letter(idx)].width = width
        for row in ws_history.iter_rows(min_row=5):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = Border(bottom=thin_gray)
            for col in (6, 11):
                cell = ws_history.cell(row=row[0].row, column=col)
                if cell.value and "Pendiente" not in str(cell.value):
                    cell.fill = PatternFill("solid", fgColor=pale_green)
            for col in (7, 12):
                cell = ws_history.cell(row=row[0].row, column=col)
                if cell.value and "No ejecutado" in str(cell.value):
                    cell.fill = PatternFill("solid", fgColor=pale_yellow)
                elif cell.value and "Pendiente" in str(cell.value):
                    cell.fill = PatternFill("solid", fgColor=pale_yellow)
                elif cell.value and "Sin registro histórico" in str(cell.value):
                    cell.fill = PatternFill("solid", fgColor=pale_red)
        if ws_history.max_row >= 5:
            table = Table(displayName="HistorialSorteos", ref=f"A4:L{ws_history.max_row}")
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True,
                                                  showColumnStripes=False, showFirstColumn=False, showLastColumn=False)
            ws_history.add_table(table)

        # Detalle: una fila por cada predicción x sorteo objetivo.
        detail_headers = [
            "ID predicción", "ID ejecución", "Tipo", "Fecha/hora generación (Madrid)",
            "Combinación predicha", "Fecha objetivo", "Posición objetivo", "Combinación ganadora",
            "Aciertos", "Fuente resultado", "Fecha/hora evaluación", "Estado ejecución", "Estado ejecución mensaje",
        ]
        style_title(ws_detail, "DETALLE DE PREDICCIONES", "Registro completo; no se sobrescribe una predicción anterior al volver a ejecutar Fuxing.")
        ws_detail.append(detail_headers)
        for cell in ws_detail[4]:
            cell.font = Font(bold=True, color=white)
            cell.fill = PatternFill("solid", fgColor=blue)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for row in predictions:
            (pred_id, run_id, generated_at, ptype, combo_json, start_date, end_date, mode,
             payload_json, draw_date, target_index, winning_combo, hits, result_source,
             evaluated_at, run_started, run_finished, run_status, run_message) = row
            ws_detail.append([
                pred_id, run_id, ptype, generated_at,
                " - ".join(f"{n:02d}" for n in json.loads(combo_json)), draw_date, target_index,
                " - ".join(f"{n:02d}" for n in json.loads(winning_combo)) if winning_combo else "",
                hits, result_source or "", evaluated_at or "", run_status, run_message or "",
            ])
        ws_detail.freeze_panes = "A5"
        ws_detail.auto_filter.ref = f"A4:M{ws_detail.max_row}"
        detail_widths = [34, 34, 12, 28, 28, 16, 16, 28, 12, 36, 28, 16, 55]
        for idx, width in enumerate(detail_widths, start=1):
            ws_detail.column_dimensions[__import__('openpyxl').utils.get_column_letter(idx)].width = width
        for row in ws_detail.iter_rows(min_row=5):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = Border(bottom=thin_gray)
            hits_cell = row[8]
            if isinstance(hits_cell.value, int):
                hits_cell.fill = PatternFill("solid", fgColor=pale_green)
        if ws_detail.max_row >= 5:
            table = Table(displayName="DetallePredicciones", ref=f"A4:M{ws_detail.max_row}")
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium4", showRowStripes=True,
                                                  showColumnStripes=False, showFirstColumn=False, showLastColumn=False)
            ws_detail.add_table(table)

        # Ejecuciones, para no confundir un día sin predicción con un proceso fallido.
        style_title(ws_runs, "EJECUCIONES FUXING", "Un registro por ejecución, incluso si no pudo completar el entrenamiento.")
        run_headers = ["ID ejecución", "Inicio (Madrid)", "Fin (Madrid)", "Modo", "Estado", "Mensaje"]
        ws_runs.append(run_headers)
        for cell in ws_runs[4]:
            cell.font = Font(bold=True, color=white)
            cell.fill = PatternFill("solid", fgColor=blue)
        for row in runs:
            ws_runs.append(list(row))
        ws_runs.freeze_panes = "A5"
        ws_runs.auto_filter.ref = f"A4:F{ws_runs.max_row}"
        for idx, width in enumerate([34, 28, 28, 14, 16, 75], start=1):
            ws_runs.column_dimensions[__import__('openpyxl').utils.get_column_letter(idx)].width = width

        for sheet in wb.worksheets:
            sheet.sheet_properties.pageSetUpPr.fitToPage = True
            sheet.page_setup.fitToWidth = 1
            sheet.page_setup.fitToHeight = 0
            sheet.sheet_view.zoomScale = 85

        tmp_path = self.xlsx_path.with_name(self.xlsx_path.stem + ".tmp.xlsx")
        try:
            wb.save(tmp_path)
            # openpyxl ha cerrado/escrito el libro. No se hace fsync sobre un descriptor
            # abierto en modo solo lectura, operación que puede fallar en Windows.
            if self.xlsx_path.exists():
                backup = self.xlsx_path.with_suffix(self.xlsx_path.suffix + ".bak")
                shutil.copy2(self.xlsx_path, backup)
            os.replace(tmp_path, self.xlsx_path)
        except Exception as exc:
            # El Excel puede estar abierto en Windows o Drive puede estar temporalmente
            # inaccesible. No se abortan los cálculos: SQLite ya conserva el histórico.
            log.warn(
                f"No se pudo actualizar el Excel ({type(exc).__name__}: {exc}). "
                "Las predicciones y aciertos siguen guardados en SQLite; cierra el Excel "
                "si está abierto y vuelve a ejecutar para regenerarlo."
            )
            return None
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
        return self.xlsx_path


def _run_fuxing_v3_core(mode="both", history_store=None, history_run_id=None):
    overall_start = time.time()
    pause_checkpoint("inicio del proceso")
    print_system_status()

    # 1. HISTÓRICO
    log.banner("PASO 1/6 — HISTÓRICO COMPLETO 1988 -> ACTUAL", "=")
    scraper = DataScraper(start_year=START_YEAR, end_year=_madrid_now().year)
    data = scraper.scrape_with_cache(force_refresh=False)
    if not data:
        log.error("Histórico vacío. Abortando.")
        return None
    log.ok(
        f"Histórico listo: {len(data):,} sorteos | "
        f"{data[0][0]:%d/%m/%Y} -> {data[-1][0]:%d/%m/%Y}"
    )

    # Seguimiento de resultados separado del motor. Actualiza sorteos reales y
    # resuelve objetivos pendientes de ejecuciones anteriores antes del entrenamiento.
    if history_store is not None:
        history_store.sync_draws(
            data,
            official_dates=set(scraper.official_selae_results.keys()),
        )
        history_path = history_store.export_excel()
        reports = history_store.export_run_reports()
        if history_path:
            log.log(f"Historial persistente actualizado antes del entrenamiento: {history_path}", "DATA")
        if reports:
            log.log(f"Informes de ejecuciones revisados tras sincronizar resultados: {len(reports)}", "DATA")

    pause_checkpoint("histórico descargado y validado")

    # 2. PREPARACIÓN
    predictor = FuxingPredictor()
    predictor.prepare(data)
    pause_checkpoint("preparación de datos terminada")

    # 3. ENTRENAMIENTO
    predictor.train_models()
    pause_checkpoint("entrenamiento de modelos terminado")

    # 4. CALIBRACIÓN + PESOS
    predictor.calibrate_and_optimize()
    pause_checkpoint("calibración y optimización terminadas")

    # 5. VALIDACIÓN
    predictor.validate(run_walk_forward=True, run_backtest=True)
    pause_checkpoint("validación y backtesting terminados")

    # 6. PREDICCIÓN DIARIA + SEMANAL
    log.banner("PASO 6/6 — PREDICCIÓN DIARIA + SEMANAL", "=")

    daily_result = {}
    weekly_result = {}

    def run_daily():
        pause_checkpoint("antes de la predicción diaria")
        log.log(">>> DIARIA: iniciando predicción", "PRED")
        try:
            r = predictor.predict(
                mc_iterations=MC_ITERATIONS,
                n_combinations=15,
            )
            daily_result.update(r)
            log.ok(">>> DIARIA: completada")
        except Exception as e:
            log.error(f">>> DIARIA: {type(e).__name__}: {e}")
            traceback.print_exc()

    def run_weekly():
        pause_checkpoint("antes de la predicción semanal")
        log.log(">>> SEMANAL: iniciando predicción", "PRED")
        try:
            engine = WeeklyEngine(
                data,
                progress_callback=lambda p, m: log.progress(p, m),
                model_probabilities=(
                    daily_result.get("probabilities")
                    if daily_result else None
                ),
            )
            r = engine.predict(
                mc_iterations=MC_ITERATIONS_SEMANAL,
                run_backtest=True,
            )
            weekly_result.update(r)
            log.ok(">>> SEMANAL: completada")
        except Exception as e:
            log.error(f">>> SEMANAL: {type(e).__name__}: {e}")
            traceback.print_exc()

    if mode in ("daily", "both"):
        run_daily()
    if mode in ("weekly", "both"):
        run_weekly()

    # No se permite publicar como ejecución correcta una predicción ausente o inválida.
    def _valid_combo(combo):
        try:
            values = [int(x) for x in combo]
            return len(values) == 6 and len(set(values)) == 6 and all(1 <= x <= 49 for x in values)
        except Exception:
            return False

    if mode in ("daily", "both") and not _valid_combo(daily_result.get("primary_combination")):
        raise RuntimeError("La predicción diaria no se ha generado o no contiene 6 números únicos entre 1 y 49.")
    if mode in ("weekly", "both") and not _valid_combo(weekly_result.get("combination")):
        raise RuntimeError("La predicción semanal no se ha generado o no contiene 6 números únicos entre 1 y 49.")

    # Registra predicciones generadas con hora de Madrid. Cada ejecución crea nuevos
    # IDs append-only; ninguna combinación anterior se modifica ni se sustituye.
    if history_store is not None and history_run_id:
        latest_actual = max(fecha.date() for fecha, _ in data)
        # Se calcula al guardar la predicción, no al comenzar el proceso: el entrenamiento
        # puede durar horas y terminar después del sorteo de las 21:30. Así no se puntúa
        # por error una combinación contra un sorteo que ya se ha celebrado.
        prediction_generated_at = _madrid_now()
        _assert_history_current_for_prediction(latest_actual, prediction_generated_at)
        first_target = _first_prediction_target_date(latest_actual, prediction_generated_at)
        target_dates = [first_target + timedelta(days=offset) for offset in range(7)]
        if mode in ("daily", "both"):
            history_store.add_prediction(
                history_run_id, "daily", daily_result["primary_combination"],
                [target_dates[0]], mode,
                payload={"combinations": daily_result.get("combinations", []),
                         "top_numbers": daily_result.get("top_numbers", []),
                         "weights": daily_result.get("weights")},
                generated_at=prediction_generated_at,
            )
        if mode in ("weekly", "both"):
            history_store.add_prediction(
                history_run_id, "weekly", weekly_result["combination"],
                target_dates, mode,
                payload={"stability_score": weekly_result.get("stability_score"),
                         "concentration_score": weekly_result.get("concentration_score"),
                         "analog_count": weekly_result.get("analog_count")},
                generated_at=prediction_generated_at,
            )

    total_elapsed = time.time() - overall_start

    # RESULTADO FINAL
    log.banner("RESULTADO FINAL", "=")

    daily_combo = daily_result.get("primary_combination")
    weekly_combo = weekly_result.get("combination")

    log.log("", "INFO")
    log.log("=" * 60, "PRED")
    log.log("  COMBINACIÓN DIARIA RECOMENDADA (próximo sorteo)", "PRED")
    log.log("=" * 60, "PRED")
    if daily_combo:
        log.log(f"  >>>  {_fmt_combo(daily_combo)}  <<<", "OK")
    else:
        log.warn("  No se pudo generar combinación diaria.")
    log.log("=" * 60, "PRED")
    log.log("", "INFO")

    top20 = daily_result.get("top_numbers", [])
    if top20:
        log.log(
            "Top-20 números (probabilidad descendente): "
            + " ".join(f"{n:02d}" for n in top20),
            "PRED",
        )

    portfolio = daily_result.get("combinations", [])
    if len(portfolio) > 1:
        log.log("Cartera completa (15 combinaciones diversas):", "PRED")
        for i, c in enumerate(portfolio, 1):
            log.log(f"  #{i:02d}  {_fmt_combo(c)}", "PRED")

    conf_set = daily_result.get("conformal_set")
    if conf_set:
        log.log(
            f"Conjunto conformal (90% cobertura): "
            f"{len(conf_set)} números -> {conf_set}",
            "PRED",
        )

    bayes_p6 = daily_result.get("bayes_p_top6")
    if bayes_p6 is not None:
        top_bayes = np.argsort(bayes_p6)[::-1][:10] + 1
        log.log(
            "Top-10 números por frecuencia posterior de aparecer entre los 6 más altos del modelo Dirichlet: "
            + " ".join(f"{n:02d}({bayes_p6[n-1]:.2f})" for n in top_bayes),
            "PRED",
        )

    log.log("", "INFO")
    log.log("=" * 60, "PRED")
    log.log("  COMBINACIÓN SEMANAL RECOMENDADA (7 sorteos)", "PRED")
    log.log("=" * 60, "PRED")
    if weekly_combo:
        log.log(f"  >>>  {_fmt_combo(weekly_combo)}  <<<", "OK")
    else:
        log.warn("  No se pudo generar combinación semanal.")
    log.log("=" * 60, "PRED")
    log.log("", "INFO")

    if weekly_result:
        log.log(
            f"Estabilidad: {weekly_result.get('stability_score', 0)*100:.2f}% · "
            f"Concentración: {weekly_result.get('concentration_score', 0)*100:.2f}% · "
            f"Analogías: {weekly_result.get('analog_count', 0)}",
            "PRED",
        )
        bt = weekly_result.get("backtest", {})
        if bt.get("tests"):
            log.log(
                f"Backtest semanal: {bt['tests']} semanas · "
                f"media hits/sorteo={bt.get('mean_hits_per_draw', 0):.3f} · "
                f"mejor semana={bt.get('mean_best_hits_week', 0):.3f} · "
                f"lift={bt.get('lift_over_random', 0):.3f}",
                "AUDIT",
            )

    log.log("", "INFO")
    log.log("Validación rigurosa:", "AUDIT")
    wf = predictor.validation_report.get("walk_forward_statistical_only", {})
    if wf.get("tests"):
        log.log(
            f"  Walk-forward estadístico: {wf['tests']} splits · "
            f"Brier={wf.get('mean_brier', 0):.6f} · "
            f"hits@12={wf.get('mean_hits_top12', 0):.3f}",
            "AUDIT",
        )
    dm = predictor.validation_report.get("diebold_mariano", {})
    if dm.get("n"):
        log.log(
            f"  Diebold-Mariano: DM={dm.get('dm_stat', 0):+.3f} "
            f"p={dm.get('p_value', 1):.3g} · "
            f"dirección={dm.get('direction', 'no disponible')} · "
            f"mejora_significativa={dm.get('significant_improvement_5pct')}",
            "AUDIT",
        )
    bt_d = predictor.backtest_report.get("daily_statistical_only", {})
    if bt_d.get("tests"):
        log.log(
            f"  Backtest diario: {bt_d['tests']} sorteos · "
            f"media={bt_d.get('mean_hits_top12', 0):.3f} · "
            f"lift={bt_d.get('lift_over_random', 0):.3f}",
            "AUDIT",
        )

    log.log("", "INFO")
    log.log(f"TIEMPO TOTAL: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)", "INFO")

    payload = {
        "generated_at": _madrid_now().isoformat(timespec="seconds"),
        "execution": {
            "historical_start": START_YEAR,
            "historical_end": _madrid_now().year,
            "historical_draws": len(data),
            "first_date": data[0][0].strftime("%Y-%m-%d"),
            "last_date": data[-1][0].strftime("%Y-%m-%d"),
            "total_elapsed_seconds": total_elapsed,
        },
        "daily": {
            "primary_combination": daily_combo,
            "combinations": daily_result.get("combinations", []),
            "top_numbers": top20,
            "conformal_set": conf_set,
            "conformal_diagnostics": daily_result.get("conformal_diagnostics"),
            "bayes_p_top6": (
                bayes_p6.tolist() if bayes_p6 is not None else None
            ),
            "weights": daily_result.get("weights"),
        },
        "weekly": {
            "combination": weekly_combo,
            "stability_score": weekly_result.get("stability_score"),
            "concentration_score": weekly_result.get("concentration_score"),
            "analog_count": weekly_result.get("analog_count"),
            "backtest": weekly_result.get("backtest"),
        },
        "validation": predictor.validation_report,
        "backtest_daily_statistical_only": predictor.backtest_report.get("daily_statistical_only"),
        "training": predictor.training_report,
        "data_integrity": scraper.last_history_audit,
    }
    _safe_json_write(RESULT_FILE, payload)
    _safe_json_write(RUN_REPORT_FILE, payload)
    log.ok(f"Resultado guardado en: {RESULT_FILE}")

    if history_store is not None and history_run_id:
        history_store.finish_run(history_run_id, "COMPLETED", "Predicciones generadas y resultado JSON guardado.")
        with history_store._db_session() as con:
            history_store._reconcile_targets(con)
        history_path = history_store.export_excel()
        if history_path:
            log.ok(f"Excel de historial actualizado: {history_path}")
        report_files = history_store.export_run_reports()
        log.log(
            f"Informes de ejecución guardados/actualizados: {len(report_files)} en {history_store.run_results_dir}",
            "DATA",
        )

    return payload


def run_fuxing_v3(mode="both"):
    """Envoltorio de auditoría/registro; delega el cálculo intacto al motor original."""
    history_store = PredictionHistoryStore()
    history_run_id = history_store.start_run(mode)
    try:
        result = _run_fuxing_v3_core(
            mode=mode,
            history_store=history_store,
            history_run_id=history_run_id,
        )
        if result is None:
            history_store.finish_run(history_run_id, "NO_RESULT", "La ejecución terminó sin payload de predicción.")
            history_store.export_excel()
            history_store.export_run_reports()
        return result
    except BaseException as exc:
        try:
            history_store.finish_run(history_run_id, "FAILED", f"{type(exc).__name__}: {exc}")
            history_store.export_excel()
            history_store.export_run_reports()
        except Exception as tracker_exc:
            log.warn(f"No se pudo cerrar/exportar el seguimiento de la ejecución fallida: {tracker_exc}")
        raise

# =====================================================================
# GUI FUXING — capa visual añadida sobre el motor definitivo.
# El motor predictivo y sus funciones no se reescriben aquí.
# =====================================================================

import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    GUI_MATPLOTLIB_AVAILABLE = True
except Exception:
    Figure = None
    FigureCanvasTkAgg = None
    GUI_MATPLOTLIB_AVAILABLE = False

try:
    import psutil
    GUI_PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    GUI_PSUTIL_AVAILABLE = False

_GUI_EVENT_QUEUE = None
_GUI_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _CooperativePauseController:
    """Pausa/reanudación cooperativa en puntos seguros del motor.

    No interrumpe por la fuerza una llamada interna de NumPy/scikit-learn/PyTorch.
    El hilo de cálculo espera en el siguiente punto de control, mientras Tk sigue
    respondiendo en el hilo principal.
    """
    def __init__(self):
        self._resume_event = threading.Event()
        self._resume_event.set()  # El motor comienza habilitado.
        self._lock = threading.Lock()
        self._pause_announced = False

    def reset(self):
        with self._lock:
            self._pause_announced = False
            self._resume_event.set()

    def request_pause(self):
        with self._lock:
            if not self._resume_event.is_set():
                return False
            self._pause_announced = False
            self._resume_event.clear()
            return True

    def is_pause_requested(self):
        return not self._resume_event.is_set()

    def resume(self):
        self._resume_event.set()

    def checkpoint(self, location="fase de cálculo"):
        if self._resume_event.is_set():
            return
        announce_pause = False
        with self._lock:
            if not self._pause_announced:
                self._pause_announced = True
                announce_pause = True
        if announce_pause:
            try:
                log.warn(f"PAUSA ACTIVA en punto seguro: {location}. El cálculo se conserva en memoria.")
            except Exception:
                pass
        # Espera en el hilo de trabajo; la interfaz Tk no queda bloqueada.
        self._resume_event.wait()
        announce_resume = False
        with self._lock:
            if self._pause_announced:
                self._pause_announced = False
                announce_resume = True
        if announce_resume:
            try:
                log.log(f"REANUDANDO cálculo desde: {location}", "INFO")
            except Exception:
                pass


_RUN_PAUSE = _CooperativePauseController()

def pause_checkpoint(location="fase de cálculo"):
    """Punto de control global para pausar y reanudar cálculos de forma segura."""
    _RUN_PAUSE.checkpoint(location)


class FuxingDesktopGUI:
    """Interfaz de escritorio para ejecutar el motor Fuxing definitivo.

    El motor se ejecuta en un hilo de trabajo. La GUI únicamente consume sus
    mensajes, pinta estados/gráficas y presenta el JSON de resultado devuelto
    por run_fuxing_v3().
    """

    BG = "#10151c"
    PANEL = "#151b23"
    PANEL_ALT = "#18202a"
    TEXT = "#e8edf2"
    MUTED = "#8da0b8"
    GREEN = "#24b36b"
    GREEN_LIGHT = "#6ee7a8"
    ORANGE = "#ff9f1c"
    RED = "#d84a52"
    CYAN = "#55d9f2"
    PURPLE = "#c59cff"
    BORDER = "#334155"

    STAGES = [
        "1. HISTÓRICO / DESCARGA",
        "2. PREPARACIÓN DE DATOS",
        "3. ENTRENAMIENTO DE MODELOS",
        "4. CALIBRACIÓN / OPTIMIZACIÓN",
        "5. VALIDACIÓN / BACKTEST",
        "6. PREDICCIÓN DIARIA / SEMANAL",
        "7. AUDITORÍA Y GUARDADO",
    ]
    STAGE_RANGES = {
        1: (0.0, 15.0), 2: (15.0, 29.0), 3: (29.0, 49.0),
        4: (49.0, 64.0), 5: (64.0, 78.0), 6: (78.0, 94.0),
        7: (94.0, 100.0),
    }

    def __init__(self, root):
        global _GUI_EVENT_QUEUE
        self.root = root
        self.events = queue.Queue()
        _GUI_EVENT_QUEUE = self.events
        self.root.title("BONOLOTO AI RESEARCH LAB - FUXING V1 DEFINITIVA")
        self.root.configure(bg=self.BG)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            try:
                self.root.geometry("1450x900")
            except tk.TclError:
                pass
        self.root.minsize(1080, 720)

        self.running = False
        self.run_mode = None
        self.pause_button = None
        self.run_started = None
        self.current_stage = 0
        self.active_blink = False
        self.stage_tiles = []
        self.result_payload = None
        self.last_error = None
        self.log_count = 0
        self.history_rows = []
        self.history_total = 0
        self.history_freq = np.zeros(MAX_NUM, dtype=float)
        self.recent_freq_per100 = np.zeros(MAX_NUM, dtype=float)
        self.daily_probability = None
        self.daily_combo = None
        self.weekly_combo = None
        self.phase_times = []
        self.phase_progress = []
        self.cpu_values = []
        self.memory_values = []
        self.val_losses = []
        self.max_log_lines = 4500
        self._last_graph_draw = 0.0
        self._last_system_sample = 0.0

        self._configure_styles()
        self._build_ui()
        self._attach_logger_bridge()
        self._paint_stage_tiles()
        self._draw_prediction_canvas()
        self.root.after(250, self._load_history_distribution)
        self._update_clock()
        self._poll_events()
        self._blink_loop()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------- UI ---------------------------------
    def _configure_styles(self):
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("TFrame", background=self.BG)
        self.style.configure("TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 9))
        self.style.configure("TLabelframe", background=self.BG, foreground=self.CYAN, bordercolor=self.BORDER)
        self.style.configure("TLabelframe.Label", background=self.BG, foreground=self.CYAN, font=("Segoe UI", 9, "bold"))
        self.style.configure("TNotebook", background=self.BG, borderwidth=0)
        self.style.configure("TNotebook.Tab", padding=(12, 5), background="#242b35", foreground=self.TEXT)
        self.style.map("TNotebook.Tab", background=[("selected", self.PANEL)], foreground=[("selected", self.CYAN)])
        self.style.configure("TButton", padding=(6, 5), font=("Segoe UI", 9), background="#252d38", foreground=self.TEXT)
        self.style.map("TButton", background=[("active", "#354355"), ("disabled", "#202630")])
        self.style.configure("Horizontal.TProgressbar", troughcolor="#252d38", background=self.GREEN, bordercolor=self.BORDER, lightcolor=self.GREEN, darkcolor=self.GREEN)
        self.style.configure("TEntry", fieldbackground=self.PANEL, foreground=self.TEXT)

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.main_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        self.config_tab = ttk.Frame(self.notebook)
        self.backtest_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Principal")
        self.notebook.add(self.stats_tab, text="Estadísticas")
        self.notebook.add(self.config_tab, text="Configuración")
        self.notebook.add(self.backtest_tab, text="Validación / Backtesting")

        monitor = ttk.LabelFrame(self.main_tab, text="Monitor de actividad y análisis en tiempo real", padding=5)
        monitor.pack(fill="both", expand=True, padx=5, pady=(5, 2))
        split = tk.PanedWindow(monitor, orient=tk.HORIZONTAL, sashwidth=5, bg=self.BORDER, bd=0, relief="flat")
        split.pack(fill="both", expand=True)
        left = tk.Frame(split, bg=self.BG)
        right = tk.Frame(split, bg=self.BG)
        split.add(left, minsize=380, width=600, stretch="always")
        split.add(right, minsize=430, width=700, stretch="always")

        self.log_text = tk.Text(
            left, wrap="word", bg=self.BG, fg=self.TEXT, insertbackground="#ffffff",
            selectbackground="#334155", relief="flat", padx=8, pady=8,
            font=("Consolas", 9), undo=False, height=16
        )
        log_scroll = ttk.Scrollbar(left, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        for tag, color in {
            "info": self.TEXT, "data": self.CYAN, "model": self.PURPLE,
            "optim": "#ffd166", "prediction": self.GREEN_LIGHT,
            "audit": "#7dd3fc", "warning": "#ffb86b", "error": "#ff6b7a",
        }.items():
            self.log_text.tag_configure(tag, foreground=color)

        self.graph_status_var = tk.StringVar(value="Esperando datos para comenzar el análisis")
        if GUI_MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(7.0, 3.65), dpi=85, facecolor=self.BG)
            self.ax_freq = self.figure.add_subplot(211)
            self.ax_progress = self.figure.add_subplot(212)
            self.ax_freq_right = None
            self.ax_progress_right = None
            self.canvas = FigureCanvasTkAgg(self.figure, master=right)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            ttk.Label(right, textvariable=self.graph_status_var, anchor="center").pack(fill="x", pady=(1, 0))
            self._style_graph_axes()
            self._draw_graphs()
        else:
            self.figure = self.ax_freq = self.ax_progress = self.canvas = None
            self.ax_freq_right = self.ax_progress_right = None
            tk.Label(
                right, text="El panel de gráficas necesita matplotlib.\nEl motor Fuxing seguirá funcionando sin él.",
                bg=self.BG, fg=self.MUTED, font=("Segoe UI", 11), justify="center"
            ).pack(fill="both", expand=True, padx=20, pady=20)
            ttk.Label(right, textvariable=self.graph_status_var, anchor="center").pack(fill="x")

        progress_frame = ttk.Frame(self.main_tab)
        progress_frame.pack(fill="x", padx=5, pady=(2, 0))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, mode="determinate")
        self.progress_bar.pack(fill="x")
        status_row = ttk.Frame(progress_frame)
        status_row.pack(fill="x", pady=(2, 0))
        self.status_var = tk.StringVar(value="Listo — esperando una operación")
        ttk.Label(status_row, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        self.elapsed_var = tk.StringVar(value="Tiempo: 00:00:00")
        ttk.Label(status_row, textvariable=self.elapsed_var, anchor="e", font=("Segoe UI", 9, "bold")).pack(side="right", padx=(8, 0))
        self.clock_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.clock_var, anchor="e").pack(fill="x")

        prediction_frame = ttk.LabelFrame(self.main_tab, text="Predicción / selección", padding=8)
        prediction_frame.pack(fill="x", padx=5, pady=(5, 3))
        activity = tk.Frame(prediction_frame, bg=self.PANEL_ALT, height=38)
        activity.pack(fill="x")
        activity.pack_propagate(False)
        self.pred_left_var = tk.StringVar(value="ESPERANDO DATOS")
        self.pred_activity_var = tk.StringVar(value="Preparado para comenzar")
        self.pred_right_var = tk.StringVar(value="ESPERANDO DATOS")
        self.pred_left_label = tk.Label(activity, textvariable=self.pred_left_var, bg=self.PANEL_ALT, fg="#ff5252", font=("Segoe UI", 12, "bold"), padx=8)
        self.pred_left_label.pack(side="left", fill="y")
        tk.Label(activity, textvariable=self.pred_activity_var, bg=self.PANEL_ALT, fg=self.TEXT, font=("Segoe UI", 12, "bold"), padx=8).pack(side="left", fill="both", expand=True)
        self.pred_right_label = tk.Label(activity, textvariable=self.pred_right_var, bg=self.PANEL_ALT, fg="#ff5252", font=("Segoe UI", 11, "bold"), padx=8)
        self.pred_right_label.pack(side="right", fill="y")
        self.result_canvas = tk.Canvas(prediction_frame, height=115, bg=self.BG, highlightthickness=0)
        self.result_canvas.pack(fill="x", pady=(4, 0))
        self.result_canvas.bind("<Configure>", lambda _e: self._draw_prediction_canvas())

        bottom = ttk.Frame(self.main_tab)
        bottom.pack(fill="x", padx=5, pady=5)
        bottom.grid_columnconfigure(0, weight=2, uniform="controls")
        bottom.grid_columnconfigure(1, weight=1, uniform="controls")
        bottom.grid_columnconfigure(2, weight=1, uniform="controls")
        ind = ttk.LabelFrame(bottom, text="Procesos individuales — estado de cada fase", padding=7)
        ind.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.stage_grid = tk.Frame(ind, bg=self.BG)
        self.stage_grid.pack(fill="both", expand=True)
        for c in range(4):
            self.stage_grid.grid_columnconfigure(c, weight=1, uniform="stage")
        for i, title in enumerate(self.STAGES):
            tile = tk.Frame(self.stage_grid, bg=self.RED, padx=2, pady=2, relief="flat")
            tile.grid(row=i // 4, column=i % 4, sticky="nsew", padx=3, pady=3)
            inner = tk.Label(tile, text=title, bg="#252d38", fg=self.TEXT, font=("Segoe UI", 8, "bold"), wraplength=190, justify="center", padx=4, pady=8, height=2)
            inner.pack(fill="both", expand=True)
            self.stage_tiles.append((tile, inner))

        auto = ttk.LabelFrame(bottom, text="Ejecución automática", padding=7)
        auto.grid(row=0, column=1, sticky="nsew", padx=3)
        tk.Button(auto, text="REALIZAR TODO EL PROCESO", command=lambda: self.start_run("both"), bg="#263442", fg="#ffffff", activebackground="#36536b", activeforeground="#ffffff", font=("Segoe UI", 10, "bold"), relief="flat", bd=0, padx=6, pady=11, wraplength=250).pack(fill="x", padx=2, pady=(3, 7))
        ttk.Label(auto, text="Genera la combinación diaria y la semanal.", wraplength=245, justify="left").pack(fill="x", padx=2, pady=(0, 3))
        self.auto_mode_var = tk.StringVar(value="Modo: diaria + semanal")
        ttk.Label(auto, textvariable=self.auto_mode_var, anchor="w", justify="left").pack(fill="x", padx=2)

        weekly = ttk.LabelFrame(bottom, text="Semanal — ejecución automática", padding=7)
        weekly.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        tk.Button(weekly, text="PREDICCIÓN SEMANAL", command=lambda: self.start_run("weekly"), bg="#163641", fg=self.CYAN, activebackground="#245a68", activeforeground="#ffffff", font=("Segoe UI", 10, "bold"), relief="flat", bd=0, padx=6, pady=11).pack(fill="x", padx=2, pady=(3, 7))
        ttk.Label(weekly, text="Ejecuta el análisis completo y presenta solo la combinación semanal.", wraplength=245, justify="left").pack(fill="x", padx=2, pady=(0, 3))
        self.weekly_mode_var = tk.StringVar(value="Modo: solo semanal")
        ttk.Label(weekly, textvariable=self.weekly_mode_var, anchor="w", justify="left").pack(fill="x", padx=2)

        controls = ttk.LabelFrame(self.main_tab, text="Control de ejecución", padding=6)
        controls.pack(fill="x", padx=5, pady=(0, 6))
        self.control_var = tk.StringVar(value="Proceso en espera")
        ttk.Label(controls, textvariable=self.control_var, anchor="w").pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(controls, text="Abrir carpeta Fuxing", command=self.open_output_folder).pack(side="right", padx=4)
        self.pause_button = tk.Button(
            controls, text="PAUSAR PROCESO", command=self.toggle_pause,
            state="disabled", bg="#9a5b12", fg="#ffffff",
            activebackground="#c77b1c", activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            padx=12, pady=5,
        )
        self.pause_button.pack(side="right", padx=4)

        self._build_stats_tab()
        self._build_config_tab()
        self._build_backtest_tab()

    def _style_graph_axes(self):
        for ax in (self.ax_freq, self.ax_progress):
            ax.set_facecolor(self.PANEL)
            ax.tick_params(colors="#cbd5e1", labelsize=7)
            ax.title.set_color(self.TEXT)
            ax.xaxis.label.set_color(self.MUTED)
            ax.yaxis.label.set_color(self.MUTED)
            for spine in ax.spines.values():
                spine.set_color(self.BORDER)
            ax.grid(True, alpha=0.17, color="#94a3b8")
        self.figure.subplots_adjust(left=0.075, right=0.96, top=0.91, bottom=0.10, hspace=0.66)

    def _build_stats_tab(self):
        outer = ttk.Frame(self.stats_tab, padding=8)
        outer.pack(fill="both", expand=True)
        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        ttk.Button(bar, text="Actualizar estadísticas", command=self.refresh_info_tabs).pack(side="left")
        self.stats_text = tk.Text(outer, wrap="word", bg=self.BG, fg=self.TEXT, insertbackground="#ffffff", font=("Consolas", 10), padx=10, pady=10)
        self.stats_text.pack(fill="both", expand=True, pady=(6, 0))
        self._put_text(self.stats_text, "Todavía no hay resultados de una ejecución en esta sesión.\n")

    def _build_config_tab(self):
        outer = ttk.Frame(self.config_tab, padding=14)
        outer.pack(fill="both", expand=True, anchor="nw")
        info = [
            ("Versión del motor", "Fuxing v3 MAXIMUM — código definitivo sin GUI"),
            ("Histórico inicial", str(START_YEAR)),
            ("Dispositivo detectado", DEVICE.upper()),
            ("PyTorch", "Disponible" if TORCH_AVAILABLE else "No disponible"),
            ("Scikit-learn", "Disponible" if SKLEARN_AVAILABLE else "No disponible"),
            ("XGBoost", "Disponible" if XGB_AVAILABLE else "No disponible"),
            ("LightGBM", "Disponible" if LGBM_AVAILABLE else "No disponible"),
            ("Numba", "Disponible" if NUMBA_AVAILABLE else "No disponible"),
            ("Iteraciones Monte Carlo diario", f"{MC_ITERATIONS:,}"),
            ("Iteraciones Monte Carlo semanal", f"{MC_ITERATIONS_SEMANAL:,}"),
            ("Carpeta de datos", str(BASE_DIR)),
            ("Base histórica", str(SQLITE_FILE)),
            ("Resultados", str(RESULTS_DIR)),
        ]
        self.config_text = tk.Text(outer, wrap="word", bg=self.BG, fg=self.TEXT, font=("Consolas", 10), padx=10, pady=10, height=20)
        self.config_text.pack(fill="both", expand=True)
        self._put_text(self.config_text, "CONFIGURACIÓN ACTIVA\n" + "=" * 76 + "\n" + "\n".join(f"{k}: {v}" for k, v in info) + "\n\nLas opciones de la interfaz no cambian los parámetros ni los algoritmos del motor definitivo.\n")

    def _build_backtest_tab(self):
        outer = ttk.Frame(self.backtest_tab, padding=8)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Validación y backtesting del motor definitivo", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.backtest_text = tk.Text(outer, wrap="word", bg=self.BG, fg=self.TEXT, font=("Consolas", 10), padx=10, pady=10)
        self.backtest_text.pack(fill="both", expand=True)
        self._put_text(self.backtest_text, "Los informes se mostrarán aquí cuando termine una ejecución.\n")

    # ----------------------- Registro / eventos ----------------------
    def _attach_logger_bridge(self):
        """Reenvía logs a la GUI sin que el hilo de cálculo toque widgets Tk."""
        original_emit = log._emit
        original_progress = log.progress

        def emit_bridge(text):
            original_emit(text)
            q = _GUI_EVENT_QUEUE
            if q is not None:
                q.put(("log", _GUI_ANSI_RE.sub("", str(text))))

        def progress_bridge(pct, msg, width=30):
            original_progress(pct, msg, width=width)
            q = _GUI_EVENT_QUEUE
            if q is not None:
                q.put(("progress", float(pct), str(msg)))

        log._emit = emit_bridge
        log.progress = progress_bridge
        self._restore_logger = (original_emit, original_progress)

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._consume_log(event[1])
                elif kind == "progress":
                    self._consume_progress(event[1], event[2])
                elif kind == "finished":
                    self._finish_run(event[1])
                elif kind == "failed":
                    self._fail_run(event[1], event[2])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _consume_log(self, line):
        if not line:
            return
        self.log_count += 1
        upper = line.upper()
        tag = "info"
        if "[ERROR" in upper or "ERROR:" in upper:
            tag = "error"
        elif "[WARNING" in upper or "WARN" in upper:
            tag = "warning"
        elif "[DATA" in upper or "[NETWORK" in upper or "HISTÓRICO" in upper or "SELAE" in upper:
            tag = "data"
        elif "[MODEL" in upper or "ENTREN" in upper or "TRANSFORMER" in upper or "RED " in upper:
            tag = "model"
        elif "[OPTIM" in upper or "CALIBR" in upper or "PESOS" in upper:
            tag = "optim"
        elif "[PRED" in upper or "COMBINACIÓN" in upper or "COMBINACION" in upper or "PREDICCIÓN" in upper:
            tag = "prediction"
        elif "[AUDIT" in upper or "AUDITOR" in upper or "VALIDACIÓN" in upper:
            tag = "audit"
        self.log_text.insert("end", line.rstrip() + "\n", tag)
        if "PAUSA ACTIVA" in upper:
            self.control_var.set("Proceso pausado en un punto seguro — pulsa REANUDAR PROCESO para continuar")
            self.status_var.set("Fuxing está en pausa; los datos y cálculos en memoria se conservan.")
        elif "REANUDANDO CÁLCULO" in upper or "REANUDANDO CALCULO" in upper:
            self.control_var.set("Proceso reanudado — Fuxing continúa calculando")
            self.status_var.set("Fuxing ha reanudado la ejecución.")
        try:
            lines = int(self.log_text.index("end-1c").split(".")[0])
            if lines > self.max_log_lines:
                self.log_text.delete("1.0", f"{lines-self.max_log_lines}.0")
        except Exception:
            pass
        self.log_text.see("end")
        self._update_activity(line)
        stage = self._detect_stage(upper)
        if stage:
            self._set_stage(stage)
        if "HISTÓRICO LISTO:" in upper or "HISTORICO LISTO:" in upper or "HISTÓRICO AUDITADO:" in upper:
            self._load_history_distribution()
        loss_match = re.search(r"val_loss\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
        if loss_match and self.run_started is not None:
            self.val_losses.append((time.monotonic() - self.run_started, float(loss_match.group(1))))
        self._record_graph_point()
        if "RESULTADO GUARDADO EN:" in upper or "RESULTADO FINAL" in upper:
            self._load_history_distribution()

    def _update_activity(self, line):
        # Conserva la línea real del registro; se eliminan prefijos y adornos ANSI.
        text = re.sub(r"^\[[^\]]+\]\s*(?:\[[^\]]+\]\s*)?", "", line).strip()
        if not text:
            return
        short = text[:86] + ("…" if len(text) > 86 else "")
        self.status_var.set(short)
        upper = text.upper()
        if not self.running:
            return
        label = "ANALIZANDO DATOS"
        if "RED NEURONAL" in upper or "TRANSFORMER" in upper or "NEURAL" in upper:
            label = "ENTRENANDO REDES NEURONALES"
        elif "ENTREN" in upper or "MODELO" in upper or "ENSEMBLE DE ÁRBOLES" in upper:
            label = "ENTRENANDO MODELOS"
        elif "CALIBR" in upper or "PESOS" in upper or "OPTIM" in upper:
            label = "CALIBRANDO Y OPTIMIZANDO"
        elif "VALIDACIÓN" in upper or "BACKTEST" in upper or "WALK-FORWARD" in upper:
            label = "VALIDANDO MODELOS"
        elif "SEMANAL" in upper:
            label = "CALCULANDO PREDICCIÓN SEMANAL"
        elif "PREDIC" in upper or "MONTE CARLO" in upper:
            label = "CALCULANDO PREDICCIÓN"
        elif "AUDIT" in upper or "GUARDADO" in upper or "EXCEL" in upper:
            label = "AUDITANDO Y GUARDANDO"
        elif "HISTÓRICO" in upper or "SORTEOS" in upper or "AÑO " in upper or "SELAE" in upper:
            label = "DESCARGANDO / VALIDANDO HISTÓRICO"
        self.pred_right_var.set(label)
        self.pred_left_var.set("ESTUDIANDO DATOS")
        self.pred_left_label.configure(fg=self.GREEN_LIGHT)
        self.pred_right_label.configure(fg=self.GREEN_LIGHT)
        self.pred_activity_var.set(short)

    def _consume_progress(self, pct, msg):
        if not self.running:
            return
        text = f"[PROGRESO {pct:5.1f}%] {msg}"
        self._consume_log(text)
        stage = max(1, self.current_stage)
        lo, hi = self.STAGE_RANGES.get(stage, (0.0, 100.0))
        mapped = lo + (hi - lo) * max(0.0, min(100.0, pct)) / 100.0
        self.progress_var.set(max(float(self.progress_var.get()), min(99.0, mapped)))
        self._record_graph_point()

    def _detect_stage(self, upper):
        # Orden específico: las expresiones de fases internas se comprueban antes
        # de los mensajes genéricos de histórico/predicción.
        if "RESULTADO FINAL" in upper or "RESULTADO GUARDADO EN:" in upper or "EXCEL DE HISTORIAL ACTUALIZADO" in upper or "INFORMES DE EJECUCIÓN GUARDADOS" in upper:
            return 7
        if "FASE 5" in upper and "PREDICCIÓN" in upper or "FASE 5" in upper and "PREDICCION" in upper:
            return 6
        if "PASO 6/6" in upper or ">>> DIARIA:" in upper or ">>> SEMANAL:" in upper:
            return 6
        if "FASE 4" in upper and ("VALIDACIÓN" in upper or "VALIDACION" in upper):
            return 5
        if "FASE 3" in upper and ("CALIBRACIÓN" in upper or "CALIBRACION" in upper):
            return 4
        if "FASE 2" in upper and "ENTRENAMIENTO" in upper:
            return 3
        if "FASE 1" in upper and ("PREPARACIÓN" in upper or "PREPARACION" in upper or "FEATURES" in upper):
            return 2
        if "PASO 1/6" in upper or "HISTÓRICO COMPLETO 1988" in upper or "HISTORICO COMPLETO 1988" in upper:
            return 1
        if "CERRANDO" in upper or "ACTUALIZANDO HISTORIAL" in upper:
            return 7
        return 0

    def _set_stage(self, stage):
        stage = max(1, min(len(self.STAGES), int(stage)))
        if stage > self.current_stage:
            self.current_stage = stage
        lo, _ = self.STAGE_RANGES.get(stage, (0.0, 100.0))
        self.progress_var.set(max(float(self.progress_var.get()), lo))
        self._paint_stage_tiles()
        self.graph_status_var.set(f"Fase {stage}/{len(self.STAGES)}: {self.STAGES[stage-1]}")
        self._record_graph_point()

    def _paint_stage_tiles(self):
        for idx, (frame, label) in enumerate(self.stage_tiles, start=1):
            if not self.running and self.result_payload is not None and not self.last_error:
                border_color, fill_color, text_color = self.GREEN, "#14532d", self.GREEN_LIGHT
            elif not self.running and self.last_error:
                if idx < self.current_stage:
                    border_color, fill_color, text_color = self.GREEN, "#14532d", self.GREEN_LIGHT
                else:
                    border_color, fill_color, text_color = self.RED, "#4a2026", "#fecaca"
            elif not self.running:
                border_color, fill_color, text_color = self.RED, "#4a2026", "#fecaca"
            elif idx < self.current_stage:
                border_color, fill_color, text_color = self.GREEN, "#14532d", self.GREEN_LIGHT
            elif idx == self.current_stage:
                border_color = self.ORANGE if self.active_blink else "#a95f12"
                fill_color = "#a85d0c" if self.active_blink else "#432b13"
                text_color = "#ffffff" if self.active_blink else "#ffd9a3"
            else:
                border_color, fill_color, text_color = self.RED, "#4a2026", "#fecaca"
            try:
                frame.configure(bg=border_color)
                label.configure(bg=fill_color, fg=text_color)
            except tk.TclError:
                pass

    def _blink_loop(self):
        if self.running:
            self.active_blink = not self.active_blink
            self._paint_stage_tiles()
        self.root.after(430, self._blink_loop)

    # ----------------------- Ejecución del motor ---------------------
    def start_run(self, mode):
        if self.running:
            messagebox.showwarning("Fuxing ya está trabajando", "Ya hay una ejecución en curso. Espera a que termine antes de iniciar otra.", parent=self.root)
            return
        if mode not in ("both", "weekly"):
            mode = "both"
        self.running = True
        _RUN_PAUSE.reset()
        self.run_mode = mode
        if self.pause_button is not None:
            self.pause_button.configure(state="normal", text="PAUSAR PROCESO")
        self.run_started = time.monotonic()
        self.current_stage = 1
        self.result_payload = None
        self.last_error = None
        self.daily_combo = None
        self.weekly_combo = None
        self.daily_probability = None
        self.phase_times = []
        self.phase_progress = []
        self.cpu_values = []
        self.memory_values = []
        self.val_losses = []
        self.progress_var.set(0.0)
        self._paint_stage_tiles()
        self.pred_left_var.set("ESTUDIANDO DATOS")
        self.pred_left_label.configure(fg=self.GREEN_LIGHT)
        self.pred_right_var.set("INICIANDO MOTOR")
        self.pred_right_label.configure(fg=self.GREEN_LIGHT)
        self.pred_activity_var.set("Inicializando Fuxing y preparando el análisis…")
        self.control_var.set("Ejecución en curso — el motor trabaja en segundo plano")
        self.status_var.set("Iniciando Fuxing…")
        self.auto_mode_var.set("En ejecución: diaria + semanal" if mode == "both" else "No seleccionada")
        self.weekly_mode_var.set("En ejecución: solo semanal" if mode == "weekly" else "No seleccionada")
        self.log_text.insert("end", "\n" + "=" * 72 + "\n", "audit")
        self.log_text.insert("end", f"INICIO GUI · modo={mode.upper()} · {datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S')}\n", "audit")
        self.log_text.insert("end", "=" * 72 + "\n", "audit")
        self.log_text.see("end")
        self._draw_prediction_canvas()
        thread = threading.Thread(target=self._run_worker, args=(mode,), daemon=True, name="Fuxing-Motor")
        thread.start()

    def toggle_pause(self):
        """Solicita una pausa cooperativa o reanuda el motor en curso."""
        if not self.running:
            return
        if _RUN_PAUSE.is_pause_requested():
            _RUN_PAUSE.resume()
            if self.pause_button is not None:
                self.pause_button.configure(text="PAUSAR PROCESO", state="normal")
            self.control_var.set("Reanudando — esperando que continúe el cálculo")
            self.status_var.set("Se ha solicitado reanudar Fuxing.")
        else:
            _RUN_PAUSE.request_pause()
            if self.pause_button is not None:
                self.pause_button.configure(text="REANUDAR PROCESO", state="normal")
            self.control_var.set("Pausa solicitada — se activará en el siguiente punto seguro")
            self.status_var.set("Esperando a que termine la operación interna actual para pausar con seguridad.")

    def _run_worker(self, mode):
        try:
            result = run_fuxing_v3(mode=mode)
            self.events.put(("finished", result))
        except BaseException as exc:
            self.events.put(("failed", f"{type(exc).__name__}: {exc}", traceback.format_exc()))

    def _finish_run(self, payload):
        self.running = False
        _RUN_PAUSE.resume()
        if self.pause_button is not None:
            self.pause_button.configure(state="disabled", text="PAUSAR PROCESO")
        self.result_payload = payload if isinstance(payload, dict) else None
        self.last_error = None
        self.progress_var.set(100.0)
        self.current_stage = len(self.STAGES)
        self._paint_stage_tiles()
        self.pred_left_var.set("PROCESO COMPLETADO")
        self.pred_left_label.configure(fg=self.GREEN_LIGHT)
        self.pred_right_var.set("RESULTADOS DISPONIBLES")
        self.pred_right_label.configure(fg=self.GREEN_LIGHT)
        self.pred_activity_var.set("Análisis terminado")
        self.control_var.set("Ejecución finalizada correctamente")
        self.status_var.set("Fuxing ha terminado; revisa las combinaciones y los informes.")
        self.graph_status_var.set("Proceso completado; gráficas actualizadas con los datos disponibles.")
        if self.result_payload:
            self.daily_combo = self.result_payload.get("daily", {}).get("primary_combination")
            self.weekly_combo = self.result_payload.get("weekly", {}).get("combination")
            probs = self.result_payload.get("daily", {}).get("bayes_p_top6")
            if probs is not None:
                try:
                    arr = np.asarray(probs, dtype=float).reshape(-1)
                    if len(arr) == MAX_NUM:
                        self.daily_probability = arr
                except Exception:
                    self.daily_probability = None
            self._draw_prediction_canvas()
        else:
            self.result_canvas.delete("all")
            self.result_canvas.create_text(20, 30, anchor="w", text="La ejecución terminó sin un resultado JSON que mostrar.", fill=self.RED, font=("Segoe UI", 11, "bold"))
        self._load_history_distribution()
        self._record_graph_point(force=True)
        self._draw_graphs()
        self.refresh_info_tabs()
        try:
            log.save_logs()
        except Exception:
            pass

    def _fail_run(self, error, details):
        self.running = False
        _RUN_PAUSE.resume()
        if self.pause_button is not None:
            self.pause_button.configure(state="disabled", text="PAUSAR PROCESO")
        self.last_error = error
        self.pred_left_var.set("ERROR DE EJECUCIÓN")
        self.pred_left_label.configure(fg="#ff6b7a")
        self.pred_right_var.set("REVISAR REGISTRO")
        self.pred_right_label.configure(fg="#ff6b7a")
        self.pred_activity_var.set(error[:100])
        self.control_var.set("La ejecución ha fallado; el detalle está en el monitor de actividad.")
        self.status_var.set(error)
        try:
            self.log_text.insert("end", "\n" + details + "\n", "error")
            self.log_text.see("end")
        except Exception:
            pass
        self._paint_stage_tiles()
        self.refresh_info_tabs()
        messagebox.showerror("Fuxing: ejecución fallida", error + "\n\nConsulta el registro de actividad para ver el detalle.", parent=self.root)
        try:
            log.save_logs()
        except Exception:
            pass

    def _on_close(self):
        if self.running:
            close = messagebox.askyesno(
                "Fuxing sigue ejecutándose",
                "El motor sigue calculando en segundo plano. Si cierras la ventana, el proceso se interrumpirá.\n\n¿Quieres cerrar igualmente?",
                parent=self.root,
            )
            if not close:
                return
        try:
            log.save_logs()
            log.close()
        except Exception:
            pass
        self.root.destroy()

    # --------------------------- Gráficas ----------------------------
    def _load_history_distribution(self):
        try:
            rows = []
            if SQLITE_FILE.exists():
                with sqlite3.connect(str(SQLITE_FILE), timeout=2.0) as con:
                    rows = con.execute("SELECT fecha,n1,n2,n3,n4,n5,n6 FROM draws ORDER BY fecha").fetchall()
            if not rows and CSV_FILE.exists():
                with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as fh:
                    import csv as _csv
                    rd = _csv.DictReader(fh)
                    for row in rd:
                        try:
                            rows.append((row.get("fecha", ""), *[int(row[f"n{i}"]) for i in range(1, 7)]))
                        except Exception:
                            continue
            counts = np.zeros(MAX_NUM, dtype=float)
            recent = np.zeros(MAX_NUM, dtype=float)
            clean_rows = []
            for row in rows:
                nums = []
                try:
                    nums = [int(x) for x in row[1:7]]
                except Exception:
                    continue
                if len(nums) != 6 or len(set(nums)) != 6 or any(n < 1 or n > MAX_NUM for n in nums):
                    continue
                clean_rows.append((str(row[0]), nums))
            for _, nums in clean_rows:
                for n in nums:
                    counts[n - 1] += 1
            last = clean_rows[-365:]
            for _, nums in last:
                for n in nums:
                    recent[n - 1] += 1
            self.history_rows = clean_rows
            self.history_total = len(clean_rows)
            self.history_freq = counts
            self.recent_freq_per100 = recent / max(1, len(last)) * 100.0
            if self.history_total:
                self.graph_status_var.set(f"Histórico local: {self.history_total:,} sorteos · muestra reciente: {len(last)} sorteos")
            self._draw_graphs()
        except Exception as exc:
            self.graph_status_var.set(f"Esperando histórico para las gráficas ({type(exc).__name__})")

    def _record_graph_point(self, force=False):
        if self.run_started is None:
            return
        elapsed = max(0.0, time.monotonic() - self.run_started)
        if not self.phase_times or force or elapsed - self.phase_times[-1] >= 0.20:
            self.phase_times.append(elapsed)
            progress = float(self.progress_var.get())
            if self.current_stage:
                lo, hi = self.STAGE_RANGES.get(self.current_stage, (0.0, 100.0))
                progress = max(progress, lo)
            self.phase_progress.append(min(100.0, progress))
            if GUI_PSUTIL_AVAILABLE:
                try:
                    proc = psutil.Process(os.getpid())
                    self.cpu_values.append((elapsed, float(proc.cpu_percent(interval=None))))
                    self.memory_values.append((elapsed, proc.memory_info().rss / (1024 ** 2)))
                except Exception:
                    pass
            if len(self.phase_times) > 2000:
                self.phase_times = self.phase_times[-1500:]
                self.phase_progress = self.phase_progress[-1500:]
            self._draw_graphs()

    def _draw_graphs(self):
        if not GUI_MATPLOTLIB_AVAILABLE or self.canvas is None:
            return
        try:
            # Elimina los ejes secundarios de la actualización anterior antes
            # de crear otros; así no se acumulan capas de Matplotlib en cada tick.
            if getattr(self, "ax_freq_right", None) is not None:
                try:
                    self.ax_freq_right.remove()
                except Exception:
                    pass
                self.ax_freq_right = None
            if getattr(self, "ax_progress_right", None) is not None:
                try:
                    self.ax_progress_right.remove()
                except Exception:
                    pass
                self.ax_progress_right = None
            ax = self.ax_freq
            ax.clear()
            self.ax_progress.clear()
            self._style_graph_axes()
            nums = np.arange(1, MAX_NUM + 1)
            if self.history_total:
                bars = ax.bar(nums, self.history_freq, color="#3a86a8", alpha=0.78, width=0.82, label="Frecuencia histórica")
                ax.set_ylabel("Apariciones", fontsize=8)
                ax.set_xlim(0, 50)
                ax.set_xticks(np.arange(1, 50, 4))
                ax.set_title(f"Frecuencia de números (1–49) · {self.history_total:,} sorteos", fontsize=9, pad=6)
                axr = ax.twinx()
                self.ax_freq_right = axr
                axr.plot(nums, self.recent_freq_per100, color="#ffd166", linewidth=1.4, marker=".", markersize=2, label="Últimos 365 · por 100 sorteos")
                axr.set_ylabel("Frecuencia reciente / 100", fontsize=7, color="#ffd166")
                axr.tick_params(axis="y", colors="#ffd166", labelsize=6)
                for spine in axr.spines.values():
                    spine.set_color(self.BORDER)
                handles, labels = ax.get_legend_handles_labels()
                h2, l2 = axr.get_legend_handles_labels()
                ax.legend(handles + h2, labels + l2, loc="upper left", fontsize=6, framealpha=0.18)
                # Marcas del resultado final, sin confundirlas con una probabilidad.
                if self.daily_combo:
                    ax.scatter(self.daily_combo, [self.history_freq[int(n)-1] for n in self.daily_combo], marker="*", s=65, color="#ff6b6b", edgecolors="#fff1f2", linewidths=0.4, label="Combinación diaria")
                if self.weekly_combo:
                    ax.scatter(self.weekly_combo, [self.history_freq[int(n)-1] for n in self.weekly_combo], marker="D", s=25, color="#6ee7a8", edgecolors="#dcfce7", linewidths=0.35, label="Combinación semanal")
            else:
                ax.set_title("Distribución histórica de señales (1–49)", fontsize=9, pad=6)
                ax.set_xlim(0, 50)
                ax.set_xticks(np.arange(1, 50, 4))
                ax.text(0.5, 0.5, "La distribución se cargará cuando Fuxing actualice el histórico.", transform=ax.transAxes, ha="center", va="center", color=self.MUTED, fontsize=8)
                ax.set_ylabel("Apariciones")
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.grid(True, axis="y", alpha=0.15)

            ax2 = self.ax_progress
            ax2.clear()
            self._style_graph_axes()
            if self.phase_times:
                x = np.asarray(self.phase_times, dtype=float) / 60.0
                y = np.asarray(self.phase_progress, dtype=float)
                ax2.plot(x, y, color=self.CYAN, linewidth=1.8, label="Progreso estimado por fases")
                ax2.set_ylim(0, 105)
                ax2.set_ylabel("Progreso (%)", fontsize=7, color=self.CYAN)
                ax2.set_xlabel("Tiempo transcurrido (minutos)", fontsize=7)
                if self.cpu_values:
                    cpu_x = np.asarray([p[0] for p in self.cpu_values]) / 60.0
                    cpu_y = np.asarray([p[1] for p in self.cpu_values])
                    axr2 = ax2.twinx()
                    self.ax_progress_right = axr2
                    axr2.plot(cpu_x, cpu_y, color="#ff9f1c", linewidth=1.0, alpha=0.8, label="CPU del proceso (%)")
                    axr2.set_ylabel("CPU (%)", fontsize=7, color="#ff9f1c")
                    axr2.tick_params(axis="y", colors="#ff9f1c", labelsize=6)
                    for spine in axr2.spines.values():
                        spine.set_color(self.BORDER)
                ax2.legend(loc="upper left", fontsize=6, framealpha=0.18)
                ax2.set_title("Progreso de ejecución y actividad del proceso", fontsize=9, pad=5)
            else:
                ax2.set_title("Progreso de ejecución y actividad del proceso", fontsize=9, pad=5)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 100)
                ax2.set_ylabel("Progreso (%)", fontsize=7)
                ax2.set_xlabel("Tiempo transcurrido (minutos)", fontsize=7)
                ax2.text(0.5, 0.5, "Inicia una ejecución para ver la evolución de las fases.", transform=ax2.transAxes, ha="center", va="center", color=self.MUTED, fontsize=8)
            ax2.tick_params(axis="both", labelsize=6)
            ax2.grid(True, alpha=0.15)
            self.canvas.draw_idle()
        except Exception as exc:
            try:
                self.graph_status_var.set(f"Gráfica no disponible: {type(exc).__name__}: {exc}")
            except Exception:
                pass

    # ---------------------- Presentación de salida -------------------
    def _draw_prediction_canvas(self):
        if not hasattr(self, "result_canvas"):
            return
        canvas = self.result_canvas
        try:
            canvas.delete("all")
            width = max(700, canvas.winfo_width())
            rows = []
            target = None
            target_end = None
            if self.result_payload:
                execution = self.result_payload.get("execution", {})
                generated = self.result_payload.get("generated_at")
                try:
                    gen_dt = datetime.fromisoformat(str(generated)) if generated else _madrid_now()
                    latest = execution.get("last_date")
                    target = _first_prediction_target_date(latest, gen_dt)
                    target_end = target + timedelta(days=6)
                except Exception:
                    target = None
            if self.run_mode in ("both", None) and self.daily_combo:
                label = "DIARIA"
                if target:
                    label += f" · objetivo {target.strftime('%d/%m/%Y')}"
                rows.append((label, self.daily_combo, "#ff6b6b"))
            if self.weekly_combo:
                label = "SEMANAL"
                if target and target_end:
                    label += f" · periodo {target.strftime('%d/%m')}–{target_end.strftime('%d/%m/%Y')}"
                rows.append((label, self.weekly_combo, self.CYAN))
            if not rows:
                canvas.create_text(width / 2, 57, text="Esperando la selección final…", fill=self.MUTED, font=("Segoe UI", 12, "bold"))
                return
            for idx, (title, combo, accent) in enumerate(rows):
                y = 31 if len(rows) == 2 and idx == 0 else (88 if len(rows) == 2 else 57)
                canvas.create_text(12, y, anchor="w", text=title, fill=accent, font=("Segoe UI", 10, "bold"))
                start_x = min(max(245, width * 0.32), width - 310)
                for j, val in enumerate(combo[:6]):
                    x = start_x + j * 47
                    r = 17
                    canvas.create_oval(x-r, y-r, x+r, y+r, fill=accent, outline="#e5e7eb", width=1)
                    canvas.create_text(x, y, text=f"{int(val):02d}", fill="#10151c", font=("Segoe UI", 11, "bold"))
        except Exception:
            pass

    def _update_clock(self):
        now = datetime.now().astimezone()
        weekdays = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
        weekday = weekdays[now.weekday()].capitalize()
        self.clock_var.set(f"{weekday}, {now:%d/%m/%Y %H:%M:%S}")
        if self.running and self.run_started is not None:
            elapsed = int(time.monotonic() - self.run_started)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            self.elapsed_var.set(f"Tiempo: {h:02d}:{m:02d}:{s:02d}")
            if time.monotonic() - self._last_graph_draw >= 2.0:
                self._last_graph_draw = time.monotonic()
                self._record_graph_point(force=True)
            if GUI_PSUTIL_AVAILABLE:
                try:
                    proc = psutil.Process(os.getpid())
                    ram = proc.memory_info().rss / (1024 ** 2)
                    cpu = proc.cpu_percent(interval=None)
                    self.graph_status_var.set(f"Fase {self.current_stage}/7 · CPU {cpu:.1f}% · RAM proceso {ram:,.0f} MB · logs {self.log_count:,}")
                except Exception:
                    pass
        self.root.after(1000, self._update_clock)

    # ---------------------- Pestañas informativas --------------------
    @staticmethod
    def _put_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def refresh_info_tabs(self):
        payload = self.result_payload or {}
        if payload:
            execution = payload.get("execution", {})
            daily = payload.get("daily", {})
            weekly = payload.get("weekly", {})
            training = payload.get("training", {})
            validation = payload.get("validation", {})
            audit = payload.get("data_integrity", {})
            summary = {
                "generado": payload.get("generated_at"),
                "ejecución": execution,
                "integridad_historica": {
                    "total_sorteos": audit.get("total_draws", execution.get("historical_draws")),
                    "fecha_inicio": audit.get("min_date", execution.get("first_date")),
                    "fecha_final": audit.get("max_date", execution.get("last_date")),
                    "recuento_oficial": audit.get("official_expected_total"),
                },
                "diaria": {
                    "combinación": daily.get("primary_combination"),
                    "top_números": daily.get("top_numbers"),
                    "pesos": daily.get("weights"),
                },
                "semanal": {
                    "combinación": weekly.get("combination"),
                    "estabilidad": weekly.get("stability_score"),
                    "concentración": weekly.get("concentration_score"),
                    "analogías": weekly.get("analog_count"),
                },
                "entrenamiento": training,
                "validación": validation,
            }
            self._put_text(self.stats_text, json.dumps(summary, ensure_ascii=False, indent=2, default=str))
            bt = {
                "backtest_diario": payload.get("backtest_daily_statistical_only"),
                "backtest_semanal": weekly.get("backtest"),
                "validación": validation,
                "advertencia": "Los resultados históricos no garantizan que la Bonoloto sea predecible."
            }
            self._put_text(self.backtest_text, json.dumps(bt, ensure_ascii=False, indent=2, default=str))
        else:
            message = self.last_error or "Todavía no hay resultados de una ejecución finalizada en esta sesión."
            self._put_text(self.stats_text, message)
            self._put_text(self.backtest_text, message)

    def open_output_folder(self):
        try:
            path = str(BASE_DIR)
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", path])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("No se pudo abrir la carpeta", f"Ruta: {BASE_DIR}\n\n{exc}", parent=self.root)


def _launch_gui():
    root = tk.Tk()
    FuxingDesktopGUI(root)
    root.mainloop()


def main():
    """Sin argumentos abre GUI; --console/--mode mantiene disponible el CLI."""
    parser = argparse.ArgumentParser(description="Fuxing Bonoloto — GUI + motor definitivo")
    parser.add_argument("--mode", choices=("daily", "weekly", "both"), default=None,
                        help="Consola: predicción diaria, semanal o ambas.")
    parser.add_argument("--console", action="store_true",
                        help="Ejecutar por consola en lugar de abrir la interfaz gráfica.")
    args = parser.parse_args()
    if args.console or args.mode is not None:
        mode = args.mode or "both"
        try:
            run_fuxing_v3(mode=mode)
        except KeyboardInterrupt:
            log.warn("Proceso interrumpido por el usuario.")
            raise SystemExit(130)
        except Exception as exc:
            log.error(f"FUXING v3 terminó con error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            log.close()
        return
    try:
        _launch_gui()
    except Exception:
        try:
            log.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
