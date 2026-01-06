#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NAT Sector Updater (IVAO Aurora) - Shanwick/Gander / OCC Oceanic

- Fetches live NAT/OTS tracks from FAA NAT page
- Terminal UI (Rich if installed; fallback to plain text)
- First-run configuration:
  - language (fr/en)
  - path to main .isc file OR a folder containing .isc (script will auto-detect)
  - OCC/Oceanic include folder name (inside <sector_root>/include/<folder>)
- Generates NAT_TRACKS.awh using Aurora [HIGH AIRWAY] format:
  - Named points written as NAME;NAME; (like your sample)
  - Lat/Lon tokens like 55/20 become N055.00.00.000;W020.00.00.000
- Patches the .isc to ensure NAT_TRACKS.awh is referenced in [HIGH AIRWAY] via F;...

NEW:
- Option to set the CONFIG FILE PATH (persistent):
  - CLI:  python nat_update.py --config "D:\\my\\nat_config.json"
  - Menu option to change config path; it saves a locator file in ~/.nat_sector_updater/

Optional dependency:
  pip install rich
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
import urllib.request


# -------------------- Optional Rich UI --------------------

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich import box

    RICH = True
except Exception:
    RICH = False


# -------------------- Constants --------------------

FAA_NAT_URL = "https://notams.aim.faa.gov/nat.html"


# -------------------- App storage (default) --------------------

APP_DIR = Path.home() / ".nat_sector_updater"
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
LOCATOR_PATH = APP_DIR / "config_location.json"  # stores the chosen config path persistently


# -------------------- i18n --------------------

I18N = {
    "fr": {
        "app_title": "NAT Sector Updater (IVAO Aurora)",
        "first_run": "Première exécution : configuration",
        "choose_lang": "Choisis la langue (fr/en)",
        "lang_invalid": "Langue invalide. Valeurs possibles: fr, en",
        "ask_isc": "Chemin du fichier secteur principal (.isc) OU dossier contenant les .isc",
        "ask_occ_folder": "Nom du dossier dans <racine secteur>/include/ (ex: OCC_Oceanic)",
        "cfg_saved": "Configuration enregistrée",
        "menu": "Menu",
        "menu_1": "1) Rafraîchir & afficher les NAT",
        "menu_2": "2) Mettre à jour le secteur (générer NAT_TRACKS.awh + patch .isc)",
        "menu_3": "3) Reconfigurer",
        "menu_4": "4) Quitter",
        "menu_5": "5) Chemin du fichier config (afficher/modifier)",
        "prompt_choice": "Choix",
        "fetching": "Récupération des NAT en direct…",
        "fetch_fail": "Impossible de récupérer les NAT.",
        "parse_fail": "Impossible de parser les NAT.",
        "no_tracks": "Aucun track trouvé.",
        "tracks_title": "Tracks trouvés",
        "isc_missing": "Le fichier .isc est introuvable.",
        "isc_unreadable": "Impossible de lire le .isc (droits/accès).",
        "writing_awh": "Écriture du fichier NAT_TRACKS.awh…",
        "patching_isc": "Modification du .isc (ajout F;...)…",
        "done": "Terminé.",
        "backup": "Backup créé",
        "confirm_overwrite": "Le fichier NAT_TRACKS.awh existe déjà. L'écraser ?",
        "paths": "Chemins",
        "isc_path": "ISC",
        "include_dir": "Include",
        "awh_path": "Fichier AWH",
        "created_dir": "Dossier créé",
        "choose_isc": "Plusieurs .isc trouvés - choisis",
        "pick_num": "Numéro du .isc à utiliser",
        "no_isc_found": "Aucun fichier .isc trouvé dans",
        "bad_path": "Chemin invalide (fichier .isc ou dossier attendu)",
        "config_path_title": "Fichier config",
        "current_config_path": "Chemin actuel",
        "ask_new_config_path": "Nouveau chemin du fichier config (vide = annuler)",
        "move_existing_config": "Déplacer l'ancien fichier config vers le nouveau chemin ?",
        "config_path_updated": "Chemin du fichier config mis à jour",
        "config_path_invalid": "Chemin de config invalide",
        "config_loaded": "Config chargée",
    },
    "en": {
        "app_title": "NAT Sector Updater (IVAO Aurora)",
        "first_run": "First run: configuration",
        "choose_lang": "Choose language (fr/en)",
        "lang_invalid": "Invalid language. Allowed: fr, en",
        "ask_isc": "Path to main sector file (.isc) OR folder containing .isc files",
        "ask_occ_folder": "Folder name inside <sector_root>/include/ (e.g. OCC_Oceanic)",
        "cfg_saved": "Configuration saved",
        "menu": "Menu",
        "menu_1": "1) Refresh & show NAT tracks",
        "menu_2": "2) Update sector (generate NAT_TRACKS.awh + patch .isc)",
        "menu_3": "3) Reconfigure",
        "menu_4": "4) Quit",
        "menu_5": "5) Config file path (show/change)",
        "prompt_choice": "Choice",
        "fetching": "Fetching live NAT tracks…",
        "fetch_fail": "Failed to fetch NAT tracks.",
        "parse_fail": "Failed to parse NAT tracks.",
        "no_tracks": "No tracks found.",
        "tracks_title": "Tracks found",
        "isc_missing": "The .isc file does not exist.",
        "isc_unreadable": "Cannot read the .isc file (permissions/access).",
        "writing_awh": "Writing NAT_TRACKS.awh…",
        "patching_isc": "Patching .isc (adding F;...)…",
        "done": "Done.",
        "backup": "Backup created",
        "confirm_overwrite": "NAT_TRACKS.awh already exists. Overwrite?",
        "paths": "Paths",
        "isc_path": "ISC",
        "include_dir": "Include",
        "awh_path": "AWH file",
        "created_dir": "Directory created",
        "choose_isc": "Multiple .isc found - pick one",
        "pick_num": "Pick .isc number to use",
        "no_isc_found": "No .isc files found in",
        "bad_path": "Invalid path (expected .isc file or folder)",
        "config_path_title": "Config file",
        "current_config_path": "Current path",
        "ask_new_config_path": "New config file path (empty = cancel)",
        "move_existing_config": "Move old config file to the new path?",
        "config_path_updated": "Config file path updated",
        "config_path_invalid": "Invalid config path",
        "config_loaded": "Config loaded",
    },
}


def tr(cfg: dict, key: str) -> str:
    lang = cfg.get("language", "fr")
    return I18N.get(lang, I18N["fr"]).get(key, key)


# -------------------- Data Model --------------------

@dataclass
class Track:
    letter: str
    route_tokens: List[str]
    validity_from_utc: Optional[datetime]
    validity_to_utc: Optional[datetime]
    east_levels: List[int]
    west_levels: List[int]

    @property
    def direction(self) -> str:
        if self.west_levels and not self.east_levels:
            return "WEST"
        if self.east_levels and not self.west_levels:
            return "EAST"
        if self.east_levels and self.west_levels:
            return "BOTH"
        return "UNK"

    def route_str(self) -> str:
        return " ".join(self.route_tokens)

    def validity_str(self) -> str:
        if not self.validity_from_utc or not self.validity_to_utc:
            return "?"
        f = self.validity_from_utc.strftime("%Y-%m-%d %H:%MZ")
        t = self.validity_to_utc.strftime("%Y-%m-%d %H:%MZ")
        return f"{f} → {t}"


# -------------------- CLI (config path option) --------------------

def parse_cli_config_path(argv: List[str]) -> Optional[Path]:
    """
    Supports:
      --config <path>
      -c <path>
    """
    for i, a in enumerate(argv):
        if a in ("--config", "-c"):
            if i + 1 < len(argv):
                return Path(argv[i + 1]).expanduser()
    return None


# -------------------- Config path locator (persistent) --------------------

def ensure_app_dir() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)


def load_locator() -> Optional[Path]:
    """
    Returns persisted config path if locator file exists and is valid-ish.
    """
    try:
        if LOCATOR_PATH.exists():
            data = json.loads(LOCATOR_PATH.read_text(encoding="utf-8"))
            p = data.get("config_path")
            if p:
                return Path(p).expanduser()
    except Exception:
        return None
    return None


def save_locator(config_path: Path) -> None:
    ensure_app_dir()
    LOCATOR_PATH.write_text(
        json.dumps({"config_path": str(config_path)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_effective_config_path(argv: List[str]) -> Path:
    # Priority: CLI > locator > default
    cli = parse_cli_config_path(argv)
    if cli:
        return cli
    loc = load_locator()
    if loc:
        return loc
    return DEFAULT_CONFIG_PATH


# -------------------- Config IO --------------------

def load_config(config_path: Path) -> dict:
    ensure_app_dir()
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(config_path: Path, cfg: dict) -> None:
    ensure_app_dir()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def prompt_input(prompt: str) -> str:
    if RICH:
        return Prompt.ask(prompt).strip()
    return input(prompt + " > ").strip()


def prompt_yesno(prompt: str, default: bool = False) -> bool:
    if RICH:
        return Confirm.ask(prompt, default=default)
    val = input(f"{prompt} [{'Y/n' if default else 'y/N'}] > ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "o", "oui", "1", "true")


def run_first_time_wizard(config_path: Path, console=None) -> dict:
    cfg: dict = {}

    if RICH and console:
        console.print(Panel(tr({"language": "fr"}, "first_run"), title="Setup", expand=False))
    else:
        print("=== First run / Première exécution ===")

    while True:
        lang = prompt_input(I18N["fr"]["choose_lang"]).lower()
        if lang in ("fr", "en"):
            cfg["language"] = lang
            break
        msg = I18N["fr"]["lang_invalid"]
        if RICH and console:
            console.print(f"[red]{msg}[/red]")
        else:
            print(msg)

    isc_path = prompt_input(tr(cfg, "ask_isc"))
    isc_path = str(Path(isc_path).expanduser())
    occ_folder = prompt_input(tr(cfg, "ask_occ_folder"))

    cfg["isc_path"] = isc_path
    cfg["occ_include_folder"] = occ_folder.strip()
    cfg["last_fetch_cache"] = {}

    save_config(config_path, cfg)
    if RICH and console:
        console.print(f"[green]{tr(cfg,'cfg_saved')}[/green] -> {config_path}")
    else:
        print(tr(cfg, "cfg_saved"), "->", config_path)
    return cfg


# -------------------- NAT Fetch / Parse --------------------

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

VALIDITY_RE = re.compile(r"^\s*([A-Z]{3})\s+(\d{2})/(\d{4})Z\s+TO\s+([A-Z]{3})\s+(\d{2})/(\d{4})Z\s*$")
TRACK_LINE_RE = re.compile(r"^\s*([A-Z])\s+(.+?)\s*$")
LEVELS_RE = re.compile(r"^\s*(EAST|WEST)\s+LVLS\s+(.*)\s*$")


def http_get_text(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")


def clean_text(raw: str) -> str:
    out = []
    for ch in raw:
        o = ord(ch)
        if ch in ("\n", "\t"):
            out.append(ch)
        elif 32 <= o <= 126:
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def parse_validity(line: str, now_utc: datetime) -> Optional[Tuple[datetime, datetime]]:
    m = VALIDITY_RE.match(line)
    if not m:
        return None
    m1, d1, hhmm1, m2, d2, hhmm2 = m.groups()
    mon1, mon2 = MONTHS.get(m1), MONTHS.get(m2)
    if not mon1 or not mon2:
        return None

    def mk_dt(mon: int, day: int, hhmm: str) -> datetime:
        hh = int(hhmm[:2])
        mm = int(hhmm[2:])
        year = now_utc.year
        dt = datetime(year, mon, day, hh, mm, tzinfo=timezone.utc)
        # year boundary best-effort
        if dt - now_utc > (180 * 24 * 3600):
            dt = datetime(year - 1, mon, day, hh, mm, tzinfo=timezone.utc)
        elif now_utc - dt > (180 * 24 * 3600):
            dt = datetime(year + 1, mon, day, hh, mm, tzinfo=timezone.utc)
        return dt

    v_from = mk_dt(mon1, int(d1), hhmm1)
    v_to = mk_dt(mon2, int(d2), hhmm2)
    return v_from, v_to


def parse_levels(s: str) -> List[int]:
    s = s.strip()
    if s.upper() == "NIL":
        return []
    vals = []
    for tok in s.split():
        if tok.isdigit():
            vals.append(int(tok))
    return vals


def looks_like_route_token(tok: str) -> bool:
    if re.fullmatch(r"\d{2}(\.\d+)?/\d{2}(\.\d+)?", tok):
        return True
    if re.fullmatch(r"[A-Z0-9]{3,8}", tok):
        return True
    return False


def parse_faa_tracks(text: str) -> List[Track]:
    now_utc = datetime.now(timezone.utc)
    cleaned = clean_text(text)
    lines = [ln.rstrip() for ln in cleaned.splitlines() if ln.strip()]

    current_validity: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    tracks: List[Track] = []

    i = 0
    while i < len(lines):
        ln = lines[i].strip()

        v = parse_validity(ln, now_utc)
        if v:
            current_validity = v
            i += 1
            continue

        tm = TRACK_LINE_RE.match(ln)
        if tm:
            letter = tm.group(1)
            rest = tm.group(2).strip()

            if rest.upper().startswith(("REMARKS", "END OF", "PART ")):
                i += 1
                continue

            tokens = rest.split()
            if not tokens or not all(looks_like_route_token(t) for t in tokens[: min(3, len(tokens))]):
                i += 1
                continue

            east_lvls: List[int] = []
            west_lvls: List[int] = []

            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()

                if VALIDITY_RE.match(nxt):
                    break

                ntm = TRACK_LINE_RE.match(nxt)
                if ntm and not nxt.upper().startswith(("REMARKS", "END OF", "PART ")):
                    break

                if nxt.upper().startswith(("REMARKS", "END OF PART", "END OF MESSAGE")):
                    break

                lm = LEVELS_RE.match(nxt)
                if lm:
                    which = lm.group(1).upper()
                    lvls = parse_levels(lm.group(2))
                    if which == "EAST":
                        east_lvls = lvls
                    else:
                        west_lvls = lvls
                j += 1

            tracks.append(
                Track(
                    letter=letter,
                    route_tokens=tokens,
                    validity_from_utc=current_validity[0],
                    validity_to_utc=current_validity[1],
                    east_levels=east_lvls,
                    west_levels=west_lvls,
                )
            )
            i = j
            continue

        i += 1

    uniq = {}
    for t in tracks:
        k = (
            t.letter,
            t.validity_from_utc.isoformat() if t.validity_from_utc else None,
            t.direction,
            t.route_str(),
        )
        uniq[k] = t
    return list(uniq.values())


# -------------------- Coordinate helpers --------------------

def dec_to_dms(value: float) -> Tuple[int, int, int, int]:
    sign = 1 if value >= 0 else -1
    v = abs(value)

    deg = int(v)
    rem = (v - deg) * 60.0
    minute = int(rem)
    rem2 = (rem - minute) * 60.0
    sec = int(rem2)
    ms = int(round((rem2 - sec) * 1000.0))

    if ms >= 1000:
        ms -= 1000
        sec += 1
    if sec >= 60:
        sec -= 60
        minute += 1
    if minute >= 60:
        minute -= 60
        deg += 1

    return deg * sign, minute, sec, ms


def format_aurora_dms_latlon(lat_deg: float, lon_deg: float) -> Tuple[str, str]:
    lat_hemi = "N" if lat_deg >= 0 else "S"
    lon_hemi = "E" if lon_deg >= 0 else "W"

    latD, latM, latS, latMS = dec_to_dms(lat_deg)
    lonD, lonM, lonS, lonMS = dec_to_dms(lon_deg)

    lat_str = f"{lat_hemi}{abs(latD):03d}.{latM:02d}.{latS:02d}.{latMS:03d}"
    lon_str = f"{lon_hemi}{abs(lonD):03d}.{lonM:02d}.{lonS:02d}.{lonMS:03d}"
    return lat_str, lon_str


def parse_latlon_token(tok: str) -> Optional[Tuple[float, float]]:
    m = re.fullmatch(r"(\d{2}(?:\.\d+)?)/(\d{2}(?:\.\d+)?)", tok)
    if not m:
        return None
    lat = float(m.group(1))
    lon = float(m.group(2))
    return lat, -lon  # West


# -------------------- Sector / ISC helpers --------------------

def read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def can_read_file(p: Path) -> bool:
    try:
        _ = p.read_bytes()[:16]
        return True
    except Exception:
        return False


def backup_file(p: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    b = p.with_suffix(p.suffix + f".bak_{ts}")
    shutil.copy2(p, b)
    return b


def find_section_indices(lines: List[str], section_name: str) -> Optional[Tuple[int, int]]:
    sec = section_name.strip().upper()
    start = None
    for idx, ln in enumerate(lines):
        if ln.strip().upper() == f"[{sec}]":
            start = idx
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.fullmatch(r"\[[A-Z0-9 _-]+\]", lines[j].strip().upper()):
            end = j
            break
    return start, end


def parse_f_refs(lines: List[str]) -> List[str]:
    refs = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("F;"):
            refs.append(s[2:].strip())
    return refs


def is_probably_isc(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".isc"


def find_isc_files_in_dir(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted(dir_path.rglob("*.isc"))


def resolve_isc_path_interactive(cfg: dict, console, raw_path: str) -> Optional[Path]:
    p = Path(raw_path).expanduser()

    if is_probably_isc(p):
        return p

    if p.exists() and p.is_dir():
        candidates = find_isc_files_in_dir(p)
        if not candidates:
            msg = f"{tr(cfg,'no_isc_found')}: {p}"
            if RICH and console:
                console.print(f"[red]{msg}[/red]")
            else:
                print(msg)
            return None

        if len(candidates) == 1:
            return candidates[0]

        if RICH and console:
            table = Table(title=tr(cfg, "choose_isc"), box=box.SIMPLE)
            table.add_column("#", style="bold")
            table.add_column("Fichier .isc" if cfg.get("language") == "fr" else ".isc file")
            for i, c in enumerate(candidates, 1):
                table.add_row(str(i), str(c))
            console.print(table)
        else:
            print(tr(cfg, "choose_isc"))
            for i, c in enumerate(candidates, 1):
                print(f"{i}) {c}")

        while True:
            sel = prompt_input(tr(cfg, "pick_num"))
            if sel.isdigit():
                idx = int(sel)
                if 1 <= idx <= len(candidates):
                    return candidates[idx - 1]

    msg = f"{tr(cfg,'bad_path')}: {p}"
    if RICH and console:
        console.print(f"[red]{msg}[/red]")
    else:
        print(msg)
    return None


def choose_airways_subdir_from_isc(isc_path: Path, include_dir: Path) -> Path:
    """
    If [HIGH AIRWAY] has refs like 'Airways\\something.awh', reuse that folder.
    Otherwise default to include_dir/Airways.
    """
    content = read_text_file(isc_path)
    lines = content.splitlines()
    sec = find_section_indices(lines, "HIGH AIRWAY")
    if sec:
        start, end = sec
        refs = parse_f_refs(lines[start + 1 : end])
        if refs:
            first = Path(refs[0].replace("\\", "/"))
            if str(first.parent) not in ("", "."):
                return include_dir / first.parent
    return include_dir / "Airways"


def patch_isc_add_high_airway_ref(isc_path: Path, rel_ref: str) -> bool:
    content = read_text_file(isc_path)
    lines = content.splitlines()

    target_line = f"F;{rel_ref}".replace("/", "\\")
    modified = False

    sec = find_section_indices(lines, "HIGH AIRWAY")
    if not sec:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append("[HIGH AIRWAY]")
        lines.append(target_line)
        modified = True
    else:
        start, end = sec
        existing = {ln.strip().replace("/", "\\") for ln in lines[start + 1 : end] if ln.strip()}
        if target_line in existing:
            return False
        lines.insert(end, target_line)
        modified = True

    if modified:
        new_text = "\n".join(lines).rstrip("\n") + "\n"
        isc_path.write_text(new_text, encoding="utf-8", errors="ignore")
    return modified


# -------------------- NAT -> AWH (your style) --------------------

def make_nat_awh(tracks: List[Track]) -> str:
    """
    NAT_TRACKS.awh using your style:
      - Named points: LAT=NAME ; LON=NAME (e.g. T;G3;NASBU;NASBU;)
      - Lat/Lon tokens like 55/20 -> N055.00.00.000 ; W020.00.00.000
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    out: List[str] = []
    out.append("// ------------------------------------------------------------------")
    out.append("// NAT tracks (generated automatically)")
    out.append(f"// Updated: {now_utc}")
    out.append(f"// Source: {FAA_NAT_URL}")
    out.append("// Format: Aurora [HIGH AIRWAY] (L/T) | names or coordinates")
    out.append("// ------------------------------------------------------------------")
    out.append("")

    def token_to_fields(tok: str) -> Optional[Tuple[str, str]]:
        tok_u = tok.strip().upper()

        ll = parse_latlon_token(tok_u)
        if ll:
            lat, lon = ll
            return format_aurora_dms_latlon(lat, lon)

        if re.fullmatch(r"[A-Z0-9]{2,8}", tok_u):
            return tok_u, tok_u

        return None

    def sort_key(t: Track):
        vf = t.validity_from_utc.timestamp() if t.validity_from_utc else 0
        return (vf, t.letter)

    for t in sorted(tracks, key=sort_key):
        aid = f"NAT_{t.letter}"

        out.append(
            f"// Track {t.letter} | {t.direction} | {t.validity_str()} | "
            f"E:{' '.join(map(str, t.east_levels)) or 'NIL'} "
            f"W:{' '.join(map(str, t.west_levels)) or 'NIL'}"
        )

        points: List[Tuple[str, str]] = []
        for tok in t.route_tokens:
            fields = token_to_fields(tok)
            if fields:
                points.append(fields)

        if not points:
            out.append("// (no drawable points)")
            out.append("")
            continue

        mid = len(points) // 2
        latL, lonL = points[mid]
        out.append(f"L;{aid};{latL};{lonL};")

        for latS, lonS in points:
            out.append(f"T;{aid};{latS};{lonS};")

        out.append("")

    return "\n".join(out).rstrip() + "\n"


# -------------------- UI helpers --------------------

def get_console():
    return Console() if RICH else None


def show_tracks(cfg: dict, console, tracks: List[Track]) -> None:
    if not tracks:
        if RICH and console:
            console.print(f"[yellow]{tr(cfg,'no_tracks')}[/yellow]")
        else:
            print(tr(cfg, "no_tracks"))
        return

    if RICH and console:
        table = Table(title=tr(cfg, "tracks_title"), box=box.SIMPLE_HEAVY)
        table.add_column("Letter", style="bold")
        table.add_column("Dir")
        table.add_column("Validity (UTC)")
        table.add_column("Levels")
        table.add_column("Route")

        for t in sorted(tracks, key=lambda x: (x.validity_from_utc or datetime.min.replace(tzinfo=timezone.utc), x.letter)):
            levels = f"E:{' '.join(map(str, t.east_levels)) or 'NIL'} | W:{' '.join(map(str, t.west_levels)) or 'NIL'}"
            table.add_row(t.letter, t.direction, t.validity_str(), levels, t.route_str())

        console.print(table)
    else:
        print("=== Tracks ===")
        for t in tracks:
            levels = f"E:{' '.join(map(str,t.east_levels)) or 'NIL'} W:{' '.join(map(str,t.west_levels)) or 'NIL'}"
            print(f"{t.letter} {t.direction} {t.validity_str()} {levels}")
            print("  ", t.route_str())


def fetch_and_parse(config_path: Path, cfg: dict, console) -> List[Track]:
    if RICH and console:
        console.print(f"[cyan]{tr(cfg,'fetching')}[/cyan]")
    else:
        print(tr(cfg, "fetching"))

    try:
        raw = http_get_text(FAA_NAT_URL, timeout=25)
    except Exception as e:
        if RICH and console:
            console.print(f"[red]{tr(cfg,'fetch_fail')}[/red] {e}")
        else:
            print(tr(cfg, "fetch_fail"), e)
        return []

    try:
        tracks = parse_faa_tracks(raw)
    except Exception as e:
        if RICH and console:
            console.print(f"[red]{tr(cfg,'parse_fail')}[/red] {e}")
        else:
            print(tr(cfg, "parse_fail"), e)
        return []

    cfg["last_fetch_cache"] = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(tracks),
    }
    save_config(config_path, cfg)
    return tracks


def update_sector(config_path: Path, cfg: dict, console, tracks: List[Track]) -> None:
    raw = cfg.get("isc_path", "")
    isc_path = resolve_isc_path_interactive(cfg, console, raw)
    if not isc_path:
        return

    if not isc_path.exists():
        if RICH and console:
            console.print(f"[red]{tr(cfg,'isc_missing')}[/red] {isc_path}")
        else:
            print(tr(cfg, "isc_missing"), isc_path)
        return

    if not can_read_file(isc_path):
        if RICH and console:
            console.print(f"[red]{tr(cfg,'isc_unreadable')}[/red] {isc_path}")
        else:
            print(tr(cfg, "isc_unreadable"), isc_path)
        return

    # Save resolved .isc path (important if config had a folder)
    cfg["isc_path"] = str(isc_path)
    save_config(config_path, cfg)

    sector_root = isc_path.parent
    occ_folder = cfg.get("occ_include_folder", "").strip()
    include_dir = sector_root / "include" / occ_folder

    if not include_dir.exists():
        include_dir.mkdir(parents=True, exist_ok=True)
        if RICH and console:
            console.print(f"[dim]{tr(cfg,'created_dir')}: {include_dir}[/dim]")
        else:
            print(tr(cfg, "created_dir"), include_dir)

    airway_dir = choose_airways_subdir_from_isc(isc_path, include_dir)
    if not airway_dir.exists():
        airway_dir.mkdir(parents=True, exist_ok=True)
        if RICH and console:
            console.print(f"[dim]{tr(cfg,'created_dir')}: {airway_dir}[/dim]")
        else:
            print(tr(cfg, "created_dir"), airway_dir)

    awh_path = airway_dir / "NAT_TRACKS.awh"

    if RICH and console:
        paths_table = Table(title=tr(cfg, "paths"), box=box.SIMPLE)
        paths_table.add_column(tr(cfg, "isc_path"))
        paths_table.add_column(tr(cfg, "include_dir"))
        paths_table.add_row(str(isc_path), str(include_dir))
        console.print(paths_table)
    else:
        print("ISC:", isc_path)
        print("Include:", include_dir)

    if awh_path.exists():
        ok = prompt_yesno(tr(cfg, "confirm_overwrite"), default=False)
        if not ok:
            return
        b = backup_file(awh_path)
        if RICH and console:
            console.print(f"[dim]{tr(cfg,'backup')}: {b}[/dim]")
        else:
            print(tr(cfg, "backup"), b)

    if RICH and console:
        console.print(f"[cyan]{tr(cfg,'writing_awh')}[/cyan]")
    else:
        print(tr(cfg, "writing_awh"))

    awh_text = make_nat_awh(tracks)
    awh_path.write_text(awh_text, encoding="utf-8")

    b2 = backup_file(isc_path)
    if RICH and console:
        console.print(f"[dim]{tr(cfg,'backup')}: {b2}[/dim]")
    else:
        print(tr(cfg, "backup"), b2)

    try:
        rel_ref = awh_path.relative_to(include_dir)
    except Exception:
        rel_ref = Path("NAT_TRACKS.awh")

    if RICH and console:
        console.print(f"[cyan]{tr(cfg,'patching_isc')}[/cyan]")
    else:
        print(tr(cfg, "patching_isc"))

    changed = patch_isc_add_high_airway_ref(isc_path, str(rel_ref).replace("/", "\\"))
    if RICH and console:
        console.print(f"[green]{tr(cfg,'done')}[/green]  (isc modified: {changed})")
        console.print(f"[green]{tr(cfg,'awh_path')}[/green]: {awh_path}")
    else:
        print(tr(cfg, "done"), "(isc modified:", changed, ")")
        print(tr(cfg, "awh_path"), awh_path)


# -------------------- Config path menu option --------------------

def show_or_change_config_path(current_cfg: dict, console, current_config_path: Path) -> Path:
    """
    Returns the (possibly updated) config path.
    Persists chosen path in locator file.
    """
    title = tr(current_cfg, "config_path_title")
    cur = str(current_config_path)

    if RICH and console:
        table = Table(title=title, box=box.SIMPLE)
        table.add_column(tr(current_cfg, "current_config_path"))
        table.add_row(cur)
        console.print(table)
    else:
        print(f"{title}: {cur}")

    newp = prompt_input(tr(current_cfg, "ask_new_config_path")).strip()
    if not newp:
        return current_config_path

    new_path = Path(newp).expanduser()

    # basic validation: must be a file path ending with .json (recommended)
    if new_path.suffix.lower() not in (".json", ""):
        msg = tr(current_cfg, "config_path_invalid")
        if RICH and console:
            console.print(f"[red]{msg}[/red] -> {new_path}")
        else:
            print(msg, "->", new_path)
        return current_config_path

    # ensure parent exists
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        msg = tr(current_cfg, "config_path_invalid")
        if RICH and console:
            console.print(f"[red]{msg}[/red] -> {new_path}")
        else:
            print(msg, "->", new_path)
        return current_config_path

    # move old config if desired and exists and different
    if current_config_path.exists() and current_config_path.resolve() != new_path.resolve():
        if prompt_yesno(tr(current_cfg, "move_existing_config"), default=True):
            try:
                new_path.write_text(current_config_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                # keep old as backup (don’t delete silently)
                backup_file(current_config_path)
            except Exception:
                # if move fails, we still set locator; user can reconfigure
                pass

    # persist new config path in locator
    save_locator(new_path)

    msg = tr(current_cfg, "config_path_updated")
    if RICH and console:
        console.print(f"[green]{msg}[/green] -> {new_path}")
    else:
        print(msg, "->", new_path)

    return new_path


# -------------------- Main --------------------

def main() -> int:
    console = get_console()

    # Determine config path (CLI > locator > default)
    config_path = get_effective_config_path(sys.argv[1:])
    ensure_app_dir()

    cfg = load_config(config_path)

    # If no config content yet, run wizard
    if not cfg.get("language") or not cfg.get("isc_path") or not cfg.get("occ_include_folder"):
        cfg = run_first_time_wizard(config_path, console=console)

    # App header
    if RICH and console:
        console.print(Panel(Text(tr(cfg, "app_title"), justify="center"), expand=False))
    else:
        print(tr(cfg, "app_title"))
        print("-" * 40)

    tracks: List[Track] = []

    while True:
        if RICH and console:
            console.print("")
            console.print(f"[bold]{tr(cfg,'menu')}[/bold]")
            console.print(tr(cfg, "menu_1"))
            console.print(tr(cfg, "menu_2"))
            console.print(tr(cfg, "menu_3"))
            console.print(tr(cfg, "menu_4"))
            console.print(tr(cfg, "menu_5"))
        else:
            print("\n" + tr(cfg, "menu"))
            print(tr(cfg, "menu_1"))
            print(tr(cfg, "menu_2"))
            print(tr(cfg, "menu_3"))
            print(tr(cfg, "menu_4"))
            print(tr(cfg, "menu_5"))

        choice = prompt_input(tr(cfg, "prompt_choice")).strip()

        if choice == "1":
            tracks = fetch_and_parse(config_path, cfg, console)
            show_tracks(cfg, console, tracks)

        elif choice == "2":
            if not tracks:
                tracks = fetch_and_parse(config_path, cfg, console)
            if not tracks:
                continue
            update_sector(config_path, cfg, console, tracks)

        elif choice == "3":
            cfg = run_first_time_wizard(config_path, console=console)

        elif choice == "4":
            return 0

        elif choice == "5":
            # change config path (persistent)
            old_path = config_path
            config_path = show_or_change_config_path(cfg, console, config_path)

            # reload cfg from new path (if exists), otherwise keep current and save it there
            new_cfg = load_config(config_path)
            if new_cfg and new_cfg.get("language"):
                cfg = new_cfg
                if RICH and console:
                    console.print(f"[dim]{tr(cfg,'config_loaded')}: {config_path}[/dim]")
                else:
                    print(tr(cfg, "config_loaded"), ":", config_path)
            else:
                # ensure current cfg is saved into the new location
                save_config(config_path, cfg)

            # (optional) keep locator updated even if user uses CLI next time
            # already done in show_or_change_config_path

            # If user changed from CLI path and wants to go back, this will persist as locator.
            _ = old_path  # silence linters

        else:
            # ignore invalid input
            continue


if __name__ == "__main__":
    raise SystemExit(main())
