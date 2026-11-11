"""Execução limpa ponta a ponta com medição de tempo por etapa.

Uso (a partir da raiz do projeto):
  python src/run_pipeline.py --dry-run                 # só lista o plano
  python src/run_pipeline.py                           # tudo, horizonte 24
  python src/run_pipeline.py --horizons 24,12          # + sensibilidade do prognóstico
  python src/run_pipeline.py --only silver_e1 gold_e2_h24
  python src/run_pipeline.py --skip custos nb05
  python src/run_pipeline.py --fast                    # teste de fumaça (poucas árvores/rotações)
  python src/run_pipeline.py --keep-going --append     # não parar no 1º erro; acrescentar ao CSV

Pré-requisito: CSV bruto em data/raw/ e data/external/failure_reports.csv preenchido.
Saídas: docs/tempos_execucao.csv, docs/tempos_modelos.csv, docs/ambiente_execucao.json,
        notebooks executados em docs/execucao/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = ROOT / "docs"
NB = ROOT / "notebooks"
PY = sys.executable


def cmd(*args) -> list[str]:
    return [PY, *map(str, args)]


def build_steps(horizons: list[int], windows: str | None = None, coverage: float | None = None):
    """Lista ordenada de (nome, descrição, tipo, payload)."""
    main_h = horizons[0]
    steps = [
        ("bronze", "CSV -> Bronze (Parquet)", "cmd", cmd(SRC / "bronze.py")),
        ("silver_e0", "Silver E0 (10 s, sem regras)", "cmd", cmd(SRC / "silver.py", "--variant", "e0")),
        ("silver_e1", "Silver E1 (1 min, qualidade)", "cmd", cmd(SRC / "silver.py", "--variant", "e1")),
    ]
    for h in horizons:
        steps.append((f"gold_e0_h{h}", f"Gold E0, horizonte {h} h", "cmd",
                      cmd(SRC / "gold.py", "--variant", "e0", "--horizon", h)))
        steps.append((f"gold_e1_h{h}", f"Gold E1, horizonte {h} h", "cmd",
                      cmd(SRC / "gold.py", "--variant", "e1", "--horizon", h)))
        # Janelas e cobertura vêm de src/config.py (ROLLING_WINDOWS_MIN, MIN_WINDOW_COVERAGE), para
        # a pasta de saída ser e2_h{h}, que é a que o notebook 02 lê. Passe --windows/--coverage
        # apenas para um experimento à parte (a pasta ganha sufixo e o notebook 02 não a enxerga).
        extra = (["--windows", windows] if windows else []) + (["--min-coverage", coverage] if coverage else [])
        steps.append((f"gold_e2_h{h}", f"Gold E2 (janelas do config.py), horizonte {h} h", "cmd",
                      cmd(SRC / "gold.py", "--variant", "e2", "--horizon", h, *extra)))
    steps.append(("tests", "Testes automatizados", "cmd", cmd("-m", "pytest", "-q", ROOT / "tests")))
    steps.append(("nb01", "Notebook 01 (EDA)", "nb", ("01_eda_metropt3.ipynb", None)))
    for h in horizons:
        steps.append((f"nb02_h{h}", f"Notebook 02 (prognóstico), horizonte {h} h", "nb",
                      ("02_modelagem_metropt3.ipynb", h)))
    steps += [
        ("nb03", "Notebook 03 (supervisionado E0–E4)", "nb", ("03_diagnostico_metropt3_v2.ipynb", None)),
        ("nb04", "Notebook 04 (sem rótulos)", "nb", ("04_deteccao_sem_rotulos_v2.ipynb", None)),
        ("nb05", "Notebook 05 (P2 degradação)", "nb", ("05_p2_degradacao.ipynb", None)),
        ("custos", "Benchmark de custos (P3)", "cmd", cmd(SRC / "benchmark_custos.py")),
        ("assets", "Tabelas e figuras do artigo", "cmd", cmd(SRC / "make_artigo_assets.py")),
    ]
    _ = main_h
    return steps


def patch_source(src: str, horizon: int | None, fast: bool) -> str:
    """Ajusta constantes do notebook (apenas na cópia executada)."""
    if horizon is not None:
        src = re.sub(r"^HORIZON\s*=\s*\d+", f"HORIZON = {horizon}", src, flags=re.M)
    src = re.sub(r"^EMBARGO\s*=\s*HORIZON\b", "EMBARGO = 24", src, flags=re.M)
    src = re.sub(r"^EMBARGO\s*=\s*\d+", "EMBARGO = 24", src, flags=re.M)
    if fast:
        src = re.sub(r"^N_TREES\s*=\s*\d+", "N_TREES = 20", src, flags=re.M)
        src = re.sub(r"^N_SHIFTS\s*=\s*\d+", "N_SHIFTS = 30", src, flags=re.M)
        src = re.sub(r"^N_SEEDS\s*=\s*\d+", "N_SEEDS = 2", src, flags=re.M)
    return src


def run_notebook(name: str, horizon: int | None, fast: bool, tag: str, env: dict) -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(NB / name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = patch_source(cell.source, horizon, fast)
    os.environ.update(env)
    NotebookClient(nb, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(NB)}}).execute()
    out = DOCS / "execucao"
    out.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out / f"{tag}_{name}")


def environment() -> dict:
    info = {"data": datetime.now().isoformat(timespec="seconds"), "so": platform.platform(),
            "python": platform.python_version(), "cpus": os.cpu_count()}
    for mod in ("pandas", "numpy", "sklearn", "lightgbm", "pyspark"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True)
        info["java"] = (out.stderr or out.stdout).splitlines()[0]
    except Exception:
        info["java"] = None
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--horizons", default="24", help="ex.: 24 ou 24,12 (o 1º é o principal)")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    steps = build_steps(horizons)
    names = [s[0] for s in steps]
    for n in (args.only or []) + args.skip:
        if n not in names:
            ap.error(f"passo desconhecido: {n}. Opções: {', '.join(names)}")

    def selected(n: str) -> bool:
        return (args.only is None or n in args.only or any(n.startswith(o + "_") for o in args.only)) \
            and n not in args.skip

    plan = [s for s in steps if selected(s[0])]
    print(f"Plano ({len(plan)} passos){' [FAST]' if args.fast else ''}:")
    for i, (n, d, *_r) in enumerate(plan, 1):
        print(f"  {i:2d}. {n:<14} {d}")
    if args.dry_run:
        return 0

    DOCS.mkdir(exist_ok=True)
    times_path = DOCS / "tempos_execucao.csv"
    models_path = DOCS / "tempos_modelos.csv"
    if not args.append:
        models_path.unlink(missing_ok=True)
    (DOCS / "ambiente_execucao.json").write_text(
        json.dumps({**environment(), "fast": args.fast, "horizontes": horizons}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    new = args.append is False or not times_path.exists()
    fh = times_path.open("w" if not args.append else "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["passo", "descricao", "inicio", "duracao_s", "status", "observacao"])

    base_env = {"PI_FIT_TIMING": "1", "PI_TIMING_FILE": str(models_path), "PYTHONPATH": str(SRC)}
    total0 = time.perf_counter()
    failed = False
    for n, d, kind, payload in plan:
        print(f"\n=== {n}: {d}")
        started = datetime.now().isoformat(timespec="seconds")
        t0 = time.perf_counter()
        status, obs = "ok", ""
        try:
            if kind == "cmd":
                env = {**os.environ, "PYTHONPATH": str(SRC)}
                if n != "tests":  # os testes não entram na medição de tempos dos modelos
                    env.update({**base_env, "PI_NOTEBOOK": n})
                r = subprocess.run(payload, cwd=ROOT, env=env)
                if r.returncode != 0:
                    status, obs = "erro", f"código de saída {r.returncode}"
            else:
                name, h = payload
                run_notebook(name, h, args.fast, n, {**base_env, "PI_NOTEBOOK": n})
        except Exception as exc:  # noqa: BLE001
            status, obs = "erro", f"{type(exc).__name__}: {str(exc)[-300:]}".replace("\n", " ")
        dt = time.perf_counter() - t0
        w.writerow([n, d, started, f"{dt:.1f}", status, obs])
        fh.flush()
        print(f"--- {n}: {status} em {dt:.1f} s {obs}")
        if status != "ok":
            failed = True
            if not args.keep_going:
                break
    fh.close()
    print(f"\nTempo total desta execução: {time.perf_counter() - total0:.1f} s")
    print(f"\nTempos em {times_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
