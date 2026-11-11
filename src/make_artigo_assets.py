"""Gera as tabelas e figuras do artigo a partir dos CSVs de resultados em docs/.

Uso:
  python src/make_artigo_assets.py                   # lê docs/, grava docs/artigo/
  python src/make_artigo_assets.py --docs docs --out docs/artigo

Arquivos esperados em docs/ (os que faltarem são pulados com um aviso):
  resultados_modelagem_none_h{2,6,12,24}.csv   prognóstico (notebook 02)
  resultados_supervisionado_e0_e4.csv          braço supervisionado (notebook 03 v2)
  resultados_sem_rotulos_{global,eventos,ablacao,deriva}.csv   (notebook 04 v2)
  resultados_p2_degradacao.csv                 (notebook 05)
  custos_p3.csv                                (benchmark_custos.py)

Saídas em --out: tabelas/*.csv, tabelas.md (para copiar no texto) e figuras/*.png.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGSIZE = (6.5, 3.9)
DPI = 200
tables: dict[str, pd.DataFrame] = {}
captions: dict[str, str] = {}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def read(docs: Path, name: str):
    p = docs / name
    if not p.exists():
        print(f"  [pulado] {name} não encontrado")
        return None
    return pd.read_csv(p)


def add_table(key: str, df: pd.DataFrame, caption: str) -> None:
    tables[key] = df
    captions[key] = caption


def savefig(fig, out: Path, name: str) -> None:
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "figuras" / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print(f"  figura: {name}.png")


# --------------------------------------------------------------------------- prognóstico
def prognostico(docs: Path, out: Path) -> None:
    frames = []
    for h in (2, 6, 12, 24):
        d = read(docs, f"resultados_modelagem_none_h{h}.csv")
        if d is None:
            continue
        d = d.copy()
        d["horizonte_h"] = h
        if "lift" not in d.columns:
            d["lift"] = d["pr_auc"] / d["prevalencia"]
        for c in ("p_pr_auc", "deteccao_por_acaso"):
            if c not in d.columns:
                d[c] = np.nan
        frames.append(d)
    if not frames:
        return
    d = pd.concat(frames, ignore_index=True)
    g = d.groupby("horizonte_h")
    t = pd.DataFrame({
        "combinações": g.size(),
        "com_lift_maior_que_1": g["lift"].apply(lambda s: int((s > 1).sum())),
        "com_p_menor_que_0,10": g["p_pr_auc"].apply(lambda s: int((s < 0.10).sum()) if s.notna().any() else np.nan),
        "menor_p": g["p_pr_auc"].min(),
        "maior_lift": g["lift"].max(),
        "detecções_observadas": g["detected"].sum(),
        "detecções_esperadas_por_acaso": g["deteccao_por_acaso"].sum(min_count=1),
    }).reset_index()
    add_table("t_prognostico", t.round(3),
              "Prognóstico (E0–E2, 3 modelos, 3 falhas de teste): resultados por horizonte contra o acaso. "
              "Células vazias: arquivo sem baseline de acaso (reexecutar o notebook 02).")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(0)
    colors = {"E0": "tab:blue", "E1": "tab:orange", "E2": "tab:green"}
    for v, c in colors.items():
        s = d[d["variante"] == v]
        x = s["horizonte_h"].astype(str).map({"2": 0, "6": 1, "12": 2, "24": 3}) + rng.uniform(-0.18, 0.18, len(s))
        sig = s["p_pr_auc"] < 0.10
        ax.scatter(x[~sig], s["lift"][~sig], s=22, alpha=0.7, color=c, label=v)
        ax.scatter(x[sig], s["lift"][sig], s=70, color=c, edgecolors="k", linewidths=1.5, zorder=3)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0,5", "1", "2", "4", "8"])
    ax.set_xticks(range(4)); ax.set_xticklabels(["2 h", "6 h", "12 h", "24 h"])
    ax.set_xlabel("Horizonte de antecipação"); ax.set_ylabel("Lift (PR-AUC / prevalência)")
    ax.set_title("Prognóstico: lift por horizonte (tracejada = acaso; borda preta = p < 0,10)", fontsize=9)
    ax.legend(title="Variante", fontsize=8)
    savefig(fig, out, "f1_prognostico")


# --------------------------------------------------------------------------- supervisionado
def supervisionado(docs: Path, out: Path) -> None:
    d = read(docs, "resultados_supervisionado_e0_e4.csv")
    if d is None:
        return
    order = [v for v in ["E0", "E1", "E2", "E3", "E4*"] if v in set(d["variante"])]
    pr = d.groupby(["variante", "falha_testada"])["pr_auc"].mean().unstack().reindex(order)
    pr.columns = [f"PR-AUC falha {c}" for c in pr.columns]
    ex = d.groupby("variante").agg(
        detecções_sustentadas=("sustentada", "sum"),
        combinações=("sustentada", "size"),
        falsos_alarmes_por_dia=("false_alarms_per_day", "mean"),
    ).reindex(order)
    t = pd.concat([pr, ex], axis=1).reset_index().round(3)
    add_table("t_supervisionado", t,
              "Braço supervisionado (detecção da falha em curso): PR-AUC média dos 3 modelos por falha, "
              "detecções sustentadas (alarme em pelo menos 10% dos minutos da falha) e falsos alarmes por dia. "
              "E4* é exploratória (só analógicos).")
    per_model = d.pivot_table(index=["variante", "modelo"], columns="falha_testada", values="pr_auc").round(3)
    per_model.columns = [f"PR-AUC falha {c}" for c in per_model.columns]
    add_table("t_supervisionado_modelos", per_model.reset_index(),
              "Braço supervisionado: PR-AUC por variante, modelo e falha testada.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    falhas = sorted(d["falha_testada"].unique())
    w = 0.8 / len(falhas)
    for i, f in enumerate(falhas):
        vals = [d[(d.variante == v) & (d.falha_testada == f)]["pr_auc"].mean() for v in order]
        ax.bar(np.arange(len(order)) + i * w - 0.4 + w / 2, vals, w, label=f"Falha {f}")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("PR-AUC (média dos 3 modelos)"); ax.set_ylim(0, 1.25)
    ax.set_title("Supervisionado: detecção por variante do pipeline (E4* exploratória)", fontsize=9)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    savefig(fig, out, "f2_supervisionado")


# --------------------------------------------------------------------------- sem rótulos
def sem_rotulos(docs: Path, out: Path) -> None:
    g = read(docs, "resultados_sem_rotulos_global.csv")
    if g is not None:
        ref = g[g["quantil"] == 99.9].drop_duplicates(["variante", "modelo"])
        cols = ["variante", "modelo", "pr_auc", "lift", "eventos_detectados"]
        for c in ("eventos_sustentados", "recall_medio", "atraso_mediano_h", "falsos_alarmes_por_dia"):
            if c in ref.columns:
                cols.append(c)
        frac = "frac_minutos_normais_em_alarme" if "frac_minutos_normais_em_alarme" in ref.columns else "pct_minutos_normais_em_alarme"
        if frac in ref.columns:
            cols.append(frac)
        t = ref[cols].sort_values("pr_auc", ascending=False).reset_index(drop=True).round(3)
        add_table("t_sem_rotulos", t,
                  "Detecção sem rótulos no quantil de referência (percentil 99,9 da validação), ordenada pela PR-AUC. "
                  "Variantes com sufixo -A (só analógicos) são exploratórias.")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        lab = (ref["variante"] + " " + ref["modelo"].replace({"IsolationForest": "IF", "Mahalanobis": "Mah"}))
        r = ref.assign(lab=lab).sort_values("pr_auc")
        ax.barh(r["lab"], r["pr_auc"], color=np.where(r["modelo"] == "IsolationForest", "tab:blue", "tab:orange"))
        ax.set_xlabel("PR-AUC (independente do limiar)")
        ax.set_title("Sem rótulos: ranking por PR-AUC", fontsize=10)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="tab:blue", label="Isolation Forest"),
                           Patch(color="tab:orange", label="Mahalanobis")], loc="lower right", fontsize=8)
        savefig(fig, out, "f3_sem_rotulos_ranking")

    a = read(docs, "resultados_sem_rotulos_ablacao.csv")
    if a is not None:
        add_table("t_ablacao", a.round(3).sort_values(["modelo", "cenario"]).reset_index(drop=True),
                  "Ablação por grupos de sensores (E1, quantil 99,9; exploratória e pós-hoc).")

    d = read(docs, "resultados_sem_rotulos_deriva.csv")
    if d is not None:
        add_table("t_deriva", d.round(3), "Falsos alarmes (episódios por dia) por mês, quantil 99,9.")
        long = d.melt(id_vars=["variante", "modelo"], var_name="mes", value_name="fa")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for (v, m), s in long.groupby(["variante", "modelo"]):
            if (v, m) in [("E1", "IsolationForest"), ("E1-A", "IsolationForest"),
                          ("E1", "Mahalanobis"), ("E1-A", "Mahalanobis")]:
                ax.plot(s["mes"], s["fa"], marker="o", label=f"{v} {m}")
        ax.set_ylabel("Falsos alarmes (episódios/dia)"); ax.set_xlabel("Mês (2020)")
        ax.set_title("Deriva: falsos alarmes por mês (E1 com e sem sensores digitais)")
        ax.legend(fontsize=8)
        savefig(fig, out, "f4_deriva")


# --------------------------------------------------------------------------- P2
def p2(docs: Path, out: Path) -> None:
    d = read(docs, "resultados_p2_degradacao.csv")
    if d is None:
        return
    m = ["eventos_sustentados", "recall_medio", "pr_auc", "minutos_de_referencia_com_score"]
    s = d.groupby(["config", "cenario", "nivel"])[m].mean().round(3).reset_index()
    add_table("t_p2", s, "P2: média entre as sementes por configuração e cenário (nível 0 = referência).")
    base = d[d["cenario"] == "base"].set_index("config")
    ymin = float(np.nanmin(d["pr_auc"])) - 0.03
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, kind, xl in zip(axes, ["faltantes", "ruido"], ["Fração de minutos removidos", "Ruído (× desvio-padrão)"]):
        for name in d["config"].unique():
            x = d[(d.config == name) & (d.cenario == kind)].groupby("nivel")["pr_auc"].agg(["mean", "std"])
            xs = [0.0] + list(x.index)
            ys = [base.loc[name, "pr_auc"]] + list(x["mean"])
            es = [0.0] + list(x["std"].fillna(0))
            line = ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=name)
            for xv, yv in zip(xs, ys):
                if pd.isna(yv):   # pipeline sem dados para ajustar o modelo: detecção zero
                    ax.plot(xv, ymin, marker="X", color=line[0].get_color(), markersize=9, clip_on=False)
                    ax.annotate("sem dados", (xv, ymin), textcoords="offset points", xytext=(-4, 8),
                                fontsize=7, ha="right", color=line[0].get_color())
        ax.set_xlabel(xl)
    axes[0].set_ylabel("PR-AUC (população de referência)")
    axes[0].legend(fontsize=7)
    fig.suptitle("P2: PR-AUC sob degradação (barras = desvio entre sementes)", fontsize=10)
    savefig(fig, out, "f5_p2_degradacao")


# --------------------------------------------------------------------------- P3
def custos(docs: Path, out: Path) -> None:
    d = read(docs, "custos_p3.csv")
    if d is None:
        return
    t = d[["experimento", "metodo", "linhas", "tempo_mediano_s", "linhas_por_s"]].copy()
    best = t.groupby(["experimento", "linhas"])["tempo_mediano_s"].transform("min")
    t["vezes_o_mais_rápido"] = (t["tempo_mediano_s"] / best).round(1)
    add_table("t_custos", t, "P3: tempo mediano por etapa e método (depende do hardware; ver custos_p3.json).")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    short = t["metodo"].str.replace(" (leitura + agregação)", "", regex=False).str.replace(" linhas (completa)", " (completa)", regex=False)
    lab = t["experimento"].str.split("_").str[0] + ". " + short
    palette = {"B_leitura": "tab:blue", "C_silver_1min": "tab:orange", "D_janelas": "tab:green"}
    ax.barh(lab[::-1], t["tempo_mediano_s"][::-1], color=t["experimento"].map(palette)[::-1])
    ax.set_xscale("log"); ax.set_xlabel("Tempo mediano (s, escala logarítmica)")
    ax.set_title("P3: custo de tempo (B = leitura, C = Silver de 1 min, D = janelas)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    savefig(fig, out, "f6_custos")


# --------------------------------------------------------------------------- pipeline e tempos
def pipeline_diagram(docs: Path, out: Path) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=color, ec="0.3", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"))

    box(1, 17, 15, 8, "CSV bruto\n218 MB\n(10 s)", "#eeeeee")
    box(21, 17, 15, 8, "Bronze\nParquet 31 MB\n(sem alterar)", "#e8c9a0")
    box(41, 26, 17, 9, "Silver E0\n10 s, sem regras", "#d9d9d9")
    box(41, 5, 17, 9, "Silver E1\n1 min, qualidade\ne cobertura", "#c0c0c8")
    box(63, 17, 15, 8, "Gold\nrótulos +\njanelas 1 h/6 h", "#f2d98a")
    box(63, 33, 15, 5, "Relatório de falhas", "#f4f4f4")
    box(83, 26, 16, 9, "Supervisionado\nE0–E4", "#b8d8b8")
    box(83, 5, 16, 9, "Sem rótulos\nIF e Mahalanobis", "#b8d0e8")
    arrow(16.5, 21, 20.5, 21); arrow(36.5, 22, 40.5, 29); arrow(36.5, 20, 40.5, 11)
    arrow(58.5, 29, 62.5, 23); arrow(58.5, 11, 62.5, 19); arrow(70.5, 32.5, 70.5, 25.6)
    arrow(78.5, 23, 82.5, 29); arrow(58.5, 8.5, 82.5, 8.5)
    savefig(fig, out, "f0_pipeline")


def tempos(docs: Path, out: Path) -> None:
    ex = read(docs, "tempos_execucao.csv")
    if ex is not None:
        ex = ex[ex["passo"] != "TOTAL"].drop_duplicates("passo", keep="last")
        t = ex[["passo", "descricao", "duracao_s", "status"]].copy()
        total = pd.DataFrame([{"passo": "TOTAL", "descricao": "soma das etapas", "duracao_s": t["duracao_s"].sum(), "status": ""}])
        add_table("t_tempos_execucao", pd.concat([t, total], ignore_index=True),
                  "Tempo de parede de cada etapa em uma execução limpa (depende do hardware; ver ambiente_execucao.json).")
    md = read(docs, "tempos_modelos.csv")
    if md is not None and len(md):
        g = (md.groupby(["estimador", "metodo"])
               .agg(chamadas=("segundos", "size"), total_s=("segundos", "sum"), mediana_s=("segundos", "median"))
               .reset_index().sort_values("total_s", ascending=False))
        add_table("t_tempos_modelos", g, "Tempo acumulado de treino (fit) e predição por tipo de modelo, somando todos os notebooks.")
        fit = g[g["metodo"] == "fit"].sort_values("total_s")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.barh(fit["estimador"], fit["total_s"], color="tab:purple")
        ax.set_xlabel("Tempo total de treino (s)"); ax.set_title("Custo de treino por modelo (todas as dobras e configurações)", fontsize=9)
        savefig(fig, out, "f7_tempos_treino")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", default="docs/artigo")
    args = ap.parse_args()
    docs, out = Path(args.docs), Path(args.out)
    (out / "tabelas").mkdir(parents=True, exist_ok=True)

    for fn in (pipeline_diagram, prognostico, supervisionado, sem_rotulos, p2, custos, tempos):
        print(fn.__name__)
        fn(docs, out)

    md = ["# Tabelas do artigo (geradas por make_artigo_assets.py)\n"]
    for key, df in tables.items():
        df.to_csv(out / "tabelas" / f"{key}.csv", index=False)
        md.append(f"## {key}\n\n{captions[key]}\n\n{md_table(df)}\n")
    (out / "tabelas.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{len(tables)} tabelas e as figuras salvas em {out}")


if __name__ == "__main__":
    main()
