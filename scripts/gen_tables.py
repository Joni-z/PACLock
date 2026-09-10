# CBraMod-style result tables generated from runs/: datasets as column groups (3 metrics each), methods as rows grouped
# non-FM / FM pretrained / FM scratch / CroFreMo, mean ± std (3 seeds) or single value, best per column in bold.
import json, glob, statistics as st, os
MULTI = ["balanced_acc", "cohen_kappa", "weighted_f1"]; BIN = ["balanced_acc", "pr_auc", "auroc"]
DS = {"tusz":("TUSZ",BIN,"seizure detection"),"chbmit":("CHB-MIT",BIN,"seizure detection"),"siena":("Siena",BIN,"seizure detection"),
      "tuev":("TUEV, 6-class",MULTI,"event classification"),"iiic":("IIIC, 6-class",MULTI,"event classification"),"tuar":("TUAR, 3-class",MULTI,"artifact"),
      "tuab":("TUAB",BIN,"abnormal EEG"),"tuep":("TUEP",BIN,"epilepsy diagnosis"),"adfd":("ADFD, 3-class",MULTI,"cohort diagnosis"),
      "caueeg":("CAUEEG, 3-class",MULTI,"cohort diagnosis"),"sleepedf":("Sleep-EDF, 5-class",MULTI,"sleep staging"),"isruc":("ISRUC, 5-class",MULTI,"sleep staging")}
MNAME = {"balanced_acc":"Bal.\\ Acc.","cohen_kappa":"Cohen's $\\kappa$","weighted_f1":"Weighted F1","pr_auc":"AUC-PR","auroc":"AUROC"}
GROUPS = [("", [("SPaRCNet","sparcnet"),("ContraWR","contrawr"),("CNN-Transformer","cnn_transformer"),("FFCL","ffcl"),("ST-Transformer","st_transformer"),("EEGNet","eegnet"),("EEGConformer","eegconformer")]),
          ("Foundation models, released pretrained weights", [("BIOT","biot_prest16"),("LaBraM-Base","labram_pretrained"),("CBraMod","cbramod_pretrained"),("EEGPT","eegpt_pretrained"),("TFM-Tokenizer","tfm_pretrained"),("CSBrain","csbrain"),("REVE-Base","reve_pretrained")]),
          ("Foundation models, trained from scratch", [("BIOT","biot_scratch"),("LaBraM-Base","labram_scratch"),("CBraMod","cbramod_scratch"),("EEGPT","eegpt_scratch"),("TFM-Tokenizer","tfm_scratch")]),
          ("", [("CroFreMo (2.7M)","cf2_v1d192")])]
def load(ds, suf):
    out = {}
    for f in sorted(glob.glob(f"runs/{ds}-{suf}/seed*/result.json")):
        try: t = json.load(open(f))["test"]
        except Exception: continue
        for k, v in t.items(): out.setdefault(k, []).append(v)
    return out
def fmt(vals, best):
    if not vals: return "--"
    m = st.mean(vals); s = f"{m:.4f}"
    if len(vals) >= 3: s += f" $\\pm$ {st.stdev(vals):.4f}"
    elif len(vals) < 3: s += f"$^{{({len(vals)})}}$"
    return f"\\textbf{{{s}}}" if best else s
def table(ds_list, label, caption):
    cols = []
    for ds in ds_list:
        cols += [(ds, m) for m in DS[ds][1]]
    data = {}
    for gname, rows in GROUPS:
        for name, suf in rows:
            for ds in ds_list:
                data[(name, suf, ds)] = load(ds, suf)
    # best per column among rows that have values
    best = {}
    for ds, m in cols:
        cands = [(st.mean(data[(n, s, ds)][m]), (n, s)) for _, rows in GROUPS for n, s in rows if data[(n, s, ds)].get(m)]
        if len(cands) >= 2: best[(ds, m)] = max(cands)[1]
    L = ["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule"]
    L.append("\\multirow{2}{*}{Method} & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS[ds][0]}}}" for ds in ds_list) + " \\\\")
    L.append(" ".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(ds_list))))
    L.append(" & " + " & ".join(MNAME[m] for _, m in cols) + " \\\\")
    L.append("\\midrule")
    first = True
    for gname, rows in GROUPS:
        lines = []
        for name, suf in rows:
            if not any(data[(name, suf, ds)] for ds in ds_list): continue
            lines.append(name + " & " + " & ".join(fmt(data[(name, suf, ds)].get(m, []), best.get((ds, m)) == (name, suf)) for ds, m in cols) + " \\\\")
        if not lines: continue
        if not first: L.append("\\midrule")
        first = False
        if gname: L.append(f"\\multicolumn{{{1+len(cols)}}}{{l}}{{\\textit{{{gname}}}}} \\\\")
        L += lines
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L)
os.makedirs("results/tables", exist_ok=True)
main = [(["tusz","chbmit"], "tab:seizure", "Seizure detection. Mean $\\pm$ std over three seeds; a superscript gives the seed count when fewer. Best per column in bold."),
        (["tuev","iiic"], "tab:events", "Epileptiform event and pattern classification. Format as in Table~\\ref{tab:seizure}.")]
app = [(["siena","tuep"], "tab:app-siena-tuep", "Siena seizure detection and TUEP epilepsy diagnosis."),
       (["tuab","tuar"], "tab:app-tuab-tuar", "TUAB abnormal-EEG detection and TUAR artifact classification."),
       (["adfd","caueeg"], "tab:app-adfd-caueeg", "ADFD and CAUEEG cohort-level diagnosis."),
       (["sleepedf","isruc"], "tab:app-sleep", "Sleep staging on Sleep-EDF and ISRUC.")]
open("results/tables/main_tables.tex","w").write("\n\n".join(table(*t) for t in main))
open("results/tables/appendix_tables.tex","w").write("\n\n".join(table(*t) for t in app))
# ablation: tokenizer x width, 3 metrics per dataset, four datasets in two tables
ABL = [("waveform-only, $d{=}192$","cf2_v1d192raw"),("duplex, $d{=}192$ (final)","cf2_v1d192"),("waveform-only, $d{=}256$","cf2_v1d256raw"),("duplex, $d{=}256$","cf2_v1d256")]
def abl_table(ds_list, label, caption):
    cols = [(ds, m) for ds in ds_list for m in DS[ds][1]]
    data = {(n, s, ds): load(ds, s) for n, s in ABL for ds in ds_list}
    best = {}
    for ds, m in cols:
        cands = [(st.mean(data[(n, s, ds)][m]), (n, s)) for n, s in ABL if data[(n, s, ds)].get(m)]
        if len(cands) >= 2: best[(ds, m)] = max(cands)[1]
    L = ["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
         "\\multirow{2}{*}{Tokenizer, width} & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS[ds][0]}}}" for ds in ds_list) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(ds_list))),
         " & " + " & ".join(MNAME[m] for _, m in cols) + " \\\\", "\\midrule"]
    for n, s in ABL:
        L.append(n + " & " + " & ".join(fmt(data[(n, s, ds)].get(m, []), best.get((ds, m)) == (n, s)) for ds, m in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L)
open("results/tables/ablation_tables.tex","w").write(abl_table(["tusz","chbmit"], "tab:abl-seizure", "Duplex vs.\\ waveform-only tokenizer in the same encoder, seizure detection. Single seeds.") + "\n\n" +
                                                    abl_table(["tuev","iiic"], "tab:abl-events", "Duplex vs.\\ waveform-only tokenizer in the same encoder, event classification. Single seeds."))
print("tables written:", [f for f in os.listdir("results/tables")])
print(open("results/tables/main_tables.tex").read()[:2500])

PRIM = {"tusz":"pr_auc","chbmit":"pr_auc","siena":"pr_auc","tuev":"cohen_kappa","iiic":"cohen_kappa","tuar":"cohen_kappa","tuab":"balanced_acc","tuep":"auroc","adfd":"balanced_acc","caueeg":"balanced_acc","sleepedf":"cohen_kappa","isruc":"cohen_kappa"}
SHORT = {"tusz":"TUSZ","chbmit":"CHB-MIT","siena":"Siena","tuev":"TUEV","iiic":"IIIC","tuar":"TUAR","tuab":"TUAB","tuep":"TUEP","adfd":"ADFD","caueeg":"CAUEEG","sleepedf":"Sleep-EDF","isruc":"ISRUC"}
def v(ds, suf, metric=None):
    d = load(ds, suf); m = metric or PRIM[ds]; return d.get(m, [])
def simple_table(rows, ds_list, label, caption, first_col="Variant", bold_best=True):
    cols = ds_list
    vals = {(r[0], ds): v(ds, r[1]) for r in rows for ds in cols}
    best = {ds: max((st.mean(vals[(r[0], ds)]), r[0]) for r in rows if vals[(r[0], ds)])[1] for ds in cols if sum(1 for r in rows if vals[(r[0], ds)]) >= 2}
    L = ["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{l" + "c"*len(cols) + "}", "\\toprule",
         first_col + " & " + " & ".join(f"{SHORT[ds]} ({MNAME[PRIM[ds]]})" for ds in cols) + " \\\\", "\\midrule"]
    for name, suf in rows:
        L.append(name + " & " + " & ".join(fmt(vals[(name, ds)], bold_best and best.get(ds) == name) for ds in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L)
design = [("no fold, $d{=}192$","cf2_v0d192"),("no fold, $d{=}192$, strength","cf2_v0d192cs"),("fold, $d{=}192$ (final)","cf2_v1d192"),("fold, $d{=}256$","cf2_v1d256"),("fold, strength, $d{=}192$","cf2_v3")]
open("results/tables/design_table.tex","w").write(simple_table(design, ["tusz","chbmit","tuev","iiic","tuar","adfd","siena","sleepedf"], "tab:design",
    "Design variants of the final model, primary metric per corpus, single seeds. ``fold'' = band rows folded into spatial attention; ``strength'' = explicit $|Z|$ feature."))
# CBraMod rows (additive) and hosts (replacement): two-row-per-host layout with absolute values
def pair_table(blocks, ds_list, label, caption):
    L = ["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{ll" + "c"*len(ds_list) + "}", "\\toprule",
         "Encoder & Tokenizer & " + " & ".join(f"{SHORT[ds]} ({MNAME[PRIM[ds]]})" for ds in ds_list) + " \\\\", "\\midrule"]
    for bi, (enc, rows) in enumerate(blocks):
        if bi: L.append("\\midrule")
        vals = {(r[0], ds): v(ds, r[1]) for r in rows for ds in ds_list}
        best = {ds: max((st.mean(vals[(r[0], ds)]), r[0]) for r in rows if vals[(r[0], ds)])[1] for ds in ds_list if sum(1 for r in rows if vals[(r[0], ds)]) >= 2}
        for ri, (name, suf) in enumerate(rows):
            L.append((enc if ri == 0 else "") + " & " + name + " & " + " & ".join(fmt(vals[(name, ds)], best.get(ds) == name) for ds in ds_list) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L)
cb = [("CBraMod (from scratch)", [("native","cbramod_scratch"),("native + 8 waveform rows","cbramod_add_raw"),("native + 8 interaction rows","cbramod_add_cpl")])]
open("results/tables/cbramod_rows_table.tex","w").write(pair_table(cb, ["tuev","tuep","iiic","chbmit","tusz"], "tab:transplant",
    "CroFreMo's rows added to CBraMod's encoder, trained from scratch: the same eight extra rows per electrode carrying waveform tokens (control) or interaction tokens. Native: three seeds; transplants: single seeds."))
hosts = [("CBraMod (4M)", [("native","cbramod_scratch"),("CroFreMo tokenizer","cbramod_crofremo_bands")]),
         ("LaBraM (5.9M)", [("native","labram_scratch"),("CroFreMo tokenizer","labram_crofremo")]),
         ("REVE (69M)", [("native","reve_scratch"),("CroFreMo tokenizer","reve_crofremo")])]
open("results/tables/hosts_table.tex","w").write(pair_table(hosts, ["tuev","iiic","chbmit","tusz"], "tab:hosts",
    "CroFreMo's tokenizer in place of the host's own, hosts trained from scratch (every electrode--row pair becomes one host channel; rows pooled after the encoder; head unchanged). Native rows: three seeds except REVE (single); transplants: single seeds."))
print("extra tables written")
