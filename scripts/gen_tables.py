# CBraMod-style result tables generated from runs/: datasets as column groups (3 metrics each), methods as rows grouped
# non-FM / FM pretrained / FM scratch / CroFreMo, mean ± std (3 seeds) or single value, best per column in bold.
import json, glob, statistics as st, os, math, hashlib
from pathlib import Path

AUDIT = {}
class MetricValues(list):
    def __init__(self):
        super().__init__()
        self.flags = set()
        self.seeds = []

def records(ds, suf):
    rows = {}
    for f in sorted(glob.glob(f"runs/{ds}-{suf}/seed*/result.json")):
        raw = Path(f).read_bytes()
        r = json.loads(raw)
        seed = int(r["seed"])
        if r["dataset"] != ds or seed in rows:
            raise ValueError(f"Mismatched dataset or duplicate seed: {f}")
        if not all(math.isfinite(float(v)) for v in r["test"].values()):
            raise ValueError(f"Non-finite reported metric: {f}")
        rows[seed] = r
        AUDIT[f] = {"sha256": hashlib.sha256(raw).hexdigest(), "seed": seed,
                    "stopped_by": r.get("stopped_by"), "verdict": r.get("verdict"),
                    "epochs_run": r.get("epochs_run"), "test": r["test"]}
    return rows

def flags(r):
    out = set()
    if r.get("stopped_by") == "time_budget": out.add("budget")
    if (r.get("verdict") or {}).get("ok") is False: out.add("flagged")
    return out

def rankable(vals):
    return bool(vals) and not getattr(vals, "flags", set())
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
    for seed, r in records(ds, suf).items():
        for k, value in r["test"].items():
            vals = out.setdefault(k, MetricValues())
            vals.append(value)
            vals.seeds.append(seed)
            vals.flags.update(flags(r))
    return out
def fmt(vals, best):
    if not vals: return "--"
    # Never improve a cell by silently dropping a failed seed. A diagnostic
    # flag suppresses the entire cell; raw values remain in table_audit.json.
    if "flagged" in getattr(vals, "flags", set()): return r"--$^{!}$"
    m = st.mean(vals); s = f"{m:.4f}"
    if len(vals) >= 3: s += f" $\\pm$ {st.stdev(vals):.4f}"
    elif len(vals) < 3: s += f"$^{{({len(vals)})}}$"
    if "budget" in getattr(vals, "flags", set()): s += r"$^{\dagger}$"
    best = best and rankable(vals)
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
        cands = [(st.mean(data[(n, s, ds)][m]), (n, s)) for _, rows in GROUPS for n, s in rows if rankable(data[(n, s, ds)].get(m))]
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
main = [(["tusz","chbmit"], "tab:seizure", "Seizure detection. Mean $\\pm$ std over three seeds; a superscript gives the seed count when fewer. Best eligible mean per column in bold. $^{!}$: cell withheld because at least one seed has a recorded diagnostic flag; $^{\\dagger}$: includes a budget-stopped run, shown but not ranked. Raw flagged results are retained in the reporting audit."),
        (["tuev","iiic"], "tab:events", "Epileptiform event and pattern classification. Format as in Table~\\ref{tab:seizure}.")]
app = [(["siena","tuep"], "tab:app-siena-tuep", "Siena seizure detection and TUEP epilepsy diagnosis."),
       (["tuab","tuar"], "tab:app-tuab-tuar", "TUAB abnormal-EEG detection and TUAR artifact classification."),
       (["adfd","caueeg"], "tab:app-adfd-caueeg", "ADFD and CAUEEG cohort-level diagnosis."),
       (["sleepedf","isruc"], "tab:app-sleep", "Sleep staging on Sleep-EDF and ISRUC.")]
open("results/tables/main_tables.tex","w").write("\n\n".join(table(*t) for t in main))
open("results/tables/appendix_tables.tex","w").write("\n\n".join(table(*t) for t in app))
# ablation: tokenizer x width, 3 metrics per dataset, four datasets in two tables
ABL = [("waveform-only, $d{=}192$","cf2_v1d192raw"),("duplex, $d{=}192$ (reference)","cf2_v1d192"),("waveform-only, $d{=}256$","cf2_v1d256raw"),("duplex, $d{=}256$","cf2_v1d256")]
def abl_table(ds_list, label, caption):
    cols = [(ds, m) for ds in ds_list for m in DS[ds][1]]
    data = {(n, s, ds): load(ds, s) for n, s in ABL for ds in ds_list}
    best = {}
    for ds, m in cols:
        cands = [(st.mean(data[(n, s, ds)][m]), (n, s)) for n, s in ABL if rankable(data[(n, s, ds)].get(m))]
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
open("results/tables/ablation_tables.tex","w").write(abl_table(["tusz","chbmit"], "tab:abl-seizure", "Duplex vs.\\ waveform-only tokenizer in the same encoder, seizure detection. Seed counts follow the main tables.") + "\n\n" +
                                                    abl_table(["tuev","iiic"], "tab:abl-events", "Duplex vs.\\ waveform-only tokenizer in the same encoder, event classification. Seed counts follow the main tables."))
print("tables written:", [f for f in os.listdir("results/tables")])
print(open("results/tables/main_tables.tex").read()[:2500])

PRIM = {"tusz":"pr_auc","chbmit":"pr_auc","siena":"pr_auc","tuev":"cohen_kappa","iiic":"cohen_kappa","tuar":"cohen_kappa","tuab":"balanced_acc","tuep":"auroc","adfd":"balanced_acc","caueeg":"balanced_acc","sleepedf":"cohen_kappa","isruc":"cohen_kappa"}
SHORT = {"tusz":"TUSZ","chbmit":"CHB-MIT","siena":"Siena","tuev":"TUEV","iiic":"IIIC","tuar":"TUAR","tuab":"TUAB","tuep":"TUEP","adfd":"ADFD","caueeg":"CAUEEG","sleepedf":"Sleep-EDF","isruc":"ISRUC"}
def v(ds, suf, metric=None):
    d = load(ds, suf); m = metric or PRIM[ds]; return d.get(m, [])
def simple_table(rows, ds_list, label, caption, first_col="Variant", bold_best=True):
    cols = ds_list
    vals = {(r[0], ds): v(ds, r[1]) for r in rows for ds in cols}
    best = {ds: max((st.mean(vals[(r[0], ds)]), r[0]) for r in rows if rankable(vals[(r[0], ds)]))[1] for ds in cols if sum(1 for r in rows if rankable(vals[(r[0], ds)])) >= 2}
    L = ["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{l" + "c"*len(cols) + "}", "\\toprule",
         first_col + " & " + " & ".join(f"{SHORT[ds]} ({MNAME[PRIM[ds]]})" for ds in cols) + " \\\\", "\\midrule"]
    for name, suf in rows:
        L.append(name + " & " + " & ".join(fmt(vals[(name, ds)], bold_best and best.get(ds) == name) for ds in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L)
design = [("no fold, $d{=}192$","cf2_v0d192"),("no fold, $d{=}192$, strength","cf2_v0d192cs"),("fold, $d{=}192$ (reference)","cf2_v1d192"),("fold, $d{=}256$","cf2_v1d256"),("fold, strength, $d{=}192$","cf2_v3")]
open("results/tables/design_table.tex","w").write(simple_table(design, ["tusz","chbmit","tuev","iiic","tuar","adfd","siena","sleepedf"], "tab:design",
    "Folded-attention design variants, primary metric per corpus. Seed counts follow the main tables. ``fold'' = band rows folded into spatial attention; ``strength'' = explicit $|Z|$ feature."))
# CBraMod rows (additive) and hosts (replacement): two-row-per-host layout with absolute values
def pair_table(blocks, ds_list, label, caption):
    L = ["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{ll" + "c"*len(ds_list) + "}", "\\toprule",
         "Encoder & Tokenizer & " + " & ".join(f"{SHORT[ds]} ({MNAME[PRIM[ds]]})" for ds in ds_list) + " \\\\", "\\midrule"]
    for bi, (enc, rows) in enumerate(blocks):
        if bi: L.append("\\midrule")
        vals = {(r[0], ds): v(ds, r[1]) for r in rows for ds in ds_list}
        best = {ds: max((st.mean(vals[(r[0], ds)]), r[0]) for r in rows if rankable(vals[(r[0], ds)]))[1] for ds in ds_list if sum(1 for r in rows if rankable(vals[(r[0], ds)])) >= 2}
        for ri, (name, suf) in enumerate(rows):
            L.append((enc if ri == 0 else "") + " & " + name + " & " + " & ".join(fmt(vals[(name, ds)], best.get(ds) == name) for ds in ds_list) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L)
cb = [("CBraMod (from scratch)", [("native","cbramod_scratch"),("native + 8 waveform rows","cbramod_add_raw"),("native + 8 interaction rows","cbramod_add_cpl")])]
open("results/tables/cbramod_rows_table.tex","w").write(pair_table(cb, ["tuev","tuep","iiic","chbmit","tusz"], "tab:transplant",
    "CroFreMo's rows added to CBraMod's encoder, trained from scratch: the same eight extra rows per electrode carrying waveform tokens (control) or interaction tokens. Means and seed counts follow the main tables."))
hosts = [("CBraMod", [("native","cbramod_scratch"),("CroFreMo tokenizer","cbramod_crofremo_bands")]),
         ("LaBraM", [("native","labram_scratch"),("CroFreMo tokenizer","labram_crofremo")]),
         ("REVE", [("native","reve_scratch"),("CroFreMo tokenizer","reve_crofremo")])]
open("results/tables/hosts_table.tex","w").write(pair_table(hosts, ["tuev","iiic","chbmit","tusz"], "tab:hosts",
    "CroFreMo's tokenizer in place of the host's own, hosts trained from scratch (every electrode--row pair becomes one host channel; rows pooled after the encoder; head unchanged). Means and seed counts follow the main tables."))
print("extra tables written")

# Additional paper tables use the same run reader; paired contrasts use seed
# intersections, never differences between means with different seed coverage.
PAIRED = {}
def paired(ds, a, b):
    aa, bb = records(ds, a), records(ds, b)
    out = MetricValues()
    for seed in sorted(aa.keys() & bb.keys()):
        out.append(aa[seed]['test'][PRIM[ds]] - bb[seed]['test'][PRIM[ds]])
        out.seeds.append(seed)
        out.flags.update(flags(aa[seed]) | flags(bb[seed]))
    PAIRED[f'{ds}:{a}-{b}'] = {'seeds': out.seeds, 'deltas': list(out),
        'mean': st.mean(out) if out else None, 'flags': sorted(out.flags)}
    return out

def grid(headers, rows, label, caption):
    lines = [r'\begin{table}[t]', r'\centering', r'\scriptsize',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{l' + 'c' * (len(headers)-1) + '}',
             r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(row) + r' \\' for row in rows]
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}}', r'\end{table}'])

def matrix(variants, datasets, label, caption):
    return grid(['Corpus (primary metric)'] + [n for n, _ in variants],
        [[SHORT[ds] + ' (' + MNAME[PRIM[ds]] + ')'] +
         [fmt(v(ds, suf), False) for _, suf in variants] for ds in datasets],
         label, caption + ' Seed counts and diagnostic markers follow Table~\\ref{tab:seizure}.')

def write(name, value):
    Path('results/tables', name + '.tex').write_text(value + '\n')

write('control_table', simple_table([
    ('Waveform reference (8 rows)', 'cf2_v1d192raw'),
    ('Own-phase control (16 rows)', 'cf2_v1d192own'),
    ('Measured alignment (16 rows)', 'cf2_v1d192')],
    ['tuev','tusz','chbmit','iiic'], 'tab:control',
    'Analytic-content control in the folded encoder. Only own-phase and measured-alignment arms match row count, gates and allocated parameters; the waveform reference has eight rows. Seed coverage follows the main tables.', bold_best=False))
contrasts = [('Folded: duplex $-$ waveform', 'cf2_v1d192','cf2_v1d192raw'),
             ('CBraMod: added interaction $-$ added waveform','cbramod_add_cpl','cbramod_add_raw'),
             ('LaBraM: replacement $-$ native','labram_crofremo','labram_scratch'),
             ('REVE: replacement $-$ native','reve_crofremo','reve_scratch')]
contrast_ds = ['tuev','tusz','chbmit','iiic','tuep']
write('content_structure_table', grid(['Intervention']+[SHORT[d] for d in contrast_ds],
    [[n]+[fmt(paired(d,a,b),False) for d in contrast_ds] for n,a,b in contrasts],
    'tab:content-structure', 'Primary-metric differences paired by seed, with the number of shared seeds shown when fewer than three; uncertainty is the sample standard deviation of paired differences. Replacement changes the whole frontend, whereas the CBraMod additive control matches the extra rows. These interventions do not isolate the same factors.'))
for d in ['tuev','tusz','chbmit','iiic']:
    paired(d,'cf2_v1d192','cf2_v1d192own')
write('arch_table', simple_table([('8 bands, tri-axial','paclock_rawtok'),
    ('1 band, tri-axial','paclock_raw_nb1'),('8 bands, flat','paclock_raw_flat')],
    ['tusz','chbmit'],'tab:arch','Architectural controls with waveform tokens. These interventions change architecture and capacity; they do not isolate coupling content.',bold_best=False))
tri = [('Waveform','paclock_rawtok'),('Coupling-only','paclock_rot2'),('Duplex','paclock_duplex')]
write('triaxial_table', matrix(tri,list(DS),'tab:coupling-triaxial',
    'Earlier tri-axial family: waveform, coupling-only and duplex tokenizers. This comparison exposes task-dependent tradeoffs; token count and parameterization are not identical across all three arms.'))
write('variants_all_table',matrix([('Tri-axial duplex','paclock_duplex')]+design,list(DS),
    'tab:variants-all','Historical folded-encoder variants and tri-axial reference. No per-corpus winner is substituted into the main model row.'))
write('pretrained_transplant_table',pair_table([('CBraMod pretrained',[
    ('Native','cbramod_pretrained'),('+ waveform','cbramod_add_rawpre'),('+ interaction','cbramod_add_cplpre')])],
    ['tuev','tuar'],'tab:transplant-pre','Additive rows with the released CBraMod checkpoint. Native and added-row arms have unequal seed coverage; a difference of these aggregate means is not a paired estimate.'))
write('pretrain_triaxial_table',matrix([('Scratch','paclock_duplex'),('60k pretraining','paclock_duplex_ptS')],
    list(DS),'tab:pretrain-triaxial','Earlier tri-axial duplex: scratch and exploratory amplitude-plus-coupling pretraining. These are not pretraining results for a new candidate.'))
write('pretrain_folded_table',matrix([('Scratch','cf2_v1d192'),('150k raw-patch pretraining','cf2_v1d192_ptR')],
    list(DS),'tab:pretrain-folded','Separate folded-encoder raw-patch pretraining experiment. Missing cells are unfinished or unavailable. The mixed-sampling-rate limitation described in the text prevents a clean comparison of pretraining objectives.'))

# Recompute class metrics from saved predictions. Keeping these diagnostic
# tables here prevents numeric edits to the manuscript from drifting from runs.
import numpy as np
CLASSES = ['SPSW','GPED','PLED','EYEM','ARTF','BCKG']
def perclass(suf, seeds):
    values = []
    for seed in seeds:
        p = Path(f'runs/tuev-{suf}/seed{seed}/test_scores.npz')
        raw = p.read_bytes()
        with np.load(p, allow_pickle=False) as data:
            y = data['y'].astype(int).reshape(-1)
            pred = data['logits'].argmax(axis=-1).reshape(-1)
        if y.shape != pred.shape or ((y<0)|(y>=6)).any():
            raise ValueError(f'Invalid class labels: {p}')
        cm = np.bincount(y*6+pred, minlength=36).reshape(6,6)
        precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(6), where=cm.sum(0)>0)
        recall = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(6), where=cm.sum(1)>0)
        f1 = np.divide(2*precision*recall,precision+recall,out=np.zeros(6),where=precision+recall>0)
        values.append(np.stack([precision,recall,f1]))
        AUDIT[str(p)] = {'sha256':hashlib.sha256(raw).hexdigest(), 'seed':seed,
                        'class_counts':cm.sum(1).tolist(), 'precision':precision.tolist(),
                        'recall':recall.tolist(), 'f1':f1.tolist()}
    return np.mean(values,axis=0)
a,b = perclass('cf2_v1d192raw',[0]), perclass('cf2_v1d192',[0])
write('perclass_table',grid(['Class','Wave P','Wave R','Wave F1','Duplex P','Duplex R','Duplex F1'],
    [[c]+[f'{x:.4f}' for x in list(a[:,i])+list(b[:,i])] for i,c in enumerate(CLASSES)],
    'tab:perclass','Historical folded-encoder TUEV class metrics, matched seed 0. These diagnostics describe one run per arm, not the multi-seed aggregate or a new candidate.'))
a,b = perclass('paclock_rawtok',[0,1,2]),perclass('paclock_duplex',[0,1,2])
write('perclass_triaxial_table',grid(['Tokenizer']+CLASSES,
    [[n]+[f'{x:.4f}' for x in arr[2]] for n,arr in [('Waveform',a),('Duplex',b)]],
    'tab:perclass-triaxial','Earlier tri-axial TUEV per-class F1, arithmetic mean of three separately scored seeds.'))
Path('results/tables/table_audit.json').write_text(json.dumps(AUDIT,indent=2,sort_keys=True)+'\n')
Path('results/tables/paired_comparisons.json').write_text(json.dumps(PAIRED,indent=2,sort_keys=True)+'\n')
print('Reporting audit:',len(AUDIT),'source artifacts; paired contrasts:',len(PAIRED))

# Optional synchronization updates tables by label, preserving surrounding
# manuscript prose. Refuse missing/duplicate labels rather than silently drift.
import argparse, re
parser = argparse.ArgumentParser(description='Generate audited tables from runs/')
parser.add_argument('--paper-dir', type=Path)
args = parser.parse_args()
if args.paper_dir:
    sources = list((args.paper_dir/'sections').glob('*.tex'))
    content = {p:p.read_text() for p in sources}
    for source in sorted(Path('results/tables').glob('*.tex')):
        for block in re.findall(r'\\begin\{table\}.*?\\end\{table\}', source.read_text(), re.S):
            label = re.search(r'\\label\{([^}]+)\}',block)[1]
            hits = []
            for p,s in content.items():
                for match in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',s,re.S):
                    if r'\label{'+label+'}' in match[0]: hits.append((p,match))
            if len(hits)!=1: raise ValueError(f'Expected one manuscript table {label}; found {len(hits)}')
            p,match=hits[0]
            content[p]=content[p][:match.start()]+block+content[p][match.end():]
    for p,s in content.items(): p.write_text(s)
    print('Synchronized manuscript tables:',args.paper_dir)
