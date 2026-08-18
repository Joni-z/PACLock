"""Add an ablation sheet to the deliverable workbook.

    sbatch slurm/run_cpu.slurm scripts.add_ablation_sheet

The nine per-corpus sheets are the frozen baseline matrix and are not touched.
Our own variants do not belong in them -- they are not baselines -- but they are
what the paper argues from, and until now they lived only in runs/. This adds
one sheet, `_消融`, carrying every PACLock variant with a finished run.

Hard rule 4 is applied but not enforced by deletion: a cell with fewer than 3
seeds is written with its seed count and greyed, so a reader can see it exists
and that it is not yet admissible, rather than seeing a blank that is
indistinguishable from "not run".
"""
import glob
import json
import os
import statistics as st

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

XLSX = "results/PACLock_baseline_matrix_filled.xlsx"
SHEET = "_消融"

DS = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc",
      "physionet_mi", "faced", "bci_iv_2a"]
DS_LABEL = {"tuab": "TUAB", "tuev": "TUEV", "tusz": "TUSZ", "chbmit": "CHB-MIT",
            "sleepedf": "Sleep-EDF", "isruc": "ISRUC",
            "physionet_mi": "PhysioNet-MI", "faced": "FACED",
            "bci_iv_2a": "BCI-IV-2a"}
KEY = {"tuab": "auroc", "tuev": "cohen_kappa", "tusz": "pr_auc",
       "chbmit": "pr_auc", "sleepedf": "cohen_kappa", "isruc": "cohen_kappa",
       "physionet_mi": "balanced_acc", "faced": "balanced_acc",
       "bci_iv_2a": "balanced_acc"}

# variant -> (label, group). Order is the order of the sheet.
ROWS = [
    ("paclock_v2",          "PACLock v2(冻结配置:pac_interaction + product)", "参照"),
    ("paclock_rawtok",      "└ raw tokenizer(去掉整个交互)",                  "tokenizer 消融"),
    ("paclock_pac",         "└ PAC 协议预处理(0.5Hz 高通,无 notch)",          "tokenizer 消融"),
    ("paclock_concat",      "└ concat 融合(SleepPACNet 式)",                  "tokenizer 消融"),
    ("paclock_pac_uniform", "└ uniform 耦合(去掉测得的 α 与 ∠Z)",             "tokenizer 消融"),
    ("paclock_rot2",        "PACLock + rotation(耦合只旋转,不缩放)",          "架构改动"),
    ("paclock_sphead",      "PACLock + spatial 头(保留电极身份)",             "架构改动"),
    ("paclock_rot_sphead",  "PACLock + rotation + spatial 头",                 "架构改动"),
    ("paclock_pt_base",     "PACLock 预训练 base(60k)",                       "预训练"),
    ("paclock_pt_large",    "PACLock 预训练 large(60k)",                      "预训练"),
    ("paclock_rawpt_large", "└ raw tokenizer 预训练 large(60k)",              "预训练"),
    ("paclock_pt_excl",     "└ 预训练池剔除 TUSZ/CHB-MIT",                     "预训练"),
    ("paclock_xyz",         "spatial_pe: xyz(蒙太奇坐标)",                    "已否决"),
    ("paclock_wn",          "每窗口归一化",                                     "已否决"),
    ("paclock_raw_large",   "raw + large 容量",                                "容量"),
    ("paclock_raw_headattn", "raw + attn 头",                                  "容量"),
]

HDR = PatternFill("solid", fgColor="D9D9D9")
GREY = PatternFill("solid", fgColor="F2F2F2")
GRP = PatternFill("solid", fgColor="EDF3FA")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")


def cell_value(ds, variant):
    ps = sorted(glob.glob("runs/%s-%s/*/result.json" % (ds, variant)))
    vals = []
    for p in ps:
        v = json.load(open(p))["test"].get(KEY[ds])
        if v is not None:
            vals.append(v)
    if not vals:
        return None, 0
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return "%.4f±%.4f" % (m, s), len(vals)


def main():
    wb = load_workbook(XLSX)
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET)

    ws["A1"] = "PACLock 自身消融 —— 与冻结 baseline 主表分开,主表未改动"
    ws["A1"].font = Font(bold=True, size=12)
    ws["A2"] = ("每格 mean±std,括号内为 seed 数。硬规则 4:少于 3 seed 不可进论文表,"
                "此处灰底标出并保留数字,以便区分「尚未达标」与「没跑」。"
                "指标:TUAB=AUROC,TUSZ/CHB-MIT=PR-AUC,TUEV/Sleep-EDF/ISRUC=kappa,"
                "其余=balanced acc。")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:K2")
    ws.row_dimensions[2].height = 30

    hdr = 4
    ws.cell(hdr, 1, "分组").fill = HDR
    ws.cell(hdr, 2, "变体").fill = HDR
    for j, ds in enumerate(DS):
        c = ws.cell(hdr, 3 + j, DS_LABEL[ds])
        c.fill, c.font, c.alignment, c.border = HDR, Font(bold=True), CENTER, BOX
    for col in (1, 2):
        ws.cell(hdr, col).font = Font(bold=True)
        ws.cell(hdr, col).border = BOX

    r = hdr + 1
    last_group = None
    n_written = n_short = 0
    for variant, label, group in ROWS:
        ws.cell(r, 1, group if group != last_group else "")
        if group != last_group:
            ws.cell(r, 1).font = Font(bold=True)
            ws.cell(r, 1).fill = GRP
        last_group = group
        ws.cell(r, 2, label).border = BOX
        ws.cell(r, 1).border = BOX
        for j, ds in enumerate(DS):
            txt, n = cell_value(ds, variant)
            c = ws.cell(r, 3 + j)
            c.border, c.alignment = BOX, CENTER
            if txt is None:
                continue
            c.value = "%s (%d)" % (txt, n)
            n_written += 1
            if n < 3:
                c.fill = GREY
                c.font = Font(italic=True, color="808080")
                n_short += 1
        r += 1

    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 42
    for j in range(len(DS)):
        ws.column_dimensions[chr(ord("C") + j)].width = 18
    ws.freeze_panes = "C5"

    wb.save(XLSX)
    print("sheet %r written: %d cells, %d of them below 3 seeds (greyed)"
          % (SHEET, n_written, n_short))
    print("sheets now:", wb.sheetnames)


if __name__ == "__main__":
    main()
