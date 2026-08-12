import openpyxl
wb = openpyxl.load_workbook("/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/results/_in.xlsx")
for sh in ["TUAB", "Sleep-EDF"]:
    ws = wb[sh]
    print("="*60); print("##", sh)
    for row in ws.iter_rows(min_row=1, max_row=40):
        vals = [(c.value if c.value is not None else "") for c in row[:6]]
        if any(str(v).strip() for v in vals):
            print("r%-3d %s" % (row[0].row, " | ".join(str(v)[:26] for v in vals)))
