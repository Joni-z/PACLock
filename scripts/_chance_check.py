import glob, json

print("=== every paclock_v2 seed, verdict status ===")
for p in sorted(glob.glob("runs/*-paclock_v2/seed*/result.json")):
    r = json.load(open(p))
    v = r["verdict"]
    cell = p.split("/")[1]
    flag = "" if v["ok"] else "  <-- COLLAPSED"
    print("  %-28s seed%s  %-13s  %s%s" % (cell, r["seed"], v["status"],
          v.get("reason", "")[:70], flag))
