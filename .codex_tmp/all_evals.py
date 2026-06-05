import os, json
from datetime import datetime

view_base = r"D:\XMMX\ST-DS Imagen\view\energy"
log_base = r"D:\XMMX\ST-DS Imagen\logs\energy"

# Collect ALL evaluation entries from ALL runs with timestamps
all_entries = []
for root, dirs, files in os.walk(view_base):
    for f in files:
        if f == "evaluation.jsonl":
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line.strip())
                    d["_run_dir"] = os.path.basename(root)
                    d["_file"] = path
                    all_entries.append(d)

# Sort by timestamp
all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

print(f"Total evaluation entries: {len(all_entries)}")
print(f"\n=== All entries sorted by timestamp (NEWEST FIRST) ===\n")
for e in all_entries[:40]:
    ts = e.get("timestamp", "?")
    run = e["_run_dir"]
    epoch = str(e.get("epoch", "?"))
    disc = e.get("disc_mean", "?")
    pred = e.get("pred_mean", "?")
    cc = e.get("cross_corr_mean", "?")
    cfid = e.get("context_fid_mean", "?")
    print(f"[{ts}] run={run}  epoch={epoch:>5s}  disc={disc}  pred={pred}  cc={cc}  cfid={cfid}")
