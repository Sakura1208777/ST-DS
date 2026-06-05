import os, json

view = r"D:\XMMX\ST-DS Imagen\view\energy"
run_dirs = []
for r, ds, fs in os.walk(view):
    for d in ds:
        if d.startswith("2026"):
            run_dirs.append((os.path.join(r, d), d))
run_dirs.sort(key=lambda x: x[1], reverse=True)

latest_dir = run_dirs[0][0]
latest_name = run_dirs[0][1]
print(f"Latest run: {latest_name}")

ef = os.path.join(latest_dir, "evaluation.jsonl")
if not os.path.exists(ef):
    print("No evaluation.jsonl found!")
    exit()

entries = []
with open(ef, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line.strip())
        entries.append(d)

# Also check if there are newer runs that I might have missed
for rd in run_dirs[:5]:
    print(f"  Found: {rd[1]}")

print(f"\n=== {latest_name} full trajectory ({len(entries)} points) ===\n")
print(f"{'epoch':>6s}  {'disc':>8s}  {'std':>6s}  {'pred':>8s}  {'cc':>8s}  {'cfid':>8s}")
print("-" * 60)
for e in entries:
    ep = str(e.get("epoch", "?"))
    disc = e.get("disc_mean", 0)
    disc_s = e.get("disc_std", 0)
    pred = e.get("pred_mean", 0)
    cc = e.get("cross_corr_mean", 0)
    cfid = e.get("context_fid_mean", 0)
    print(f"{ep:>6s}  {disc:8.4f}  {disc_s:6.4f}  {pred:8.4f}  {cc:8.4f}  {cfid:8.4f}")

# Calculate disc rate of change per 20 epochs
print(f"\n=== disc changes per 20-epoch interval ===")
prev_disc = None
prev_ep = None
for e in entries:
    disc = e.get("disc_mean", 0)
    ep = e.get("epoch", 0)
    if isinstance(ep, str):
        continue
    if prev_disc is not None:
        delta = disc - prev_disc
        direction = "↑" if delta > 0 else "↓"
        print(f"  {prev_ep:3d} -> {ep:3d}:  {prev_disc:.4f} -> {disc:.4f}  ({delta:+.4f}) {direction}")
    prev_disc = disc
    prev_ep = ep
