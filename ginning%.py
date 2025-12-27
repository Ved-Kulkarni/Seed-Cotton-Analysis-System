import csv

input_dest = "/Users/ved_kulkarni_144/Desktop/cotton_output_data/sheet_updated.csv"
output_dest = "/Users/ved_kulkarni_144/Desktop/cotton_output_data/final_results.csv"

def safe(x):
  try: return float(x)
  except: return None

rows = list(csv.reader(open(input_dest, "r", encoding="utf-8")))
header = rows[0]

Q_COL = header.index("Best Count")
F_COL = 5

header += ["Calculated Seed Weight", "Lint Percentage"]

for r in rows[1:]:
  f = safe(r[F_COL])
  q = safe(r[Q_COL])

  if f is None or q is None:
    r += ["", ""]
    continue

  S = round(f * q, 2)
  T = round(100 * (60 - S) / 60, 2)

  r += [str(S), f"{T}%"]

with open(output_dest, "w", newline="", encoding="utf-8") as f:
  csv.writer(f).writerows(rows)

print("final_results.csv created")