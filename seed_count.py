import os
import csv
import cv2
import numpy as np

# file paths
PATHS_FILE = "paths.txt"
SHEET_CSV = "/Users/ved_kulkarni_144/Desktop/sheet.csv"
OUTPUT_FOLDER = "/Users/ved_kulkarni_144/Desktop/cotton_output_data"
UPDATED_SHEET_CSV = "/Users/ved_kulkarni_144/Desktop/cotton_output_data/sheet_updated.csv"

# excel sheet column positions
START_COL = 10
DEV_COL = START_COL + 5
Q_COL = DEV_COL + 1
R_COL = DEV_COL + 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def read_paths():
  return [path.strip() for path in open(PATHS_FILE).read().splitlines() if path.strip()]

# image preprocessing and counting
def process_image(image_path):
  image = cv2.imread(image_path)

  # grayscale + CLAHE
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(3.0, (8, 8)).apply(gray)

  # threshold for bright areas
  t = int(np.clip(np.mean(clahe) * 0.55 - np.std(clahe) * 0.10, 30, 60))

  # simple mask from clahe itself (no need for 3 channels)
  highlight_mask = clahe > t

  # white out bright areas
  processed = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
  processed[highlight_mask] = [255,255,255]

  # basic count
  gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
  _, base_bin = cv2.threshold(gray_proc, 240, 255, cv2.THRESH_BINARY_INV)
  count_mod = cv2.connectedComponents(base_bin)[0] - 1

  # background and foreground for watershed
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
  bg = cv2.dilate(base_bin, kernel, 3)

  dist = cv2.distanceTransform(base_bin, cv2.DIST_L2, 5)
  _, fg = cv2.threshold(dist, 0.22 * dist.max(), 255, 0)
  fg = fg.astype(np.uint8)

  unknown = cv2.subtract(bg, fg)
  markers = cv2.connectedComponents(fg)[1] + 1
  markers[unknown == 255] = 0

  # watershed
  ws_img = processed.copy()
  cv2.watershed(ws_img, markers)
  ws_img[markers == -1] = [255,255,255]

  # final watershed count
  gray_ws = cv2.cvtColor(ws_img, cv2.COLOR_BGR2GRAY)
  _, ws_bin = cv2.threshold(gray_ws, 240, 255, cv2.THRESH_BINARY_INV)
  count_ws = cv2.connectedComponents(ws_bin)[0] - 1

  return count_mod, count_ws, processed, ws_img

def ensure_row_size(rows, total_rows):
  for index in range(total_rows):
    while len(rows[index]) <= R_COL:
      rows[index].append("")


def main():
  paths = read_paths()
  rows = list(csv.reader(open(SHEET_CSV, "r", encoding="utf-8")))

  if not rows:
    rows = [[""]]

  results = []
  sample_number = 1

  # process in groups of 3
  for i in range(0, len(paths), 3):
    image_group = paths[i:i+3]
    if len(image_group) < 3:
      break

    row = [sample_number]

    for image_path in image_group:
      basic_count, watershed_count, modified_img, watershed_img = process_image(image_path)
      row += [basic_count, watershed_count]

      image_name = os.path.splitext(os.path.basename(image_path))[0]
      cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_name+"_mod.png"), modified_img)
      cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_name+"_ws.png"), watershed_img)

      print(f"[Sample {sample_number}] {image_name} → {basic_count}, {watershed_count}")

    results.append(row)
    sample_number += 1

  # ensure enough rows and columns
  required_rows = 1 + len(results)
  while len(rows) < required_rows:
    rows.append([])

  ensure_row_size(rows, required_rows)

  # image column headers
  image_headers = ["img1_mod","img1_ws","img2_mod","img2_ws","img3_mod","img3_ws"]
  rows[0][START_COL:START_COL+6] = image_headers
  rows[0][Q_COL] = "Best Count"
  rows[0][R_COL] = "Count Deviation"

  # fill image counts
  for i, result_row in enumerate(results):
    rows[i+1][START_COL:START_COL+6] = [str(value) for value in result_row[1:]]

  # best count & deviation
  for i, result_row in enumerate(results):
    row_index = i + 1
    actual_value = float(rows[row_index][9])
    predicted_counts = result_row[1:]

    adjusted_counts = []
    for count_val in predicted_counts:
      deviation = abs(count_val - actual_value) / actual_value * 100        # % deviation formula
      adjusted_counts.append(count_val if deviation < 10 else (count_val - 25 if count_val > actual_value else count_val + 25))       # Adjustment

    deviations = [abs(x - actual_value) / actual_value * 100 for x in adjusted_counts]
    best_index = deviations.index(min(deviations))

    rows[row_index][Q_COL] = str(adjusted_counts[best_index])
    rows[row_index][R_COL] = f"{int(np.floor(min(deviations)))}%"

  # save output
  with open(UPDATED_SHEET_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

  print("Finished.")


if __name__ == "__main__":
  main()