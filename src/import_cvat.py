"""Import CVAT 'MOT 1.1' exports into data/annotations/vid_XXX.txt.

Drop the zips exported from CVAT into data/annotations/ (any file name that contains vid_XXX)
and run this. Each zip's gt/gt.txt is written as data/annotations/vid_XXX.txt with the class
column remapped BY NAME (gt/labels.txt) to our order (vehicle=1, two_wheeler=2, person=3), so a
CVAT project whose labels were created in another order still imports correctly. The zip is
then moved to data/cvat_exports/.

Usage:
  python -m src.import_cvat
"""
import re
import shutil
import zipfile
from pathlib import Path

LABELS = ["vehicle", "two_wheeler", "person"]


def convert(zip_path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.read("gt/labels.txt").decode().splitlines() if n.strip()]
        rows = z.read("gt/gt.txt").decode().splitlines()
    unknown = [n for n in names if n not in LABELS]
    if unknown:
        raise SystemExit(f"{zip_path.name}: unknown labels {unknown} (expected {LABELS})")
    remap = {i + 1: LABELS.index(n) + 1 for i, n in enumerate(names)}
    out = []
    for line in rows:
        p = line.strip().split(",")
        if len(p) < 8:
            continue
        p[7] = str(remap[int(float(p[7]))])
        out.append(",".join(p))
    note = "" if names == LABELS else f"labels remapped from order {names}"
    return "\n".join(out) + "\n", note


def main() -> None:
    ann, done = Path("data/annotations"), Path("data/cvat_exports")
    done.mkdir(parents=True, exist_ok=True)
    zips = sorted(ann.glob("*.zip"))
    if not zips:
        print("no zips in data/annotations")
    for z in zips:
        m = re.search(r"vid_\d{3}", z.name)
        if not m:
            print(f"SKIP {z.name}: no vid_XXX in the file name - rename it")
            continue
        text, note = convert(z)
        (ann / f"{m.group()}.txt").write_text(text)
        shutil.move(str(z), done / z.name)
        print(f"{z.name} -> {ann / (m.group() + '.txt')} ({text.count(chr(10))} boxes) {note}")


if __name__ == "__main__":
    main()
