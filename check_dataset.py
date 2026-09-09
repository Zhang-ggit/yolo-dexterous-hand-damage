"""检查图像、标签、数量和跨集合完全相同的图像。"""
import argparse
import hashlib
import math
from collections import Counter
from experiment_utils import DATA, NAMES, ROOT, local_path, image_files, save_json

def check_dataset(root=DATA):
    from PIL import Image
    root = local_path(root)
    report, errors, warnings, hashes = {}, [], [], {}
    for split, expected in (("train",4000), ("val",500), ("test",500)):
        images = image_files(root / split / "images")
        labels = root / split / "labels"
        counts, empty = Counter(), 0
        if len(images) != expected:
            errors.append(f"{split}：预期{expected}张，实际{len(images)}张")
        orphan = {p.stem for p in labels.glob("*.txt")} - {p.stem for p in images}
        if orphan:
            errors.append(f"{split}：{len(orphan)}个标签没有对应图片")
        for path in images:
            try:
                with Image.open(path) as im:
                    im.verify()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in hashes and hashes[digest][0] != split:
                    warnings.append(f"跨集合重复图片：{hashes[digest][1]} 与 {path.name}")
                hashes.setdefault(digest, (split,path.name))
                lines = [x for x in (labels / f"{path.stem}.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
                empty += not lines
                fingers = set()
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError("检测标签须恰好5列")
                    cls = int(parts[0])
                    x,y,w,h = map(float,parts[1:])
                    if not (0 <= cls < 15 and all(math.isfinite(v) for v in (x,y,w,h))
                            and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("类别或坐标超出范围")
                    if cls // 3 in fingers:
                        raise ValueError("同一手指重复标注")
                    fingers.add(cls // 3)
                    counts[NAMES[cls]] += 1
            except (ValueError,OSError) as exc:
                errors.append(f"{split}/{path.name}：{exc}")
        report[split] = {"images":len(images), "empty_labels":empty,
                         "objects_per_class":{name:counts[name] for name in NAMES}}
    return {"splits":report, "errors":errors, "warnings":warnings}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=local_path, default=DATA)
    p.add_argument("--output", type=local_path, default=ROOT / "runs/dataset_check.json")
    args = p.parse_args()
    report = check_dataset(args.data_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_json(args.output,report)
    for split,row in report["splits"].items():
        print(f"{split}：{row['images']}张，空标签{row['empty_labels']}个")
    print(f"错误{len(report['errors'])}项，警告{len(report['warnings'])}项；详情：{args.output}")
    for message in (report["errors"] + report["warnings"])[:10]:
        print(message)
    raise SystemExit(bool(report["errors"]))

if __name__ == "__main__":
    main()

