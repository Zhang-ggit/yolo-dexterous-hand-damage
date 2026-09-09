"""实验公共设置：相对路径以脚本目录为基准，仓库移动后无需修改。"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "hand_data_det"
WEIGHTS = ROOT / "hand_det_weights/weights/best.pt"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NAMES = [f"{s}_{f}" for f in ("thumb", "index", "middle", "ring", "pinky")
         for s in ("full", "slightly_damaged", "severely_damaged")]

def local_path(value):
    """外部文件通过命令行指定绝对路径；仓库内文件无需改路径。"""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()

def image_files(folder):
    folder = local_path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f"没有图片：{folder}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError(f"存在同名不同扩展名图片，无法唯一匹配标签：{folder}")
    return files

def dataset_yaml(data_root=DATA):
    """生成本机配置，避免YOLO全局datasets_dir影响相对路径。"""
    root = local_path(data_root)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            if not (root / split / kind).is_dir():
                raise FileNotFoundError(f"目录缺失：{root / split / kind}")
    target = ROOT / "runs/hand_data_local.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = f"path: {json.dumps(root.as_posix())}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n"
    text += "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES))
    target.write_text(text, encoding="utf-8")
    return target

def select_device(value=None):
    import torch
    return value if value is not None else ("0" if torch.cuda.is_available() else "cpu")

def load_detector(weights=WEIGHTS):
    from ultralytics import YOLO
    path = local_path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"权重不存在：{path}；请下载配套权重或通过--weights指定")
    model = YOLO(str(path), task="detect")
    if model.task != "detect" or model.names != dict(enumerate(NAMES)):
        raise ValueError("权重任务或类别编号与本实验15类不一致")
    return model

def new_output(value):
    path = local_path(value)
    candidate, index = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{index}")
        index += 1
    candidate.mkdir(parents=True)
    return candidate

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

