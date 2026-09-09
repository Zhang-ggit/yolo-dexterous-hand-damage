"""单张灵巧手图片 -> 五指中文缺损状态。

实验约定：每张图片包含一只灵巧手；未检出即记为完全缺失。
同一手指有多个预测时，使用置信度最高的预测。
"""

import argparse
from pathlib import Path
from experiment_utils import DATA, WEIGHTS, local_path, image_files, load_detector


FINGERS = (("thumb", "拇指"), ("index", "食指"), ("middle", "中指"),
           ("ring", "无名指"), ("pinky", "小指"))
STATES = (("full", "完整"), ("slightly_damaged", "轻度缺损（缺一段）"),
          ("severely_damaged", "重度缺损（缺两段）"))
CLASS_INFO = {
    f"{state}_{finger}": (finger, description)
    for finger, _ in FINGERS for state, description in STATES
}


def summarize_detections(detections, conf=0.35):
    """将 (类别名称, 置信度) 序列转为五行文本，可独立于模型验证。"""
    if not 0 <= conf <= 1:
        raise ValueError("置信度阈值必须在 [0, 1] 内")
    best = {}
    for class_name, score in detections:
        if class_name not in CLASS_INFO:
            raise ValueError(f"未知检测类别：{class_name}")
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError("预测置信度必须在 [0, 1] 内")
        if score < conf:
            continue
        finger, description = CLASS_INFO[class_name]
        if finger not in best or score > best[finger][0]:
            best[finger] = (score, description)
    return "\n".join(
        f"{chinese}：{best[finger][1] if finger in best else '完全缺失'}"
        for finger, chinese in FINGERS
    )


def predict_hand_status(image_path, weights, conf=0.35, imgsz=640, device=None):
    """加载本地检测权重，返回一张图片的五指状态文本。"""
    image_path, weights = local_path(image_path), local_path(weights)
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在：{weights}")
    if not 0 <= conf <= 1 or imgsz <= 0:
        raise ValueError("conf 须在 [0, 1] 内，imgsz 须为正整数")

    model = load_detector(weights)
    if model.task != "detect":
        raise ValueError("请使用目标检测权重")
    if set(model.names.values()) != set(CLASS_INFO):
        raise ValueError("模型类别与本实验的15个手指缺损类别不一致")
    options = {"device": device} if device is not None else {}
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz,
                            verbose=False, save=False, **options)
    if len(results) != 1:
        raise ValueError("请输入单张图片")
    result = results[0]
    detections = []
    if result.boxes is not None:
        detections = [
            (model.names[int(cls)], float(score))
            for cls, score in zip(result.boxes.cls.cpu().tolist(),
                                  result.boxes.conf.cpu().tolist())
        ]
    return summarize_detections(detections, conf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 不传图片时使用测试集第一张；外部图片传绝对路径，仓库内传相对路径即可。
    parser.add_argument("image", nargs="?", type=local_path, help="单张图片路径，默认测试集第一张")
    parser.add_argument("--weights", type=local_path, default=WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值，默认0.35")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="例如 0 或 cpu；省略时由模型自动选择")
    parser.add_argument("--output", type=local_path, help="可选：保存为UTF-8文本文件")
    args = parser.parse_args()
    try:
        if args.image is None:
            args.image = image_files(DATA / "test/images")[0]
        report = predict_hand_status(args.image, args.weights, args.conf, args.imgsz, args.device)
        if args.output:
            if args.output.resolve() in {args.image.resolve(), args.weights.resolve(), Path(__file__).resolve()}:
                raise ValueError("输出文件不能覆盖输入图片、权重或脚本")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        parser.exit(1, f"错误：{exc}\n")
    print(report)


if __name__ == "__main__":
    main()
