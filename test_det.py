#!/usr/bin/env python3
"""YOLO 目标检测整图严格匹配评测。

图片判对条件：预测数量与真实数量相等，并且存在一一对应关系，
使每一对框的类别相同、IoU >= 指定阈值。
无目标图片请保留空的同名标签文件；缺失标签文件会报错。
本指标是自定义整图正确率，不是 YOLO 原生 mAP。
"""

import argparse
from pathlib import Path
from experiment_utils import (DATA, WEIGHTS, ROOT, local_path, dataset_yaml,
                              select_device, load_detector, new_output, save_json)


def read_detection_label(label_path, img_w, img_h):
    """读取检测标签：每行必须为 class x_center y_center width height。"""
    import torch
    classes, boxes = [], []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("每行必须恰好有 5 列，请使用检测标签")
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            if cls < 0 or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("类别必须非负；中心坐标须在 [0,1]，宽高须在 (0,1]")
        except ValueError as exc:
            raise ValueError(f"标签格式错误：{label_path} 第 {line_no} 行：{exc}") from exc
        classes.append(cls)
        boxes.append([
            max(0.0, (xc - w / 2) * img_w),
            max(0.0, (yc - h / 2) * img_h),
            min(float(img_w), (xc + w / 2) * img_w),
            min(float(img_h), (yc + h / 2) * img_h),
        ])
    return classes, torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def is_exact_match(gt_classes, pred_classes, iou_matrix, iou_thresh):
    """判断是否存在满足类别和 IoU 条件的一一匹配。矩阵形状为预测数×真实数。"""
    if sorted(gt_classes) != sorted(pred_classes):
        return False
    if not gt_classes:
        return True

    # 每个预测框只能选择同类别、IoU 达标的真实框。
    candidates = [
        [g for g, gt_cls in enumerate(gt_classes)
         if gt_cls == pred_cls and float(iou_matrix[p][g]) >= iou_thresh]
        for p, pred_cls in enumerate(pred_classes)
    ]
    gt_owner = [-1] * len(gt_classes)

    def assign(pred_idx, visited):
        for gt_idx in candidates[pred_idx]:
            if visited[gt_idx]:
                continue
            visited[gt_idx] = True
            # 若真实框已被占用，尝试为原预测框重新安排匹配。
            if gt_owner[gt_idx] == -1 or assign(gt_owner[gt_idx], visited):
                gt_owner[gt_idx] = pred_idx
                return True
        return False

    return all(assign(p, [False] * len(gt_classes)) for p in range(len(pred_classes)))


def evaluate_det_exact_match(
    model_path, img_dir, label_dir,
    conf=0.35, iou_thresh=0.5, imgsz=640, batch_size=8, device=None,
):
    import torch
    from tqdm import tqdm
    from ultralytics.utils.metrics import box_iou
    if not 0 <= conf <= 1 or not 0 <= iou_thresh <= 1:
        raise ValueError("置信度阈值和 IoU 阈值必须在 [0,1] 内")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    img_dir, label_dir = local_path(img_dir), local_path(label_dir)
    if not img_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("图片目录或标签目录不存在")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_paths = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not img_paths:
        raise ValueError(f"没有找到测试图片：{img_dir}")

    # 先检查全部标签，防止缺失标签导致测试集分母悄悄变小。
    missing = [p.name for p in img_paths if not (label_dir / f"{p.stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(
            f"有 {len(missing)} 张图片缺少标签文件，例如：{', '.join(missing[:5])}。"
            "确认无目标的图片应有空的同名 .txt 文件。"
        )

    device = select_device(device)
    model = load_detector(model_path)
    if model.task != "detect":
        raise ValueError("请使用目标检测模型权重")

    total, correct, errors = 0, 0, []
    print(f"开始检测评测 | 图片数: {len(img_paths)} | conf: {conf} | 匹配IoU: {iou_thresh}")

    for start in tqdm(range(0, len(img_paths), batch_size), desc="批次推理", unit="batch"):
        chunk = img_paths[start:start + batch_size]
        with torch.inference_mode():
            results = model.predict(
                source=[str(p) for p in chunk],
                conf=conf,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                verbose=False,
                save=False,
                augment=False,
            )
        if len(results) != len(chunk):
            raise RuntimeError("预测结果数与输入图片数不一致，停止评测")

        for res, img_path in zip(results, chunk):
            img_h, img_w = res.orig_shape
            gt_classes, gt_boxes = read_detection_label(
                label_dir / f"{img_path.stem}.txt", img_w, img_h
            )
            if any(cls not in model.names for cls in gt_classes):
                raise ValueError(f"{img_path.name} 的真实类别超出了模型类别范围")
            pred_classes = []
            pred_boxes = torch.empty((0, 4), dtype=torch.float32)
            if res.boxes is not None:
                pred_classes = res.boxes.cls.cpu().int().tolist()
                pred_boxes = res.boxes.xyxy.cpu().float()

            ious = box_iou(pred_boxes, gt_boxes)
            matched = is_exact_match(gt_classes, pred_classes, ious, iou_thresh)
            total += 1
            correct += int(matched)
            if not matched:
                if len(gt_classes) != len(pred_classes):
                    reason = "预测数量与真实数量不同"
                elif sorted(gt_classes) != sorted(pred_classes):
                    reason = "类别或各类别数量不一致"
                else:
                    reason = "不存在满足 IoU 阈值的同类别一一匹配"
                errors.append((img_path.name, sorted(gt_classes), sorted(pred_classes), reason))

    accuracy = correct / total
    print("\n" + "=" * 65)
    print("验证完成")
    print(f"总评测图片数     : {total}")
    print(f"严格匹配正确数   : {correct}")
    print(f"严格匹配正确率   : {accuracy:.4f} ({accuracy:.2%})")
    print(f"错误图片数       : {total - correct}")
    print(f"预测置信度阈值   : {conf}")
    print(f"匹配条件         : 数量相等、类别相同、IoU >= {iou_thresh}、一一匹配")
    print("=" * 65)
    for name, gt, pred, reason in errors[:5]:
        print(f"错误示例：{name}\n  真实类别: {gt}\n  预测类别: {pred}\n  原因: {reason}")
    return {"total": total, "correct": correct, "accuracy": accuracy,
            "conf": conf, "match_iou": iou_thresh, "errors": errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # 默认数据、权重无需改路径；使用自训权重时传 --weights。
    p.add_argument("--weights", type=local_path, default=WEIGHTS)
    p.add_argument("--data-root", type=local_path, default=DATA)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--mode", choices=("all", "map", "exact"), default="all")
    p.add_argument("--conf", type=float, default=0.35, help="仅用于严格匹配；mAP用低阈值计算PR曲线")
    p.add_argument("--match-iou", type=float, default=0.5, help="严格匹配框IoU阈值")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=local_path, default=ROOT / "runs/evaluate")
    args = p.parse_args()
    if not (0 <= args.conf <= 1 and 0 <= args.match_iou <= 1) or min(args.batch,args.imgsz) <= 0:
        p.error("阈值须在[0,1]内，batch和imgsz须为正数")
    config = dataset_yaml(args.data_root)
    out = new_output(args.output)
    report = {"weights":str(args.weights), "split":args.split, "imgsz":args.imgsz}
    if args.mode in ("all", "map"):
        model = load_detector(args.weights)
        metrics = model.val(data=str(config), split=args.split, imgsz=args.imgsz,
                            batch=args.batch, device=select_device(args.device), workers=0,
                            conf=0.001, plots=True, project=str(out), name="map", exist_ok=False)
        report["standard_metrics"] = {k:float(v) for k,v in metrics.results_dict.items()}
        save_json(out / "metrics.json", report)
        del model
    if args.mode in ("all", "exact"):
        report["exact_match"] = evaluate_det_exact_match(
            args.weights, args.data_root / args.split / "images", args.data_root / args.split / "labels",
            conf=args.conf, iou_thresh=args.match_iou, imgsz=args.imgsz,
            batch_size=args.batch, device=args.device)
    save_json(out / "metrics.json", report)
    print(f"评估报告：{out / 'metrics.json'}")


if __name__ == "__main__":
    main()
