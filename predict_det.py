"""单图/文件夹检测，输出检测框图片、五指文本和汇总CSV。"""
import argparse
import csv
import json
from experiment_utils import (DATA, WEIGHTS, ROOT, local_path, image_files,
                              load_detector, new_output, EXTENSIONS)
from predict_hand_status import summarize_detections

def main():
    p = argparse.ArgumentParser(description=__doc__)
    # 默认测试集；其他位置用--source传入，路径不必写进代码。
    p.add_argument("--source", type=local_path, default=DATA / "test/images")
    p.add_argument("--weights", type=local_path, default=WEIGHTS)
    p.add_argument("--output", type=local_path, default=ROOT / "runs/predict")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0, help="最多处理几张，0表示全部")
    p.add_argument("--errors", type=local_path, help="test_det.py生成的metrics.json，只显示其中错误图片")
    args = p.parse_args()
    if not 0 <= args.conf <= 1 or min(args.batch,args.imgsz) <= 0 or args.limit < 0:
        p.error("参数范围不合法")
    if args.source.is_dir():
        paths = image_files(args.source)
    elif args.source.is_file() and args.source.suffix.lower() in EXTENSIONS:
        paths = [args.source]
    else:
        raise FileNotFoundError(f"图片或图片目录不存在：{args.source}")
    if args.errors:
        data = json.loads(args.errors.read_text(encoding="utf-8"))
        if "exact_match" not in data:
            raise ValueError("评估报告没有exact_match，请先运行严格匹配评估")
        names = {row[0] for row in data["exact_match"]["errors"]}
        missing = names - {path.name for path in paths}
        if missing:
            raise ValueError("部分错误图片不在source中，请指定评估时使用的图片目录")
        paths = [path for path in paths if path.name in names]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print("没有需要输出的图片")
        return
    model = load_detector(args.weights)
    out = new_output(args.output)
    (out / "images").mkdir()
    (out / "texts").mkdir()
    options = {"device":args.device} if args.device is not None else {}
    with (out / "status.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["图片", "拇指", "食指", "中指", "无名指", "小指"])
        for start in range(0,len(paths),args.batch):
            chunk = paths[start:start+args.batch]
            results = model.predict(source=[str(path) for path in chunk], conf=args.conf,
                                    imgsz=args.imgsz, batch=args.batch, verbose=False, save=False, **options)
            if len(results) != len(chunk):
                raise RuntimeError("返回结果数与输入图片数不一致")
            for path,result in zip(chunk,results):
                detections = [] if result.boxes is None else [
                    (model.names[int(cls)],float(score))
                    for cls,score in zip(result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist())]
                report = summarize_detections(detections,args.conf)
                (out / "texts" / f"{path.stem}.txt").write_text(report+"\n",encoding="utf-8")
                result.save(filename=str(out / "images" / f"{path.stem}.jpg"))
                writer.writerow([path.name]+[line.split("：",1)[1] for line in report.splitlines()])
            print(f"已处理 {min(start+args.batch,len(paths))}/{len(paths)}")
    print(f"检测结果：{out}")

if __name__ == "__main__":
    main()

