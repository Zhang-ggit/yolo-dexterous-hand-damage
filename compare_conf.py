"""仅在验证集比较置信度阈值，输出整图严格匹配正确率。"""
import argparse
import csv
from experiment_utils import DATA, WEIGHTS, ROOT, local_path, new_output, save_json
from test_det import evaluate_det_exact_match

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=local_path,default=DATA)
    p.add_argument("--weights",type=local_path,default=WEIGHTS)
    p.add_argument("--thresholds",type=float,nargs="+",default=[0.25,0.35,0.50])
    p.add_argument("--match-iou",type=float,default=0.5)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--device",default=None)
    p.add_argument("--output",type=local_path,default=ROOT / "runs/compare_conf")
    args = p.parse_args()
    if (any(not 0 <= v <= 1 for v in args.thresholds)
            or not 0 <= args.match_iou <= 1 or min(args.batch,args.imgsz) <= 0):
        p.error("阈值须在[0,1]内，batch和imgsz须为正数")
    out, rows = new_output(args.output), []
    for threshold in args.thresholds:
        result = evaluate_det_exact_match(args.weights,args.data_root / "val/images",
                args.data_root / "val/labels",conf=threshold,iou_thresh=args.match_iou,
                batch_size=args.batch,imgsz=args.imgsz,device=args.device)
        rows.append(result)
        save_json(out / "details.json",{"split":"val","weights":str(args.weights),"results":rows})
    with (out / "comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=["conf","match_iou","total","correct","accuracy"],extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"验证集对比结果：{out}。选定阈值后固定参数，再进行测试集评估。")

if __name__ == "__main__":
    main()

