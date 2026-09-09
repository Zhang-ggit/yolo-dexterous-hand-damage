"""读取训练results.csv，绘制损失曲线与检测性能曲线。"""
import argparse
import csv
from experiment_utils import ROOT, local_path, new_output

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",type=local_path,default=ROOT / "hand_det_weights/results.csv")
    p.add_argument("--output",type=local_path,default=ROOT / "runs/plots")
    args = p.parse_args()
    with args.csv.open(encoding="utf-8-sig",newline="") as f:
        rows = [{k.strip():v for k,v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        raise ValueError("CSV没有训练记录")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = new_output(args.output)
    epochs = [float(row["epoch"]) for row in rows]
    for label,prefix in (("loss","train/"),("metrics","metrics/")):
        fig,ax = plt.subplots(figsize=(9,5))
        for key in rows[0]:
            if (label == "loss" and key.endswith("loss") and key.startswith(("train/","val/"))) or (label == "metrics" and key.startswith(prefix)):
                ax.plot(epochs,[float(row[key]) for row in rows],label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{label}.png",dpi=180)
        plt.close(fig)
    print(f"曲线保存至：{out}")

if __name__ == "__main__":
    main()

