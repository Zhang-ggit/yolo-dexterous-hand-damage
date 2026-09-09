"""训练灵巧手缺损检测模型。完整仓库下载后默认路径无需修改。"""
import argparse
from experiment_utils import ROOT, DATA, local_path, dataset_yaml, select_device

def main():
    p = argparse.ArgumentParser(description=__doc__)
    # 数据移到仓库外时传入 --data-root，不需要修改脚本。
    p.add_argument("--data-root", type=local_path, default=DATA)
    p.add_argument("--model", default="yolo26n.pt", help="首次使用默认模型需联网下载；也可指定本地权重")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16, help="默认16；显存不足时可改为8或4")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="默认单GPU，无CUDA时CPU；服务器可指定0,1")
    p.add_argument("--workers", type=int, default=0, help="默认0便于Windows运行")
    p.add_argument("--project", type=local_path, default=ROOT / "runs/train")
    p.add_argument("--name", default="hand_det")
    args = p.parse_args()
    if min(args.epochs, args.batch, args.imgsz) <= 0 or args.workers < 0:
        p.error("epochs、batch、imgsz须为正数，workers须非负")
    config = dataset_yaml(args.data_root)
    from ultralytics import YOLO
    device = select_device(args.device)
    model_path = local_path(args.model)
    if not model_path.is_file() and args.model != "yolo26n.pt":
        raise FileNotFoundError(f"初始权重不存在：{model_path}")
    print(f"设备：{device}；数据配置：{config}")
    model = YOLO(str(model_path), task="detect")
    model.train(data=str(config), task="detect", imgsz=args.imgsz, epochs=args.epochs,
                batch=args.batch, device=device, project=str(args.project), name=args.name,
                workers=args.workers, amp=device != "cpu", exist_ok=False, cache=False,
                save=True, plots=True, seed=0)
    print(f"训练完成，最佳权重：{model.trainer.save_dir / 'weights/best.pt'}")

if __name__ == "__main__":
    main()

