# 第五章 基于 YOLO 的灵巧手缺损状态检测

本仓库是《机器人工程实验案例教程》第五章的配套工程，面向机器人工程、自动化、机械和电气专业学生。实验使用 YOLO26n 检测灵巧手各根手指的位置与缺损等级，并生成按拇指、食指、中指、无名指、小指排列的中文状态报告。

教程按“配置环境 → 检查数据 → 训练模型 → 图像检测 → 评价与分析”的顺序展开。完成后，应能解释检测框、类别、置信度和评价指标的含义，并保留可复核的实验结果。

## 1. 实验任务与配套文件

数据由 Blender 渲染生成，训练集、验证集、测试集分别包含 4000、500、500 张图像，每张图像对应一只灵巧手。每根手指有完整、轻度缺损（缺一段）、重度缺损（缺两段）三种可标注状态，共 15 个检测类别，编号与名称见 `hand_det.yaml`。

完全缺失的手指没有实体框，不设第 16 类。文本程序对同一手指保留置信度最高的达标预测；同分时保留先返回的候选，没有达标预测时输出“完全缺失”。漏检或遮挡也可能触发这一输出，因此应结合原图与标注解释结果。本实验完成二维检测与状态识别，不涉及三维重建或机器人闭环控制。

| 文件或目录 | 用途 |
|---|---|
| `hand_data_det/` | 数据集，各子集包含 `images/` 和 `labels/` |
| `hand_det.yaml` | 数据配置与 15 类编号定义 |
| `train_det.py` | 模型训练 |
| `predict_hand_status.py` | 单图五指中文状态报告 |
| `predict_det.py` | 单图或批量检测，保存带框图、文本和 CSV |
| `test_det.py` | 标准检测指标与整图严格匹配评估 |
| `compare_conf.py` | 验证集置信度阈值对比 |
| `plot_results.py` | 读取训练 CSV 并绘制曲线 |
| `check_dataset.py` | 数据完整性与标签检查 |
| `experiment_utils.py` | 公共路径、配置及模型加载逻辑 |
| `test_workflow.py` | 路径与判定规则检查 |
| `hand_det_weights/` | 作者配套权重及原始训练记录 |

## 2. 获取项目与配置环境

### 下载项目

从 [GitHub 仓库](https://github.com/Zhang-ggit/yolo-dexterous-hand-damage) 选择 **Code → Download ZIP** 并完整解压；已安装 Git 时也可使用：

```bat
git clone https://github.com/Zhang-ggit/yolo-dexterous-hand-damage.git
cd yolo-dexterous-hand-damage
```

以下以 Windows 和 Anaconda Prompt 为操作环境。ZIP 解压后，先进入实际项目目录，例如：

```bat
cd /d D:\yolo-dexterous-hand-damage-main
```

后续命令均在项目根目录执行。目录可以更换，无需修改 Python 源码中的数据路径。

### 安装并检查依赖

安装 Anaconda 或 Miniconda 后，在 Anaconda Prompt 中执行：

```bat
setup_windows.bat
test_windows.bat
conda activate hand-yolo26-win
```

安装脚本创建或复用 Python 3.10 环境 `hand-yolo26-win`，安装 PyTorch 2.5.1、torchvision 0.20.1 的 CUDA 12.1 构建，以及 `requirements.txt` 固定的 Ultralytics 8.4.106 等依赖。大安装包保存在 `wheelhouse/`；下载中断后保留 `.part` 文件，重新运行安装脚本可尝试续传。

测试脚本检查软件版本、CUDA、NumPy 与 PyTorch 数据转换及一次配套权重推理。看到检查通过后再开展实验；每次新开终端，先进入项目目录并激活上述环境。不要使用其他环境的 `python.exe` 绝对路径运行脚本。

文档的完整训练平台为 RTX 4060 Laptop GPU（约 8 GB 显存）。所选 PyTorch 构建须与显卡及驱动兼容；仅 CPU 运行较慢，可采用以下独立环境：

```bat
conda create -n hand-yolo26-cpu python=3.10 -y
conda activate hand-yolo26-cpu
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

## 3. 检查数据与标签

保持以下数据结构：

```text
hand_data_det/
├─ train/
│  ├─ images/
│  └─ labels/
├─ val/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/
```

图像与标签通过同名文件对应，例如 `sample.jpg` 和 `sample.txt`。每个手指目标占一行，标签格式为：

```text
class_id x_center y_center width height
```

类别编号为 0–14，后四项为相对于图像宽高归一化的中心坐标、框宽和框高，不是三维坐标。无可标注目标的图像应保留空标签文件。

运行检查：

```bat
python check_dataset.py
```

脚本检查图像可读性、同名标签、类别与坐标范围、同指重复标注和跨集合完全相同的文件，输出 `runs/dataset_check.json`。预期图像数为 4000/500/500，空标签数为 8/1/0。先处理检查报告中的错误，再训练；数值检查通过后仍应抽查图像和标签的语义是否一致。

训练与评估脚本会自动生成包含本机绝对路径的 `runs/hand_data_local.yaml`，无需手动修改教学模板。数据放在其他位置时可传入 `--data-root "D:/datasets/hand_data_det"`；该目录应保留完整的 train、val、test 结构。

## 4. 训练模型与查看曲线

训练从通用预训练权重 `yolo26n.pt` 初始化，再学习本实验的 15 类手指目标。首次使用时可能联网下载该文件，也可提前放到仓库根目录。

先进行一轮流程检查：

```bat
python train_det.py --epochs 1 --batch 16 --workers 0 --name smoke
```

确认数据读取、反向传播和权重保存正常后，执行完整训练。下面显式采用文档中本机完整实验的参数：

```bat
python train_det.py --epochs 100 --batch 32 --workers 8 --imgsz 640 --name hand_det
```

`epochs` 是训练轮数，`batch` 是每批图像数量，`workers` 是数据加载进程数，`imgsz` 是输入尺寸设置。当前脚本的默认值为 100、16、0、640；自动选择首张 CUDA 显卡，无可用 CUDA 时使用 CPU。显存不足时将 batch 降为 16、8 或 4；数据加载异常时先设 `--workers 0`。

输出位于 `runs/train/hand_det`，重复运行会自动增加目录编号。**后续示例中的 `hand_det` 必须替换为终端打印的实际训练目录。** 主要产物为：

- `weights/best.pt`：后续推理和评估使用的最佳权重。
- `weights/last.pt`：最后一轮权重。
- `results.csv`、`results.png`：逐轮记录及曲线。
- `args.yaml`：本次训练配置。

绘制本次训练的损失和验证指标曲线：

```bat
python plot_results.py --csv runs/train/hand_det/results.csv
```

图像保存至终端打印的 `runs/plots*` 目录。观察训练与验证损失、mAP 的变化；一轮短训练只能证明流程可运行，不能代表最终精度。

教材记录的 RTX 4060 Laptop GPU 实验采用 batch=32、workers=8、100 轮，CSV 累计时间为 3985.61 秒（66.43 分钟），末轮验证集 mAP50 与 mAP50-95 为 0.99335、0.93168。这是特定实验的验证成绩，学生应记录自己的耗时与结果，不应将其作为独立测试集成绩。

## 5. 使用训练结果进行检测

后续命令显式指定本次训练的 `best.pt`，使检测结果与训练记录对应。

单图检测默认选取测试集按文件名排序后的第一张图：

```bat
python predict_hand_status.py --weights runs/train/hand_det/weights/best.pt
```

指定图像并保存五指中文报告，将 `实际图片.jpg` 替换为真实文件名：

```bat
python predict_hand_status.py "hand_data_det/test/images/实际图片.jpg" --weights runs/train/hand_det/weights/best.pt --conf 0.35 --output runs/status.txt
```

先批量检查前 16 张测试图，再处理完整测试集：

```bat
python predict_det.py --weights runs/train/hand_det/weights/best.pt --limit 16
python predict_det.py --weights runs/train/hand_det/weights/best.pt
```

输出目录 `runs/predict*` 中，`images/` 保存带框图，`texts/` 保存每图中文状态，`status.csv` 汇总五指状态。带框图保留原始检测结果，文本按同指最高置信度归并，因此两者目标数量可能不同。

使用 `--source "图片或文件夹路径"` 可更换输入，`--device cpu` 或 `--device 0` 可指定设备。自选图像无需标签即可推理，但计算监督评价指标需要真实标签。

## 6. 验证集参数对比与测试集评估

### 在验证集选择置信度阈值

固定同一权重、输入尺寸和匹配 IoU，只比较置信度阈值：

```bat
python compare_conf.py --weights runs/train/hand_det/weights/best.pt --thresholds 0.25 0.35 0.5
```

结果保存在 `runs/compare_conf*` 的 `comparison.csv` 和 `details.json`，比较指标为验证集整图严格匹配正确率。阈值降低可能减少漏检，也可能增加误检。选择完成后固定参数，再进行测试集评价。

### 在测试集生成报告

下面以 conf=0.35、匹配 IoU=0.50、imgsz=640 为例。若验证集选择了其他置信度，请同步替换本节评估与可视化命令中的 `--conf`。

```bat
python test_det.py --weights runs/train/hand_det/weights/best.pt --conf 0.35 --match-iou 0.5 --imgsz 640
```

默认在 test 集同时计算两类指标：

| 指标 | 含义 |
|---|---|
| P、R | 标准检测精确率与召回率 |
| mAP50 | IoU=0.50 条件下的平均精度均值 |
| mAP50-95 | IoU=0.50–0.95、步长 0.05 的平均精度均值 |
| 整图严格匹配正确率 | 全图预测与标签数量相等，且存在类别相同、IoU 达标的一一对应 |

标准评价内部使用 `conf=0.001` 收集预测形成 P-R 曲线；命令行 `--conf` 只影响整图严格匹配。标准 P、R 不能标为“固定 0.35 下的 P、R”。`--match-iou` 是评价匹配条件。整图严格匹配评价原始检测集合，不能改称五指文本准确率。

报告保存至 `runs/evaluate*/metrics.json`，标准评价图在 `map/` 子目录。报告包含权重路径、数据子集、输入尺寸以及严格匹配的阈值与错误清单。仅计算一类指标时，可加 `--mode map` 或 `--mode exact`。

### 查看错误案例

使用上一步终端打印的实际报告路径：

```bat
python predict_det.py --weights runs/train/hand_det/weights/best.pt --errors runs/evaluate/metrics.json --conf 0.35 --imgsz 640
```

重复运行后目录可能变为 `runs/evaluate_2`，应据此修改路径。只运行 `--mode map` 不会生成严格匹配错误清单；上述默认完整评估会生成。若评估的是 val 集，可视化时增加 `--source hand_data_det/val/images`。

结合真实标注检查漏检、背景误检、手指身份混淆、缺损等级混淆、定位偏差和重复预测。文本看似正确时，仍可能因框偏移或多余预测而无法通过整图严格匹配。

## 7. 整理实验报告

报告应包含任务与标签定义、环境及训练配置、损失与验证曲线、五指图文对照、测试指标和典型错误分析。保留对应的 `args.yaml`、`results.csv`、`best.pt` 和 `metrics.json`，注明代码版本、权重来源、子集、尺寸、阈值与实际耗时。

训练验证成绩和独立测试成绩应分别报告。数据来自 Blender，渲染测试集表现不能直接代表真实摄像机或机器人现场表现；扩展到实物时需要另行采集、标注和评价真实图像。

## 补充：直接使用作者已训练模型

完成环境配置后，也可使用仓库自带的 `hand_det_weights/weights/best.pt`，跳过训练，直接运行：

```bat
python predict_hand_status.py
python predict_det.py --limit 16
python test_det.py
python plot_results.py
```

前三条默认加载作者配套权重，最后一条绘制 `hand_det_weights/results.csv` 中已有的训练记录。需要检测自己的图片时，给单图脚本传入图片路径，或给批量脚本增加 `--source`。

教材记录的配套权重测试集 mAP50 为 0.98015、mAP50-95 为 0.79223，整图严格匹配为 387/500（77.40%，conf=0.35、匹配 IoU=0.50、imgsz=640）。这组结果与前述学生自训模型的验证成绩属于不同权重和评价子集。
