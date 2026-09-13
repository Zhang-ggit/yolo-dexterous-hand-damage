# 基于 YOLO 的灵巧手缺损状态检测

本仓库是《机器人工程实验案例教程》第五章的配套工程，面向机器人工程、自动化、机械和电气专业学生，使用 YOLO26n 检测灵巧手五根手指的位置与缺损状态。仓库已包含数据集、可直接推理的训练权重、训练记录，以及数据检查、推理、评估和训练脚本。

> **学生不训练模型也能完成基础实验。** 默认推理和评估脚本会自动加载 `hand_det_weights/weights/best.pt`。只有“重新训练模型”一节中的命令会更新模型参数。

## 任务定义

每张图像包含一只灵巧手。五根手指分别具有三种可检测状态，共 15 个检测类别：

- 完整：`full_*`
- 轻度缺损（缺一段）：`slightly_damaged_*`
- 重度缺损（缺两段）：`severely_damaged_*`

其中 `*` 为 `thumb`、`index`、`middle`、`ring` 或 `pinky`。

“完全缺失”不是第 16 个检测类别，因为完全缺失的手指没有可标注实体，也没有检测框。后处理程序按以下规则生成固定的五指中文报告：

1. 对每根手指保留置信度达到阈值的候选。
2. 同一手指出现多个候选时，选择置信度最高者；同分时保留模型先返回的候选。
3. 某根手指没有达标候选时，输出“完全缺失”。

因此，“完全缺失”也可能由漏检、遮挡或置信度过低引起，不能脱离原图和真实标签解释。

## 零训练快速开始

以下步骤使用仓库自带的已训练权重，不会训练模型。

### 1. 下载并进入项目

在 GitHub 页面选择 **Code → Download ZIP** 并完整解压，也可以运行：

```bash
git clone https://github.com/Zhang-ggit/yolo-dexterous-hand-damage.git
cd yolo-dexterous-hand-damage
```

后续命令均在仓库根目录执行。脚本根据自身位置解析数据和权重路径，项目放在其他磁盘或含中文、空格的目录中时通常不需要修改源码。

### 2. 安装并验证 Windows 环境

在 **Anaconda Prompt** 中进入项目目录，然后运行：

```bat
setup_windows.bat
test_windows.bat
conda activate hand-yolo26-win
```

`setup_windows.bat` 创建或复用 Python 3.10 环境并安装依赖；`test_windows.bat` 检查 PyTorch、CUDA、NumPy、OpenCV 和 Ultralytics，并使用配套 `best.pt` 完成一次单图推理。两个批处理文件都不会启动训练。

安装脚本面向 Windows 和 NVIDIA 显卡，使用 PyTorch 2.5.1、torchvision 0.20.1 与 CUDA 12.1 构建。仅使用 CPU 时，建议手动创建环境并安装 CPU 构建：

```bat
conda create -n hand-yolo26 python=3.10 -y
conda activate hand-yolo26
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

CUDA 12.1 指 PyTorch 构建携带的运行时版本，不要求另行安装同名 CUDA Toolkit。若显卡需要其他 PyTorch/CUDA 组合，应选择彼此兼容的版本并重新验证环境。

### 3. 运行一次单图检测

```bash
python predict_hand_status.py
```

未指定图片时，脚本自动选择测试集按文件名排序后的第一张图，并输出五根手指的中文状态。

指定图片并保存文本：

```bash
python predict_hand_status.py "hand_data_det/test/images/实际图片.jpg" --output runs/status.txt
```

请把 `实际图片.jpg` 替换为真实文件名。仓库外的图片可以使用绝对路径。

### 4. 批量生成检测结果

先处理前 16 张测试图：

```bash
python predict_det.py --limit 16
```

确认输出正常后处理完整测试集：

```bash
python predict_det.py
```

每次运行会创建新的 `runs/predict*` 目录，其中包含：

- `images/`：带检测框的图像；
- `texts/`：每张图像的五指中文状态；
- `status.csv`：全部图像的状态汇总。

指定其他图像目录、单张图片或推理设备：

```bash
python predict_det.py --source "hand_data_det/val/images"
python predict_det.py --source "其他图片/示例.jpg" --device 0
python predict_det.py --source "其他图片" --device cpu
```

自选图片没有标签时可以推理和展示，但不能据此计算准确率、召回率或 mAP。

## 哪些命令不需要训练

| 命令 | 是否会训练 | 是否可直接使用配套权重 | 说明 |
|---|:---:|:---:|---|
| `setup_windows.bat` | 否 | 是 | 安装环境，不更新模型参数 |
| `test_windows.bat` | 否 | 是 | 环境检查和一次配套权重推理 |
| `python check_dataset.py` | 否 | 不涉及 | 检查图片、标签、类别和重复文件 |
| `python predict_hand_status.py` | 否 | 是 | 单图五指状态输出 |
| `python predict_det.py` | 否 | 是 | 单图或文件夹批量检测 |
| `python test_det.py` | 否 | 是 | 标准检测指标与整图严格匹配评估 |
| `python compare_conf.py ...` | 否 | 是 | 在验证集重复推理并比较置信度阈值 |
| `python plot_results.py` | 否 | 是 | 绘制仓库已有的训练记录，不重新训练 |
| `python test_workflow.py` | 否 | 不涉及 | 检查路径、状态归并与匹配逻辑 |
| `python train_det.py ...` | **是** | 否 | 从通用预训练模型开始训练或微调 |

以下两类命令不能无条件照抄：

- 包含 `runs/train/hand_det/weights/best.pt` 的命令要求该文件已经由学生训练产生；如果只使用配套权重，请省略 `--weights`，或改为 `--weights hand_det_weights/weights/best.pt`。
- Word 中的 `model.train(...)`、`model.predict(...)` 和匹配函数代码是程序原理示例，不是直接粘贴到终端执行的命令。

## 数据检查

```bash
python check_dataset.py
```

脚本检查以下内容：

- 图片是否可读取，图片与标签是否同名对应；
- YOLO 标签是否为 `class_id x_center y_center width height` 五列；
- 类别编号与归一化坐标是否合法；
- 同一图片是否重复标注同一根手指；
- 是否存在空标签或跨数据集完全相同的文件。

当前数据集包含训练集 4000 张、验证集 500 张、测试集 500 张。训练集、验证集和测试集分别有 8、1、0 个空标签。空标签表示确认没有可标注目标，不应删除。跨集合文件检查只能发现字节完全相同的图片，不能排除相近视角或相似场景。

## 使用配套权重评估

直接运行全部评估：

```bash
python test_det.py
```

该命令默认使用 `hand_det_weights/weights/best.pt` 和测试集，同时输出两类结果：

- 标准检测指标：P、R、mAP50、mAP50-95 及评估图；
- 自定义整图严格匹配正确率：预测数与真实数相同，并且全部目标可按类别相同、IoU 达标的条件一一匹配。

分别运行标准评估和严格匹配评估：

```bash
python test_det.py --mode map
python test_det.py --mode exact --conf 0.35 --match-iou 0.5
python test_det.py --mode map --split val
```

标准 mAP 评估内部使用 `conf=0.001` 收集预测以形成 P-R 曲线；命令行参数 `--conf` 只用于整图严格匹配。`--match-iou` 是评估时的框匹配条件，不是预测阶段的 NMS 参数。

结果保存在新的 `runs/evaluate*` 目录。配套权重在 500 张测试图上的记录如下：

| 指标 | 结果 |
|---|---:|
| 标准精确率 P | 0.96425 |
| 标准召回率 R | 0.92944 |
| mAP50 | 0.98015 |
| mAP50-95 | 0.79223 |
| 整图严格匹配正确率 | 387 / 500（77.40%） |

这些结果对应测试集、`imgsz=640`，以及严格匹配的 `conf=0.35`、`match_iou=0.50`。整图严格匹配正确率不是 mAP，也不是五指文本准确率。

### 查看严格匹配错误样例

先生成带错误清单的评估报告：

```bash
python test_det.py --mode exact
```

再将终端打印的实际 `metrics.json` 路径传给：

```bash
python predict_det.py --errors runs/evaluate/metrics.json
```

如果输出目录已自动编号，例如 `runs/evaluate_2`，必须使用该次运行的真实路径。评估和可视化还应保持相同的 `--weights`、`--source`、`--conf` 与 `--imgsz`。

## 无需训练的参数与曲线实验

在验证集比较置信度阈值：

```bash
python compare_conf.py --thresholds 0.25 0.35 0.5
```

该命令不会训练模型，但会对配套权重进行多次验证集推理。它生成 `comparison.csv` 和 `details.json`。阈值应在验证集选择，选定后再固定参数进行一次测试集评估，避免用测试集反复调参。

绘制配套权重已有的训练曲线：

```bash
python plot_results.py
```

默认读取 `hand_det_weights/results.csv`，因此学生无需训练。只有绘制学生自己的训练记录时才需要指定新的 CSV：

```bash
python plot_results.py --csv runs/train/hand_det/results.csv
```

## 可选实验：重新训练模型

这一节中的命令会训练模型。只完成配套权重的基础实验时可以跳过。

训练脚本默认使用 `yolo26n.pt` 作为通用预训练初始化。首次运行可能联网下载该文件；也可以提前放到仓库根目录。先做一轮流程检查：

```bash
python train_det.py --epochs 1 --batch 32 --workers 8 --name smoke
```

RTX 4060 Laptop GPU（约 8 GB）上的完整训练记录使用：

```bash
python train_det.py --epochs 100 --batch 32 --workers 8 --imgsz 640
```

显存不足时优先减小批大小；Windows 数据加载异常时可先将 `workers` 改为 0：

```bash
python train_det.py --epochs 100 --batch 16 --workers 0 --imgsz 640
```

训练输出写入 `runs/train/hand_det*`，重复运行会创建带编号的新目录，不覆盖配套权重。训练结束后以终端打印的实际路径为准。使用新权重时必须显式指定：

```bash
python predict_hand_status.py --weights runs/train/hand_det/weights/best.pt
python test_det.py --weights runs/train/hand_det/weights/best.pt
```

文档中的本机完整训练记录为 100 轮、`batch=32`、`workers=8`、`imgsz=640`，累计训练时间 3985.61 秒（66.43 分钟）；验证集 P、R、mAP50、mAP50-95 分别为 0.98283、0.97337、0.99335、0.93168。训练时间和指标只适用于该次硬件、数据划分与配置，学生应记录自己的实际结果。

## 路径与输出约定

项目的核心目录如下：

```text
仓库目录/
├─ *.py
├─ requirements.txt
├─ hand_det.yaml
├─ hand_data_det/
│  ├─ train/images/  train/labels/
│  ├─ val/images/    val/labels/
│  └─ test/images/   test/labels/
└─ hand_det_weights/
   ├─ weights/best.pt
   └─ results.csv
```

训练和评估脚本会生成 `runs/hand_data_local.yaml`，其中写入当前计算机的数据集绝对路径。`hand_det.yaml` 是供阅读和直接调用 YOLO CLI 时使用的模板；通过本仓库 Python 脚本运行时无需修改它。

数据或权重在仓库外时使用命令行参数：

```bash
python check_dataset.py --data-root "D:/datasets/hand_data_det"
python test_det.py --data-root "D:/datasets/hand_data_det" --weights "D:/models/best.pt"
python predict_det.py --source "D:/images" --weights "D:/models/best.pt"
```

外部权重必须是目标检测模型，并具有与本实验一致的 15 类名称和编号，否则脚本会拒绝加载。

## 文件说明

| 文件 | 用途 |
|---|---|
| `predict_hand_status.py` | 单图推理并输出五指中文状态 |
| `predict_det.py` | 单图或文件夹检测，保存带框图、文本和 CSV |
| `test_det.py` | 标准 mAP 与整图严格匹配评估 |
| `compare_conf.py` | 在验证集比较严格匹配置信度阈值 |
| `plot_results.py` | 从训练 CSV 绘制损失和性能曲线 |
| `train_det.py` | 可选的模型训练入口 |
| `check_dataset.py` | 数据完整性、标签格式和重复文件检查 |
| `experiment_utils.py` | 公共路径、配置、设备和模型加载逻辑 |
| `test_workflow.py` | 不依赖真实推理的规则与路径回归检查 |
| `setup_windows.bat` | 创建 Windows Conda 环境并安装依赖 |
| `test_windows.bat` | 检查环境并运行配套权重单图推理 |

## 常见问题

**为什么运行成功不等于识别正确？** 运行成功只说明环境、路径和接口正常。模型仍可能漏检、误检、混淆类别或产生偏移框，应结合带框图和真实标签分析。

**为什么 mAP 很高，整图严格匹配仍较低？** mAP 按目标和类别汇总；整图严格匹配要求一张图中的所有目标同时满足数量、类别与 IoU 条件，一处错误就会使整张图不通过。

**为什么状态文本与带框图中的目标数不完全一致？** 带框图展示模型保留的检测目标；文本会把同一手指的多个候选归并为一个状态，并为没有达标候选的手指补上“完全缺失”。

**可以把渲染数据上的结果当作真实机器人性能吗？** 不可以。真实摄像机图像还包含材料反射、传感器噪声、运动模糊和照明差异，部署前需要建立独立的真实数据测试集。

**如何快速检查代码逻辑？** 运行：

```bash
python test_workflow.py
```

## 复现实验时应记录

至少保存代码版本、数据划分、权重来源、Python/PyTorch/Ultralytics 版本、设备、`epochs`、`batch`、`imgsz`、`workers`、随机种子、实际输出目录、运行时间和评价条件。比较两组模型时，除权重外应尽量保持输入集合、尺寸、阈值和指标口径一致。

数据与代码的许可仍待项目作者明确；在许可确定前，请勿假设可用于商业发布或再次分发。
