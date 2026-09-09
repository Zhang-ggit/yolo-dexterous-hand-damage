# 第五章 基于YOLO的灵巧手缺损状态检测

本仓库用于《机器人工程实验案例教程》第五章，面向大二下及以上学生。
仓库地址：https://github.com/Zhang-ggit/yolo-dexterous-hand-damage
在仓库页面选择Code → Download ZIP并完整解压，或使用Git克隆整个仓库。
图像由Blender渲染，训练/验证/测试集分别为4000/500/500张。
每张图按一只灵巧手处理。五根手指分别具有完整、轻度缺损（缺一段）、
重度缺损（缺两段）三种可标注状态，共15个检测类别。

实验约定：某手指没有置信度达标的检测结果，即输出“完全缺失”。
完全缺失没有检测框，不作为第16类训练。相同手指存在多个预测时，
选置信度最高者；同分时选模型返回的第一个。
因此漏检、遮挡也可能被本实验规则转换为“完全缺失”，需在误差分析中说明。

## 1. 文件说明

| 文件 | 用途 |
|---|---|
| train_det.py | 训练，保留原文件名，已移除服务器路径与强制双GPU |
| test_det.py | 原严格匹配评估，以及新增的标准mAP评估 |
| predict_hand_status.py | 输入单图，输出五指中文状态 |
| predict_det.py | 单图/文件夹检测，保存带框图片、状态文本与汇总CSV |
| check_dataset.py | 检查图片、标签、类别统计及跨集合相同文件 |
| compare_conf.py | 在验证集比较置信度阈值 |
| plot_results.py | 将训练CSV绘制为损失和性能曲线 |
| experiment_utils.py | 公共路径、配置和模型加载逻辑 |
| test_workflow.py | 不依赖YOLO的路径及判定规则回归检查 |
| hand_det.yaml | 供教学阅读的数据配置模板 |
| hand_det_weights/weights/best.pt | 作者提供的已训练权重 |
| hand_det_weights/results.csv | 作者原始训练记录 |

目录结构保持如下即可，无需修改代码路径：

    仓库目录/
      *.py
      requirements.txt
      hand_det.yaml
      hand_data_det/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/
      hand_det_weights/
        weights/best.pt
        results.csv

## 2. 环境准备

Windows快速测试：打开Anaconda Prompt，进入仓库目录，先执行
`setup_windows.bat`，成功后执行`test_windows.bat`。
这会创建或复用独立环境hand-yolo26-win，安装依赖并检测NumPy/PyTorch转换、
GPU计算及配套权重的单图推理。脚本失败会停止，便于定位错误。
脚本不自动启动训练；测试成功后可执行：

    conda run --no-capture-output -n hand-yolo26-win python train_det.py --epochs 1 --batch 16 --name windows_smoke

Windows安装脚本会先通过download_windows.bat从PyTorch官方站点下载两个Windows/Python3.10
安装包到wheelhouse目录，再本地安装。下载中的文件以.part结尾；失败后再次运行setup_windows.bat
会尝试从断点续传，已经完成的.whl文件直接复用。首次从旧版pip临时下载切换到这一流程时，
不会自动复用pip临时目录中的半成品。其余小型依赖仍需联网安装。
可单独运行download_windows.bat下载这两个大文件。此功能提高中断后的可恢复性，不保证下载提速。

以作者服务器的Python 3.10环境为基准，使用独立的教学环境。
下面是从服务器清单筛选的安装方案，已由用户在Windows完成单图推理与1轮训练验证；
完整测试集评估及batch=16训练仍待验证。
推荐在新环境安装，不要求更改作者已有的yolo26环境。

    conda create -n hand-yolo26 python=3.10 -y
    conda activate hand-yolo26

按作者的PyTorch 2.5.1 / torchvision 0.20.1 / CUDA 12.1组合安装，
以下命令适用于支持该构建的Windows/Linux与NVIDIA显卡：

    python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

这组版本对应[PyTorch官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。
CUDA 12.1是此PyTorch构建使用的运行时版本，不要求照抄原环境里额外安装的CUDA 13工具包。
显卡必须受所选构建支持；如果学生电脑需要更新的CUDA/PyTorch，应另行选择配套版本并修改
requirements.txt中torch与torchvision两行，重新验证，不应强行套用这组旧版本。

仅CPU环境使用以下命令替代上述cu121安装命令：

    python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

随后在仓库目录执行（其余底层依赖由pip自动解析安装）：

    python -m pip install -r requirements.txt
    python -m pip check
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

作者提供的原环境记录：Python 3.10.20、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、
Ultralytics 8.4.106、cuDNN 9.1.0、两张RTX 4090。权重元数据中的Ultralytics版本一致。
requirements.txt固定核心与主要运行依赖，不是全部传递依赖的跨平台锁定文件。

本实验不处理音频，不需要torchaudio；不手工指定cuda-toolkit、CUDA 13包、Triton、
NCCL等底层组件，由所选PyTorch构建按平台管理所需依赖。不要照搬整个服务器pip清单。

原环境同时存在opencv-python 5.0.0.93和opencv-python-headless 4.13.0.92。
[OpenCV包说明](https://pypi.org/project/opencv-python/)要求同一环境只选一个cv2提供包。
仅凭安装清单无法判断实际加载版本。教学候选环境统一选择opencv-python 4.13.0.92，
用于学生桌面环境，并避免与Ultralytics对opencv-python的依赖重复安装。
这是有意调整，需要实际验证，不等同于声称原训练使用了这一版本。

[YOLO26官方说明](https://docs.ultralytics.com/models/yolo26/)介绍了本实验使用的模型。
离线推理需提前安装依赖并下载配套best.pt；首次训练默认模型还需联网下载yolo26n.pt，
也可预先把它放在仓库根目录。

## 3. 路径为什么不必修改

脚本通过自身文件位置确定仓库根目录，不依赖终端当前目录或作者服务器目录。
训练和评估会自动生成runs/hand_data_local.yaml，写入当前电脑的数据集绝对路径。
移动整个仓库后再次运行会重新生成，无需手动维护。请不要并发启动使用不同数据根目录的
训练/评估任务，因为它们共享这一份自动生成的配置文件。

所有命令行中的相对文件路径也按仓库根目录解释，支持空格和中文路径（有空格时加引号）。
如果数据或权重放在仓库外，使用--data-root或--weights传入绝对路径。
这些参数在代码中有注释，不必修改源码。

hand_det.yaml是教学模板。若绕过Python脚本，直接使用yolo命令，
需要按模板注释将path改为本机数据集绝对路径。
原hand_det_weights/args.yaml中的服务器路径仅保留为历史训练记录，本项目脚本不使用它。

## 4. 基础实验：已有权重检测

先检查数据：

    python check_dataset.py

无参数运行时，单图脚本自动使用测试集第一张图片：

    python predict_hand_status.py

指定图片并保存文本（把图片路径替换为实际图片）：

    python predict_hand_status.py "hand_data_det/test/images/实际图片.jpg" --output runs/status.txt

检查前16张测试图，生成带框图、每图五指状态文本、status.csv：

    python predict_det.py --limit 16

处理完整测试集：

    python predict_det.py

通过--source可指定其他图片目录或单张图片。图中的类别名使用权重中的英文名称，
文本和CSV输出中文五指状态。可视化展示模型原始检测框，文本按最高置信度归并为每指一个状态。

## 5. 训练实验

    python train_det.py

默认：YOLO26n、640像素、100轮、batch=16、workers=0。
自动使用首张CUDA显卡，否则使用CPU。CPU能运行但训练速度需实测。
默认批大小是起点，不保证适配所有游戏本；出现显存不足时减小：

    python train_det.py --batch 4

先做1轮流程验证：

    python train_det.py --epochs 1 --batch 16 --name smoke

服务器双GPU可手动指定：

    python train_det.py --device 0,1 --batch 64 --workers 8

新训练结果放在runs/train/hand_det，重复运行由YOLO自动增加目录编号，
不会覆盖作者提供的权重。训练结束会打印实际best.pt路径。
后续使用自训权重时，根据打印结果指定--weights，例如：

    python predict_hand_status.py --weights runs/train/hand_det/weights/best.pt
    python test_det.py --weights runs/train/hand_det/weights/best.pt

默认评估始终使用作者提供的best.pt，不会自动切换到最新训练结果。
作者报告服务器训练约23分钟；学生电脑时间需记录硬件、批大小和实际耗时后填写。

## 6. 评估与错误分析

    python test_det.py

默认在test集输出两类指标：
- 标准精确率、召回率、mAP50、mAP50-95及评估图。
- 自定义整图严格匹配正确率：预测与真实框数量相等、类别一致，
  并存在IoU达到阈值的一一对应匹配。保留原脚本的匹配算法。

标准mAP评估使用conf=0.001收集预测以形成PR曲线；--conf默认0.35，
仅用于严格匹配评估。标准评估的P/R由Ultralytics定义，不应说成固定0.35下的P/R。
--match-iou默认0.5，是严格匹配条件，不是预测时的NMS参数。
整图严格匹配正确率不是mAP，也不是逐指状态正确率。

    python test_det.py --mode map
    python test_det.py --mode exact --conf 0.35 --match-iou 0.5
    python test_det.py --split val

结果保存至runs/evaluate（再次运行增加后缀）内的metrics.json；
标准指标图在其map子目录，严格匹配错误列表保存在JSON中。
检查错误图片（按实际生成目录替换报告路径）：

    python predict_det.py --errors runs/evaluate/metrics.json

如评估使用了自训权重、验证集或其他置信度，可视化时也传入相同--weights、--source和--conf。
例如验证集评估须配--source hand_data_det/val/images。
文件缺少标签会报错；无目标图应有空标签文件，不应删除它们。

## 7. 参数对比与训练曲线

置信度对比固定使用val集，避免用测试集调参：

    python compare_conf.py --thresholds 0.25 0.35 0.5

对每个阈值重新运行严格匹配评估，生成comparison.csv与details.json。
选定阈值后固定它，再用test_det.py进行一次最终测试集评估。

绘制作者的训练曲线：

    python plot_results.py

绘制自己的训练记录：

    python plot_results.py --csv runs/train/hand_det/results.csv

训练和标准评估本身也会生成结果图。实验报告可记录训练设置、
训练时间、标准指标、严格匹配正确率及典型错误案例。

## 8. 当前验证与发布说明

已对5000张图片进行可读性、标签格式、类别与同指重复标注检查。
训练集有8个空标签、验证集1个、测试集0个；空标签按无可见标注目标处理。
未发现完全相同文件跨集合重复；这不等同于检查了相近视角或同场景数据泄漏。

本地逻辑检查：

    python test_workflow.py

用户提供的Windows日志已确认：RTX 4060 Laptop GPU（8GB）、Python 3.10.21、
PyTorch 2.5.1+cu121、Ultralytics 8.4.106、OpenCV 4.13.0环境下，NumPy转换、
GPU计算、配套权重单图推理及1轮训练成功。该次训练batch=4，训练器报告0.034小时，
这是流程验证，不代表100轮训练后的检测精度。按用户要求，默认训练批大小现改为16；
batch=16的速度和显存占用尚未实测，完整test_det.py评估尚待验证。

.gitignore已排除临时文件、新实验输出、缓存和中间权重，
保留数据集与配套best.pt。数据/代码许可尚待作者确定，下载入口见本文开头。

