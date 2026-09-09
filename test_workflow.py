"""不依赖YOLO的回归检查：路径移动、中文状态规则、严格匹配和命令行。"""
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from experiment_utils import ROOT
from predict_hand_status import summarize_detections
from test_det import is_exact_match

class WorkflowTests(unittest.TestCase):
    def test_missing_and_conflicting_predictions(self):
        self.assertEqual(summarize_detections([]).count("完全缺失"),5)
        report = summarize_detections([("full_thumb",0.9),("severely_damaged_thumb",0.8),
                                      ("slightly_damaged_index",0.8),("full_middle",0.2)])
        self.assertIn("拇指：完整",report)
        self.assertIn("食指：轻度缺损（缺一段）",report)
        self.assertIn("中指：完全缺失",report)
        self.assertEqual(len(report.splitlines()),5)

    def test_invalid_detection(self):
        with self.assertRaises(ValueError):
            summarize_detections([("unknown",0.9)])
        with self.assertRaises(ValueError):
            summarize_detections([],conf=2)

    def test_matching_requires_reassignment(self):
        # 贪心匹配会失败；必须把第一个预测改配给第二个真实框。
        self.assertTrue(is_exact_match([0,0],[0,0],[[0.9,0.8],[0.7,0.1]],0.5))
        self.assertFalse(is_exact_match([0,0],[0,0],[[0.9,0.1],[0.7,0.1]],0.5))
        self.assertFalse(is_exact_match([0],[1],[[1]],0.5))
        self.assertFalse(is_exact_match([0],[],[],0.5))
        self.assertTrue(is_exact_match([],[],[],0.5))

    def test_relocated_repository(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="移动仓库 ",dir=ROOT / "tmp") as tmp:
            path = Path(tmp)
            shutil.copyfile(ROOT / "experiment_utils.py",path / "experiment_utils.py")
            spec = importlib.util.spec_from_file_location("relocated",path / "experiment_utils.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for split in ("train","val","test"):
                for kind in ("images","labels"):
                    (path / "hand_data_det" / split / kind).mkdir(parents=True)
            config = module.dataset_yaml()
            self.assertEqual(module.ROOT,path.resolve())
            self.assertEqual(module.local_path("hand_data_det"),path.resolve() / "hand_data_det")
            import json
            first_line = config.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(json.loads(first_line.removeprefix("path: ")),(path / "hand_data_det").as_posix())

    def test_cli_help_from_other_directory(self):
        for name in ("train_det","test_det","predict_hand_status","predict_det",
                     "check_dataset","compare_conf","plot_results"):
            result = subprocess.run([sys.executable,str(ROOT / f"{name}.py"),"--help"],
                                    cwd=ROOT.parent,capture_output=True)
            self.assertEqual(result.returncode,0,(name,result.stderr))

if __name__ == "__main__":
    unittest.main()

