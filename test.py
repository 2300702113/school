import unittest


from logic_core import BusinessLogicAnalyzer


# 🛠️ 1. 构造 YOLO 数据环境
class MockTensor:
    def __init__(self, data):
        self.data = data
    def cpu(self): return self
    def numpy(self): return self.data

class MockBox:
    def __init__(self, cls_id, coords=None):
        self.cls = [cls_id]
        self.xyxy = [MockTensor(coords if coords else [0, 0, 0, 0])]

class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes


# =========================================================
# 🧪 2. 编写全覆盖测试用例
# =========================================================
class TestCompleteBusinessLogic(unittest.TestCase):

    def setUp(self):
        """测试前置准备：定义不同模型的 names 字典"""
        self.helmet_dict = {0: 'person', 1: 'helmet'}
        self.bad_helmet_dict = {0: 'car', 1: 'dog'} # 缺少 person 和 helmet 的坏字典
        self.behavior_dict = {
            0: 'hand-raising', 1: 'reading', 2: 'writing', 
            3: 'using phone', 4: 'bowing the head', 5: 'leaning over the table'
        }

    # ---------------- 模块 A: 路由分发器测试 ----------------
    def test_router_dispatch(self):
        """测试语句：验证各种 method_name 是否能正确路由"""
        results = [MockResult(boxes=[])]
        
        # 1. 路由不存在的方法
        msg, status = BusinessLogicAnalyzer.analyze("不存在的方法", results, {})
        self.assertEqual(status, "INFO")
        
        # 2. 路由正确的三个方法 (这里只测路由通不通，不测具体逻辑)
        _, s1 = BusinessLogicAnalyzer.analyze("头盔佩戴规范检测 (交集判定法)", results, self.helmet_dict)
        _, s2 = BusinessLogicAnalyzer.analyze("单人行为专注度分析", results, self.behavior_dict)
        _, s3 = BusinessLogicAnalyzer.analyze("多人课堂行为统计", results, self.behavior_dict)
        self.assertIn(s1, ["INFO", "WARNING", "DANGER", "SUCCESS"])
        self.assertIn(s2, ["INFO", "WARNING", "DANGER", "SUCCESS"])
        self.assertIn(s3, ["INFO", "WARNING", "DANGER", "SUCCESS"])

    # ---------------- 模块 B: 二维平面交集算法测试 ----------------
    def test_intersection_conditions(self):
        """测试条件：覆盖交集算法的全部逻辑分支 (T/T, F/T, T/F)"""
        boxA = [0, 0, 100, 100]
        # 条件 1: 宽>0 且 高>0 -> True
        self.assertTrue(BusinessLogicAnalyzer.check_intersection(boxA, [50, 50, 150, 150]))
        # 条件 2: 宽<=0 -> False (X轴分离)
        self.assertFalse(BusinessLogicAnalyzer.check_intersection(boxA, [150, 0, 250, 100]))
        # 条件 3: 宽>0 且 高<=0 -> False (Y轴分离)
        self.assertFalse(BusinessLogicAnalyzer.check_intersection(boxA, [0, 150, 100, 250]))

    # ---------------- 模块 C: 头盔佩戴检测测试 ----------------
    def test_helmet_missing_classes(self):
        """测试条件：模型字典里根本没有人和头盔"""
        msg, status = BusinessLogicAnalyzer._analyze_helmet([MockResult([])], self.bad_helmet_dict)
        self.assertEqual(status, "WARNING")
        self.assertIn("未找到", msg)

    def test_helmet_no_person(self):
        """测试条件：画面里没人"""
        msg, status = BusinessLogicAnalyzer._analyze_helmet([MockResult([])], self.helmet_dict)
        self.assertEqual(status, "INFO")
        self.assertIn("未检测到人员", msg)

    def test_helmet_safe_and_violation(self):
        """测试条件：画面里2个人，1个戴了，1个没戴"""
        boxes = [
            MockBox(0, [0, 0, 100, 100]),     # 人 A
            MockBox(1, [20, 0, 80, 50]),      # 头盔 (在人A头上)
            MockBox(0, [200, 200, 300, 300])  # 人 B (没戴头盔)
        ]
        msg, status = BusinessLogicAnalyzer._analyze_helmet([MockResult(boxes)], self.helmet_dict)
        self.assertEqual(status, "DANGER")    # 只要有1个没戴，就是 DANGER
        self.assertIn("发现 1 人未佩戴", msg)

    def test_helmet_all_safe(self):
        """测试条件：画面所有人全部佩戴"""
        boxes = [
            MockBox(0, [0, 0, 100, 100]), 
            MockBox(1, [20, 0, 80, 50])
        ]
        msg, status = BusinessLogicAnalyzer._analyze_helmet([MockResult(boxes)], self.helmet_dict)
        self.assertEqual(status, "SUCCESS")

    # ---------------- 模块 D: 单人行为分析测试 ----------------
    def test_single_behavior_no_person(self):
        """测试分支：单人分析 - 无人"""
        msg, status = BusinessLogicAnalyzer._analyze_behavior([MockResult([])], self.behavior_dict)
        self.assertEqual(status, "INFO")

    def test_single_behavior_too_many(self):
        """测试分支：单人分析 - 超过1人报警"""
        boxes = [MockBox(1), MockBox(2)] # 2个人
        msg, status = BusinessLogicAnalyzer._analyze_behavior([MockResult(boxes)], self.behavior_dict)
        self.assertEqual(status, "WARNING")
        self.assertIn("检测到多人", msg)

    def test_single_behavior_success_and_unknown(self):
        """测试分支：单人分析 - 成功识别与未知动作处理"""
        # 正常识别
        msg1, s1 = BusinessLogicAnalyzer._analyze_behavior([MockResult([MockBox(1)])], self.behavior_dict)
        self.assertEqual(s1, "SUCCESS")
        self.assertIn("reading", msg1)

        # 未知动作ID (比如模型输出了字典里没有的分类 99)
        msg2, s2 = BusinessLogicAnalyzer._analyze_behavior([MockResult([MockBox(99)])], self.behavior_dict)
        self.assertIn("未知动作", msg2)

    # ---------------- 模块 E: 多人课堂统计测试 (🆕 新增功能测试) ----------------
    def test_group_behavior_no_person(self):
        """测试分支：多人统计 - 画面无学生"""
        msg, status = BusinessLogicAnalyzer._analyze_group_behavior([MockResult([])], self.behavior_dict)
        self.assertEqual(status, "INFO")
        self.assertIn("未检测到学生", msg)

    def test_group_behavior_safe_class(self):
        """测试分支：多人统计 - 班级纪律良好 (无人违规)"""
        # 3个 hand-raising, 2个 reading
        boxes = [MockBox(0), MockBox(0), MockBox(0), MockBox(1), MockBox(1)]
        msg, status = BusinessLogicAnalyzer._analyze_group_behavior([MockResult(boxes)], self.behavior_dict)
        self.assertEqual(status, "INFO")
        self.assertIn("hand-raising: 3人", msg)
        self.assertIn("reading: 2人", msg)

    def test_group_behavior_warning_class(self):
        """测试分支：多人统计 - 出现玩手机、低头等违纪动作"""
        # 1个 reading, 1个 using phone (ID=3), 1个未知动作(ID=99)
        boxes = [MockBox(1), MockBox(3), MockBox(99)]
        msg, status = BusinessLogicAnalyzer._analyze_group_behavior([MockResult(boxes)], self.behavior_dict)
        self.assertEqual(status, "WARNING")  # 因为有 using phone，必须触发 WARNING
        self.assertIn("using phone: 1人", msg)
        self.assertIn("未知动作: 1人", msg)


if __name__ == '__main__':
    # 启动单元测试，并在控制台打印详细信息 (verbosity=2)
    unittest.main(verbosity=2)