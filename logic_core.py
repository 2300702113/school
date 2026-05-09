import cv2
from ultralytics import YOLO

class VisionDataLayer:
    """数据交互层：负责底层资源调配 (YOLO模型、OpenCV视频流)"""
    def __init__(self):
        self.model = None
        self.cap = None

    def load_model(self, model_path):
        self.model = YOLO(model_path)
        return self.model.names

    def load_video(self, video_path):
        self.release_video()
        self.cap = cv2.VideoCapture(video_path)
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.isOpened() else 0

    def load_webcam(self, camera_id=0):
        self.release_video()
        self.cap = cv2.VideoCapture(camera_id)
        return self.cap.isOpened()

    def read_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            return ret, frame, current_frame
        return False, None, 0

    def set_frame_position(self, frame_position):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

    def predict(self, source, conf_threshold, target_classes):
        if self.model:
            return self.model.predict(source=source, conf=conf_threshold, classes=target_classes, verbose=False)
        return None

    def release_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class BusinessLogicAnalyzer:
    """业务逻辑层：负责算法判定和规则过滤"""
    
    @staticmethod
    def check_intersection(boxA, boxB):
        """核心算法：二维平面交集判断"""
        overlap_x_min = max(boxA[0], boxB[0])
        overlap_y_min = max(boxA[1], boxB[1])
        overlap_x_max = min(boxA[2], boxB[2])
        overlap_y_max = min(boxA[3], boxB[3])
        return (overlap_x_max - overlap_x_min) > 0 and (overlap_y_max - overlap_y_min) > 0
    
    @classmethod
    def analyze(cls, method_name, results, class_names_dict):
        """路由分发：根据选定的业务方法，返回 (文本提示, 状态级别)"""
        if method_name == "头盔佩戴规范检测 (交集判定法)":
            return cls._analyze_helmet(results, class_names_dict)
        elif method_name == "单人行为专注度分析":
            return cls._analyze_behavior(results, class_names_dict)
        elif method_name == "多人课堂行为统计": # 🆕 增加路由分发
            return cls._analyze_group_behavior(results, class_names_dict)
            
        return "未知业务逻辑", "INFO"



    @classmethod
    def _analyze_helmet(cls, results, names_dict):
        person_id, helmet_id = -1, -1
        for k, v in names_dict.items():
            name_lower = v.lower()
            if 'person' in name_lower or '人' in name_lower: person_id = k
            elif 'helmet' in name_lower or '帽' in name_lower: helmet_id = k

        if person_id == -1 or helmet_id == -1:
            return "⚠️ 警告：当前模型未找到 'person' 或 'helmet' 类别！", "WARNING"

        persons_boxes, helmets_boxes = [], []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy()
            if cls_id == person_id: persons_boxes.append(coords)
            elif cls_id == helmet_id: helmets_boxes.append(coords)

        total_persons = len(persons_boxes)
        no_helmet_count = sum(1 for p_box in persons_boxes if not any(cls.check_intersection(p_box, h_box) for h_box in helmets_boxes))

        if total_persons == 0:
            return "🟢 待机中: 画面中未检测到人员", "INFO"
        elif no_helmet_count > 0:
            return f"🚨 违规警报: 画面总人数 {total_persons} 人 | 发现 {no_helmet_count} 人未佩戴安全帽！", "DANGER"
        else:
            return f"✅ 状态合规: 画面总人数 {total_persons} 人 | 全员穿戴规范", "SUCCESS"

    @classmethod
    def _analyze_behavior(cls, results, names_dict):
        num_people = len(results[0].boxes)
        if num_people == 0:
            return "🟢 待机中: 画面中未检测到学生", "INFO"
        elif num_people > 1:
            return f"⚠️ 警告: 检测到多人 ({num_people}人)，请确保画面中只有单名学生！", "WARNING"
        else:
            cls_id = int(results[0].boxes[0].cls[0])
            action_name = names_dict.get(cls_id, "未知动作")
            return f"✅ 行为分析: 当前学生正在 【{action_name}】", "SUCCESS"
        
    @classmethod
    def _analyze_group_behavior(cls, results, names_dict):
        """🆕 业务逻辑 C：多人课堂行为分类统计"""
        boxes = results[0].boxes
        num_people = len(boxes)

        if num_people == 0:
            return "🟢 待机中: 画面中未检测到学生", "INFO"

        # 1. 使用字典来统计每个动作的人数
        action_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            action_name = names_dict.get(cls_id, "未知动作")
            # 如果字典里没有这个动作，默认为0并加1；如果有，直接加1
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

        # 2. 将统计结果拼接成友好的字符串
        details = []
        for action, count in action_counts.items():
            details.append(f"{action}: {count}人")
            
        # 比如拼接成 "reading: 3人 | writing: 2人"
        details_str = " | ".join(details) 

        # 3. 可以在这里加个彩蛋：如果有人玩手机或低头，改变状态栏颜色报警
        warning_actions = ['using phone', 'bowing the head', 'leaning over the table']
        has_warning = any(action in action_counts for action in warning_actions)

        if has_warning:
            return f"⚠️ 课堂提醒: 总计 {num_people} 人 -> {details_str}", "WARNING"
        else:
            return f"📊 课堂统计: 总计 {num_people} 人 -> {details_str}", "INFO"