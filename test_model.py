import sys
import os
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                               QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                               QSlider, QInputDialog, QMessageBox, QGroupBox, 
                               QCheckBox, QScrollArea, QComboBox) # 🆕 引入 QComboBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from ultralytics import YOLO

class modelDetectionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能视觉通用识别系统 v3.0 (业务集成版)")
        self.resize(1200, 800) # 略微放大窗口以适应新布局

        self.CONF_THRESHOLD = 0.5  
        self.model = None 
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame) 
        self.is_paused = False
        
        self.class_checkboxes = {}

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ================= 左侧：综合显示面板 (🆕 重新布局为垂直结构) =================
        left_panel = QWidget()
        left_v_layout = QVBoxLayout(left_panel)
        left_v_layout.setContentsMargins(0, 0, 0, 0) 

        # 🆕 1. 顶部：方法选择栏
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("🚀 当前业务逻辑:"))
        self.method_selector = QComboBox()


        self.method_selector.addItems([
            "头盔佩戴规范检测 (交集判定法)", 
            "单人行为专注度分析" # 🆕 新增的行为检测方法
        ])


        self.method_selector.setStyleSheet("font-size: 15px; padding: 5px; font-weight: bold;")
        header_layout.addWidget(self.method_selector, stretch=1)
        left_v_layout.addLayout(header_layout)

        # 2. 中部：画面显示区
        self.display_label = QLabel("请先在右侧加载 AI 模型 (.pt 文件)...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #1e1e1e; color: #a0a0a0; font-size: 24px; font-weight: bold; border-radius: 5px;")
        left_v_layout.addWidget(self.display_label, stretch=1) 

        # 3. 进度条
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False) 
        self.video_slider.sliderPressed.connect(self.slider_pressed)   
        self.video_slider.sliderReleased.connect(self.slider_released) 
        left_v_layout.addWidget(self.video_slider)

        # 🆕 4. 底部：状态信息播报栏
        self.status_bar = QLabel("系统就绪 | 等待接入视频源...")
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("""
            background-color: #2c3e50; color: white; 
            font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;
        """)
        left_v_layout.addWidget(self.status_bar)

        main_layout.addWidget(left_panel, stretch=5) 

        # ================= 右侧：控制面板区 (保持原样) =================
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15) 

        btn_style = """
            QPushButton {
                background-color: #2b5c8f; color: white; border-radius: 8px; 
                font-size: 16px; font-weight: bold; padding: 12px;
            }
            QPushButton:hover { background-color: #3b7cbf; }
            QPushButton:disabled { background-color: #555555; color: #888888; } 
        """

        self.lbl_model_status = QLabel("当前模型: 未加载 ❌")
        self.lbl_model_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c;")
        right_layout.addWidget(self.lbl_model_status)

        self.btn_load_model = QPushButton("📂 选择 AI 模型 (.pt)")
        self.btn_load_model.setStyleSheet(btn_style.replace("#2b5c8f", "#8e44ad").replace("#3b7cbf", "#9b59b6")) 
        self.btn_load_model.clicked.connect(self.select_model)
        right_layout.addWidget(self.btn_load_model)

        self.class_group = QGroupBox("🎯 指定检测类别")
        self.class_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 2px solid #bdc3c7; border-radius: 8px; margin-top: 15px; background-color: #ffffff;}
            QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #2c3e50; }
        """)
        group_layout = QVBoxLayout()
        toggle_layout = QHBoxLayout()
        
        btn_toggle_style = """
            QPushButton { background-color: #ecf0f1; color: #2c3e50; border-radius: 4px; font-size: 13px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; }
            QPushButton:hover { background-color: #d5dbdb; }
        """
        self.btn_select_all = QPushButton("✅ 全选")
        self.btn_select_all.setStyleSheet(btn_toggle_style)
        self.btn_select_all.clicked.connect(self.select_all_classes)
        self.btn_deselect_all = QPushButton("❌ 全不选")
        self.btn_deselect_all.setStyleSheet(btn_toggle_style)
        self.btn_deselect_all.clicked.connect(self.deselect_all_classes)
        toggle_layout.addWidget(self.btn_select_all)
        toggle_layout.addWidget(self.btn_deselect_all)
        group_layout.addLayout(toggle_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background-color: transparent;")
        
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: transparent;")
        self.class_layout = QVBoxLayout(self.scroll_content)
        self.class_layout.setSpacing(2) 
        
        self.scroll_area.setWidget(self.scroll_content)
        group_layout.addWidget(self.scroll_area)
        self.class_group.setLayout(group_layout)
        right_layout.addWidget(self.class_group, stretch=1) 

        self.lbl_conf = QLabel(f"当前置信度阈值: {self.CONF_THRESHOLD}")
        self.lbl_conf.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(self.lbl_conf)

        self.btn_conf = QPushButton("⚙️ 设置置信度")
        self.btn_conf.setStyleSheet(btn_style.replace("#2b5c8f", "#d35400").replace("#3b7cbf", "#e67e22")) 
        self.btn_conf.clicked.connect(self.set_confidence)
        right_layout.addWidget(self.btn_conf)

        self.btn_video = QPushButton("🎬 导入监控视频")
        self.btn_video.setStyleSheet(btn_style)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_video.setEnabled(False)
        right_layout.addWidget(self.btn_video)

        self.btn_pause = QPushButton("⏸️ 暂停播放")
        self.btn_pause.setStyleSheet(btn_style)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setEnabled(False) 
        right_layout.addWidget(self.btn_pause)

        self.btn_image = QPushButton("🖼️ 导入单张图片")
        self.btn_image.setStyleSheet(btn_style)
        self.btn_image.clicked.connect(self.load_image)
        self.btn_image.setEnabled(False)
        right_layout.addWidget(self.btn_image)

        self.btn_webcam = QPushButton("📷 开启本地摄像头")
        self.btn_webcam.setStyleSheet(btn_style)
        self.btn_webcam.clicked.connect(self.load_webcam)
        self.btn_webcam.setEnabled(False)
        right_layout.addWidget(self.btn_webcam)

        main_layout.addLayout(right_layout, stretch=1)


        # 依次给下面的所有按钮都加上这一行：
        self.btn_select_all.setFocusPolicy(Qt.NoFocus)
        self.btn_deselect_all.setFocusPolicy(Qt.NoFocus)
        self.btn_conf.setFocusPolicy(Qt.NoFocus)
        self.btn_video.setFocusPolicy(Qt.NoFocus)
        self.btn_pause.setFocusPolicy(Qt.NoFocus)
        self.btn_image.setFocusPolicy(Qt.NoFocus)
        self.btn_webcam.setFocusPolicy(Qt.NoFocus)
        
        # 甚至连下拉框最好也加上
        self.method_selector.setFocusPolicy(Qt.NoFocus)

        # ================= 🆕 终极焦点接管 =================
        # 让主窗口自己变成一个可以接收键盘事件的实体，并在开局强行霸占焦点！
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    # ---------------- 🆕 全局键盘事件拦截 ----------------
    def keyPressEvent(self, event):
        """监听键盘按键，实现快捷键功能"""
        # 如果按下的是空格键
        if event.key() == Qt.Key_Space:
            # 只有当暂停按钮处于激活状态（说明视频正在播或已暂停）时，空格键才生效
            if self.btn_pause.isEnabled():
                self.toggle_pause()
        else:
            # 如果按的不是空格键，就按照默认逻辑处理
            super().keyPressEvent(event)

    # ---------------- UI 组件交互逻辑 ----------------
    def select_all_classes(self):
        for cb in self.class_checkboxes.values(): cb.setChecked(True)

    def deselect_all_classes(self):
        for cb in self.class_checkboxes.values(): cb.setChecked(False)

    def get_selected_classes(self):
        selected_classes = [cls_id for cls_id, cb in self.class_checkboxes.items() if cb.isChecked()]
        return selected_classes if len(selected_classes) > 0 else [9999]

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 模型", "", "PyTorch Models (*.pt)")
        if file_path:
            self.stop_video() 
            self.lbl_model_status.setText("⏳ 正在加载并解析模型...")
            self.lbl_model_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #f39c12;")
            QApplication.processEvents() 

            try:
                self.model = YOLO(file_path)
                for i in reversed(range(self.class_layout.count())): 
                    widget = self.class_layout.itemAt(i).widget()
                    if widget: widget.deleteLater()
                self.class_checkboxes.clear()
                
                for cls_id, cls_name in self.model.names.items():
                    cb = QCheckBox(f"[{cls_id}] {cls_name}")
                    cb.setChecked(True) 
                    cb.setStyleSheet("""
                        QCheckBox { font-size: 15px; padding: 6px; border-radius: 5px; color: #7f8c8d;}
                        QCheckBox:hover { background-color: #f1f2f6; }
                        QCheckBox:checked { background-color: #e3f2fd; color: #2980b9; font-weight: bold; }
                    """)
                    self.class_layout.addWidget(cb)
                    self.class_checkboxes[cls_id] = cb

                model_name = os.path.basename(file_path)
                self.lbl_model_status.setText(f"✅ 当前模型: {model_name}")
                self.lbl_model_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
                self.display_label.setText(f"模型加载成功！\n\n💡 提示：在执行头盔检测前，请确保右侧列表中的\n [person] 和 [Helmet] 处于勾选状态。")
                
                self.btn_video.setEnabled(True)
                self.btn_image.setEnabled(True)
                self.btn_webcam.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "模型加载失败", f"无法加载该模型，错误信息：\n{str(e)}")
                self.lbl_model_status.setText("❌ 模型加载失败")
                self.lbl_model_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c;")

    def set_confidence(self):
        value, ok = QInputDialog.getDouble(self, "设置置信度", "请输入新的阈值:", self.CONF_THRESHOLD, 0.01, 1.0, 2)
        if ok:
            self.CONF_THRESHOLD = value
            self.lbl_conf.setText(f"当前置信度阈值: {self.CONF_THRESHOLD:.2f}")

    def toggle_pause(self):
        if self.cap is not None and self.cap.isOpened():
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.is_paused:
                if current_frame >= total_frames - 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.video_slider.setValue(0)
                self.is_paused = False
                self.btn_pause.setText("⏸️ 暂停播放")
                self.timer.start(30)
            else:
                self.is_paused = True
                self.btn_pause.setText("▶️ 继续播放")
                self.timer.stop()

    def slider_pressed(self):
        self.timer.stop()

    def slider_released(self):
        if self.cap:
            frame_position = self.video_slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            if not self.is_paused:
                self.timer.start(30) 
            else:
                self.update_frame()  

    # ---------------- 媒体控制与核心流转 ----------------
    def load_video(self):
        self.stop_video()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_slider.setMaximum(total_frames)
            self.video_slider.setValue(0)
            self.video_slider.setEnabled(True)
            self.is_paused = False
            self.btn_pause.setText("⏸️ 暂停播放")
            self.btn_pause.setEnabled(True)
            self.timer.start(30) 

    def load_webcam(self):
        self.stop_video()
        self.cap = cv2.VideoCapture(0) 
        self.video_slider.setEnabled(False) 
        self.btn_pause.setEnabled(False) 
        self.timer.start(30)

    def load_image(self):
        self.stop_video() 
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path and self.model is not None:
            self.display_label.setText("正在检测中...")
            self.video_slider.setEnabled(False) 
            self.btn_pause.setEnabled(False) 
            
            target_classes = self.get_selected_classes()
            results = self.model.predict(source=file_path, conf=self.CONF_THRESHOLD, classes=target_classes, verbose=False)
            
            # 单张图片同样调用分析逻辑更新状态栏
            self.analyze_and_update_status(results)
            self.show_image(results[0].plot())

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened() and self.model is not None:
            ret, frame = self.cap.read()
            if ret:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.video_slider.blockSignals(True) 
                self.video_slider.setValue(current_frame)
                self.video_slider.blockSignals(False)

                target_classes = self.get_selected_classes()
                results = self.model.predict(source=frame, conf=self.CONF_THRESHOLD, classes=target_classes, verbose=False)
                
                # 视频流实时调用分析逻辑更新状态栏
                self.analyze_and_update_status(results)
                self.show_image(results[0].plot())
            else:
                self.timer.stop()
                self.is_paused = True
                self.btn_pause.setText("▶️ 重新播放")
                self.video_slider.blockSignals(True)
                self.video_slider.setValue(self.video_slider.maximum())
                self.video_slider.blockSignals(False)
                
    def show_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(pixmap)

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_slider.setValue(0) 
        self.is_paused = False
        self.btn_pause.setText("⏸️ 暂停播放")
        self.btn_pause.setEnabled(False) 

    # ================= 🆕 核心业务逻辑区域 =================

    def check_intersection(self, boxA, boxB):
        """判断两个矩形框在二维平面上是否有交集"""
        overlap_x_min = max(boxA[0], boxB[0])
        overlap_y_min = max(boxA[1], boxB[1])
        overlap_x_max = min(boxA[2], boxB[2])
        overlap_y_max = min(boxA[3], boxB[3])

        if (overlap_x_max - overlap_x_min) > 0 and (overlap_y_max - overlap_y_min) > 0:
            return True
        return False


    def analyze_and_update_status(self, results):
        """解析 YOLO 结果，根据下拉框选择的方法执行不同逻辑，并更新状态栏"""
        current_method = self.method_selector.currentText()

        # =========================================================
        # 逻辑 A：头盔佩戴规范检测
        # =========================================================
        if current_method == "头盔佩戴规范检测 (交集判定法)":
            person_id = -1
            helmet_id = -1  
            for k, v in self.model.names.items():
                name_lower = v.lower()
                if 'person' in name_lower or '人' in name_lower: person_id = k
                elif 'helmet' in name_lower or '帽' in name_lower: helmet_id = k

            if person_id == -1 or helmet_id == -1:
                self.status_bar.setText("⚠️ 警告：当前模型未找到 'person' 或 'helmet' 类别！")
                self.status_bar.setStyleSheet("background-color: #f39c12; color: white; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;")
                return

            persons_boxes = []
            helmets_boxes = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                coords = box.xyxy[0].cpu().numpy()
                if cls_id == person_id: persons_boxes.append(coords)
                elif cls_id == helmet_id: helmets_boxes.append(coords)

            total_persons = len(persons_boxes)
            no_helmet_count = 0
            for p_box in persons_boxes:
                has_helmet = False
                for h_box in helmets_boxes:
                    if self.check_intersection(p_box, h_box):
                        has_helmet = True
                        break
                if not has_helmet:
                    no_helmet_count += 1

            if total_persons == 0:
                self.status_bar.setText("🟢 待机中: 画面中未检测到人员")
                self.status_bar.setStyleSheet("background-color: #2c3e50; color: white; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;")
            elif no_helmet_count > 0:
                self.status_bar.setText(f"🚨 违规警报: 画面总人数 {total_persons} 人 | 发现 {no_helmet_count} 人未佩戴安全帽！")
                self.status_bar.setStyleSheet("background-color: #c0392b; color: white; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;")
            else:
                self.status_bar.setText(f"✅ 状态合规: 画面总人数 {total_persons} 人 | 全员穿戴规范")
                self.status_bar.setStyleSheet("background-color: #27ae60; color: white; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;")

        # =========================================================
        # 🆕 逻辑 B：单人行为专注度分析
        # =========================================================
        elif current_method == "单人行为专注度分析":
            # 在行为检测模型中，检测到的总框数通常就是人数
            num_people = len(results[0].boxes)

            if num_people == 0:
                self.status_bar.setText("🟢 待机中: 画面中未检测到学生")
                self.status_bar.setStyleSheet("background-color: #2c3e50; color: white; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;")
            
            elif num_people > 1:
                self.status_bar.setText(f"⚠️ 警告: 检测到多人 ({num_people}人)，请确保画面中只有单名学生！")
                self.status_bar.setStyleSheet("background-color: #f39c12; color: white; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;")
            
            else:
                # 画面中只有 1 个人，获取他正在做的动作
                box = results[0].boxes[0]
                cls_id = int(box.cls[0])
                # 直接通过字典拿到类似 "reading", "writing" 的字符串
                action_name = self.model.names[cls_id] 
                
                # 你可以根据具体的动作进行中文映射，或者直接显示英文
                self.status_bar.setText(f"✅ 行为分析: 当前学生正在 【{action_name}】")
                self.status_bar.setStyleSheet("background-color: #27ae60; color: white; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = modelDetectionSystem()
    window.show()
    sys.exit(app.exec())