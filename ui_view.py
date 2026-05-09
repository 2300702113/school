from PySide6.QtWidgets import (QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                               QHBoxLayout, QWidget, QSlider, QGroupBox, 
                               QCheckBox, QScrollArea, QComboBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

class DetectionUI(QMainWindow):
    """纯视图层：仅负责界面排版与视觉更新"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能视觉通用识别系统 v4.1 (架构分离版)")
        self.resize(1200, 800)
        self.class_checkboxes = {}
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ====== 左侧：显示面板 ======
        left_panel = QWidget()
        left_v_layout = QVBoxLayout(left_panel)
        left_v_layout.setContentsMargins(0, 0, 0, 0) 

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("🚀 当前业务逻辑:"))
        self.method_selector = QComboBox()

        # 找到这行代码并更新
        self.method_selector.addItems([
            "头盔佩戴规范检测 (交集判定法)", 
            "单人行为专注度分析",
            "多人课堂行为统计"  # 🆕 新增这个选项
        ])

        self.method_selector.setStyleSheet("font-size: 15px; padding: 5px; font-weight: bold;")
        header_layout.addWidget(self.method_selector, stretch=1)
        left_v_layout.addLayout(header_layout)

        self.display_label = QLabel("请先在右侧加载 AI 模型 (.pt 文件)...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #1e1e1e; color: #a0a0a0; font-size: 24px; font-weight: bold; border-radius: 5px;")
        left_v_layout.addWidget(self.display_label, stretch=1) 

        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False) 
        left_v_layout.addWidget(self.video_slider)

        self.status_bar = QLabel("系统就绪 | 等待接入视频源...")
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.update_status_style("INFO")
        left_v_layout.addWidget(self.status_bar)

        main_layout.addWidget(left_panel, stretch=5) 

        # ====== 右侧：控制面板 ======
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15) 

        btn_style = """
            QPushButton { background-color: #2b5c8f; color: white; border-radius: 8px; font-size: 16px; font-weight: bold; padding: 12px; }
            QPushButton:hover { background-color: #3b7cbf; }
            QPushButton:disabled { background-color: #555555; color: #888888; } 
        """

        self.lbl_model_status = QLabel("当前模型: 未加载 ❌")
        self.lbl_model_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c;")
        right_layout.addWidget(self.lbl_model_status)

        self.btn_load_model = QPushButton("📂 选择 AI 模型 (.pt)")
        self.btn_load_model.setStyleSheet(btn_style.replace("#2b5c8f", "#8e44ad").replace("#3b7cbf", "#9b59b6")) 
        right_layout.addWidget(self.btn_load_model)

        self.class_group = QGroupBox("🎯 指定检测类别")
        self.class_group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #bdc3c7; border-radius: 8px; margin-top: 15px; background-color: #ffffff;} QGroupBox::title { left: 10px; color: #2c3e50; }")
        group_layout = QVBoxLayout()
        
        toggle_layout = QHBoxLayout()
        btn_toggle_style = "QPushButton { background-color: #ecf0f1; color: #2c3e50; border-radius: 4px; font-size: 13px; font-weight: bold; padding: 5px; border: 1px solid #bdc3c7; } QPushButton:hover { background-color: #d5dbdb; }"
        self.btn_select_all = QPushButton("✅ 全选")
        self.btn_select_all.setStyleSheet(btn_toggle_style)
        self.btn_deselect_all = QPushButton("❌ 全不选")
        self.btn_deselect_all.setStyleSheet(btn_toggle_style)
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

        self.lbl_conf = QLabel("当前置信度阈值: 0.5")
        self.lbl_conf.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(self.lbl_conf)

        self.btn_conf = QPushButton("⚙️ 设置置信度")
        self.btn_conf.setStyleSheet(btn_style.replace("#2b5c8f", "#d35400").replace("#3b7cbf", "#e67e22")) 
        right_layout.addWidget(self.btn_conf)

        self.btn_video = QPushButton("🎬 导入监控视频")
        self.btn_video.setStyleSheet(btn_style)
        right_layout.addWidget(self.btn_video)

        self.btn_pause = QPushButton("⏸️ 暂停播放")
        self.btn_pause.setStyleSheet(btn_style)
        right_layout.addWidget(self.btn_pause)

        self.btn_image = QPushButton("🖼️ 导入单张图片")
        self.btn_image.setStyleSheet(btn_style)
        right_layout.addWidget(self.btn_image)

        self.btn_webcam = QPushButton("📷 开启本地摄像头")
        self.btn_webcam.setStyleSheet(btn_style)
        right_layout.addWidget(self.btn_webcam)

        self.set_media_buttons_enabled(False)
        main_layout.addLayout(right_layout, stretch=1)

        # 剥夺焦点机制
        for btn in [self.btn_load_model, self.btn_select_all, self.btn_deselect_all, 
                    self.btn_conf, self.btn_video, self.btn_pause, self.btn_image, self.btn_webcam]:
            btn.setFocusPolicy(Qt.NoFocus)
        self.method_selector.setFocusPolicy(Qt.NoFocus)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    # ---------------- UI 渲染与状态更新接口 ----------------
    def update_status_style(self, level):
        styles = {
            "INFO": "background-color: #2c3e50; color: white;",
            "SUCCESS": "background-color: #27ae60; color: white;",
            "WARNING": "background-color: #f39c12; color: white;",
            "DANGER": "background-color: #c0392b; color: white;"
        }
        base_style = "font-size: 18px; font-weight: bold; padding: 12px; border-radius: 5px;"
        self.status_bar.setStyleSheet(styles.get(level, styles["INFO"]) + base_style)

    def set_media_buttons_enabled(self, enabled):
        self.btn_video.setEnabled(enabled)
        self.btn_image.setEnabled(enabled)
        self.btn_webcam.setEnabled(enabled)

    def render_class_checkboxes(self, names_dict):
        for i in reversed(range(self.class_layout.count())): 
            widget = self.class_layout.itemAt(i).widget()
            if widget: widget.deleteLater()
        self.class_checkboxes.clear()
        
        for cls_id, cls_name in names_dict.items():
            cb = QCheckBox(f"[{cls_id}] {cls_name}")
            cb.setChecked(True) 
            cb.setStyleSheet("QCheckBox { font-size: 15px; padding: 6px; border-radius: 5px; color: #7f8c8d;} QCheckBox:hover { background-color: #f1f2f6; } QCheckBox:checked { background-color: #e3f2fd; color: #2980b9; font-weight: bold; }")
            self.class_layout.addWidget(cb)
            self.class_checkboxes[cls_id] = cb

    def check_all_classes(self, checked=True):
        for cb in self.class_checkboxes.values(): 
            cb.setChecked(checked)

    def get_selected_classes(self):
        selected = [cls_id for cls_id, cb in self.class_checkboxes.items() if cb.isChecked()]
        return selected if selected else [9999]

    def render_display_image(self, frame_rgb):
        h, w, ch = frame_rgb.shape
        qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(pixmap)