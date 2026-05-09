import sys
import os
import cv2
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
from PySide6.QtCore import Qt, QTimer

# 导入我们刚刚分离的两个核心模块
from ui_view import DetectionUI
from logic_core import VisionDataLayer, BusinessLogicAnalyzer

class AppController(DetectionUI):
    """事件控制层：统筹 UI 视图与底层数据的交互"""
    def __init__(self):
        super().__init__()
        
        # 初始化核心变量与数据层
        self.CONF_THRESHOLD = 0.5  
        self.is_paused = False
        self.data_layer = VisionDataLayer()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame) 

        # 绑定所有的按钮事件
        self.bind_events()

    def bind_events(self):
        self.btn_load_model.clicked.connect(self.select_model)
        self.btn_select_all.clicked.connect(lambda: self.check_all_classes(True))
        self.btn_deselect_all.clicked.connect(lambda: self.check_all_classes(False))
        self.btn_conf.clicked.connect(self.set_confidence)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_webcam.clicked.connect(self.load_webcam)
        self.btn_image.clicked.connect(self.load_image)
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.video_slider.sliderPressed.connect(self.slider_pressed)
        self.video_slider.sliderReleased.connect(self.slider_released)

    def keyPressEvent(self, event):
        """接管空格键快捷暂停功能"""
        if event.key() == Qt.Key_Space and self.btn_pause.isEnabled():
            self.toggle_pause()
        else:
            super().keyPressEvent(event)

    # ---------------- 控制器核心调度逻辑 ----------------
    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 模型", "", "PyTorch Models (*.pt)")
        if file_path:
            self.stop_video()
            self.lbl_model_status.setText("⏳ 正在加载并解析模型...")
            QApplication.processEvents() 

            try:
                # 1. 调动数据层加载模型
                names_dict = self.data_layer.load_model(file_path)
                # 2. 调动 UI 层渲染多选框
                self.render_class_checkboxes(names_dict)
                
                model_name = os.path.basename(file_path)
                self.lbl_model_status.setText(f"✅ 当前模型: {model_name}")
                self.lbl_model_status.setStyleSheet("font-size: 16px; font-weight: bold; color: #27ae60;")
                self.display_label.setText("模型加载成功！请导入媒体开始检测。")
                self.set_media_buttons_enabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "模型加载失败", f"无法加载模型：\n{str(e)}")
                self.lbl_model_status.setText("❌ 模型加载失败")

    def set_confidence(self):
        value, ok = QInputDialog.getDouble(self, "设置置信度", "请输入新的阈值:", self.CONF_THRESHOLD, 0.01, 1.0, 2)
        if ok:
            self.CONF_THRESHOLD = value
            self.lbl_conf.setText(f"当前置信度阈值: {self.CONF_THRESHOLD:.2f}")

    def load_video(self):
        self.stop_video()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_path:
            total_frames = self.data_layer.load_video(file_path)
            self.video_slider.setMaximum(total_frames)
            self.video_slider.setValue(0)
            self.video_slider.setEnabled(True)
            self.is_paused = False
            self.btn_pause.setText("⏸️ 暂停播放")
            self.btn_pause.setEnabled(True)
            self.timer.start(30) 

    def load_webcam(self):
        self.stop_video()
        if self.data_layer.load_webcam(0):
            self.video_slider.setEnabled(False) 
            self.btn_pause.setEnabled(False) 
            self.timer.start(30)

    def load_image(self):
        self.stop_video() 
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path and self.data_layer.model:
            self.display_label.setText("正在检测中...")
            self.video_slider.setEnabled(False) 
            self.btn_pause.setEnabled(False) 
            
            # 单图推理调度
            results = self.data_layer.predict(file_path, self.CONF_THRESHOLD, self.get_selected_classes())
            self.dispatch_logic_and_render(results)

    def update_frame(self):
        """定时器钩子：循环拉取视频帧并调度"""
        ret, frame, current_frame = self.data_layer.read_frame()
        
        if ret:
            self.video_slider.blockSignals(True) 
            self.video_slider.setValue(current_frame)
            self.video_slider.blockSignals(False)

            results = self.data_layer.predict(frame, self.CONF_THRESHOLD, self.get_selected_classes())
            self.dispatch_logic_and_render(results)
        else:
            self.timer.stop()
            self.is_paused = True
            self.btn_pause.setText("▶️ 重新播放")
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(self.video_slider.maximum())
            self.video_slider.blockSignals(False)

    def dispatch_logic_and_render(self, results):
        """将数据送去业务层分析，并将结果交还 UI 层渲染"""
        if not results: return

        # 1. 调用业务层 (Pure Math/Logic)
        status_msg, status_level = BusinessLogicAnalyzer.analyze(
            method_name=self.method_selector.currentText(),
            results=results,
            class_names_dict=self.data_layer.model.names
        )

        # 2. 调用 UI 层渲染文本和颜色
        self.status_bar.setText(status_msg)
        self.update_status_style(status_level)

        # 3. 将 OpenCV BGR 帧转换为 RGB，并交由 UI 层绘制
        frame_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        self.render_display_image(frame_rgb)

    # ---------------- 播放状态控制 ----------------
    def toggle_pause(self):
        if self.data_layer.cap and self.data_layer.cap.isOpened():
            if self.is_paused:
                if self.video_slider.value() >= self.video_slider.maximum() - 1:
                    self.data_layer.set_frame_position(0)
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
        self.data_layer.set_frame_position(self.video_slider.value())
        if not self.is_paused:
            self.timer.start(30) 
        else:
            self.update_frame()

    def stop_video(self):
        self.timer.stop()
        self.data_layer.release_video()
        self.video_slider.setValue(0) 
        self.is_paused = False
        self.btn_pause.setText("⏸️ 暂停播放")
        self.btn_pause.setEnabled(False) 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppController() # 启动控制器实例
    window.show()
    sys.exit(app.exec())