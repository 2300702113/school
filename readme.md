# 👁️ 智能视觉通用识别系统 (Smart Vision General Detection System)

![Version](https://img.shields.io/badge/version-v4.1-blue.svg)
![Architecture](https://img.shields.io/badge/architecture-MVC-orange.svg)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)
![Framework](https://img.shields.io/badge/framework-PySide6%20%7C%20Ultralytics%20YOLO-lightgrey.svg)

## 📖 项目简介 (Overview)

本项目是一个基于 **MVC 架构**与 **YOLO 动态模型调度** 的多场景视觉合规巡检系统。摒弃了传统视觉项目“硬编码”和“单纯画框计数”的痛点，本系统引入了**空间关系推理（AABB交集与嵌套）**，能够实现从“认识物体”到“理解业务逻辑”的跨越。

系统目前已完美落地两大垂直场景，并支持通过解耦的逻辑层快速横向扩展：
1. **智慧工地 PPE 检测**：基于二维平面交集的安全帽/反光衣佩戴规范检测。
2. **智慧教培行为分析**：单人专注度解析与多人课堂违纪（玩手机、低头等）动态聚合报警。

## ✨ 核心特性 (Key Features)

- **🧩 纯粹的 MVC 架构解耦**：UI 视图 (`ui_view.py`)、数据交互 (`logic_core.py/VisionDataLayer`) 与业务逻辑 (`logic_core.py/BusinessLogicAnalyzer`) 彻底分离，支持并行开发与无损维护。
- **🧠 动态 Agent 逻辑分发**：支持热插拔任意 `.pt` YOLO 模型，自动解析模型类别（Names Dict），并根据前端指令动态挂载不同的长链推理逻辑。
- **📐 空间逻辑推理引擎**：内置 AABB 二维平面交集碰撞检测算法，精准实现目标间的关系绑定（如：人与头盔的物理绑定关系）。
- **🛡️ 工业级测试保障**：针对核心逻辑层配备了基于 Mock 数据的极限单元测试，实现 **100% 语句覆盖率** 与 **100% 条件/分支覆盖率**。
- **🚀 多路并发压测支持**：内置无头（Headless）多进程压测脚本，支持对内存、CPU 和 FPS 的极限性能摸底。

## 📂 项目结构 (Project Structure)

```text
📦 PROJECT_ROOT
 ┣ 📂 datasets/               # 数据集存放目录 (已被 .gitignore 忽略)
 ┣ 📂 runs/                   # YOLO 训练/预测输出 (已被 .gitignore 忽略)
 ┣ 📜 main_controller.py      # 【C】控制器：统筹 UI 与底层逻辑的事件流转
 ┣ 📜 ui_view.py              # 【V】视图层：PySide6 纯界面排版与视觉渲染
 ┣ 📜 logic_core.py           # 【M】模型层：YOLO 视频流调度与纯数学/规则算法
 ┣ 📜 test_full_coverage.py   # 核心单元测试脚本 (Mock 数据注入与全分支覆盖)
 ┣ 📜 benchmark.py            # 多进程并发极限压测脚本
 ┣ 📜 .gitignore              # Git 忽略配置 (屏蔽大文件与隐私数据)
 ┗ 📜 README.md               # 项目说明文档