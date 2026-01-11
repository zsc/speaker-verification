# 实时声纹识别 Demo

该目录包含了一个完整的声纹识别演示系统，提供**实时 Web 界面**和**命令行工具 (CLI)** 两种模式。

## 功能特性

*   **静音检测 (VAD)**: 使用 [Silero VAD](https://github.com/snakers4/silero-vad) 稳健地检测语音片段。
*   **声纹识别**: 使用 [3D-Speaker's ERes2Net](https://github.com/alibaba-damo-academy/3D-Speaker) 模型提取声纹特征向量（Embedding）。
*   **音频增强 (可选)**: 支持 [CMGAN](https://github.com/ruizhecao96/CMGAN) 降噪算法，在识别前对音频进行预处理（需手动放置模型）。
*   **实时可视化**:
    *   **距离矩阵**: 实时计算并可视化最近 8 段语音之间的余弦距离（Cosine Distance）。
    *   **事件时间线**: 实时显示 VAD 状态（语音开始/结束）和模型处理进程。
*   **双模式支持**:
    *   **Web Demo**: 基于 Flask + Socket.IO，支持麦克风实时输入。
    *   **CLI 模式**: 支持对本地音频文件进行批处理分析。

## 安装说明

演示系统会自动处理大部分依赖和模型。

1.  **安装 Python 依赖**:
    推荐使用提供的启动脚本，它会在运行前自动检查并安装依赖：
    ```bash
    ./demo/start_demo.sh
    ```
    或者手动安装：
    ```bash
    pip install -r demo/requirements.txt
    ```

2.  **模型获取**:
    *   **ERes2Net (声纹模型)**: 首次运行时会自动从 ModelScope (魔搭社区) 下载。
    *   **Silero VAD**: 通过 Torch Hub 自动下载。
    *   **CMGAN (降噪模型)**: 可选。如果你有预训练模型，请将其命名为 `ckpt` 并放置在项目根目录的 `./model/` 文件夹下。如果缺失，系统将自动跳过降噪步骤，直接使用原始音频。

## 使用说明

### 1. Web Demo (麦克风实时模式)

启动 Web 服务器：
```bash
./demo/start_demo.sh
```
*   **访问地址**: 在浏览器中打开 [http://localhost:5053](http://localhost:5053)。
*   **操作步骤**:
    1.  点击 **"Start Microphone"** 开启麦克风。
    2.  开始说话。
    3.  系统会自动检测完整语句（VAD 切分），处理完成后更新**距离矩阵**热力图。
    4.  **绿色/红色**事件表示语音开始/结束，**橙色**事件表示正在提取声纹。

### 2. CLI 模式 (文件处理模式)

对现有的音频文件（WAV 格式）进行分析，自动切分并计算段落间的声纹差异。

**命令格式:**
```bash
python demo/cli.py <音频文件路径> [选项]
```

**示例:**
```bash
python demo/cli.py data/test.wav
```

**可选参数:**
*   `input_file`: 输入 WAV 文件的路径。
*   `--threshold`: VAD 概率阈值 (默认: 0.5)。值越高越严格。
*   `--min_silence`: 判定为语句结束的最小静音时长 (单位: 毫秒，默认: 500)。

**输出示例:**
CLI 将输出检测到的片段信息及余弦距离矩阵（数值越小表示声音越相似）：
```text
Distance Matrix:
       Seg 1    Seg 2    Seg 3 
Seg 1   0.000    0.152    0.866 
Seg 2   0.152    0.000    0.742 
Seg 3   0.866    0.742    0.000 
```

## 目录结构

*   `server.py`: Flask + Socket.IO 后端服务器。
*   `cli.py`: 命令行工具入口。
*   `model_loader.py`: 统一的模型加载模块（VAD, ERes2Net, CMGAN）。
*   `templates/index.html`: 前端界面。
*   `static/js/main.js`: 前端逻辑（音频处理、WebSocket 通信、绘图）。
*   `requirements.txt`: 演示系统专用的 Python 依赖清单。