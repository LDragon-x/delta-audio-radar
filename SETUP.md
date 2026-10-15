# 三角洲音频雷达 — 新机装机单（SETUP）

> 2026-09-27 打包。此包 = `delta-audio-radar`（主项目）+ `delta-gunsmith`（枪匠 Web 应用）。
> 未打入：testdata/（可用 tool/fetch 脚本重下）、data/gunsmith/raw/lza_*（理包站原始资产，可重下）、node_modules（npm install 重建）。

## 1. 环境

1. **Python 3.12**（装时勾 Add to PATH）
   2. 依赖：
   ```
   pip install soundcard numpy keyboard mouse yt-dlp imageio-ffmpeg
   ```
   （requirements.txt 只有 soundcard / numpy，另外四个是采集器和下载脚本要用的）
3. **Node 20+**（只有玩枪匠应用才需要）
4. **VB-Cable**（音频过道用，免费版即可）：装完重启，然后
   ```
   python tool/monitor.py --list
   ```
   找到 CABLE 设备名填进 config.json
5. 可选：Ollama + `qwen3:4b`（贾维斯的 LLM 层，没有也能跑）

## 2. 验收（必须 PASS 才算装好）

```
cd delta-audio-radar
python main.py --demo
```
输出 `DEMO PASS`（12 项断言，含热键掩蔽、自身声纹剔除检查）。FAIL 就别往下走，回来找我。

## 3. 游戏音频设置（照这个设，不然方位不准）

- 耳机，**HRTF 空间音效开**
- 杜比/DTS 之类的虚拟环绕**关掉**
- 24bit / 48000Hz

## 4. 常用入口

| 要干什么 | 命令 |
| --- | --- |
| 实时面板（方位/人数/枪型） | `python main.py` |
| 音频过道（游戏→VB-Cable→耳机） | `python tool/monitor.py` |
| 样本采集器 | `python tool/collector.py` |
| 枪型识别 | `python tool/guntype.py` |
| 环境音档案 | `python tool/ambient.py` |
| 容器占卜师（格型→品质预判） | `python tool/container_oracle.py` |
| 物价刷新 | `python tool/fetch_market.py` |

枪匠应用：`cd delta-gunsmith && npm install && npm run dev`

## 5. 遗留待办（想起来就搞）

- `samples/guns/library.json` 里 5 个枪名还是"_待确认"，B 站视频里核对后直接改名
- `data/kb/operators.json` 的干员技能 CD 还是 null，局内看到了填
- `deliverables/` 里两个对比 wav（脚步段/枪声段），戴耳机 A/B 听一下，把感受告诉我，调 EQ 参数
- **声纹库（防线3）已停用**（2026-09-27 真机标定：12 频带 80ms 指纹对大类声音不可分，
  见 JOURNAL 当日条目）。代码/工具/库都留着（self_reload 126 条 + self_firemode 7 条），
  v0.2 换高分辨率特征（倒谱/长尾/混响尾巴）再试。自己的声音现在由
  **热键掩蔽全覆盖**（R/B 换弹切模式 + WASD 移动脚步 + 鼠标左键开火）+ 规则层兜。
  想微调掩蔽窗口就在 config.json `hotkey_mask` 里改
- 然后才是正戏：网格标定 + 屏幕 overlay

## 6. 留痕

全部过程文档在 `docs/`：JOURNAL（工程日志）、DECISIONS（决策记录）、ROADMAP、HARVEST、UI_SPEC、OPERATOR。接着写，别断。
