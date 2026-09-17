# qsmy-deepseek-locator

> 用 DeepSeek 视觉模型做**物体定位**：给它一张图和一句「找什么」，
> 拿回 **0.0~1.0 的归一化坐标**（框 / 点）与中文名称，需要的话直接把框和标签画回图上。

```python
from qsmy_deepseek_locator import locate_to_file

# 最省事：图片 + 「找什么」+ 输出路径，回来时标注图已经写好
result = locate_to_file("photo.png", "红色圆形", "annotated.png")
print(result.annotated_path, result.labels)     # annotated.png ['红色圆形']
```

要自己掌控坐标与绘制：

```python
from qsmy_deepseek_locator import locate, draw

result = locate("photo.png", "红色圆形")
for d in result:
    print(d.label, d.bbox, d.center)      # 红色圆形 (0.101, 0.205, 0.298, 0.402) (0.1995, 0.3035)

draw("photo.png", result).save("annotated.png")
```

命令行同样一行：

```bash
qsmy-deepseek-locator photo.png -t "红色圆形" -o annotated.png
```

---

## 1. 安装

```bash
pip install qsmy-deepseek-locator
```

要改源码、跑测试就用可编辑安装：

```bash
git clone https://github.com/QsmyHyly/qsmy-deepseek-locator.git
cd qsmy-deepseek-locator
python -m venv .venv && .venv\Scripts\activate      # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -e ".[dev]"
```

依赖只有三个：`openai`（DeepSeek 走 OpenAI 兼容协议）、`Pillow`（读图 / 打标）、`requests`（下载图片 URL）。

跑自测（187 个用例，**全程离线、不花 API**）：

```bash
python -m pytest tests -q
```

## 2. 配 API Key

```bash
# Windows（持久生效）
setx DEEPSEEK_API_KEY "sk-你的key"
# macOS / Linux
export DEEPSEEK_API_KEY="sk-你的key"
```

也可以不设环境变量，直接传参：`Locator(api_key="sk-...")` 或 `locate(..., api_key="sk-...")`。
其余可配项见 [`.env.example`](https://github.com/QsmyHyly/qsmy-deepseek-locator/blob/main/.env.example)。

> **没有 Key 会怎样**：直接抛 `MissingAPIKeyError`，并告诉你三种配法。
> 本库刻意**不提供**「无 Key 时返回假数据」的降级 —— 演示程序这样做很方便，
> 但库不行：用户会拿着一堆看起来正常的假坐标当真结果。

## 3. 三种用法

### 3.1 一行式出图（最省事）

给「图片 + 找什么 + 输出路径」，函数回来时标注图已经躺在磁盘上了：

```python
from qsmy_deepseek_locator import locate_to_file

result = locate_to_file("photo.png", "红色圆形", "runs/photo_annotated.png")
print(result.annotated_path)   # runs/photo_annotated.png（没写扩展名会自动补 .png）
print(result.labels)           # ['红色圆形']
```

可选参数（全部关键字，按需给）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `api_key` | 读 `DEEPSEEK_API_KEY` | **传了就只用它**，不再看环境变量 |
| `thinking` | `False` | 默认显式关闭思考（实测不掉准确率、耗时约省一半）；传 `None` 交回环境变量决定 |
| `image_detail` | `"original"` | `low/high/original/auto`；传 `None` 表示不发送该字段 |
| `use_tools` | `False` | 工具（Agent）调用：v0.1 未实现，传 `True` 会**当场报错**而不是被静默忽略 |
| `prompt` | `None` | 直接给**整段**用户消息（给了就忽略第二个位置参数 `target`）。⚠️ 它不会自动套上本库那句「请找出图中所有的…」包装句式，所以「找什么」请走 `target` |
| `model` / `base_url` / `timeout` / `max_tokens` / `max_side` / `system_prompt` | 见第 6 节 | 与 `Locator.locate()` 同名同义 |
| `colors` / `box_width` / `point_radius` / `font_size` / `draw_label` / `scale_to_image` | 见 `drawing.draw` | 绘制样式 |

几条已定好的行为，不必去猜：

- **输出路径先校验、后调模型** —— 路径拼错不该等花掉一次 API 调用才发现；父目录会自动创建；
- 扩展名决定格式（`.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff/.gif`），**认不出的后缀直接报错**，
  绝不偷偷存成别的格式（文件名写着 `.jpg` 内容却是 PNG，是最难排查的一类问题）；
- 模型没找到目标**照样出图**（内容等于原图），这不是失败，`result.detections` 为空而已；
- 画的是**原图**，所以输出分辨率始终等于输入分辨率。

### 3.2 一行式（只要坐标）

```python
from qsmy_deepseek_locator import locate

result = locate("photo.png", "画面里的人")      # 本地路径 / URL / bytes / PIL.Image / data URL 都行
print(result.summary())                        # {'total': 3, 'bbox_count': 3, ...}
```

### 3.3 复用定位器（多图批量时用这个）

```python
from qsmy_deepseek_locator import Locator, draw

locator = Locator(thinking=False)              # 关掉思考：更快、更省 token（实测不掉准确率）
for path in ["a.png", "b.png", "c.png"]:
    result = locator.locate(path, "按钮")
    draw(path, result).save(path.replace(".png", "_annotated.png"))
    print(path, result.labels)
```

`locate()` / `locate_to_file()` 每次都会重新读环境变量、新建客户端；批量场景请自己建
`Locator`，再调用它的 `locate()` 或 `locate_to_file()`（后者同样会把落盘路径写进 `result.annotated_path`）。

### 3.4 命令行

```bash
qsmy-deepseek-locator photo.png -t "登录按钮"                 # 打印坐标 + 生成 photo_annotated.png
qsmy-deepseek-locator photo.png -t "人" --print-json          # 只吐 JSON（可直接管道给 jq）
qsmy-deepseek-locator photo.png -t "人" --json r.json         # 连证据一起存盘
qsmy-deepseek-locator photo.png -t "人" --no-thinking         # 关思考，更快
qsmy-deepseek-locator photo.png --show-reasoning              # 实时看模型的思考过程
qsmy-deepseek-locator photo.png -t "人" --log                 # 开调试日志（请求体/响应体，见 7.1）
qsmy-deepseek-locator https://example.com/a.jpg -t "商品"     # 直接给 URL

qsmy-deepseek-locator bench --count 5 --n-shapes 3            # 跑准确率评测（见第 5 节）
qsmy-deepseek-locator bench --images-only --count 3           # 只造图不调模型（不花钱）
```

退出码：`0` 成功、`1` 运行期错误（缺 Key / 图片读不了 / 接口报错）、`2` 命令行用法错误。

## 4. 坐标约定（本库最要紧的一条）

**所有坐标都是 0.0~1.0 的相对比例，小数位数不设上限。**

```python
d.bbox              # (x1, y1, x2, y2) 归一化，左上角到右下角
d.point             # (x, y) 归一化
d.center            # 框的几何中心 / 点本身
d.to_pixels(1920, 1080)   # {'bbox_px': (...), 'point_px': None, 'center_px': (...)}
```

为什么不是像素：**模型根本看不到图片的真实分辨率**。服务端会先把图缩放再喂给它，
而且不回传缩放后的尺寸；模型报的「像素」落在它每次自己编的画布上
（实测同一张图三次调用分别给出 1000x750 / 1000x800 / 1024x768）。
缩放本身是纯线性等比的，所以**相对比例是这个链路上唯一可靠的量**。

两条兜底措施：

- 模型偶尔输出 `0~1000` 旧刻度（视觉大模型圈的常见约定），整批会被**无损除以 1000** 换回来，
  并在 `result.warnings` 里留一条说明；
- 输出像是**像素坐标**时（越界），本库**只告警不猜测** —— 越界本身就是「提示词没被遵守」的情报，
  悄悄夹紧等于把情报抹掉。绘制时才夹到边界，保证画得出来。

## 5. 自带评测（改提示词之前请先跑它）

准确率几乎完全由**提示词口径**决定。同一批 15 个目标的 A/B 实测：

| 提示词版本 | 检出率 | 平均 IoU | 输出像素坐标的图片 |
|---|---|---|---|
| 旧：说「用归一化值」又说「无需考虑分辨率」 | **40%** | 0.655 | 3/5 |
| 新：显式换算公式 + 禁像素值 + 交代刻度不确定 | **100%** | 0.897 | 0/5 |

改一句话就是 60 个百分点，所以「改完提示词到底变好没有」必须能自动判分：

```bash
$ qsmy-deepseek-locator bench --count 5 --n-shapes 3 --annotate
图片与真值：runs/benchmark/images
图片 5 张 | 真值 15 个 | 预测 15 个
检出率 100.0%（15/15）   精确率 100.0%   平均 IoU 0.874
标签准确率 100.0%（颜色 100.0% / 形状 100.0%）
平均耗时 2.3s/张   带告警的图片 0 张
报告：runs/benchmark/report.json
```

（上面这段是本库的实跑输出：`deepseek-flash`、思考开启、默认参数、种子 42。
换成 `--no-thinking` 会更快 —— 另一轮 2 张图的实测是 1.8s/张、平均 IoU 0.915，准确率不掉。）

它用代码生成「已知答案」的几何图形图，真值顺手算出来，再按 IoU（阈值 0.5）匹配预测框。
产物落在 `runs/benchmark/`：`images/`（图 + `ground_truth.json`）、`annotated/`（预测画回图）、`report.json`。

编程接口：`from qsmy_deepseek_locator.benchmark import run_benchmark, evaluate_sample`。

## 6. API 速查

```python
from qsmy_deepseek_locator import (
    Locator, Detection, draw, save_annotated, locate_to_file, __version__,
)

# 一行式出图（模块级函数 = 建临时 Locator 再调下面的方法）
result = locate_to_file(
    "photo.png",           # 图片：路径 / URL / bytes / PIL.Image / data URL
    "红色圆形",             # 找什么（target，就是那句用户提示词）
    "out/annotated.png",   # 输出文件的完整路径（含文件名）
    api_key=None,          # 传了就不读 DEEPSEEK_API_KEY
    thinking=False,        # 默认显式关闭思考
    image_detail="original",  # 默认 original；None = 不发送该字段
    use_tools=False,       # v0.1 只能是 False
)

locator = Locator(
    api_key=None,          # 默认读 DEEPSEEK_API_KEY
    model="deepseek-flash",
    thinking=False,        # None=沿用服务端默认；False=关思考（更快更省）
    reasoning_effort=None, # low/medium/high/xhigh/max
    image_detail=None,     # low/high/original/auto；默认不发送该字段
    max_tokens=None,       # 输出上限（含思考 token）
    timeout=300,            # 单次请求超时（秒）；恒走流式，超时按「两次数据之间的静默」算
    max_side=None,         # 发送前把图缩到最长边不超过它（省流量，不影响坐标精度）
)
```

上面这些名字都会被 `Settings.merged()` 收下（真实签名是 `Locator(*, settings=None, client=None, max_side=None, **overrides)`），
所以除了它们，还可以直接传 `client=`（自备客户端）、`settings=`（整份配置）、`max_retries=`、`log_file=`（调试日志）。

```python
result = locator.locate(
    "photo.png",           # 路径 / URL / bytes / PIL.Image / data URL
    "红色圆形",             # 找什么（省略 = 识别主要物体）
    prompt=None,           # 直接给完整用户消息（给了就忽略 target）
    system_prompt=None,    # 覆盖系统提示词 —— 承载坐标口径，慎改
    on_event=print,        # 流式事件回调：reasoning/content/tool_call/finish/usage/model（见第 7 节）
)
```

`LocateResult` 上有什么：

| 字段 / 方法 | 说明 |
|---|---|
| `result.detections` | `list[Detection]`，主数据；也可直接 `for d in result` |
| `result.bboxes` / `.points` / `.labels` / `.centers` | 按类型取出的便捷视图 |
| `result.find("红")` | 按标签子串筛 |
| `result.annotated_path` | 标注图落盘路径；只有 `locate_to_file()` 会填，其余入口恒为 `None` |
| `result.warnings` | 旧刻度换算 / 坐标越界 / 没解析到坐标等告警 |
| `result.text` / `.reasoning` | 模型正文 / 思考过程（**证据**，排查时全靠它） |
| `result.usage` / `.duration_ms` / `.model` | token 用量、耗时、实际模型 |
| `result.to_dict()` / `.to_json()` / `.save(path)` | 序列化 |
| `result.describe()` | 人类可读的多行摘要（CLI 默认输出） |

打标：

```python
draw(image, result)                       # -> PIL.Image（不改动入参图）
draw(image, result, box_width=4, font_size=26, draw_label=True)
draw(image, result, scale_to_image=True)  # 线宽/字号按图片尺寸自动推（大图不再细到看不见）
save_annotated(image, result, path="out.png")     # -> Path
```

## 7. 看过程：流式事件

本库内部**恒走流式**（原因见下一节 FAQ），所以「模型正在想什么 / 正在写什么 / 正在调哪个工具」
一路都是现成的，只是默认收完流才把结果交给你。想实时看，三层粒度随便挑：

| 粒度 | 入口 | 适合 |
|---|---|---|
| 一条龙 | `locate_to_file(img, target, "out.png", on_event=cb)` | 只要标注图，进度顺手打一下 |
| 结构化 + 进度 | `Locator.locate(img, target, on_event=cb)` | 要 `result`，同时想看过程 |
| 只要事件流 | `DeepSeekVisionClient().stream(messages, ...)` | 自己做分栏显示 / 自己接工具往返 |

三层拿到的是**同一批事件**，共六种：

| `type` | 字段 | 说明 |
|---|---|---|
| `reasoning` | `text` | 思考内容的一个片段 |
| `content` | `text` | 正文的一个片段 |
| `tool_call` | `index` / `id` / `name` / `arguments` | 工具调用的一个分片，`arguments` 是**增量** |
| `finish` | `reason` | `stop` / `length` / `tool_calls` |
| `usage` | `usage` | token 用量（只在最后一个 chunk，兼容服务可能不给） |
| `model` | `model` | 服务端实际使用的模型名（去重后只来一条） |

三点必须知道：

- 片段**切分是任意的**（按 token，不按字/句），拼起来才是完整内容；一次定位调用实测 122 条事件。
- `tool_call` 的 `arguments` 是**逐字符**吐的（实测一次 47 个分片），要 `json.loads` 得自己按 `index` 拼；
  `complete()` 已经替你拼好，放在 `ChatReply.tool_calls`。
- 别自己写解包 —— 现成的示例直接抄：

```bash
python examples/stream_events.py            # 定位请求的事件流，逐条带时间戳 + 首字延迟
python examples/stream_events.py --tools    # 带 tools 的请求：工具调用一片片吐出来、再拼回去
python examples/stream_events.py --raw      # 事件的 JSON 原样打印
```

关于工具的边界：`tools` / `tool_choice` 由你**原样透传**给服务端，`ChatReply.tool_calls`
给你拼好的调用请求，但**本库不声明工具、也不执行工具** —— 要不要跑、跑完怎么把结果发回去，
是调用方的事。`Locator.locate` 这一路不带 `tools`，所以它不会有 `tool_call` 事件；
`use_tools=True` 依旧直接抛 `NotImplementedError`（v0.1 没有 Agent 循环）。

### 7.1 调试日志：把请求体和响应体落盘

上面那套事件是「实时看」，调试日志是「事后查」—— 它把**网络层**的报文与响应写成
一行一个 JSON 的文件（JSONL）。**默认全程关闭**，一行参数开启：

```python
locate("photo.png", "红色圆形", log_file="runs/logs/run.jsonl")
```

四种开法（越靠前越优先）：`log_file=` 参数 / `Locator(log_file=...)` / 环境变量
`QSML_LOG_FILE` / CLI 的 `--log-file PATH`（`--log` 则自动落到
`runs/logs/qsml-<时间戳>.jsonl`）。`log_file=False` 是**明确关闭**，用来盖掉环境变量里开着的日志。

一次调用会写下这些行：

| `event` | 内容 |
|---|---|
| `request` | 完整请求体（messages、stream 参数、tools…）+ 脱敏后的配置 |
| `event` | 每个流式事件（思考 / 正文 / 工具调用 / …），与 `on_event` 收到的是同一批 |
| `chunk` | 原始 chunk —— **只在 `DebugLog(path, chunks=True)` 时有** |
| `reply` | 拼好的完整响应体（正文 / 思考 / 工具调用 / usage / 结束原因） |
| `result` | 解析后的结构化结果（坐标、告警、原始项） |
| `error` | 任何异常（含空正文那类），原样抛出前先记一笔 |

两条安全线：

- **图片不进日志**。报文里的 data URL 会被换成 `data:image/png;base64,（省略 N 字符）`，
  只保留「类型 + 体积」—— 否则一张 1200x900 的图就是几百 KB base64，日志比图还大且没法读。
- **API Key 不进日志**。配置行走 `config.redacted()` 脱敏，只留首尾几位。

⚠️ 除此之外日志里**有完整的模型输入输出**（提示词、思考过程、坐标），适合自己排查，
别默认往公共 CI artifact 或别人的机器上丢。日志写失败不影响识别（只往 stderr 提醒一次）。

要连原始 chunk 一起记（排查「服务端是不是发了奇怪的字段」），自己构造对象传进去：

```python
from qsmy_deepseek_locator import DebugLog
locate("photo.png", "红色圆形", log_file=DebugLog("runs/logs/full.jsonl", chunks=True))
```

---

## 8. 常见问题

**Q：返回「正文是空的」（`EmptyResponseError`）怎么办？**
A：九成是思考 token 吃光了输出上限（此时 HTTP 仍是 200，`content` 就成了空串）。
调大 `max_tokens`、或关掉思考（`thinking=False`）、或降 `reasoning_effort`。异常信息里就写着这三条。

**Q：调用会不会超时？需要自己开流式吗？**
A：不用管，**本库内部恒走流式**（报文里固定带 `stream: true` 与 `stream_options.include_usage`），
`locate` / `locate_to_file` / CLI 全是同一条路径，只是收完流之后一次性把结果交给你。
流式对超时的意义是实测过的：同一张图、同一份报文、`timeout=2` 秒时，
**流式跑了 7.42 秒正常返回**（2880 个 chunk，相邻 chunk 最大间隔 507ms），
**非流式 2.14 秒就被 `APITimeoutError` 打断** —— 换句话说，关掉流式会让本来能成的请求直接失败。
代价是它的超时口径是「两次数据之间的静默」而不是总时长：模型迟迟不吐第一个字时照样会被打断，
所以 `timeout` 不要设得太贴 —— 默认就是 **300s**（连接/写入超时也用它）。
真挂住时的最坏等待是 `timeout × (max_retries + 1)`，默认即 300s × 3，
想收紧就传 `timeout=60` 或设 `QSML_TIMEOUT`。

**Q：模型一个目标都没找到，是报错吗？**
A：不是。`result.empty` 为真、`warnings` 里会说清是「模型明确回了空数组」还是「正文里没有坐标」。
后者通常意味着提示词没被遵守，该改提示词而不是重试。

**Q：坐标看着偏了 / 报了越界告警？**
A：先看 `result.warnings` 与 `result.text`。越界基本等于模型给了像素坐标，
通常是 system 提示词被改过 —— 坐标口径写在 `prompts.py` 里，请不要在 `target` 里另写一套。

**Q：能换成别的模型 / 别的厂商吗？**
A：`model` 与 `base_url` 都能改，但请先用 `bench` 验证。
特别注意官方另一档 `deepseek-v4-pro` **不支持图像理解**：图片会被静默丢弃，
HTTP 照样返回 200，只能从 `usage.prompt_tokens` 没涨看出来。

**Q：`image_detail` 能提高定位精度吗？**
A：不能。每张图服务端最多只算 384 token，大图无论如何都会被缩到约 800x800。
它改的是「缩放发生在哪一层」，不是模型真正看到的像素数。

**Q：为什么极扁 / 极长的图上定位很差？**
A：那是模型能力的边界，不是库的缺陷：长边被缩到约 1000 后，密集小目标只剩几像素。
详见 [`docs/API-NOTES.md`](https://github.com/QsmyHyly/qsmy-deepseek-locator/blob/main/docs/API-NOTES.md) 第 9 节。

**Q：想实时看到模型的思考、正文、工具调用，有现成的代码吗？**
A：有，见第 7 节。一句话版：给 `locate` / `locate_to_file` 传 `on_event=你的回调`，
或者用最细的一层 `DeepSeekVisionClient().stream(messages)`。
现成可跑的示例是 `examples/stream_events.py`（加 `--tools` 演示工具调用分片，加 `--raw` 打事件 JSON）。

**Q：异常该怎么兜？网络中途断了抛什么？**
A：全都继承 `LocatorError`，`except LocatorError` 一把兜住即可，具体的子类见第 6 节。
**流跑到一半**才断（服务端断连、读超时、流里回一个 error 事件）也算 —— 本库会把它包成
`APIError`，原始异常挂在 `__cause__` 上，不会丢。所以 CLI 那种「接口报错就退出码 1 加一句
错误：…」的承诺，对中途失败同样成立。

**Q：出问题了，想看到底发出去什么、模型回了什么？**
A：开调试日志，见 7.1 节：`locate("photo.png", "红色圆形", log_file="runs/logs/run.jsonl")`，
或 CLI 加 `--log`。请求体、流式事件、完整响应体、解析结果、异常都会写成 JSONL；
图片 data URL 会省略成占位符，API Key 会脱敏。**默认不开。**

**Q：模型要调用工具时，本库会替我执行吗？**
A：**不会**。`tools` 原样透传、调用请求拼好放在 `ChatReply.tool_calls`，到这儿为止 ——
执行工具、把结果发回去、决定要不要再来一轮，全是调用方的事。
`locate` / `locate_to_file` 的 `use_tools=True` 会直接抛 `NotImplementedError`（v0.1 没有 Agent 循环）。

## 9. 文档

- [`docs/API-NOTES.md`](https://github.com/QsmyHyly/qsmy-deepseek-locator/blob/main/docs/API-NOTES.md) —— DeepSeek 接口事实与踩坑记录（**这个库为什么长这样**）。
  代码里凡是为某条坑做了特殊处理的地方，都用 `@doc docs/API-NOTES.md#<锚点>` 指回对应小节。
- 其余说明按「文档就近写在代码里」的原则放在模块头注释：
  `prompts.py`（提示词为什么这么写）、`parsing.py`（刻度兜底与为何不猜）、
  `images.py`（编码策略）、`drawing.py`（中文字体）、`benchmark.py`（评测口径）、
  `debuglog.py`（调试日志记什么、为什么不记图片）。

## 10. 与 `deepseek-vision-annotation` 的关系

本库是从那个演示项目里**抽出来的核心**，两者的分工：

| | deepseek-vision-annotation | 本库 |
|---|---|---|
| 形态 | 完整演示程序（FastAPI + 网页对比 + 历史记录 + 工具执行框架） | 可 `pip install` 的库 |
| 交互 | 浏览器界面、SSE 流式控制台 | Python API + CLI |
| 无 Key 时 | 进 Mock 模式，页面照样能演示 | **直接报错**（不给假数据） |
| 坐标口径 / 提示词 / 打标逻辑 | 同一套，已在本库中保留 | 同一套 |

## 11. 许可证

[MIT](https://github.com/QsmyHyly/qsmy-deepseek-locator/blob/main/LICENSE)。`docs/` 中的接口事实整理自 DeepSeek 官方文档与实际调用观测，
以官方站点 <https://api-docs.deepseek.com> 为准。
## 12. 已知限制

发布前做过一轮逐文件的对抗性复审，下面这些是**确认存在、但 0.1.0 没有修**的。
写在这里，免得你踩到了以为是自己的用法不对：

- **`Locator(thinking=True)` 会被 `locate_to_file()` 的函数默认值盖掉。** 后者的默认值是
  `thinking=False, image_detail="original"`（一行式入口图快），而它是**函数默认值**这一层，
  于是会盖过构造时传的设置。想沿用构造参数就显式写 `thinking=None, image_detail=None`。
  这是 0.1.0 里唯一一处「函数默认值赢了构造参数」的地方，与第 6 节写的优先级相反，计划在 0.2 改掉。
- **刻度兜底只看数值，看不出坐标的来源。** 整批最大值落在 `1.0 < max ≤ 1000` 时，一律按
  0~1000 旧口径除以 1000。**默认提示词路径下这是安全的**：库的 system_prompt 已明确告诉模型
  它拿不到真实分辨率、统一按 0.0~1.0 输出，模型因此给不出真实像素坐标（它"报的像素"落在自己
  每次编的画布上，实测同一张图三次给出 1000x750 / 1000x800 / 1024x768）。**但覆盖
  `system_prompt` 去要像素坐标、或把别的模型（如 Qwen2.5-VL，它返回真实像素）的输出喂给
  `parse_detections` 时，这一步会改错数据。** 换算一定会留下告警，但告警只陈述"已除以 1000"、
  不替你判断该不该除 —— 请对照结果里的 `image_size` 复核。
- **这个换算是整批判定的**（`max()` 取自所有检测项的所有坐标）：混着给（一部分 >1000、
  一部分没超）时，整批都会按旧口径处理或整批都不处理，不会逐项各判各的。
- **输出被截断时，提示语指的是提示词，而不是输出预算。** `finish_reason == "length"`
  且正文里没解析出坐标时，`warnings` 说的是「正文里没有可解析的坐标，可检查提示词」——
  真实原因往往是思考 token 吃光了 `max_tokens`。看到这条时请一并调大 `max_tokens`
  或关掉思考（`thinking=False`），别只改提示词。
- **`max_side=None`（默认）时图片零重编码，因此不做图像内容校验。** 把一个非图片文件
  （比如 `.png` 后缀的 HTML）喂进去，本库不会在本地拦下它，报错会来自服务端。
  想严格拦截就传 `max_side=`（例如 `max_side=1600`），那条路径会真的解码图片。
- **定位是「框出大概位置」，不是像素级分割。** 不做 NMS、不去重，同一个目标可能出现两个框；
  框的精度受模型限制（每张图服务端只算 384 token）。
- **同一张图多次调用，返回的目标集合不保证一致。** 本库不发送任何采样参数
  （`temperature` / `top_p` / `seed` 一个都没设），每次调用都是一次独立采样。实测同一张图、
  同一目标连跑三次，目标数给出过 `17 / 10 / 8` 这样的差别，命名与粒度也跟着变；
  **反倒是同一个目标的位置相对稳**（实测同一栋楼三轮中心点相差 1.5~2%）。所以结果看着"飘"时，
  先怀疑「这一次框了哪些目标」，而不是「坐标算错了」。要复现性就自己多跑几次按 label 聚类投票。
- **标注图的标签会自动避让，落点不保证和框的位置一一对应。** 标签默认画在框上方，贴图片
  边缘时翻进框内侧、被别的标签压住时向下错开，四边都保证不越出画布 —— 想完全固定位置，
  就自己拿 `Detection` 列表用 PIL 画。
- **0.1.0 没有 Agent / 工具执行循环。** `use_tools=True` 直接抛 `NotImplementedError`；
  `client.complete(..., tools=[...])` 能拿到完整工具调用参数，但本库不替你执行。
- **实测只跑过 CPython 3.11 与 3.12。** `requires-python = ">=3.9"` 是按语法静态核对的
  （全部模块都有 `from __future__ import annotations`，没有 3.10+ 独有语法），没有真在 3.9 / 3.10 上跑过。
