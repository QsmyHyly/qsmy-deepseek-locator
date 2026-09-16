# qsmy-deepseek-locator

> 用 DeepSeek 视觉模型做**物体定位**：给它一张图和一句「找什么」，
> 拿回 **0.0~1.0 的归一化坐标**（框 / 点）与中文名称，需要的话直接把框和标签画回图上。

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

本地开发（当前阶段就用这个，尚未发布 PyPI）：

```bash
cd qsmy-deepseek-locator
python -m venv .venv && .venv\Scripts\activate      # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -e ".[dev]"
```

依赖只有三个：`openai`（DeepSeek 走 OpenAI 兼容协议）、`Pillow`（读图 / 打标）、`requests`（下载图片 URL）。

## 2. 配 API Key

```bash
# Windows（持久生效）
setx DEEPSEEK_API_KEY "sk-你的key"
# macOS / Linux
export DEEPSEEK_API_KEY="sk-你的key"
```

也可以不设环境变量，直接传参：`Locator(api_key="sk-...")` 或 `locate(..., api_key="sk-...")`。
其余可配项见 [`.env.example`](./.env.example)。

> **没有 Key 会怎样**：直接抛 `MissingAPIKeyError`，并告诉你三种配法。
> 本库刻意**不提供**「无 Key 时返回假数据」的降级 —— 演示程序这样做很方便，
> 但库不行：用户会拿着一堆看起来正常的假坐标当真结果。

## 3. 三种用法

### 3.1 一行式

```python
from qsmy_deepseek_locator import locate

result = locate("photo.png", "画面里的人")      # 本地路径 / URL / bytes / PIL.Image / data URL 都行
print(result.summary())                        # {'total': 3, 'bbox_count': 3, ...}
```

### 3.2 复用定位器（多图批量时用这个）

```python
from qsmy_deepseek_locator import Locator, draw

locator = Locator(thinking=False)              # 关掉思考：更快、更省 token（实测不掉准确率）
for path in ["a.png", "b.png", "c.png"]:
    result = locator.locate(path, "按钮")
    draw(path, result).save(path.replace(".png", "_annotated.png"))
    print(path, result.labels)
```

`locate()` 每次都会重新读环境变量、新建客户端；批量场景请自己建 `Locator`。

### 3.3 命令行

```bash
qsmy-deepseek-locator photo.png -t "登录按钮"                 # 打印坐标 + 生成 photo_annotated.png
qsmy-deepseek-locator photo.png -t "人" --print-json          # 只吐 JSON（可直接管道给 jq）
qsmy-deepseek-locator photo.png -t "人" --json r.json         # 连证据一起存盘
qsmy-deepseek-locator photo.png -t "人" --no-thinking         # 关思考，更快
qsmy-deepseek-locator photo.png --show-reasoning              # 实时看模型的思考过程
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
qsmy-deepseek-locator bench --count 5 --n-shapes 3 --annotate
# 图片 5 张 | 真值 15 个 | 预测 15 个
# 检出率 100.0%（15/15）   精确率 100.0%   平均 IoU 0.897
# 标签准确率 100.0%（颜色 100.0% / 形状 100.0%）
# 平均耗时 4.2s/张   带告警的图片 0 张
```

它用代码生成「已知答案」的几何图形图，真值顺手算出来，再按 IoU（阈值 0.5）匹配预测框。
产物落在 `runs/benchmark/`：`images/`（图 + `ground_truth.json`）、`annotated/`（预测画回图）、`report.json`。

编程接口：`from qsmy_deepseek_locator.benchmark import run_benchmark, evaluate_sample`。

## 6. API 速查

```python
from qsmy_deepseek_locator import Locator, Detection, draw, save_annotated, __version__

locator = Locator(
    api_key=None,          # 默认读 DEEPSEEK_API_KEY
    model="deepseek-flash",
    thinking=False,        # None=沿用服务端默认；False=关思考（更快更省）
    reasoning_effort=None, # low/medium/high/xhigh/max
    image_detail=None,     # low/high/original/auto；默认不发送该字段
    max_tokens=None,       # 输出上限（含思考 token）
    timeout=120,
    max_side=None,         # 发送前把图缩到最长边不超过它（省流量，不影响坐标精度）
)

result = locator.locate(
    "photo.png",           # 路径 / URL / bytes / PIL.Image / data URL
    "红色圆形",             # 找什么（省略 = 识别主要物体）
    prompt=None,           # 直接给完整用户消息（给了就忽略 target）
    system_prompt=None,    # 覆盖系统提示词 —— 承载坐标口径，慎改
    on_event=print,        # 流式事件回调：{"type": "reasoning"|"content"|"finish"|...}
)
```

`LocateResult` 上有什么：

| 字段 / 方法 | 说明 |
|---|---|
| `result.detections` | `list[Detection]`，主数据；也可直接 `for d in result` |
| `result.bboxes` / `.points` / `.labels` / `.centers` | 按类型取出的便捷视图 |
| `result.find("红")` | 按标签子串筛 |
| `result.warnings` | 旧刻度换算 / 坐标越界 / 没解析到坐标等告警 |
| `result.text` / `.reasoning` | 模型正文 / 思考过程（**证据**，排查时全靠它） |
| `result.usage` / `.duration_ms` / `.model` | token 用量、耗时、实际模型 |
| `result.to_dict()` / `.to_json()` / `.save(path)` | 序列化 |
| `result.describe()` | 人类可读的多行摘要（CLI 默认输出） |

打标：

```python
draw(image, result)                       # -> PIL.Image（不改动入参图）
draw(image, result, box_width=4, font_size=26, draw_label=True)
save_annotated(image, result, path="out.png")     # -> Path
```

## 7. 常见问题

**Q：返回「正文是空的」（`EmptyResponseError`）怎么办？**
A：九成是思考 token 吃光了输出上限（此时 HTTP 仍是 200，`content` 就成了空串）。
调大 `max_tokens`、或关掉思考（`thinking=False`）、或降 `reasoning_effort`。异常信息里就写着这三条。

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
详见 [`docs/API-NOTES.md`](./docs/API-NOTES.md) 第 9 节。

## 8. 文档

- [`docs/API-NOTES.md`](./docs/API-NOTES.md) —— DeepSeek 接口事实与踩坑记录（**这个库为什么长这样**）。
  代码里凡是为某条坑做了特殊处理的地方，都用 `@doc docs/API-NOTES.md#<锚点>` 指回对应小节。
- 其余说明按「文档就近写在代码里」的原则放在模块头注释：
  `prompts.py`（提示词为什么这么写）、`parsing.py`（刻度兜底与为何不猜）、
  `images.py`（编码策略）、`drawing.py`（中文字体）、`benchmark.py`（评测口径）。

## 9. 与 `deepseek-vision-annotation` 的关系

本库是从那个演示项目里**抽出来的核心**，两者的分工：

| | deepseek-vision-annotation | 本库 |
|---|---|---|
| 形态 | 完整演示程序（FastAPI + 网页对比 + 历史记录 + 工具执行框架） | 可 `pip install` 的库 |
| 交互 | 浏览器界面、SSE 流式控制台 | Python API + CLI |
| 无 Key 时 | 进 Mock 模式，页面照样能演示 | **直接报错**（不给假数据） |
| 坐标口径 / 提示词 / 打标逻辑 | 同一套，已在本库中保留 | 同一套 |

## 10. 许可证

[MIT](./LICENSE)。`docs/` 中的接口事实整理自 DeepSeek 官方文档与实际调用观测，
以官方站点 <https://api-docs.deepseek.com> 为准。
