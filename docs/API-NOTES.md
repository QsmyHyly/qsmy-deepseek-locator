# DeepSeek 接口事实与踩坑记录

> 本文件解决什么问题：**这个库为什么长这样**。
> 下面每一条都是实测结论（不是抄文档），代码里凡是为此做了特殊处理的写法，
> 都会用 `@doc docs/API-NOTES.md#<锚点>` 指回对应小节。
> 换模型 / 换服务商前请重跑一遍 `qsmy-deepseek-locator bench` 再下结论。

---

## 1. 模型与端点

| 项 | 值 |
|---|---|
| 端点 | `POST https://api.deepseek.com/chat/completions`（OpenAI 兼容） |
| 认证 | `Authorization: Bearer $DEEPSEEK_API_KEY` |
| 本项目用的模型 | `deepseek-flash`（= DeepSeek-V4.1-Flash） |
| 不能用的模型 | `deepseek-v4-pro` —— **不支持图像理解**，见第 7 节 |

`GET /models` 返回的列表**不等于**可用模型清单：一些带到期日的实验模型能直接调用但不列在里面。
本库只发 `chat/completions` 一条路，因此任何 OpenAI 兼容服务都可以用 `base_url` 指过去。

@doc 对应实现：`config.py`（DEFAULT_BASE_URL / DEFAULT_MODEL）

---

## 2. 传图的三种方式

| 方式 | 写法 | 限制 |
|---|---|---|
| base64 内联 | `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` | 计入请求体总量；单图 ≤ 32 MiB |
| 外部 URL | `{"type":"image_url","image_url":{"url":"https://..."}}` | URL ≤ 8192 字符；≤ 32 MiB；60 秒内要下载完 |
| Files API | `{"type":"file","file_id":"..."}` | 单图可到 64 MiB，适合多请求复用同一张图 |

- **图片只能出现在 `user` 消息里**；放进 `system` / `assistant` 会直接 400。
- 本库默认走 base64 内联：本地图片 / PIL 图 / bytes 都能用，不需要先把图传到某个公网地址。
  不需要缩放时**不重新编码**，直接把原始字节 base64 上去（见 `images.py` 的模块说明）。

@doc 对应实现：`client.py`（image_part / build_messages）、`images.py`（to_data_url）

---

## 3. 图片 token 与尺寸

- **每张图最多只算 384 token**：比约 384x384 小的会被放大，更大的会被缩到约 800x800。
  所以 2000x2000 和 5000x5000 的 token 数一样 —— **靠堆分辨率换精度是没用的**。
- 于是 `image_url.detail`（low / high / original / auto）改的只是「缩放发生在哪一层」，
  不是模型真正看到的像素数。本库默认**根本不发送这个字段**，报文与已验证过的链路逐字节一致。
- 单请求最多 600 张图；含 15 张及以上时单边最大 4096 像素（否则 8192）。

@doc 对应实现：`config.py`（IMAGE_DETAILS）、`client.py`（image_part 只在合法时带 detail）

---

## 4. 思考模式与 reasoning_effort

- DeepSeek 是**思考模型**：先产 `reasoning_content`（思维链）再产 `content`（正文），默认开启。
- `thinking` **不是** Chat Completions 的顶层字段，必须走 `extra_body`：

  ```python
  client.chat.completions.create(
      model="deepseek-flash",
      messages=[...],
      extra_body={"thinking": {"type": "disabled"}},   # 关闭思考
      reasoning_effort="low",                          # 强度反而是顶层参数
  )
  ```

  直接传 `thinking=...` 会被 SDK 以「未知关键字」拒绝。
- `reasoning_effort` 合法值：`low` / `medium` / `high` / `xhigh` / `max`；
  本库对非法值**一律不传**（宁可走服务端默认，也不要 400）。
- 关闭思考时 `temperature` 之类参数才有效；思考模式下传了也不生效。
- 参考项目的 A/B 实测：同一批图关掉思考**快 2.2 倍**（4.9s → 2.2s），
  检出率与标签准确率都是 100%，平均 IoU 0.842 → 0.873。定位类任务关掉通常不亏。

@doc 对应实现：`client.py`（thinking_payload / resolve_thinking / build_request）

---

## 5. 思考 token 会吃掉 max_tokens

**症状**：HTTP 200，但 `message.content` 是空字符串。
**原因**：思考 token 与正文**共用**输出上限。上限被思考占满时，正文一个字都轮不到。

实测：`max_tokens=700` 时 reasoning 用满 700、正文为空；提到 2500 就正常。
`usage.completion_tokens_details.reasoning_tokens` 能直接看到思考花了多少。

**对策**（本库怎么处理的）：
1. 绝不把「空正文」当成「图里没有目标」——那是两回事。空正文一律抛
   `EmptyResponseError`，异常信息里点名 `max_tokens` 与 `thinking=False` 两条出路。
2. 默认**不发送** `max_tokens`，交回服务端默认；要设就设够（`--max-tokens 4096`）。
3. 定位类任务优先考虑 `thinking=False`，从根上避免这个问题。

@doc 对应实现：`client.py`（DeepSeekVisionClient.complete 的空正文检查、_empty_hint）

---

## 6. 流式必须同时读 reasoning_content

只读 `delta.content` 会误判成「流式没有输出」——先到的增量**全是思考内容**。
最后一个 chunk 还可能带一个 `choices=[]`、只装 `usage` 的包，
直接取 `chunk.choices[0]` 会 IndexError。

本库的处理：`client.stream()` 把每个 chunk 拆成
`{"type": "reasoning"|"content"|"tool_call"|"finish"|"usage"|"model"}` 事件，
既吞掉了空 choices 的坑，也让 CLI 能实时显示进度。
`stream_options={"include_usage": True}` 用来顺带拿 usage；遇到不认这个字段的兼容服务会自动去掉重试一次
（那只影响「能不能顺手拿到 usage」，不该让整轮识别失败）。

还有一个反直觉的地方：**服务端在每个 chunk 上都带 `model` 字段**。
照单全收的话一次调用会甩出上百条一模一样的 model 事件（实测 110 条），
所以 `stream()` 只在模型名**第一次出现**时发一条。

@doc 对应实现：`client.py`（_events_from_chunk / _create / stream 里的 model 去重）

---

## 6.1 工具调用也是流式的，而且一个字符一个 chunk

给 `deepseek-flash` 带上 `tools` 参数后，本机真跑的返回形状如下（流式）：

```
第 1 个分片  {"index": 0, "id": "call_00_xxx", "type": "function",
              "function": {"name": "crop_region", "arguments": ""}}
第 2..N 分片 {"index": 0, "id": null, "type": null,
              "function": {"name": null, "arguments": "<一个字符>"}}
最后一个分片 {"delta": {"content": ""}, "finish_reason": "tool_calls"}
```

要点：

1. **id 与函数名只在第一个分片里**，之后全是 `null`。累积时用后到的 null 覆盖会丢字段。
2. `arguments` 是**增量**、而且碎得离谱：一次 `{"name": "红色圆形", "x1": 0.13, ...}`
   实测被拆成 47 个 chunk（一个字符一个），**必须按 `index` 自己拼**才能 `json.loads`。
3. 多工具并发时按 `index` 区分，分片可能交错到达，别假设某个工具的分片是连续的。
4. 结束原因是 `tool_calls`（不是 `stop`），而且此时 `content` 常常是**空串** ——
   有工具调用时正文为空是正常的，不能当成「空正文」报错（见第 5 节的 `EmptyResponseError`）。

本库的处理：`client.stream()` 发 `{"type": "tool_call", "index", "id", "name", "arguments"}`
事件（arguments 为增量）；`complete()` 用 `_accumulate_tool_calls` 按 index 拼回完整对象，
放进 `ChatReply.tool_calls`，并跳过「空正文」检查。
`tools` / `tool_choice` 由调用方原样透传 —— **本库不声明工具、也不执行工具**。

@doc 对应实现：`client.py`（_tool_call_events / _accumulate_tool_calls）、`locate.py`（use_tools 的报错文案）

---

## 7. 非视觉模型不会报错，只会静默丢图

把图片发给不支持视觉的模型（例如 `deepseek-v4-pro`）时，实测 **HTTP 仍是 200**，
图片被悄悄丢掉，模型回一句「我无法处理图片」。

⇒ 判断图片到底有没有被接收，**看 `prompt_tokens` 有没有涨**，别指望状态码。
（本库把 `usage` 原样放进 `LocateResult.usage`，就是为了留这条排查路径。）

所以本库的默认模型固定为 `deepseek-flash`；`--model` 是逃生口，不是推荐用法。

@doc 对应实现：`config.py`（DEFAULT_MODEL 的注释）、`locate.py`（LocateResult.usage）

---

## 8. 坐标口径：为什么必须是 0.0~1.0，以及提示词有多要命

**模型看不到图片的真实分辨率**：服务端先缩放再喂给它，且**不回传缩放后的尺寸**。
它输出的「像素」落在它每次自己编的画布上 —— 实测同一张图三次调用分别给出
1000x750 / 1000x800 / 1024x768。像素值既不准确也不可复现。

缩放本身是**纯线性、等比、无补边**的（拟合 R²≈1、截距≈0），所以**相对比例不受影响**，
这就是全库统一用 0.0~1.0 的原因。

参考项目的 A/B 实测（5 张图 × 3 目标 = 15 个真值）：

| 提示词版本 | 检出率 | 平均 IoU | 输出像素坐标的图片 |
|---|---|---|---|
| 旧：说「使用归一化值」又说「无需考虑分辨率」 | **40%** (6/15) | 0.655 | 3/5 |
| 新：显式换算公式 + 禁止像素值 + 交代刻度不确定 | **100%** (15/15) | 0.897 | 0/5 |

结论有两条，都很反直觉：
1. 模型「看图 + 认颜色形状」一直很强（两版标签准确率都是 100%），失败**几乎全来自坐标约定没被遵守**；
2. 提示词里**自相矛盾**比写得少更致命。改一句话就是 60 个百分点的差别。

本库据此做了三件事：提示词写死口径（`prompts.py`）；解析层对 0~1000 旧刻度做无损兜底、
对像素坐标**告警而不猜测**（`parsing.py`）；自带自动判分的评测（`benchmark.py`），
让「改了提示词到底有没有变好」有据可查。

@doc 对应实现：`prompts.py`（DEFAULT_SYSTEM_PROMPT）、`parsing.py`（normalize_to_unit）、`benchmark.py`

---

## 9. 模型能力的已知边界（不是 bug，别当 bug 修）

极端长宽比的密集小目标上，模型会「颜色全认对、位置基本全错」：

| 图 | 思考开 | 思考关 |
|---|---|---|
| 900x900 常规圆点阵 | 检出 9/9 | 7/9 |
| 2400x300（8:1） | 检出 0/9 | 1/9 |
| 800x2400（1:3） | 检出 3/9 | 3/9 |

原因：服务端把长边缩到约 1000 后，圆点直径只剩 5~6 像素，已接近有效分辨率下限。
**这是模型能力的边界**，提示词与代码都救不回来。演示或验收时别把它当成库的缺陷。
