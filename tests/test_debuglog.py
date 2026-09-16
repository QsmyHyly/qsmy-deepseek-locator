"""调试日志的自测：默认不开、开了写什么、写不进去会怎样。

全部离线：网络层用「把 _create 打桩成返回预设 chunk」的方式绕过，不打真实接口。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from qsmy_deepseek_locator import DebugLog, Locator, Settings, locate
from qsmy_deepseek_locator.client import DeepSeekVisionClient
from qsmy_deepseek_locator.debuglog import _fallback, _plain, coerce_log, default_log_path
from qsmy_deepseek_locator.errors import APIError

REPLY = '[{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "红色圆形"}]'


def _chunk(*, content=None, reasoning=None, finish=None, usage=None):
    delta = NS(content=content, reasoning_content=reasoning, tool_calls=None)
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=usage, model="fake")


def _fake_stream_client():
    """一个 DeepSeekVisionClient，但发请求那一步被换成了预设 chunk。"""
    client = DeepSeekVisionClient(Settings(api_key="test-key"))
    chunks = [_chunk(reasoning="看图"), _chunk(content=REPLY), _chunk(finish="stop")]
    client._create = lambda kwargs, settings: iter(chunks)
    return client


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _events(records: list[dict]) -> list[str]:
    return [r["event"] for r in records]


# --------------------------------------------------------------------------- #
# coerce_log：参数怎么变成「开 / 不开」
# --------------------------------------------------------------------------- #
def test_coerce_log_off_by_default():
    assert coerce_log(None) is None
    assert coerce_log(False) is None
    assert coerce_log("") is None  # .env 里写 QSML_LOG_FILE= 是常见写法


def test_coerce_log_true_uses_default_path():
    log = coerce_log(True)
    assert log is not None
    assert log.path == default_log_path()
    assert str(log.path).replace("\\", "/").startswith("runs/logs/qsml-")
    assert log.path.suffix == ".jsonl"


def test_coerce_log_path_and_instance():
    log = coerce_log("runs/logs/x.jsonl")
    assert log is not None and log.path == Path("runs/logs/x.jsonl")
    mine = DebugLog("custom.jsonl", chunks=True)
    assert coerce_log(mine) is mine  # 传对象就原样用，chunks 等细节归调用方


def test_coerce_log_rejects_nonsense():
    with pytest.raises(ValueError, match="log_file"):
        coerce_log(123)


# --------------------------------------------------------------------------- #
# 默认不开
# --------------------------------------------------------------------------- #
def test_locate_off_by_default(tmp_path, sample_png, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = locate(sample_png, "圆形", client=_fake_stream_client())
    assert len(result) == 1
    assert not (tmp_path / "runs").exists()  # 没开日志就不该凭空造目录


# --------------------------------------------------------------------------- #
# 开了之后写了什么
# --------------------------------------------------------------------------- #
def test_log_records_the_whole_story(tmp_path, sample_png):
    path = tmp_path / "run.jsonl"
    locator = Locator(client=_fake_stream_client())
    locator.locate(sample_png, "红色圆形", log_file=path)
    records = _read(path)
    assert _events(records) == ["request", "event", "event", "event", "event", "reply", "result"]
    # 事件行里能直接看到思考与正文
    kinds = [r["data"].get("type") for r in records if r["event"] == "event"]
    assert kinds == ["model", "reasoning", "content", "finish"]
    reply = next(r["data"] for r in records if r["event"] == "reply")
    assert reply["text"] == REPLY
    assert reply["reasoning"] == "看图"
    result = next(r["data"] for r in records if r["event"] == "result")
    assert result["detections"][0]["label"] == "红色圆形"
    assert result["counts"]["total"] == 1
    assert "text" not in result  # 正文在 reply 行里，result 行不重复记一遍


def test_log_is_appended_not_overwritten(tmp_path, sample_png):
    path = tmp_path / "run.jsonl"
    locator = Locator(client=_fake_stream_client())
    locator.locate(sample_png, "圆形", log_file=path)
    first = len(_read(path))
    locator.locate(sample_png, "圆形", log_file=path)
    assert len(_read(path)) == first * 2  # 两次调用各记各的，追加不覆盖


def test_request_body_has_the_prompt_and_redacted_image(tmp_path, sample_png):
    path = tmp_path / "run.jsonl"
    Locator(client=_fake_stream_client()).locate(sample_png, "红色圆形", log_file=path)
    body = next(r["data"]["body"] for r in _read(path) if r["event"] == "request")
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    text_part = body["messages"][-1]["content"][-1]["text"]
    assert "红色圆形" in text_part
    image_url = body["messages"][-1]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert "省略" in image_url and "字符" in image_url
    # 原始 base64 一个字都不该留在日志里（日志文件比图还大就没意义了）
    assert len(image_url) < 200


def test_api_key_never_reaches_the_log(tmp_path, sample_png):
    path = tmp_path / "run.jsonl"
    Locator(api_key="sk-abcdef1234567890", client=_fake_stream_client()).locate(
        sample_png, "圆形", log_file=path,
    )
    text = path.read_text(encoding="utf-8")
    assert "sk-abcdef1234567890" not in text
    settings = next(r["data"]["settings"] for r in _read(path) if r["event"] == "request")
    assert settings["api_key"].startswith("sk-abc")  # redacted() 只留首尾


def test_error_is_logged_then_reraised(tmp_path, sample_png, monkeypatch):
    path = tmp_path / "run.jsonl"
    client = _fake_stream_client()

    def boom(kwargs, settings):
        raise RuntimeError("网络炸了")

    monkeypatch.setattr(client, "_create", boom)
    # 底层异常会被包成本库的 APIError，原始异常挂在 __cause__ 上：
    # 调用方既能按 LocatorError 一把兜住，也还能拿到原始类型去查根因。
    with pytest.raises(APIError, match="网络炸了") as excinfo:
        Locator(client=client).locate(sample_png, "圆形", log_file=path)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    records = _read(path)
    assert _events(records) == ["request", "error"]  # 报文照样留下了
    assert records[-1]["data"]["type"] == "RuntimeError"  # 日志里记的是**原始**类型


def test_env_var_turns_logging_on(tmp_path, sample_png, monkeypatch):
    path = tmp_path / "from_env.jsonl"
    monkeypatch.setenv("QSML_LOG_FILE", str(path))
    Locator(client=_fake_stream_client()).locate(sample_png, "圆形")
    assert path.exists()


def test_false_overrides_the_env_var(tmp_path, sample_png, monkeypatch):
    monkeypatch.setenv("QSML_LOG_FILE", str(tmp_path / "from_env.jsonl"))
    Locator(client=_fake_stream_client()).locate(sample_png, "圆形", log_file=False)
    assert not (tmp_path / "from_env.jsonl").exists()


def test_locator_constructor_enables_logging(tmp_path, sample_png):
    """Locator(log_file=...) 走的是 Settings 覆盖那条路（log_file 是 Settings 字段）。"""
    path = tmp_path / "ctor.jsonl"
    Locator(client=_fake_stream_client(), log_file=str(path)).locate(sample_png, "圆形")
    assert path.exists()


def test_true_uses_the_auto_path(tmp_path, sample_png, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Locator(client=_fake_stream_client()).locate(sample_png, "圆形", log_file=True)
    written = list((tmp_path / "runs" / "logs").glob("qsml-*.jsonl"))
    assert len(written) == 1


def test_log_writes_only_the_network_layer_when_chunks_off(tmp_path, sample_png):
    path = tmp_path / "run.jsonl"
    log = DebugLog(path)
    Locator(client=_fake_stream_client()).locate(sample_png, "圆形", log_file=log)
    assert "chunk" not in _events(_read(path))


def test_chunks_true_also_records_raw_chunks(tmp_path, sample_png):
    path = tmp_path / "run.jsonl"
    log = DebugLog(path, chunks=True)
    Locator(client=_fake_stream_client()).locate(sample_png, "圆形", log_file=log)
    chunks = [r for r in _read(path) if r["event"] == "chunk"]
    assert len(chunks) == 3
    assert chunks[0]["data"]["model"] == "fake"


def test_logging_off_does_not_pass_log_to_the_client(sample_png, fake_client):
    """自备客户端不必为了「不用日志」而改签名 —— 这条守住那个承诺。"""
    client = fake_client("[ ]")
    Locator(client=client).locate(sample_png, "圆形")
    assert client.calls[0]["log"] is None


def test_logging_on_passes_the_log_down(sample_png, fake_client, tmp_path):
    client = fake_client("[ ]")
    Locator(client=client).locate(sample_png, "圆形", log_file=tmp_path / "x.jsonl")
    assert isinstance(client.calls[0]["log"], DebugLog)


# --------------------------------------------------------------------------- #
# 序列化与容错
# --------------------------------------------------------------------------- #
def test_fallback_handles_arbitrary_objects():
    class Dumpable:
        def model_dump(self):
            return {"ok": 1}

    class Plain:
        def __init__(self):
            self.value = 2

    class Opaque:
        __slots__ = ()

    assert _fallback(Dumpable()) == {"ok": 1}
    assert _fallback(Plain()) == {"value": 2}
    assert isinstance(_fallback(Opaque()), str)  # 最后一定有 str() 兜底


def test_plain_keeps_short_strings_and_caps_depth():
    assert _plain("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"
    deep: object = "叶子"
    for _ in range(20):
        deep = [deep]
    assert "省略" in json.dumps(_plain(deep), ensure_ascii=False)


def test_write_failure_warns_but_does_not_raise(tmp_path, capsys):
    """日志写不出去（这里让路径指向一个目录）不能反过来把识别搞挂。"""
    log = DebugLog(tmp_path)
    log.write("request", {"a": 1})  # 不抛异常
    assert "调试日志写入失败" in capsys.readouterr().err
    log.write("request", {"a": 2})  # 只提醒一次，不刷屏
    assert capsys.readouterr().err == ""

