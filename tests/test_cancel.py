"""取消机制（P1-4）的自测：cancel_event 要在**发请求之前**和**流式途中**都能喊停。

反馈原文是「全同步阻塞、无取消机制」——安卓上只能靠"超时即 kill 整个 Python 进程"绕过，
代价是一次误杀会把整个 worker 带走。库里加一个 threading.Event 之后，
调用方至少有一条体面的退出路径。

为什么不自己造一个假的线程去打断：cancel_event 是**调用方主动 set** 的开关，
本库只在检查点上读它 —— 所以自测里直接预先 set() 就能精确复现两种情况，
不必依赖调度与时序（那类测试在 CI 上会偶发红）。
"""

from __future__ import annotations

import threading

import pytest

from qsmy_deepseek_locator import Locator, locate
from qsmy_deepseek_locator.errors import CancelledError, LocatorError

BOX = '[{"bbox_2d": [0.25, 0.25, 0.75, 0.75], "label": "方块"}]'


class _EventFiringClient:
    """假客户端：complete() 期间按预设节奏回调若干事件。"""

    def __init__(self, text=BOX, events=("content", "content", "finish"), on_first=None):
        self.text = text
        self.events = events
        self.on_first = on_first          # 第一次事件时执行（用来中途 set cancel_event）
        self.calls = 0

    def complete(self, messages, *, settings=None, on_event=None, log=None):
        self.calls += 1
        from qsmy_deepseek_locator import ChatReply
        for index, kind in enumerate(self.events):
            if on_event is not None:
                on_event({"type": kind, "text": self.text[:4]})
            if index == 0 and self.on_first is not None:
                self.on_first()
        return ChatReply(text=self.text, model="fake")


def test_cancelled_before_start_never_calls_the_model(sample_png):
    """进来时已经 set 了 —— 一次 API 调用都不该花（"点了取消还扣钱"是最招骂的 bug）。"""
    client = _EventFiringClient()
    event = threading.Event()
    event.set()
    locator = Locator(client=client, api_key="sk-test")
    with pytest.raises(CancelledError):
        locator.locate(sample_png, "方块", cancel_event=event)
    assert client.calls == 0


def test_cancelled_mid_stream(sample_png):
    """流式途中 set —— 下一个事件到达时就得停，不许把整轮跑完。"""
    event = threading.Event()
    client = _EventFiringClient(on_first=event.set)
    locator = Locator(client=client, api_key="sk-test")
    with pytest.raises(CancelledError):
        locator.locate(sample_png, "方块", cancel_event=event)
    assert client.calls == 1, "第一轮该发出去（取消是中途发生的），但不该有第二轮"


def test_cancelled_after_the_model_returns(sample_png):
    """事件回调之后、结果返回之前 set —— 那段空档也要查一次。

    这一段没有事件经过，如果只在 on_event 上查，取消会「晚一轮才生效」，
    调用方看到的是"点了取消，图还是画出来了"。
    """
    event = threading.Event()
    client = _EventFiringClient(events=())          # 一个事件都不回调
    locator = Locator(client=client, api_key="sk-test")
    event.set()                                     # 进门前就 set，走的是"进门前检查"那条路
    with pytest.raises(CancelledError):
        locator.locate(sample_png, "方块", cancel_event=event)


def test_no_cancel_event_keeps_old_behavior(sample_png):
    """**负向对照**：不传 cancel_event 时行为与 0.1.2 完全一样（连回调都不包装）。"""
    seen = []
    client = _EventFiringClient()
    locator = Locator(client=client, api_key="sk-test")
    result = locator.locate(sample_png, "方块", on_event=seen.append)
    assert len(result) == 1
    assert seen, "不传 cancel_event 时 on_event 必须原样工作"


def test_callbacks_are_not_swallowed_when_not_cancelled(sample_png):
    """包装后的回调仍要把事件透出去，且顺序不变。"""
    seen = []
    client = _EventFiringClient(events=("reasoning", "content", "finish"))
    locator = Locator(client=client, api_key="sk-test")
    locator.locate(sample_png, "方块", on_event=seen.append, cancel_event=threading.Event())
    assert [event["type"] for event in seen] == ["reasoning", "content", "finish"]


def test_locate_to_file_also_accepts_cancel_event(fake_client, sample_png, tmp_path):
    """一行式入口同样要能取消（返回前 set 就会在进门前被拦下）。"""
    event = threading.Event()
    event.set()
    client = fake_client(BOX)
    locator = Locator(client=client, api_key="sk-test")
    with pytest.raises(CancelledError):
        locator.locate_to_file(sample_png, "方块", tmp_path / "o.png", cancel_event=event)
    assert client.calls == []


def test_module_level_locate_passes_it_through(fake_client, sample_png):
    """模块级 locate() 的分流不能把 cancel_event 漏进 Settings（那会直接 TypeError）。"""
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledError):
        locate(sample_png, "方块", client=fake_client(BOX), api_key="sk-test", cancel_event=event)


def test_cancelled_error_is_a_locator_error():
    """取消也是本库异常体系里的东西：except LocatorError 兜得住，不必单开分支。"""
    assert issubclass(CancelledError, LocatorError)
    assert not issubclass(CancelledError, InterruptedError), \
        "别让它被误当成系统级中断（KeyboardInterrupt / InterruptedError）去处理"
