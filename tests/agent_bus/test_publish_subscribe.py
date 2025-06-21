from core.agent_bus import publish, subscribe, reset


def test_publish_and_subscribe():
    reset("a")
    publish("a", {"text": "hello"})
    msgs = list(subscribe("a"))
    assert msgs == [{"text": "hello"}]
    assert list(subscribe("a")) == []
