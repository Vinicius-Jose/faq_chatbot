from app.utils.tools import singleton


@singleton
class TestSingleton:
    pass


def test_singleton() -> None:
    instance_a = TestSingleton()
    instance_b = TestSingleton()
    assert instance_a == instance_b
