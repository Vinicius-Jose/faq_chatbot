import time
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--delay",
        action="store",
        default=0,
        type=float,
        help="Delay time between tests",
    )


@pytest.fixture(autouse=True)
def delay_between_tests(request: pytest.FixtureRequest):
    delay = request.config.getoption("--delay")
    if delay > 0:
        time.sleep(delay)
