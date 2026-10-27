"""The WebBackend protocol is the seam the real pipeline backend will implement."""

from tubitak_rag.interfaces.web.backend import WebBackend
from tubitak_rag.interfaces.web.fake_backend import FakeBackend


def test_fake_backend_satisfies_protocol() -> None:
    assert isinstance(FakeBackend(), WebBackend)


def test_object_missing_methods_does_not_satisfy_protocol() -> None:
    class Partial:
        is_demo = False

        def ask(self) -> None: ...

    assert not isinstance(Partial(), WebBackend)
