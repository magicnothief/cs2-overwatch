"""Tests for picking, fetching and talking to the prebuilt llama.cpp judge engine.

Nothing here downloads or starts a real engine: the order builds are tried in,
the checksum gate, unpacking, and the prompt are what can go wrong quietly.
"""

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from overwatch.downloads import fetch
from overwatch.layers.l4_judge import server
from overwatch.layers.l4_judge.judge import DEFAULT_MODEL


def _names(builds: list[server.Build]) -> list[str]:
    return [b.name for b in builds]


@pytest.mark.parametrize(
    ("cuda", "expected"),
    [
        (None, ["vulkan", "cpu"]),
        (12, ["vulkan", "cuda-12", "cpu"]),
        (13, ["vulkan", "cuda-13", "cuda-12", "cpu"]),
    ],
)
def test_vulkan_comes_first_then_cuda_and_cpu_last(
    monkeypatch: pytest.MonkeyPatch, cuda: int | None, expected: list[str]
) -> None:
    monkeypatch.setattr(server, "nvidia_cuda_major", lambda: cuda)
    for os_name in ("linux", "windows"):
        assert _names(server.candidates("auto", os_name)) == expected


def test_cuda_can_be_put_first_on_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "nvidia_cuda_major", lambda: 13)
    assert _names(server.candidates("auto", "linux", prefer_cuda=True)) == [
        "cuda-13",
        "cuda-12",
        "vulkan",
        "cpu",
    ]
    monkeypatch.setattr(server, "nvidia_cuda_major", lambda: None)
    assert _names(server.candidates("auto", "linux", prefer_cuda=True)) == [
        "vulkan",
        "cpu",
    ]


def test_cpu_only_means_the_cpu_build_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "nvidia_cuda_major", lambda: 13)
    assert _names(server.candidates(0, "linux")) == ["cpu"]
    assert _names(server.candidates("0", "windows")) == ["cpu"]


def test_every_build_is_pinned_to_the_release_with_a_checksum() -> None:
    for (os_name, name), build in server.BUILDS.items():
        assert build.name == name
        for file, sha256 in build.files:
            assert len(sha256) == 64
            assert file.endswith(".zip" if os_name == "windows" else ".tar.gz")
            if file.startswith("llama-"):
                assert server.RELEASE in file


def test_devices_are_read_from_the_engines_own_listing() -> None:
    listing = """0.00.000.248 I srv  llama_server: initializing ...
Available devices:
  Vulkan0: NVIDIA GeForce RTX 3060 (12534 MiB, 11539 MiB free)
  Vulkan1: AMD Radeon(TM) Graphics (2048 MiB, 1900 MiB free)
"""
    found = server._DEVICE.findall(listing)
    assert found == [
        ("Vulkan0", "NVIDIA GeForce RTX 3060", "12534", "11539"),
        ("Vulkan1", "AMD Radeon(TM) Graphics", "2048", "1900"),
    ]
    assert server._DEVICE.findall("Available devices:\n  (none)\n") == []


def _archive(tmp_path: Path) -> tuple[Path, str]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("llama-b1/llama-server", "#!/bin/sh\n")
        z.writestr("llama-b1/llama-server.exe", "MZ")
        z.writestr("llama-b1/libggml.so", "")
    path = tmp_path / "release" / "llama-test.zip"
    path.parent.mkdir()
    path.write_bytes(buffer.getvalue())
    return path, hashlib.sha256(buffer.getvalue()).hexdigest()


def test_a_download_with_the_wrong_checksum_is_thrown_away(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    dest = tmp_path / "got.zip"
    with pytest.raises(RuntimeError, match="checksum"):
        fetch(archive.as_uri(), dest, "0" * 64)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_an_engine_is_downloaded_once_and_unpacked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, sha256 = _archive(tmp_path)
    monkeypatch.setattr(server, "BASE_URL", archive.parent.as_uri() + "/")
    build = server.Build("cpu", ((archive.name, sha256),))
    engines = tmp_path / "engines"
    exe = server.install(build, engines)
    assert exe.parent.name == "llama-b1"
    assert not (engines / archive.name).exists()  # the archive is not kept
    archive.unlink()  # a second call must not need it
    assert server.install(build, engines) == exe


def test_the_prompt_is_the_one_the_judge_was_evaluated_with() -> None:
    """The GGUF's own chat template, rendered as llama.cpp's Jinja formatter does."""
    gguf = pytest.importorskip("gguf")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    if not DEFAULT_MODEL.exists():
        pytest.skip("no judge model here")
    field = gguf.GGUFReader(DEFAULT_MODEL).fields["tokenizer.chat_template"]
    template = bytes(field.parts[field.data[0]]).decode()

    def fail(message: str) -> None:
        raise ValueError(message)

    rendered = (
        ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        .from_string(template)
        .render(
            messages=[
                {"role": "system", "content": "the rules"},
                {"role": "user", "content": "the evidence"},
            ],
            add_generation_prompt=True,
            bos_token="",
            eos_token="<|im_end|>",
            raise_exception=fail,
        )
    )
    assert rendered == server.render_prompt("the rules", "the evidence")


def test_a_dropped_download_is_retried_and_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first connection dies halfway; the retry asks for the rest only."""
    import urllib.error

    from overwatch import downloads

    payload = bytes(range(256)) * 64
    asked: list[str | None] = []

    class Response(io.BytesIO):
        def __init__(self, body: bytes, status: int, total: int) -> None:
            super().__init__(body)
            self.status = status
            self.headers = {"Content-Length": str(total)}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def urlopen(request, timeout):
        asked.append(request.headers.get("Range"))
        if len(asked) == 1:  # half the file arrives, then the connection drops
            part = tmp_path / "file.bin.part"
            part.write_bytes(payload[:5000])
            raise urllib.error.URLError("connection reset")
        start = int(request.headers["Range"].split("=")[1].rstrip("-"))
        return Response(payload[start:], 206, len(payload) - start)

    monkeypatch.setattr(downloads.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(downloads.time, "sleep", lambda _s: None)
    got = downloads.fetch(
        "https://example.invalid/file.bin",
        tmp_path / "file.bin",
        hashlib.sha256(payload).hexdigest(),
    )
    assert got.read_bytes() == payload
    assert asked == [None, "bytes=5000-"]
