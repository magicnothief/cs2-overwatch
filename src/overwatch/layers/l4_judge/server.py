"""Run the judge on llama.cpp's own prebuilt server, downloaded for this machine.

The judge used to run through llama-cpp-python, which has to be compiled for the
GPU it will use: CMake, plus the CUDA toolkit or the Vulkan SDK. That is fine on
the machine this was developed on and a wall for everyone else. llama.cpp
publishes ready builds of `llama-server` for every combination that matters, so
the app downloads the one that fits and talks to it over localhost:

    any GPU         Vulkan: a 30 MB download that runs on NVIDIA, AMD and Intel
    NVIDIA          CUDA 13 or 12 (whichever the driver runs), if Vulkan fails;
                    first instead with prefer_cuda (~25% faster, a 600 MB download)
    anything        CPU, always last: it cannot fail for want of a GPU

Each build is tried in that order until one starts and loads the model; the
report says which one ran (Engine.describe()). The release is pinned, and each
file is checked against the SHA-256 GitHub published for it.

The prompt is rendered here, not by the server, so it is byte-for-byte the one
the fine-tune was evaluated with (see render_prompt), and the answer is held to
the verdict's JSON schema by the server's grammar.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from overwatch.downloads import Progress, fetch, unpack

#: The llama.cpp release every download comes from.
RELEASE = "b11177"
BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/"
#: How long a server may take to load a model before it counts as failed.
START_TIMEOUT = 180.0


@dataclass(frozen=True)
class Build:
    """One way to run llama.cpp: its files, and what it needs from the machine."""

    name: str  # "cuda-13", "cuda-12", "vulkan", "cpu"
    files: tuple[tuple[str, str], ...]  # (release file, sha256)

    @property
    def gpu(self) -> bool:
        return self.name != "cpu"

    @property
    def size_note(self) -> str:
        return {"cuda-13": "~600 MB", "cuda-12": "~700 MB"}.get(self.name, "~30 MB")


_R = RELEASE
BUILDS: dict[tuple[str, str], Build] = {
    ("linux", "cuda-13"): Build(
        "cuda-13",
        (
            (
                f"llama-{_R}-bin-ubuntu-cuda-13.4-x64.tar.gz",
                "5d73f877871fddf1ca3909551301c33da488e7bfd55fe2e45377362b307a7c04",
            ),
            (
                f"cudart-llama-{_R}-bin-ubuntu-cuda-13.4-x64.tar.gz",
                "af086512f877a0b86f0607c687868ef0321ad8070b45513823ad0f261d7118a1",
            ),
        ),
    ),
    ("linux", "cuda-12"): Build(
        "cuda-12",
        (
            (
                f"llama-{_R}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                "7f584b4911be091788468412022dde72c8337af8db880d5f748e87563b25a82a",
            ),
            (
                f"cudart-llama-{_R}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                "cea160366caea83923d76a676fc8591033d92138f35250cd3dc8b11f2edd57d8",
            ),
        ),
    ),
    ("linux", "vulkan"): Build(
        "vulkan",
        (
            (
                f"llama-{_R}-bin-ubuntu-vulkan-x64.tar.gz",
                "1557dbc00d446cb21d7e9771d98106495f55d8cef4faff85ee479eef3698d0c3",
            ),
        ),
    ),
    ("linux", "cpu"): Build(
        "cpu",
        (
            (
                f"llama-{_R}-bin-ubuntu-x64.tar.gz",
                "9e088583293c4c104953ead0dd8957f4e00dca318c7f241c039198e5eb2e2e54",
            ),
        ),
    ),
    ("windows", "cuda-13"): Build(
        "cuda-13",
        (
            (
                f"llama-{_R}-bin-win-cuda-13.4-x64.zip",
                "35821fa553fc69feef4c3861456249ff6b5a282da67c4f8634688ad024149621",
            ),
            (
                "cudart-llama-bin-win-cuda-13.4-x64.zip",
                "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
            ),
        ),
    ),
    ("windows", "cuda-12"): Build(
        "cuda-12",
        (
            (
                f"llama-{_R}-bin-win-cuda-12.4-x64.zip",
                "14e756ba453e29db57578c1e5791245fe05c893671d3b334e08482ba1a0946bb",
            ),
            (
                "cudart-llama-bin-win-cuda-12.4-x64.zip",
                "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
            ),
        ),
    ),
    ("windows", "vulkan"): Build(
        "vulkan",
        (
            (
                f"llama-{_R}-bin-win-vulkan-x64.zip",
                "6e4e7a8cea0d2dbd0a745b55f6b42d46d8130f8df4791938802f8269dfc205d5",
            ),
        ),
    ),
    ("windows", "cpu"): Build(
        "cpu",
        (
            (
                f"llama-{_R}-bin-win-cpu-x64.zip",
                "7f11f910c9ef590782a0a0edbaa99aac1665823ab7bbd63830eca97c39e8defc",
            ),
        ),
    ),
}


def this_os() -> str:
    system = platform.system().lower()
    if system not in ("linux", "windows") or platform.machine().lower() not in (
        "x86_64",
        "amd64",
    ):
        msg = f"no prebuilt judge engine for {platform.system()} {platform.machine()}"
        raise RuntimeError(msg)
    return system


@functools.cache
def nvidia_cuda_major() -> int | None:
    """The newest CUDA major version the NVIDIA driver runs, or None without one."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe], capture_output=True, text=True, timeout=15, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.", out)
    return int(found.group(1)) if found else None


def candidates(
    gpu_layers: int | str = "auto",
    os_name: str | None = None,
    *,
    prefer_cuda: bool = False,
) -> list[Build]:
    """The builds to try, best first. gpu_layers 0 means the CPU build only.

    Vulkan comes first on every GPU: on an RTX 3060 it gave the same verdicts as
    CUDA at 2.1 s per case against 1.6, for a 30 MB download against 600 MB.
    """
    os_name = os_name or this_os()
    if str(gpu_layers) == "0":
        return [BUILDS[os_name, "cpu"]]
    cuda_builds = []
    cuda = nvidia_cuda_major()
    if cuda is not None and cuda >= 13:
        cuda_builds.append("cuda-13")
    if cuda is not None and cuda >= 12:
        cuda_builds.append("cuda-12")  # a newer driver still runs it
    order = [*cuda_builds, "vulkan"] if prefer_cuda else ["vulkan", *cuda_builds]
    order.append("cpu")
    seen: list[Build] = []
    for name in order:
        build = BUILDS[os_name, name]
        if build not in seen:
            seen.append(build)
    return seen


# --------------------------------------------------------------------- download


def install(build: Build, engines: Path, progress: Progress | None = None) -> Path:
    """The build's llama-server executable, downloading it the first time."""
    tell = progress or (lambda _message: None)
    home = engines / f"llama-{RELEASE}-{build.name}"
    exe = _find_server(home)
    if exe is not None:
        return exe
    home.mkdir(parents=True, exist_ok=True)
    tell(f"Downloading the judge engine ({build.name}, {build.size_note}, once)")
    for name, sha256 in build.files:
        archive = engines / name
        if not archive.exists():
            fetch(BASE_URL + name, archive, sha256, tell)
        unpack(archive, home)
        archive.unlink()
    exe = _find_server(home)
    if exe is None:
        msg = f"no llama-server in {home}"
        raise RuntimeError(msg)
    if os.name != "nt":
        exe.chmod(exe.stat().st_mode | 0o111)
    return exe


def _find_server(home: Path) -> Path | None:
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    found = sorted(home.rglob(name)) if home.exists() else []
    return found[0] if found else None


# ----------------------------------------------------------------------- server


def render_prompt(system: str, user: str) -> str:
    """The Qwen chat prompt, exactly as the judge was evaluated with.

    This is what the GGUF's own chat template produces with its defaults (checked
    in tests against llama-cpp-python's rendering): the assistant turn opens a
    `<think>` block, and the grammar makes the answer start straight away.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass(frozen=True)
class Device:
    """A GPU as llama.cpp sees it."""

    id: str  # "CUDA0", "Vulkan0"
    name: str
    total_mib: int
    free_mib: int


_DEVICE = re.compile(r"^\s+(\w+?\d+): (.+) \((\d+) MiB, (\d+) MiB free\)", re.MULTILINE)


def devices(exe: Path, env: dict[str, str]) -> list[Device]:
    """The GPUs this build can use on this machine (none, for a CPU build)."""
    try:
        out = subprocess.run(
            [str(exe), "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        Device(i, name.strip(), int(total), int(free))
        for i, name, total, free in _DEVICE.findall(out.stdout + out.stderr)
    ]


def needed_mib(model: Path, context: int) -> int:
    """VRAM to hold the whole model: its weights, the context cache, and scratch."""
    return int(model.stat().st_size / 2**20 * 1.1) + context // 4 + 768


@dataclass
class Engine:
    """A running llama-server with one model loaded."""

    build: Build
    process: subprocess.Popen
    port: int
    log: Path
    device: Device | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def uses_gpu(self) -> bool:
        return self.device is not None

    def describe(self) -> str:
        where = (
            f"GPU: {self.device.name} via {self.build.name}"
            if self.device is not None
            else "CPU"
        )
        return f"{where} (llama.cpp {RELEASE})"

    def post(self, path: str, body: dict, timeout: float = 600) -> dict:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def start(
    model: Path,
    *,
    engines: Path,
    gpu_layers: int | str = "auto",
    context: int = 4096,
    threads: int | None = None,
    only: str | None = None,
    prefer_cuda: bool = False,
    progress: Progress | None = None,
) -> Engine:
    """Start llama-server with the model, trying the builds best first.

    A GPU build is used only when it sees a GPU with room for the whole model:
    a card busy with something else (a fine-tuning run, a game) is left alone,
    and the judge runs on the CPU instead. `only` names one build ("cuda-13",
    "vulkan", ...) to use and no other.
    """
    tell = progress or (lambda _message: None)
    notes: list[str] = []
    builds = (
        [BUILDS[this_os(), only]]
        if only is not None
        else candidates(gpu_layers, prefer_cuda=prefer_cuda)
    )
    for build in builds:
        try:
            exe = install(build, engines, tell)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            notes.append(f"{build.name}: {exc}")
            continue
        env = _environment(exe, build)
        device = None
        if build.gpu:
            usable = devices(exe, env)
            need = needed_mib(model, context)
            roomy = [d for d in usable if d.free_mib >= need]
            if not roomy:
                notes.append(
                    f"{build.name}: "
                    + (
                        f"no GPU with {need} MiB free"
                        if usable
                        else "sees no GPU on this machine"
                    )
                )
                continue
            device = max(roomy, key=lambda d: d.free_mib)
        tell(f"Starting the judge ({device.name if device else 'CPU'})")
        engine = _launch(exe, build, device, model, context, threads, env, engines)
        if engine is not None:
            engine.notes = notes
            return engine
        notes.append(f"{build.name}: did not start (see {engines / 'server.log'})")
    msg = "the judge engine would not start: " + "; ".join(notes)
    raise RuntimeError(msg)


def _die_with_parent() -> None:  # pragma: no cover - runs in the child
    """Linux: have the kernel stop the engine if this app dies without cleaning up."""
    try:
        ctypes.CDLL("libc.so.6").prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM
    except OSError:
        pass


def _environment(exe: Path, build: Build) -> dict[str, str]:
    env = dict(os.environ)
    if os.name != "nt":  # the build's own libraries sit next to the executable
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            p for p in (str(exe.parent), env.get("LD_LIBRARY_PATH")) if p
        )
    if not build.gpu:  # a CPU build must not touch a GPU, even for scratch work
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["GGML_VK_VISIBLE_DEVICES"] = ""
    return env


def _launch(
    exe: Path,
    build: Build,
    device: Device | None,
    model: Path,
    context: int,
    threads: int | None,
    env: dict[str, str],
    engines: Path,
) -> Engine | None:
    port = _free_port()
    args = [
        str(exe),
        "--model", str(model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(context),
        "--parallel", "1",
        "--no-ui",
    ]  # fmt: skip
    if device is not None:
        args += ["--device", device.id, "--n-gpu-layers", "999"]
    else:
        args += ["--n-gpu-layers", "0"]
    if threads:
        args += ["--threads", str(threads)]
    log = engines / "server.log"
    engines.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as sink:
        process = subprocess.Popen(
            args,
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            preexec_fn=_die_with_parent if os.name != "nt" else None,  # noqa: PLW1509
        )
    engine = Engine(build, process, port, log, device)
    atexit.register(engine.close)
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return engine
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.3)
    engine.close()
    return None


__all__ = [
    "BUILDS",
    "RELEASE",
    "Build",
    "Device",
    "Engine",
    "candidates",
    "devices",
    "install",
    "nvidia_cuda_major",
    "render_prompt",
    "start",
]
