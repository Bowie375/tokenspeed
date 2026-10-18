# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Fixed token inputs and reference provenance for the homework evaluations."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Fixture:
    name: str
    input_ids: list[int]
    continuation_ids: list[int]
    kind: str
    formatted_text: str

    def __post_init__(self):
        if not self.input_ids or any(
            not isinstance(x, int) or x < 0
            for x in self.input_ids + self.continuation_ids
        ):
            raise ValueError("fixture token IDs must be nonempty/nonnegative integers")
        if self.kind not in ("raw", "chat"):
            raise ValueError("fixture kind must be raw or chat")


def exact_length_ids(pattern: list[int], *, length: int) -> list[int]:
    """Repeat an existing token pattern to exactly length tokens without re-tokenizing."""
    if not pattern or length <= 0:
        raise ValueError("pattern must be nonempty and length positive")
    return (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]


def build_fixtures(
    tokenizer, *, lengths: list[int], continuation_steps: int, include_chat: bool
) -> list[Fixture]:
    """Build raw controlled lengths and optionally four semantically varied chat prompts."""
    if continuation_steps < 1 or len(set(lengths)) != len(lengths):
        raise ValueError("continuation_steps must be positive and lengths unique")
    pattern = tokenizer.encode(
        "The red fox counts 1 2 3. 红色狐狸数数字。 ", add_special_tokens=False
    )
    continuation = exact_length_ids(
        tokenizer.encode(" The answer is four.", add_special_tokens=False),
        length=continuation_steps,
    )
    result = [
        Fixture(
            f"raw_{length}",
            exact_length_ids(pattern, length=length),
            continuation,
            "raw",
            "",
        )
        for length in lengths
    ]
    if include_chat:
        prompts = {
            "chinese": "请用一句话解释什么是递归。",
            "english": "Explain recursion in one sentence.",
            "arithmetic": "What is 17 + 25? Answer with the number only.",
            "pattern": "Continue this pattern for twenty items: red blue red blue",
        }
        for name, text in prompts.items():
            rendered = tokenizer.apply_chat_template(
                [dict(role="user", content=text)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            result.append(
                Fixture(
                    name,
                    tokenizer.encode(rendered, add_special_tokens=False),
                    continuation,
                    "chat",
                    rendered,
                )
            )
    return result


def save_fixtures(path: Path, fixtures: list[Fixture]):
    """Save token IDs and explicit teacher-forced continuations as UTF-8 JSON."""
    if len({f.name for f in fixtures}) != len(fixtures):
        raise ValueError("fixture names must be unique")
    Path(path).write_text(
        json.dumps(
            [dataclasses.asdict(f) for f in fixtures], indent=2, ensure_ascii=False
        )
        + "\n"
    )


def load_fixtures(path: Path) -> list[Fixture]:
    """Restore fixed inputs without invoking a tokenizer or changing token order."""
    return [Fixture(**item) for item in json.loads(Path(path).read_text())]


def environment_manifest(
    root: Path, checkpoint: Path, *, seed: int, settings: dict
) -> dict:
    """Record source, environment and small checkpoint-file hashes without importing NPU code."""
    packages = {}
    for name in (
        "torch",
        "torch_npu",
        "transformers",
        "tokenspeed-scheduler",
        "nbformat",
        "nbclient",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in checkpoint.iterdir()
        if p.is_file()
        and p.name in ("config.json", "tokenizer_config.json", "chat_template.jinja")
    }
    metadata = checkpoint / ".cache/huggingface/download/config.json.metadata"
    revision = metadata.read_text().splitlines()[0] if metadata.exists() else None
    return dict(
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        worktree_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ),
        checkpoint=str(checkpoint.resolve()),
        checkpoint_revision=revision,
        checkpoint_file_hashes=hashes,
        python=platform.python_version(),
        packages=packages,
        seed=seed,
        settings=settings,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        reference="HF CPU FP32 eager",
    )
