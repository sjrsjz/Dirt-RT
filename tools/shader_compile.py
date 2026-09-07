"""Shared GLSL include expansion for source-contract compilation audits."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
GLSLANG = Path("E:/VulkanSDK/Bin/glslangValidator.exe")


def expand(path: Path) -> str:
    parts = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r'^\s*#include\s+"([^"]+)"\s*$', line)
        if match:
            name = match[1]
            child = (ROOT / "shaders" / name.lstrip("/")
                     if name.startswith("/") else path.parent / name)
            parts.append(expand(child))
        else:
            parts.append(line)
    return "\n".join(parts)
