"""Apply the flashinfer + raft source patches required to build FreeKV.

Mirrors the sed patches documented in old_README.md, but as Python for
robustness (tolerant to whitespace variations, idempotent).

Usage: python patch_deps.py <repo_root>
"""
import os
import re
import sys


def patch_flashinfer(repo_root: str) -> None:
    path = os.path.join(
        repo_root, "3rdparty/flashinfer/include/flashinfer/utils.cuh"
    )
    with open(path, "r") as f:
        content = f.read()

    if "group_size == 5" in content and "group_size == 7" in content:
        print(f"[flashinfer] already patched: {path}")
        return

    pattern = re.compile(
        r"(\}\s*else\s+if\s*\(group_size\s*==\s*4\)\s*\{\s*\\\s*\n"
        r"\s*constexpr\s+size_t\s+GROUP_SIZE\s*=\s*4\s*;\s*\\\s*\n"
        r"\s*__VA_ARGS__\s*\\\s*\n)",
        re.MULTILINE,
    )
    addition = (
        "  } else if (group_size == 5) {                             \\\n"
        "    constexpr size_t GROUP_SIZE = 5;                         \\\n"
        "    __VA_ARGS__                                              \\\n"
        "  } else if (group_size == 7) {                             \\\n"
        "    constexpr size_t GROUP_SIZE = 7;                         \\\n"
        "    __VA_ARGS__                                              \\\n"
    )
    new_content, n = pattern.subn(lambda m: m.group(1) + addition, content, count=1)
    if n == 0:
        raise RuntimeError(
            f"[flashinfer] could not locate group_size == 4 block in {path}"
        )
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[flashinfer] patched {path}")


def patch_raft(repo_root: str) -> None:
    path = os.path.join(
        repo_root, "3rdparty/raft/cpp/include/raft/util/vectorized.cuh"
    )
    with open(path, "r") as f:
        content = f.read()

    if "IOType<__nv_bfloat16, 1>" in content:
        print(f"[raft] already patched: {path}")
        return

    # 1) add include
    if "#include <cuda_bf16.h>" not in content:
        content = content.replace(
            "#include <cuda_fp16.h>",
            "#include <cuda_fp16.h>\n#include <cuda_bf16.h>",
            1,
        )

    # 2) insert bfloat16 IOType specializations right after the
    #    `struct IOType<__half2, 4> { ... };` block.
    anchor_re = re.compile(
        r"(struct\s+IOType<__half2,\s*4>\s*\{[^}]*\};)", re.MULTILINE | re.DOTALL
    )
    addition = """

template <>
struct IOType<__nv_bfloat16, 1> {
  typedef __nv_bfloat16 Type;
};
template <>
struct IOType<__nv_bfloat16, 2> {
  typedef __nv_bfloat162 Type;
};
template <>
struct IOType<__nv_bfloat16, 4> {
  typedef uint2 Type;
};
template <>
struct IOType<__nv_bfloat16, 8> {
  typedef uint4 Type;
};
template <>
struct IOType<__nv_bfloat162, 1> {
  typedef __nv_bfloat162 Type;
};
template <>
struct IOType<__nv_bfloat162, 2> {
  typedef uint2 Type;
};
template <>
struct IOType<__nv_bfloat162, 4> {
  typedef uint4 Type;
};
"""
    new_content, n = anchor_re.subn(lambda m: m.group(1) + addition, content, count=1)
    if n == 0:
        raise RuntimeError(
            f"[raft] could not locate IOType<__half2, 4> anchor in {path}"
        )
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[raft] patched {path}")


def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "/opt/tafkv"
    patch_flashinfer(repo_root)
    patch_raft(repo_root)


if __name__ == "__main__":
    main()
