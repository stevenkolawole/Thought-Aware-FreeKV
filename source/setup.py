from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import shutil
import subprocess
import torch


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        env = os.environ.copy()

        cudacxx = env.get("CUDACXX")
        if not cudacxx:
            detected = shutil.which("nvcc")
            if not detected:
                for candidate in (
                    "/usr/local/cuda/bin/nvcc",
                    "/usr/local/cuda-12.9/bin/nvcc",
                    "/usr/local/cuda-12.8/bin/nvcc",
                    "/usr/local/cuda-12.7/bin/nvcc",
                    "/usr/local/cuda-12.6/bin/nvcc",
                    "/usr/local/cuda-12.5/bin/nvcc",
                    "/usr/local/cuda-12.4/bin/nvcc",
                ):
                    if os.path.exists(candidate):
                        detected = candidate
                        break
            cudacxx = detected

        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DBUILD_SHARED_LIBS=ON",
            "-GNinja",
        ]
        if cudacxx:
            env["CUDACXX"] = cudacxx
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={cudacxx}")

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", ext.name], cwd=build_temp, env=env
        )


setup(
    name="freekv",
    packages=find_packages(),
    ext_modules=[CMakeExtension("freekv_cpp", "freekv_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
