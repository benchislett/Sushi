import os
import subprocess
import sys
import glob
import shutil

import nanobind
from setuptools import Extension, setup
from setuptools.build_meta import *  # noqa: F403,F401
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):  # type: ignore
    def __init__(self, name: str, sourcedir: str = "") -> None:
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):  # type: ignore
    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dnanobind_DIR={nanobind.cmake_dir()}",
        ]

        build_args = []

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", ext.name] + build_args,
            cwd=self.build_temp,
        )

        # Find the built .so file - CMake puts it in the output directory we specified
        dest_path = self.get_ext_fullpath(ext.name)
        dest_dir = os.path.dirname(dest_path)

        # Look for .so files in the output directory (extdir)
        built_extensions = glob.glob(os.path.join(extdir, f"{ext.name}*.so"))

        # Also check the build temp directory
        if not built_extensions:
            built_extensions = glob.glob(
                os.path.join(self.build_temp, f"{ext.name}*.so")
            )

        # Also check for files with different naming patterns
        if not built_extensions:
            built_extensions = glob.glob(os.path.join(extdir, f"*{ext.name}*.so"))

        if not built_extensions:
            built_extensions = glob.glob(
                os.path.join(self.build_temp, f"*{ext.name}*.so")
            )

        if built_extensions:
            # Ensure destination directory exists
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.copy2(built_extensions[0], dest_path)
            print(f"Copied {built_extensions[0]} to {dest_path}")


# Create the extension
setup(
    ext_modules=[CMakeExtension("sushi_core", sourcedir="src")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
