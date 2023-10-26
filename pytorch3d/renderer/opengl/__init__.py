# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# If we can access EGL, import MeshRasterizerOpenGL.
def _can_import_egl():
    import os
    import warnings

    try:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import OpenGL.EGL
    except (AttributeError, ImportError, ModuleNotFoundError):
        warnings.warn(
            "Can't import EGL, not importing MeshRasterizerOpenGL. This might happen if"
            " your Python application imported OpenGL with a non-EGL backend before"
            " importing PyTorch3D, or if you don't have pyopengl installed as part"
            " of your Python distribution."
        )
        return False

    return True


if _can_import_egl():
    from .rasterizer_opengl import MeshRasterizerOpenGL

__all__ = [k for k in globals().keys() if not k.startswith("_")]
