# TACTO renderer-only adaptation

Source: official `tacto==0.0.3` wheel, downloaded from the Tsinghua PyPI mirror.
Upstream: https://github.com/facebookresearch/tacto
The upstream MIT license is preserved in `LICENSE`.

This directory includes only `renderer.py` and `config_digit.yml`. The public entry point is
`from tacto_renderer import Renderer` after adding `vendor/` to `sys.path`.

Changes to upstream renderer:

1. Remove the unused `import pybullet as p`. Its only uses in this file were already commented out upstream.
2. Replace removed NumPy alias `np.float` with `float` in background calibration.

`__init__.py` exports only `Renderer`. The upstream `tacto.Sensor` API, PyBullet integration,
and URDF loading are outside this adaptation. This avoids requiring a source build of PyBullet
and the old `urdfpy` dependency on `networkx==2.2` under Python 3.12 / Apple Silicon.

`scripts/simulation_smoke.py` supplies MuJoCo sphere-plane geometric indentation to the
upstream `Renderer.render_from_depth` optical renderer. This is a minimal integration,
not a calibrated soft-body gel or full manipulation scene.
