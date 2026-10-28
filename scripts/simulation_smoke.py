"""MuJoCo contact plus the official TACTO optical renderer, without PyBullet."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
from tacto_renderer import Renderer


def main():
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'configs/contact_scene.xml'))
    data = mujoco.MjData(model)
    maximum_contacts, maximum_force = 0, 0.
    for _ in range(500):
        mujoco.mj_step(model, data)
        maximum_contacts = max(maximum_contacts, data.ncon)
        for contact in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact, force)
            maximum_force = max(maximum_force, float(force[0]))
    assert data.ncon > 0 and maximum_force > 0
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, 0, force)
    assert np.isfinite(data.qpos).all() and force[0] > 0
    # Minimal adapter: sphere/plane intersection from MuJoCo geometry gives depth.
    # This is a geometric indentation approximation, not a soft-body gel model.
    radius = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'sphere'), 0])
    center = data.qpos[:3]
    xx, yy = np.meshgrid(np.linspace(-0.01, 0.01, 120), np.linspace(-0.015, 0.015, 160))
    radial_squared = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    depth = np.maximum(0., np.sqrt(np.maximum(0., radius ** 2 - radial_squared)) - center[2])
    assert depth.max() > 0
    renderer = Renderer(120, 160, None, str(ROOT / 'vendor/tacto_renderer/config_digit.yml'))
    try:
        empty, _ = renderer.render_from_depth(np.zeros_like(depth), noise=False, calibration=False)
        touch, rendered_depth = renderer.render_from_depth(depth, noise=False, calibration=False)
        difference = float(np.mean(np.abs(touch.astype(float) - empty.astype(float))))
        assert touch.shape == (160, 120, 3) and np.isfinite(rendered_depth).all()
        assert difference > 0.01, 'Optical image did not change with contact'
        Image.fromarray(empty).save(output / 'tacto_no_contact.png')
        Image.fromarray(touch).save(output / 'tacto_contact.png')
        np.save(output / 'contact_indentation.npy', depth, allow_pickle=False)
    finally:
        renderer.r.delete()
    with mujoco.Renderer(model, height=240, width=320) as scene:
        scene.update_scene(data, camera='overview')
        rgb = scene.render()
        assert rgb.shape == (240, 320, 3) and rgb.std() > 1
        Image.fromarray(rgb).save(output / 'mujoco_scene.png')
    report = {'status': 'passed', 'physics': f'MuJoCo {mujoco.__version__}',
              'tactile_renderer': 'TACTO 0.0.3 vendored renderer only (MIT)',
              'physics_steps': 500, 'simulation_time_seconds': float(data.time),
              'maximum_contacts': maximum_contacts, 'maximum_normal_force_n': maximum_force,
              'settled_normal_force_n': float(force[0]),
              'maximum_indentation_m': float(depth.max()),
              'contact_vs_no_contact_rgb_mean_absolute_difference': difference,
              'rgb_shape': list(touch.shape),
              'checks': ['physical_contact', 'positive_contact_force', 'positive_geometric_indentation',
                         'TACTO_optical_image_changes_with_contact', 'MuJoCo_scene_render'],
              'limitations': 'Geometric sphere-plane depth adapter, not calibrated elastomer deformation. Renderer-only TACTO; PyBullet Sensor API is not installed.'}
    (output / 'simulation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
