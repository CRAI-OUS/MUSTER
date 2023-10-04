import pymuster
import numpy as np


def test_registration():
    # Create a smal 3d image time series
    img = np.random.rand(4, 1, 10, 10, 10)

    deform_reg = pymuster.Registration(
        stages_iterations=[5],
        stages_img_scales=[2],
        stages_deform_scales=[2],
        image_size=[10, 10, 10],
        pix_dim=[1, 1, 1],
        device="cpu",
        verbose=False,
    )

    out = deform_reg.fit(img)
    deform_matrix = out["deform_field"]
    assert deform_matrix.shape == (4, 4, 3, 10, 10, 10)
    assert deform_matrix.dtype == np.float32


if __name__ == "__main__":
    test_registration()
