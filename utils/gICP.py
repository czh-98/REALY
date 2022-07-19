"""
This file is part of the repo: https://github.com/czh-98/REALY
If you find the code useful, please cite our paper:
"REALY: Rethinking the Evaluation of 3D Face Reconstruction"
European Conference on Computer Vision 2022
Code: https://github.com/czh-98/REALY
Copyright (c) [2021-2022] [Tencent AI Lab]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from utils.util import keypoints_region_map
from utils.util import get_barycentric_coordinates, fit_icp_RT


def global_rigid_align_7_kpt(predicted_vertices=None, REALY_HIFI3D_keypoints=None, template_topology="HIFI3D"):
    """

    Args:
        predicted_vertices: vertices from the predicted mesh, shape: nx3
        REALY_HIFI3D_keypoints: keypoints from the REALY HIFI3D mesh, shape: 85x3
        template_topology: topology of the predicted mesh, default: HIFI3D

    Returns: gICP aligned vertices of the predicted mesh

    """
    assert REALY_HIFI3D_keypoints.shape[0] == 85

    keypoints = keypoints_region_map["seven_keypoints"]
    predicted_keypoints = get_barycentric_coordinates(predicted_vertices, template_topology)[keypoints]

    trans, scale, R, t = fit_icp_RT(predicted_keypoints.T, REALY_HIFI3D_keypoints[keypoints].T, with_scale=True)

    rigid_align = np.matmul(predicted_vertices * scale, R.T) + t

    return rigid_align
