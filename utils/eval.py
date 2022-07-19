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
import trimesh
from utils.NICP import nICP_with_keypoints
from utils.NICP import nICP_without_keypoints
import cv2


def bidirectional_evaluation_pipeline(
    predicted_mesh,
    REALY_scan_region_mesh,
    predicted_keypoints_region,
    REALY_HIFI3D_keypoints_region,
    region="mouth",
    visualize_error_map=True,
):
    """

    Args:
        predicted_mesh: vertices from the regional aligned predicted mesh, shape: nx3
        REALY_scan_region_mesh: mesh from the REALY scan of specific region, <V,F>, Trimesh
        predicted_keypoints_region: regional keypoints from the predicted mesh specific region, shape: n1x3
        REALY_HIFI3D_keypoints_region: regional keypoints from the REALY HIFI3D mesh of specific region, shape: n1x3
        region: selected face region for alignment, str
        visualize_error_map: whether return the error map on ground-truth scan regions, bool: True

    Returns: the reconstruction error NMSE of the predicted mesh@region and error map scan

    """

    # init hyper-parameters on non-rigid alignment
    epsilon = 1.0
    gamma = 1
    alpha = 50
    beta = 0.5

    # 2nd-stage
    alpha2 = 50 * 3
    kp_w1 = 50.0  # key point weight-forhead
    kp_w2 = 50.0  # key point weight-others

    # Step1: find the nn-map of the ground truth vertices on the regional-aligned predicted mesh
    # to ensure the correspondence share the same vertex number as the ground-truth

    REALY_scan_region_vertices = REALY_scan_region_mesh.vertices
    REALY_scan_region_triangles = REALY_scan_region_mesh.faces
    correspondence_on_prediction, _, _ = predicted_mesh.nearest.on_surface(REALY_scan_region_vertices)

    # use REALY HIFI3D keypoints to find the keypoint index on REALY scan
    _, REALY_scan_keypoints_index = REALY_scan_region_mesh.nearest.vertex(REALY_HIFI3D_keypoints_region)

    # Step2: non-rigid alignment that deform the REALY scan region into the predicted mesh
    # through the coarse correspondence built from the nn-map in Step1
    if region == "cheek":
        deformation_tool = nICP_without_keypoints(
            REALY_scan_region_vertices.T,
            correspondence_on_prediction.T,
            REALY_scan_region_triangles.T,
            epsilon,
            gamma,
            alpha,
            beta,
        )
    else:
        deformation_tool = nICP_with_keypoints(
            REALY_scan_region_vertices.T,
            correspondence_on_prediction.T,
            REALY_scan_region_triangles.T,
            epsilon,
            gamma,
            alpha,
            beta,
            REALY_scan_keypoints_index,
            predicted_keypoints_region.T,
            kp_w1,
            kp_w2,
            alpha2,
        )

    deformation_tool.apply()

    # Step3: update the correspondence between the ground truth scan and the predicted mesh
    REALY_scan_deformed = trimesh.Trimesh(
        vertices=deformation_tool.ver_src.T, faces=REALY_scan_region_triangles, process=False
    )

    correspondence_on_prediction_updated, _, _ = predicted_mesh.nearest.on_surface(REALY_scan_deformed.vertices)

    # Step4: calculate the NMSE error
    # between the regional aligned predicted mesh and the original ground truth scan region
    normalized_mean_square_error = np.sqrt(
        np.sum((correspondence_on_prediction_updated - REALY_scan_region_vertices) ** 2, axis=-1)
    ).mean()

    if visualize_error_map is True:
        mean_square_error_on_vertices = np.sqrt(
            np.sum((correspondence_on_prediction_updated - REALY_scan_region_vertices) ** 2, axis=-1)
        )
        mean_square_error_on_vertices = (mean_square_error_on_vertices - 0) / 0.5
        mean_square_error_on_vertices[mean_square_error_on_vertices > 1] = 1.0

        rgb_values_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET).reshape(256, 3)[::-1]
        vertices_colors = rgb_values_map[(mean_square_error_on_vertices * 255).astype(np.int32)]

        rgb_gt_region = trimesh.Trimesh(
            vertices=REALY_scan_region_vertices,
            faces=REALY_scan_region_triangles,
            vertex_colors=vertices_colors,
            process=False,
        )

        return normalized_mean_square_error, REALY_scan_deformed, rgb_gt_region

    return normalized_mean_square_error, REALY_scan_deformed
