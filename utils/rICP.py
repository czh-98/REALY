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

from utils.util import keypoints_region_map
from utils.util import get_barycentric_coordinates, fit_icp_RT
from utils.gICP import global_rigid_align_7_kpt as gicp


def region_icp_one_region(
    predicted_vertices,
    REALY_scan_region_vertices,
    predicted_keypoints,
    REALY_HIFI3D_keypoints,
    predicted_triangles,
    max_iteration=100,
    region="mouth",
):
    """

    Args:
        predicted_vertices: vertices from the predicted mesh, shape: nx3
        REALY_scan_region_vertices: vertices from the REALY scan of specific region, shape: Nx3
        predicted_keypoints: regional keypoints from the predicted mesh specific region, shape: n1x3
        REALY_HIFI3D_keypoints: regional keypoints from the REALY HIFI3D mesh of specific region, shape: n1x3
        predicted_triangles: triangles from the predicted mesh, shape: mx3, start from 0
        max_iteration: max iterations for rICP alignment, default: 100
        region: selected face region for alignment

    Returns: rICP aligned vertices of the predicted mesh@region

    """

    assert predicted_triangles.min() == 0

    # init
    error_pre = 0.0
    cur_iter = 0.0

    # set weight
    # repeat the keypoints[x rate] to adjust the weight between the keypoints to other vertices
    if region == "nose" or region == "mouth" or region == "forehead":
        rate = REALY_scan_region_vertices.shape[0] / REALY_HIFI3D_keypoints.shape[0]
    elif region == "cheek":
        rate = REALY_scan_region_vertices.shape[0] / REALY_HIFI3D_keypoints.shape[0] / 4
    else:
        raise Exception("Undefined region error")

    predicted_keypoints_correspondence = np.vstack([predicted_keypoints] * int(rate))
    REALY_HIFI3D_keypoints_correspondence = np.vstack([REALY_HIFI3D_keypoints] * int(rate))

    while True:
        predicted_mesh = trimesh.Trimesh(vertices=predicted_vertices, faces=predicted_triangles, process=False)
        correspondence_on_prediction, _, _ = predicted_mesh.nearest.on_surface(REALY_scan_region_vertices)

        correspondence_on_prediction = np.vstack((correspondence_on_prediction, predicted_keypoints_correspondence))
        correspondence_on_ground_truth = np.vstack((REALY_scan_region_vertices, REALY_HIFI3D_keypoints_correspondence))

        trans, _, R, t = fit_icp_RT(correspondence_on_prediction.T, correspondence_on_ground_truth.T, with_scale=False)

        # rigid align on seg regions
        correspondence_on_prediction_aligned = np.matmul(correspondence_on_prediction, R.T) + t
        correspondence_on_prediction_aligned = correspondence_on_prediction_aligned.astype(np.float32)

        error = np.sum((correspondence_on_prediction_aligned - correspondence_on_ground_truth) ** 2, axis=1).mean()

        # early stop if alignment converged
        if abs(error - error_pre) < 1e-6 or cur_iter >= max_iteration:
            break

        cur_iter += 1
        error_pre = error

        # update prediction and correspondence

        # update predicted vertices
        predicted_vertices = np.matmul(predicted_vertices, R.T) + t
        predicted_vertices = predicted_vertices.astype(np.float32)

        # rigid align on seg regions
        predicted_keypoints_correspondence = np.matmul(predicted_keypoints_correspondence, R.T) + t
        predicted_keypoints_correspondence = predicted_keypoints_correspondence.astype(np.float32)

    return predicted_vertices


def region_icp_all(
    predicted_mesh=None,
    REALY_scan_region_dict=None,
    REALY_HIFI3D_keypoints=None,
    template_topology="HIFI3D",
    max_iteration=100,
    pred_mask_face=None,
):
    """

    Args:
        predicted_mesh: the predicted mesh, obj <V,F>
        REALY_scan_region_dict: dict of the ground-truth scan regions from REALY, v_rx3
        REALY_HIFI3D_keypoints: keypoints from the REALY HIFI3D mesh, shape: 85x3
        template_topology: topology of the predicted mesh, default: HIFI3D
        max_iteration: max iterations for rICP alignment, default: 100
        pred_mask_face: region-of-interest mask of predicted mesh, shape nx1, 0,1

    Returns: the rICP results of predicted mesh <V,F> to four ground-truth regions in REALY

    """

    # Step1: gICP, global-align the prediction mesh to the ground-truth scan
    # We use 7 keypoint for global alignment for efficient
    predicted_mesh["v"] = gicp(predicted_mesh["v"], REALY_HIFI3D_keypoints, template_topology=template_topology)

    predicted_vertex = predicted_mesh["v"]

    if pred_mask_face is None:
        pred_mask_face = np.ones([predicted_vertex.shape[0], 1])
    else:
        pred_mask_face = pred_mask_face

    predicted_keypoints = get_barycentric_coordinates(predicted_vertex, template_topology)

    REALY_HIFI3D_scan_region_list = ["forehead", "mouth", "nose", "cheek"]
    rigid_regions = ["forehead", "mouth", "nose_part", "cheek"]

    # Step2: rICP, for each region, regionally align the predicted mesh to corresponding scan regions
    regional_aligned_vertices = {}
    regional_aligned_triangle = {}
    for region, rigid_region in zip(REALY_HIFI3D_scan_region_list, rigid_regions):

        predicted_vertex = predicted_mesh["v"]

        predicted_keypoints_region = predicted_keypoints[keypoints_region_map[rigid_region]]
        REALY_HIFI3D_keypoints_region = REALY_HIFI3D_keypoints[keypoints_region_map[rigid_region]]
        REALY_scan_region_vertices = REALY_scan_region_dict[region]

        predicted_vertex_regional_aligned = region_icp_one_region(
            predicted_vertex,
            REALY_scan_region_vertices,
            predicted_keypoints_region,
            REALY_HIFI3D_keypoints_region,
            predicted_mesh["fv"] - 1,
            max_iteration=max_iteration,
            region=region,
        )

        regional_aligned_vertices[region] = predicted_vertex_regional_aligned

        # this is only for visualization
        predicted_masked_triangle = []
        for tri in predicted_mesh["fv"]:
            if pred_mask_face[tri - 1].all() == 1:
                predicted_masked_triangle.append(tri)
        regional_aligned_triangle[region] = np.array(predicted_masked_triangle, dtype=np.int32)

    # return the regional-aligned predicted mesh
    return regional_aligned_vertices, regional_aligned_triangle
