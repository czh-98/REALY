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
import os
import json
from utils.io_obj import read

keypoints_region_map = {
    "forehead": [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    "mouth": [
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        64,
    ],
    "nose": [27, 28, 29, 30, 31, 32, 33, 34, 35],
    "nose_part": [27, 28, 29, 30, 31, 35, 32, 33, 34, 70, 71],
    "cheek": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 76, 77, 36, 41, 40, 39, 42, 47, 46, 45, 7, 8, 9],
    "seven_keypoints": [36, 39, 42, 45, 33, 48, 54],
}


def get_barycentric_coordinates(vertices, template_topology="HIFI3D"):
    """

    Args:
        vertices: vertices from the topology-consistent mesh, shape: nx3
        template_topology: topology of the topology-consistent mesh, default: HIFI3D

    Returns: 85 keypoints of the mesh, shape: 85x3

    """
    bary_root = r"./data/"
    obj_root = r"./data/"
    bary_path = os.path.join(bary_root, "%s.txt" % template_topology)
    obj_path = os.path.join(obj_root, "%s.obj" % template_topology)

    assert os.path.exists(bary_path) and os.path.exists(obj_path)

    template_mesh = read(obj_path)
    template_triangles = template_mesh["fv"] - 1

    with open(bary_path, "r") as f:
        barycentric_coordinates = json.load(f)

    barycentric_coordinates = np.array(barycentric_coordinates)

    triangle_index = np.array(barycentric_coordinates[:, 0], dtype=np.int32)
    keypoints_coordinates = vertices[template_triangles[triangle_index]]
    weight = np.stack(
        [
            barycentric_coordinates[:, 1],
            barycentric_coordinates[:, 2],
            1.0 - barycentric_coordinates[:, 1] - barycentric_coordinates[:, 2],
        ]
    ).T.reshape(-1, 3, 1)

    keypoints = keypoints_coordinates * weight
    keypoints = np.sum(keypoints, axis=1)

    assert keypoints.shape[0] == 85

    return keypoints


def fit_icp_RT(source, target, with_scale=True):
    """

    Args:
        source: float vertices, shape: n1x3
        target: fixed vertices, shape: n1x3
        with_scale: whether use the scale factor, bool

    Returns: transformation matrix, scale, rotation matrix, translation matrix

    """

    assert source.shape[0] == 3

    npoint = source.shape[1]
    means = np.mean(source, 1)
    meant = np.mean(target, 1)
    s1 = source - np.tile(means, (npoint, 1)).transpose()
    t1 = target - np.tile(meant, (npoint, 1)).transpose()
    W = t1.dot(s1.transpose())
    U, sig, V = np.linalg.svd(W)
    rotation = U.dot(V)

    scale = np.sum(np.sum(abs(t1))) / np.sum(np.sum(abs(rotation.dot(s1)))) if with_scale else 1.0

    translation = target - scale * rotation.dot(source)
    translation = np.mean(translation, 1)

    trans = np.zeros((4, 4))
    trans[3, 3] = 1
    trans[:3, 0:3] = scale * rotation[:, 0:3]
    trans[:3, 3] = translation[:]

    return trans, scale, rotation, translation
