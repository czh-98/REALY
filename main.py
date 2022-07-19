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
from glob import glob
from utils.io_obj import read, write
import trimesh
from utils.util import keypoints_region_map
from utils.util import get_barycentric_coordinates
import argparse
from tqdm import tqdm
from utils.rICP import region_icp_all as rICP_all
from utils.gICP import global_rigid_align_7_kpt as gicp
from utils.eval import bidirectional_evaluation_pipeline as bi_eval

import warnings

warnings.filterwarnings("ignore")


def get_statistic_metric(args):
    REALY_error_save_path = os.path.join(args.save, "REALY_error.txt")

    error_at_nose_list = []
    error_at_mouth_list = []
    error_at_forehead_list = []
    error_at_cheek_list = []

    with open(REALY_error_save_path, "r") as f:
        for line in f:
            subject_id, _, error_at_nose, error_at_mouth, error_at_forehead, error_at_cheek = line.replace(
                "\n", ""
            ).split("\t")
            error_at_nose_list.append(float(error_at_nose))
            error_at_mouth_list.append(float(error_at_mouth))
            error_at_forehead_list.append(float(error_at_forehead))
            error_at_cheek_list.append(float(error_at_cheek))

    error_at_nose_list = np.array(error_at_nose_list)
    error_at_mouth_list = np.array(error_at_mouth_list)
    error_at_forehead_list = np.array(error_at_forehead_list)
    error_at_cheek_list = np.array(error_at_cheek_list)

    # average
    print("average nmse@nose = %.3f" % error_at_nose_list.mean())
    print("average nmse@mouth = %.3f" % error_at_mouth_list.mean())
    print("average nmse@forehead = %.3f" % error_at_forehead_list.mean())
    print("average nmse@cheek = %.3f" % error_at_cheek_list.mean())

    # std
    std = lambda data: data.std() * np.sqrt(data.shape[0]) / np.sqrt(data.shape[0] - 1)
    print("standard deviation nmse@nose = %.3f" % std(error_at_nose_list))
    print("standard deviation nmse@mouth = %.3f" % std(error_at_mouth_list))
    print("standard deviation nmse@forehead = %.3f" % std(error_at_forehead_list))
    print("standard deviation nmse@cheek = %.3f" % std(error_at_cheek_list))

    # all
    print(
        "average nmse@all = %.3f"
        % (
            (
                error_at_nose_list.mean()
                + error_at_mouth_list.mean()
                + error_at_forehead_list.mean()
                + error_at_cheek_list.mean()
            )
            / 4.0
        )
    )

    return


def REALY_eval_all(args):
    REALY_keypoints_root = args.REALY_HIFI3D_keypoints
    REALY_scan_region_root = args.REALY_scan_region
    prediction_mesh_root = args.prediction
    template_topology = args.template_topology
    template_vertices_mask = args.template_mask

    aligned_predicted_mesh_save_root = os.path.join(args.save, "region_align_save")
    os.makedirs(aligned_predicted_mesh_save_root, exist_ok=True)

    REALY_error_save_path = os.path.join(args.save, "REALY_error.txt")

    metrical_scale = np.loadtxt(args.scale_path)

    if template_vertices_mask is not None:
        pred_mask_face = np.loadtxt(template_vertices_mask, delimiter=",", dtype=np.int32)
        pred_mask_face = pred_mask_face.reshape(-1, 1)
    else:
        pred_mask_face = None

    predicted_mesh_subjects = glob(os.path.join(prediction_mesh_root, "*.obj"))
    predicted_mesh_subjects.sort(key=lambda x: int(os.path.split(x)[1].replace("_", "").replace(".obj", "")))

    with tqdm(total=len(predicted_mesh_subjects)) as pbar:
        with open(REALY_error_save_path, "w") as f_w:
            for subject_path in predicted_mesh_subjects:
                pbar.update(1)
                _, subject = os.path.split(subject_path)
                subject = subject.replace(".obj", "")
                subject_id = int(subject.split("_")[0])

                # read 3D meshes for evaluation
                REALY_keypoints_path = os.path.join(REALY_keypoints_root, str(subject_id) + ".obj")
                predicted_mesh_path = os.path.join(prediction_mesh_root, subject + ".obj")
                predicted_mesh = read(predicted_mesh_path)
                REALY_HIFI3D_keypoints = read(REALY_keypoints_path)["v"]

                REALY_scan_region_path = os.path.join(REALY_scan_region_root, str(subject_id))
                REALY_scan_nose_mesh = read(os.path.join(REALY_scan_region_path, "nose.obj"))
                REALY_scan_mouth_mesh = read(os.path.join(REALY_scan_region_path, "mouth.obj"))
                REALY_scan_forehead_mesh = read(os.path.join(REALY_scan_region_path, "forehead.obj"))
                REALY_scan_cheek_mesh = read(os.path.join(REALY_scan_region_path, "cheek.obj"))

                REALY_scan_nose_vertices = REALY_scan_nose_mesh["v"]
                REALY_scan_mouth_vertices = REALY_scan_mouth_mesh["v"]
                REALY_scan_forehead_vertices = REALY_scan_forehead_mesh["v"]
                REALY_scan_cheek_vertices = REALY_scan_cheek_mesh["v"]

                REALY_scan_nose_triangles = REALY_scan_nose_mesh["fv"]
                REALY_scan_mouth_triangles = REALY_scan_mouth_mesh["fv"]
                REALY_scan_forehead_triangles = REALY_scan_forehead_mesh["fv"]
                REALY_scan_cheek_triangles = REALY_scan_cheek_mesh["fv"]

                # Step1: gICP, global-align the prediction mesh to the ground-truth scan
                # We use 7 keypoint for global alignment for efficient
                # Here is used to save the globally rigid aligned mesh
                predicted_vertices_global_aligned = gicp(
                    predicted_mesh["v"], REALY_HIFI3D_keypoints, template_topology=template_topology
                )

                # Step2: rICP, regional align the prediction mesh to the ground-truth scan region
                REALY_scan_region_vertices_dict = {
                    "nose": REALY_scan_nose_vertices,
                    "mouth": REALY_scan_mouth_vertices,
                    "forehead": REALY_scan_forehead_vertices,
                    "cheek": REALY_scan_cheek_vertices,
                }

                REALY_scan_region_triangles_dict = {
                    "nose": REALY_scan_nose_triangles,
                    "mouth": REALY_scan_mouth_triangles,
                    "forehead": REALY_scan_forehead_triangles,
                    "cheek": REALY_scan_cheek_triangles,
                }

                region_list = ["nose", "mouth", "forehead", "cheek"]

                # for each region, regional aligned
                regional_aligned_vertices_dict, regional_aligned_triangles_dict = rICP_all(
                    predicted_mesh=predicted_mesh,
                    REALY_scan_region_dict=REALY_scan_region_vertices_dict,
                    REALY_HIFI3D_keypoints=REALY_HIFI3D_keypoints,
                    template_topology=template_topology,
                    max_iteration=100,
                    pred_mask_face=pred_mask_face,
                )

                align_save_path = os.path.join(aligned_predicted_mesh_save_root, subject)

                os.makedirs(align_save_path, exist_ok=True)

                write(
                    os.path.join(align_save_path, "at_global.obj"),
                    predicted_vertices_global_aligned,
                    regional_aligned_triangles_dict["forehead"],
                )

                write(
                    os.path.join(align_save_path, "at_nose.obj"),
                    regional_aligned_vertices_dict["nose"],
                    regional_aligned_triangles_dict["nose"],
                )
                write(
                    os.path.join(align_save_path, "at_mouth.obj"),
                    regional_aligned_vertices_dict["mouth"],
                    regional_aligned_triangles_dict["mouth"],
                )
                write(
                    os.path.join(align_save_path, "at_forehead.obj"),
                    regional_aligned_vertices_dict["forehead"],
                    regional_aligned_triangles_dict["forehead"],
                )
                write(
                    os.path.join(align_save_path, "at_cheek.obj"),
                    regional_aligned_vertices_dict["cheek"],
                    regional_aligned_triangles_dict["cheek"],
                )

                keypoints_map = {
                    "forehead": keypoints_region_map["forehead"],
                    "cheek": None,
                    "nose": keypoints_region_map["nose"],
                    "mouth": keypoints_region_map["mouth"],
                }

                f_w.write(subject)
                f_w.write("\t")

                # bICP and calculate errors for each region
                for region in region_list:
                    predicted_keypoints = get_barycentric_coordinates(
                        regional_aligned_vertices_dict[region], template_topology
                    )
                    predicted_keypoints_region = predicted_keypoints[keypoints_map[region]]
                    REALY_HIFI3D_keypoints_region = REALY_HIFI3D_keypoints[keypoints_map[region]]

                    regional_aligned_pd = trimesh.Trimesh(
                        vertices=regional_aligned_vertices_dict[region],
                        faces=regional_aligned_triangles_dict[region] - 1,
                        process=False,
                    )
                    ground_truth_region = trimesh.Trimesh(
                        vertices=REALY_scan_region_vertices_dict[region],
                        faces=REALY_scan_region_triangles_dict[region] - 1,
                        process=True,
                    )

                    error, deformation, error_map = bi_eval(
                        regional_aligned_pd,
                        ground_truth_region,
                        predicted_keypoints_region,
                        REALY_HIFI3D_keypoints_region,
                        region,
                        visualize_error_map=True,
                    )

                    f_w.write("\t")
                    f_w.write(str(error * metrical_scale[subject_id - 1])[:9])  # id start from 1
                    deformation.export(os.path.join(align_save_path, "deform_%s.obj" % region))
                    error_map.export(os.path.join(align_save_path, "error_%s.obj" % region))

                f_w.write("\n")
                f_w.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--REALY_HIFI3D_keypoints",
        help="PATH to the ground-truth HIFI3D mesh",
        default="./data/REALY_HIFI3D_keypoints",
        type=str,
    )
    parser.add_argument(
        "--REALY_scan_region",
        help="PATH to the ground-truth scan region",
        default="./data/REALY_scan_region",
        type=str,
    )

    parser.add_argument(
        "--prediction",
        help="PATH to the predicted mesh",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--template_mask",
        help='PATH to the face mask of template vertices, save as ".txt" format, values: 0 or 1',
        default=None,
    )
    parser.add_argument("--template_topology", help="template topology", default="HIFI3D++", type=str)

    parser.add_argument(
        "--scale_path", help="PATH to the metrical scale file", default="./data/metrical_scale.txt", type=str
    )
    parser.add_argument(
        "--save",
        help="PATH to save the aligned meshes, deformation meshes, error map, and NMSE results",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    REALY_eval_all(args)
    get_statistic_metric(args)
