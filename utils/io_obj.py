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
import re


def read(file_path):
    """

    Args:
        file_path: obj file path

    Returns: obj dict: <V,F>

    """
    vertices = []  # v
    vertices_texture = []  # vt
    vertices_normal = []  # vn

    vertices_rgb = []

    # notice that faces are converted to triangular faces
    face_v = []  # f 1 2 3
    face_vt = []  # f 1/1 2/2 3/3
    face_vn = []  # f 1/1/1 2/2/2 3/3/3

    lines = open(file_path, "r").readlines()
    for line in lines:
        line = re.sub(" +", " ", line)
        if line.startswith("v "):
            toks = line.strip().split(" ")[1:]
            try:
                vertices.append([float(toks[0]), float(toks[1]), float(toks[2])])
                if len(toks) > 3:
                    vertices_rgb.append([float(toks[3]), float(toks[4]), float(toks[5])])
            except Exception:
                print(toks)
        elif line.startswith("vt "):
            toks = line.strip().split(" ")[1:]
            vertices_texture.append([float(toks[0]), float(toks[1])])
        elif line.startswith("vn "):
            toks = line.strip().split(" ")[1:]
            vertices_normal.append([float(toks[0]), float(toks[1]), float(toks[2])])
        elif line.startswith("f "):
            toks = line.strip().split(" ")[1:]
            if len(toks) == 3:  # triangular faces
                faces1 = toks[0].split("/")
                faces2 = toks[1].split("/")
                faces3 = toks[2].split("/")

                face_v.append([faces1[0], faces2[0], faces3[0]])
                if len(faces1) >= 2:
                    face_vt.append([faces1[1], faces2[1], faces3[1]])
                if len(faces1) >= 3:
                    if len(faces1[2]) == 0:
                        continue
                    face_vn.append([faces1[2], faces2[2], faces3[2]])
            if len(toks) == 4:  # quad faces
                faces1 = toks[0].split("/")
                faces2 = toks[1].split("/")
                faces3 = toks[2].split("/")
                faces4 = toks[3].split("/")

                face_v.append([faces1[0], faces2[0], faces3[0]])
                face_v.append([faces1[0], faces3[0], faces4[0]])
                if len(faces1) >= 2:
                    face_vt.append([faces1[1], faces2[1], faces3[1]])
                    face_vt.append([faces1[1], faces3[1], faces4[1]])
                if len(faces1) >= 3:
                    if len(faces1[2]) == 0:
                        continue
                    face_vn.append([faces1[2], faces2[2], faces3[2]])
                    face_vn.append([faces1[2], faces3[2], faces4[2]])

    results = {}
    results["v"] = np.array(vertices, np.float32)
    if len(vertices_texture) > 0:
        results["vt"] = np.array(vertices_texture, np.float32)
    if len(vertices_normal) > 0:
        results["vn"] = np.array(vertices_normal, np.float32)
    if len(face_v) > 0:
        results["fv"] = np.array(face_v, np.int32)

    if len(face_vt) > 0:
        results["fvt"] = np.array(face_vt, np.int32)
    if len(face_vn) > 0:
        results["fvn"] = np.array(face_vn, np.int32)

    if len(vertices_rgb) > 0:
        results["v_rgb"] = np.array(vertices_rgb)

    return results


def write(filename, v, f=None, vt=None, fvt=None, vn=None, fvn=None):
    """

    Args:
        filename: obj save path
        v: vertices
        f: triangles
        vt: texture coordinate
        fvt: triangle uv texture
        vn: normal coordinate
        fvn: triangle uv normal

    Returns: None

    """
    with open(filename, "w") as fp:

        for x, y, z in v:
            fp.write("v %f %f %f\n" % (x, y, z))

        if f is not None:

            if vt is not None:
                for u, v in vt:
                    fp.write("vt %f %f\n" % (u, v))

            if vn is not None:
                for x, y, z in vn:
                    fp.write("vn %f %f %f\n" % (x, y, z))

            if fvt is None and fvn is None:  # f only
                for v1, v2, v3 in f:
                    fp.write("f %d %d %d\n" % (v1, v2, v3))
            elif fvn is None:
                for (v1, v2, v3), (t1, t2, t3) in zip(f, fvt):
                    fp.write("f %d/%d %d/%d %d/%d\n" % (v1, t1, v2, t2, v3, t3))
            else:
                for (v1, v2, v3), (t1, t2, t3), (n1, n2, n3) in zip(f, fvt, fvn):
                    fp.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (v1, t1, n1, v2, t2, n2, v3, t3, n3))
