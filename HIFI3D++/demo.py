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

import scipy.io


def HIFI3D_plus_demo():
    HIFI3D_plus = scipy.io.loadmat("../../REALY_data_release/HIFI3D++.mat")
    print(HIFI3D_plus.keys())

    # shape mean, 1 x (20481x3)
    print(HIFI3D_plus["mu_shape"].shape)

    # shape basis, 526 x (20481x3)
    print(HIFI3D_plus["basis_shape"].shape)

    # expression basis, 203 x (20481x3)
    print(HIFI3D_plus["basis_exp"].shape)

    # eigenvalue, 526 x 1
    print(HIFI3D_plus["EVs"].shape)

    # triangles, 40832 x 3
    print(HIFI3D_plus["tri"].shape)

    # facial masks [0,1], 1 x 20481
    print(HIFI3D_plus["mask_face"].shape)

    # 86 keypoint index for RGB(-D) fitting, 1 x 86, c.f. https://github.com/tencent-ailab/hifi3dface
    print(HIFI3D_plus["keypoints"].shape)


if __name__ == "__main__":
    HIFI3D_plus_demo()
