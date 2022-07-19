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
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
from sksparse.cholmod import cholesky_AAt


def spsolve(sparse_A, dense_b):
    factor = cholesky_AAt(sparse_A.T)
    return factor(sparse_A.T.dot(dense_b)).toarray()


def triangles_to_edge_vertex_adjacent_matrix(triangles):  # starting from 1
    v1, v2, v3 = np.split(triangles, 3, axis=0)
    p_list = []
    tri_t = triangles.T
    for v1, v2, v3 in tri_t:
        if v1 < v2:
            p_list.append([v1, v2])
        else:
            p_list.append([v2, v1])

        if v1 < v3:
            p_list.append([v1, v3])
        else:
            p_list.append([v3, v1])

        if v2 < v3:
            p_list.append([v2, v3])
        else:
            p_list.append([v3, v2])

    p = np.stack(p_list, axis=1)

    pair_list = np.split(p, p.shape[1], axis=1)
    pair_tuple_list = [(int(elem[0]), int(elem[1])) for elem in pair_list]
    pair_set = set(pair_tuple_list)
    pair_array = np.array(list(pair_set))
    edge_numbers = np.arange(len(pair_array))

    index_i = np.concatenate([edge_numbers] * 2, axis=0)
    index_j = np.concatenate([pair_array[:, 0], pair_array[:, 1]], axis=0)
    data = np.ones([len(pair_set)])
    data = np.concatenate([data, -data], axis=0)

    ev_adj = scipy.sparse.csc_matrix((data, (index_i, index_j)), shape=[len(edge_numbers), np.amax(triangles) + 1])
    return ev_adj


def sparse_matrix_from_vertices(cur_src):
    # cur_src: 3xN
    one_row = np.ones([1, cur_src.shape[1]], np.float32)
    cur_src_ = np.concatenate([cur_src, one_row], axis=0)

    Di = np.arange(cur_src.shape[1])
    Dj = np.stack([Di * 4, Di * 4 + 1, Di * 4 + 2, Di * 4 + 3], axis=0)
    Dj = np.reshape(Dj, [-1])
    Di = np.stack([Di] * 4, axis=0)
    Di = np.reshape(Di, [-1])
    cur_src_vector = np.reshape(cur_src_, [-1])
    D = scipy.sparse.csc_matrix(
        (cur_src_vector, (Di, Dj)), shape=[cur_src.shape[1], cur_src.shape[1] * 4], dtype=np.float32
    )
    return D


class NNSearch(object):
    def __init__(self, ver_dst, n=100):  # target vertices
        self.kd_tree = scipy.spatial.KDTree(ver_dst.T, n)

    def find_nearest_neighbors(self, ver_src):
        # kd-tree to find NN
        N_src = ver_src.shape[1]
        nn_distances, nn_indices = self.kd_tree.query(ver_src.T, 1, p=2)
        nn_indices = np.array(nn_indices, np.int32)
        nn_distances = np.array(nn_distances, np.float32)
        return nn_indices, nn_distances


class nICP_without_keypoints(object):
    def __init__(self, ver_src, ver_dst, tri_src, epsilon, gamma, alpha, beta):
        self.ver_src = ver_src  # 3 x N
        self.ver_dst = ver_dst  # 3 x N
        self.tri_src = tri_src  # 3 x NT
        self.epsilon = epsilon  # optimization ending threshold
        self.gamma = float(gamma)  # stiffness loss weight for translation
        self.alpha = float(alpha)  # exponential decay for each step
        self.beta = float(beta)  # landmark loss weight

        # G: weighting for balancing rotation and translation in stiffness loss
        self.G = np.diag(np.array([1.0, 1.0, 1.0, self.gamma], np.float32))

        # homogeneous ver_src
        # M: build a node-arc incidence matrix for M
        self.M = triangles_to_edge_vertex_adjacent_matrix(tri_src)  # E x N

        # A: large matrix for complete cost function
        self.A1 = scipy.sparse.kron(self.M, self.G)  # 4E x 4N

        # B: large matrix for complete cost function
        self.B1 = np.zeros([self.A1.shape[0], 3], np.float32)
        self.B1 = scipy.sparse.csc_matrix(self.B1)

        self.nn_search = NNSearch(ver_dst)

    def apply(self):
        cur_src = self.ver_src
        cur_X = np.zeros([cur_src.shape[1] * 4, 3])
        X = np.ones([cur_src.shape[1] * 4, 3])  # initialization

        for decay in range(3):
            cur_alpha = self.alpha * np.exp(-0.5 * decay)
            epsilon = self.epsilon * pow(0.5, decay)
            step = 0
            while True:
                delta = np.linalg.norm(X - cur_X)
                if delta <= epsilon:
                    break

                cur_X = X.copy()

                # find nn
                nn_indices, nn_distances = self.nn_search.find_nearest_neighbors(cur_src)
                cur_nn_ver_dst = self.ver_dst[:, nn_indices]

                cur_D = sparse_matrix_from_vertices(cur_src)  # N x 4N
                threshold = max(np.mean(nn_distances) * 2, 1)

                cur_weight = (nn_distances < threshold).astype(np.float32)

                A2 = cur_D.multiply(cur_weight[:, np.newaxis])
                B2 = cur_nn_ver_dst.T
                B2 = np.multiply(B2, cur_weight[:, np.newaxis])

                # convert to sparse matrix
                A2 = scipy.sparse.csc_matrix(A2)
                B2 = scipy.sparse.csc_matrix(B2)

                A = scipy.sparse.vstack([self.A1.multiply(cur_alpha), A2])
                B = scipy.sparse.vstack([self.B1, B2])

                X = spsolve(A, B)

                cur_src = cur_D * X
                cur_src = cur_src.transpose()
                step += 1

        self.ver_src = cur_src


class nICP_with_keypoints(object):
    def __init__(
        self, ver_src, ver_dst, tri_src, epsilon, gamma, alpha, beta, src_pts_KP_idx, tgt_pts_KP, kp_w1, kp_w2, alpha2
    ):
        self.ver_src = ver_src  # 3 x N
        self.ver_dst = ver_dst  # 3 x N
        self.tri_src = tri_src  # 3 x NT
        self.epsilon = epsilon  # optimization ending threshold
        self.gamma = float(gamma)  # stiffness loss weight for translation
        self.alpha = float(alpha)  # exponential decay for each step
        self.beta = float(beta)  # landmark loss weight

        self.src_pts_KP_idx = np.array(src_pts_KP_idx)
        self.tgt_pts_KP = tgt_pts_KP
        self.kp_w1 = float(kp_w1)
        self.kp_w2 = float(kp_w2)
        self.alpha2 = float(alpha2)

        # G: weighting for balancing rotation and translation in stiffness loss
        self.G = np.diag(np.array([1.0, 1.0, 1.0, self.gamma], np.float32))
        self.G_KP = np.diag(np.array([2.0, 2.0, 3.0, self.gamma * 2], np.float32))

        # homogeneous ver_src
        # M: build a node-arc incidence matrix for M
        self.M = triangles_to_edge_vertex_adjacent_matrix(tri_src)  # E x N

        # A: large matrix for complete cost function
        self.A1 = scipy.sparse.kron(self.M, self.G)  # 4E x 4N
        self.A1_KP = scipy.sparse.kron(self.M, self.G_KP)  # 4E x 4N

        # B: large matrix for complete cost function
        self.B1 = np.zeros([self.A1.shape[0], 3], np.float32)
        self.B1 = scipy.sparse.csc_matrix(self.B1)

        self.nn_search = NNSearch(ver_dst)

    def apply(self):

        cur_src = self.ver_src
        cur_X = np.zeros([cur_src.shape[1] * 4, 3])
        X = np.ones([cur_src.shape[1] * 4, 3])  # initialization

        count = 0
        for decay in range(4):
            if decay == 0:
                cur_alpha = self.alpha2
            else:
                cur_alpha = self.alpha * np.exp(-0.5 * (decay - 1))
            epsilon = self.epsilon * pow(0.5, (decay - 1))

            step = 0
            while True:
                delta = np.linalg.norm(X - cur_X)

                if (delta <= epsilon) or (step > 10):
                    break

                cur_X = X.copy()

                # find nn
                nn_indices, nn_distances = self.nn_search.find_nearest_neighbors(cur_src)
                cur_nn_ver_dst = self.ver_dst[:, nn_indices]

                cur_D = sparse_matrix_from_vertices(cur_src)  # N x 4N
                threshold = max(np.mean(nn_distances) * 2, 1)
                cur_weight = (nn_distances < threshold).astype(np.float32)

                A2 = cur_D.multiply(cur_weight[:, np.newaxis])
                B2 = cur_nn_ver_dst.T
                B2 = np.multiply(B2, cur_weight[:, np.newaxis])

                # convert to sparse matrix
                A2 = scipy.sparse.csc_matrix(A2)
                B2 = scipy.sparse.csc_matrix(B2)

                kp_weight = np.zeros([cur_D.shape[0], 1])
                if decay == 0:
                    kp_weight[self.src_pts_KP_idx, :] = self.kp_w1
                else:
                    kp_weight[self.src_pts_KP_idx, :] = self.kp_w1 * np.exp(-0.9 * count) / 10.0

                cur_D_KP = cur_D.copy()  # N x 4N
                A3 = cur_D_KP.multiply(kp_weight)

                tgt_pts_KP_full = cur_nn_ver_dst.copy()
                tgt_pts_KP_full[:, self.src_pts_KP_idx] = self.tgt_pts_KP

                B3 = tgt_pts_KP_full.T
                B3 = np.multiply(B3, kp_weight)

                # convert to sparse matrix
                A3 = scipy.sparse.csc_matrix(A3)
                B3 = scipy.sparse.csc_matrix(B3)

                if decay == 0:
                    A = scipy.sparse.vstack([self.A1_KP.multiply(cur_alpha), A3])
                    B = scipy.sparse.vstack([self.B1, B3])
                else:
                    A = scipy.sparse.vstack([self.A1.multiply(cur_alpha), A2, A3])
                    B = scipy.sparse.vstack([self.B1, B2, B3])

                X = spsolve(A, B)

                cur_src = cur_D * X
                cur_src = cur_src.transpose()

                step += 1
                count += 1

                if decay == 0:
                    break

        self.ver_src = cur_src
