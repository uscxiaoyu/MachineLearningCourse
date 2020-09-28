import numpy as np
import pandas as pd
import networkx as nx


def distance(xi, xj, p=2):
    return np.sum((np.abs(xi - xj)) ** p) ** (1 / p)


def generate_kd_tree(X, y):
    """
    X: ndarray
    y: ndarray
    """
    k = X.shape[1]  # X的维度k
    kd_tree = nx.DiGraph()
    node_id = 0
    no_tag_nodes = [node_id]
    kd_tree.add_node(node_id)
    kd_tree.nodes[node_id]["X"] = X
    kd_tree.nodes[node_id]["y"] = y
    kd_tree.nodes[node_id]["node_type"] = "root"
    i = 0
    while no_tag_nodes:
        new_nodes = []
        dim = i % k  # 当前的维度
        for node in no_tag_nodes:
            c_X = kd_tree.nodes[node]["X"]
            c_y = kd_tree.nodes[node]["y"]
            x_dim = c_X[:, dim]
            if len(x_dim) >= 2:  # 如果有2个以上样本，则继续分
                kd_tree.nodes[node]["dim"] = dim  # 结点的切分维度
                s_indices = np.argsort(x_dim)
                m = int(len(s_indices) / 2)  # 中间的索引
                l_indices = s_indices[:m]  # 左子区域
                m_idx = s_indices[m]  # 留在结点上
                r_indices = s_indices[m + 1 :]  # 右子区域
                l_X, l_y = c_X[l_indices], c_y[l_indices]
                r_X, r_y = c_X[r_indices], c_y[r_indices]
                if l_y.size > 0:
                    node_id += 1
                    kd_tree.add_edge(node, node_id)
                    kd_tree.nodes[node_id]["X"] = l_X
                    kd_tree.nodes[node_id]["y"] = l_y
                    kd_tree.nodes[node_id]["node_type"] = "non_leaf"
                    kd_tree.nodes[node]["l_succ"] = node_id
                    new_nodes.append(node_id)

                if r_y.size > 0:
                    node_id += 1
                    kd_tree.add_edge(node, node_id)
                    kd_tree.nodes[node_id]["X"] = r_X
                    kd_tree.nodes[node_id]["y"] = r_y
                    kd_tree.nodes[node_id]["node_type"] = "non_leaf"
                    kd_tree.nodes[node]["r_succ"] = node_id
                    new_nodes.append(node_id)

                kd_tree.nodes[node]["point"] = (c_X[m_idx], c_y[m_idx])
            else:
                kd_tree.nodes[node]["node_type"] = "leaf"
                kd_tree.nodes[node]["point"] = (c_X[0], c_y[0])

        i += 1
        no_tag_nodes = new_nodes

    return kd_tree


def search_kd_tree(x, node, kd_tree):
    """
    搜索node在哪个区域(叶结点)
    """
    if kd_tree.nodes[node]["node_type"] != "leaf":
        dim = kd_tree.nodes[node]["dim"]
        median = kd_tree.nodes[node]["point"][0][dim]
        if x[dim] == median:  # 点在内部结点上，同时搜索左右两个子结点
            return [
                search_kd_tree(x, kd_tree.nodes[node]["l_succ"], kd_tree),
                search_kd_tree(x, kd_tree.nodes[node]["r_succ"], kd_tree),
            ]
        elif x[dim] < median:  # 左子节点
            return search_kd_tree(x, kd_tree.nodes[node]["l_succ"], kd_tree)
        else:  # 右子结点
            return search_kd_tree(x, kd_tree.nodes[node]["r_succ"], kd_tree)
    else:
        return node


def flatten(a_list, result=[]):
    for a in a_list:
        if isinstance(a, list):
            result = flatten(a, result=result)
        else:
            result.append(a)

    return result


def find_k_neighbors(x, node, k, kd_tree):
    """
    从叶结点x回退
    k_list保存离x最近的k个点
    """
    dist_node_x = distance(x, kd_tree.nodes[node]["point"][0])
    k_list = [[dist_node_x, node]]  # 保存k个最近邻居
    back_list = [node]  # 回退历史
    while kd_tree.nodes[node]["node_type"] != "root":
        p_node = list(kd_tree.predeccessors(node))[0]
        if p_node not in back_list:  # 如果p_node没有在回退过程中被评估
            back_list.append(p_node)
            dim = kd_tree.nodes[p_node]["dim"]  # 父结点的切分维度
            p_x = kd_tree.nodes[p_node]["point"][0]  # 父结点保存的数据x
            dist_pnode_x = distance(x, p_x)  # x到p_node的距离
            dist_div_x = np.abs(p_x[dim] - x[dim])  # x 到 p_node所在切割面的距离
            # 决定是否往k_list中添加p_node
            if len(k_list) < k or dist_pnode_x < k_list[-1][0]:  # 不足k个近邻 or 小于k_list中距离最远的点
                k_list.append([dist_pnode_x, p_node])
                k_list = sorted(k_list)[:k]

            # 决定是否向上回退或者往node兄弟结点搜索
            if dist_div_x < k_list[-1][0]:  # 到父结点分割平面的距离小于k_list中最远的点
                sibling_nodes = list(kd_tree.successors(p_node))
                sibling_nodes.remove(node)
                t_node = sibling_nodes[0]  # node的兄弟结点
                node = search_kd_tree(x, t_node, kd_tree)  # t_node距x最近的叶结点, 由node往上回退
                back_list.append(node)
                dist_node_x = distance(x, kd_tree.nodes[node]["point"][0])
                if len(k_list) < k or dist_node_x < k_list[-1][0]:  # 决定是否将该叶结点加入到k_list
                    k_list.append([dist_node_x, node])
                    k_list = sorted(k_list)[:k]
            else:  # 如果父结点分割面的距离大于当前最近距离，则设置当前结点为p_node，继续下一轮循环
                node = p_node
        else:  # 如果p_node结点已被遍历
            node = p_node

    return k_list
