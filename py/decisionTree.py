import networkx as nx
import numpy as np
import pandas as pd
from copy import deepcopy


class DecisionTreeClassifier:
    '''
    算法实现思想：树的生成过程为结点的新增及其标记(name)过程，因此需判断何时新增结点？新增的节点是否满足叶结点的条件？
        新增的节点应选择哪个属性作为其标记？
    默认train_data的最后一列为标签，其它列为特征
    '''
    def __init__(self, train_data, epsilon, alpha, mode='id.3'):
        if type(train_data) != pd.core.frame.DataFrame:
            raise(TypeError, '请使用pandas.DataFrame组织数据!')

        self.feature = train_data[train_data.columns[:-1]]
        self.label = train_data[train_data.columns[-1]]
        self.alpha = alpha  # 正则化阈值
        self.epsilon = epsilon  # 信息增益(比)最低阈值
        self.tree = nx.DiGraph()
        self.gener_nodeId = self.nodeId_gener(root=0)
        self.root = next(self.gener_nodeId)
        self.no_name_nodes = [0]  # 尚未分类的节点，动态变化
        self.mode = mode
        self.tree.add_node(0)
        self.tree.nodes[0]["X"] = self.feature
        self.tree.nodes[0]["y"] = self.label

    def nodeId_gener(self, root=0, end=1e5):
        nId = root
        while nId < end:
            yield nId
            nId += 1

    def info_gain(self, X, y, col_name):
        '''
        计算信息增益
        '''
        Xi = X[col_name].values
        y = y.values
        prob_y = {c: np.sum(y == c)/y.size for c in np.unique(y)}  # 计算y的概率分布
        H_y = -np.sum([prob_y[p]*np.log2(prob_y[p]) for p in prob_y])  # 计算y的经验熵
        Xi_y_dict = {xi: y[Xi == xi] for xi in np.unique(Xi)}  # 找出对应xi的y
        probs = {xi: {c: np.sum(Xi_y_dict[xi] == c)/Xi_y_dict[xi].size
                        for c in np.unique(Xi_y_dict[xi])}
                    for xi in Xi_y_dict}  # 各xi下y的概率分布
        H_yi = {xi: -np.sum([probs[xi][p]*np.log2(probs[xi][p])
                        for p in probs[xi]])
                    for xi in probs}  # 计算xi对应的y的经验熵
        H_y_x = np.sum([H_yi[xi]*(Xi_y_dict[xi].size / y.size)
                        for xi in H_yi])  # 计算Xi对y的经验条件熵
        return H_y - H_y_x

    def info_grain_ratio(self, X, y, col_name):
        '''
        计算信息增益比
        '''
        Xi = X[col_name].values
        y = y[y.columns[0]].values
        ig = self.info_gain(X, y, col_name)
        prob_xi_y = {xi: len(y[Xi == xi])/len(y) for xi in np.unique(Xi)}  # 找出对应xi的y
        H_y = -np.sum([prob_xi_y[xi] * np.log2(prob_xi_y[xi])
                       for xi in prob_xi_y])
        return ig / H_y

    def generate_successors(self, node):
        '''
        从属性列表中选出目标属性，并和当前备用节点node建立联系
        node的属性:
            name -- 续分属性attribute或者leaf_node
            category -- 对应样本的类别
            X -- 特征向量
            y -- 类别
        '''
        X = self.tree.nodes[node]["X"]
        y = self.tree.nodes[node]["y"]
        new_add_nodes = []
        frequency_y = [(v, np.sum(y[y.values == v]))
                       for v in np.unique(y.values)]
        max_y = max(frequency_y, key=lambda x: x[1])[0]
        # case 1: 如果无续分属性或样本同属一类, 当前节点置为叶结点
        if X.size == 0 or len(frequency_y) == 1:
            self.tree.nodes[node]["name"] = 'leaf_node'
            self.tree.nodes[node]["category"] = max_y
        else:  # 存在续分属性
            attr_set = set(X.columns.values)  # 节点对应的属性
            if self.mode == "id.3":
                info_gain_list = [(attr, self.info_gain(X, y, attr))
                                  for attr in attr_set]  # 计算各属性的信息增益
            elif self.mode == "c4.5":
                info_gain_list = [(attr, self.info_gain_ratio(X, y, attr))
                                  for attr in attr_set]  # 计算各属性的信息增益比
            else:
                raise(ValueError, "请选择正确的分类准则('id.3'或'c4.5')")
            # 选择信息增益(比)最大的属性
            # print(info_gain_list)
            target_attr, target_info_gain = max(info_gain_list, key=lambda x: x[1])
            # case 2: 如果信息增益小于阈值或者target_attr所有值相同，则不继续生成结点，当前节点重置为叶结点
            if len(np.unique(X[target_attr].values)) == 1 or target_info_gain < self.epsilon:
                self.tree.nodes[node]["name"] = 'leaf_node'
                self.tree.nodes[node]["category"] = max_y
            else:  # case 3: 继续往下增加结点, 当前节点名称置为续分属性名
                self.tree.nodes[node]["name"] = target_attr  # 续分属性名
                Xi = X[target_attr]
                retain_cols = deepcopy(X.columns.values.tolist())
                retain_cols.remove(target_attr)  # 子结点数据不包含上一级分裂属性
                for xi in np.unique(Xi.values):
                    self.node_id = next(self.gener_nodeId)
                    self.tree.add_edge(node, self.node_id, value=xi)  # 添加连边
                    # 获取目标值，且删除该属性
                    self.tree.nodes[self.node_id]["X"] = X[Xi == xi][retain_cols]
                    # 根据X的索引获取对应的y
                    self.tree.nodes[self.node_id]["y"] = y[Xi == xi]
                    new_add_nodes.append(self.node_id)

        return new_add_nodes

    def train_tree(self):
        '''
        生成决策树
        '''
        while self.no_name_nodes:
            new_nodes = []
            for node in self.no_name_nodes:
                new_add_nodes = self.generate_successors(node)
                new_nodes.extend(new_add_nodes)

            self.no_name_nodes = new_nodes

    def loss(self, tree):
        '''
        损失函数
        '''
        CT_list = []
        for i in tree:
            if tree.nodes[i]['name'] == 'leaf_node':  # 叶结点
                y = tree.nodes[i]['y'].values[:, 0]
                frequency = np.array([np.sum(y == c) for c in np.unique(y)])
                CT_list.append(np.dot(frequency, np.log2(frequency/y.size)))

        return -np.sum(CT_list) + self.alpha*len(CT_list)

    def select_nodes(self, node, tree):
        """
        判断目标叶结点及其兄弟结点是否都为叶结点, 如果是，则返回(父结点, 子节点们)
        """
        prede_node = list(tree.predecessors(node))[0]
        succe_nodes = list(tree.successors(node))
        labels = np.array([tree.nodes[i]['name'] == 'leaf_node' for i in succe_nodes])
        if labels.all():  # 兄弟结点需为叶结点
            return []
        else:
            return [prede_node, succe_nodes]

    def evaluate_pruning(self, parent_node, leaf_nodes):
        """
        判断是否进行一次剪枝操作
        C(T_A) - C(T_B) <= 0, 则剪枝
        """
        parent_y = self.tree.nodes[parent_node]["y"].values[:, 0]
        p_freq = np.array([np.sum(parent_y == c) for c in np.unique(parent_y)])
        p_entropy = np.dot(p_freq, np.log2(p_freq/parent_y.size))  # 父节点的熵
        l_entropy = 0  # 叶节点的熵
        for i in leaf_nodes:
            i_y = self.tree.nodes[i]["y"].values[:, 0]
            i_freq = np.array([np.sum(i_y == c) for c in np.unique(i_y)])
            l_entropy += np.dot(i_freq, np.log2(i_freq/i_y.size))

        return p_entropy - l_entropy + self.alpha * (len(leaf_nodes) - 1)

    def prune_tree(self):
        """
        利用正则化剪枝: 取所有兄弟节点都为叶节点的子树考虑
        """
        tree = deepcopy(self.tree)
        c_subtrees = []  # 待评估子树
        candidate_nodes = []  # 评估候选结点
        for node in tree:
            if node not in candidate_nodes and tree.nodes[node]['name'] == 'leaf_node':
                c_subtree = self.select_nodes(node, tree)
                if c_subtree:  # 如果不为空
                    parent_node, succe_nodes = c_subtree
                    c_subtrees.append([parent_node, succe_nodes])
                    candidate_nodes.extend([parent_node, *succe_nodes])

        while c_subtrees:  # 无待评估子树时停止迭代
            parent_node, leaf_nodes = c_subtrees.pop(0)  # 取出并删除第1个元素
            if self.evaluate_pruning(parent_node, leaf_nodes) <= 0:  # 判断是否剪枝
                tree.remove_nodes_from(leaf_nodes)  # 删除叶结点
                tree.nodes[parent_node]["name"] = 'leaf_node'  # 重置其父结点为叶结点
                parent_y = self.tree.nodes[parent_node]["y"].values[:, 0]
                max_y = max([(c, np.sum(parent_y == c))
                             for c in np.unique(parent_y)], lambda x: x[1])[0]
                tree.nodes[parent_node]["category"] = max_y  # 新叶结点对应的类别
                # 评估回缩的节点及其兄弟结点是否满足候选子树条件
                new_sub_tree = self.select_nodes(parent_node, tree)
                if new_sub_tree:
                    c_subtrees.append(new_sub_tree)

        return tree

    def predict(self, x_dict, tree=None):
        """
        x_dict: 字典
        """
        if not tree:
            tree = self.tree

        node = 0
        while True:
            xi = tree.nodes[node]["name"]
            value = x_dict[xi]
            for i in tree.successors(node):
                if tree[node][i]['value'] == value:
                    break
            else:  # 如果没有相等的值
                print(f'目标属性值不存在{tree.nodes[node]["name"]} = {value}')
                break

            if tree.nodes[i]['name'] == 'leaf_node':
                return tree.nodes[i]['category']
            else:
                node = i


if __name__ == '__main__':
    '''
    [x]
    色泽：乌黑-0, 青绿-1, 浅白-2
    根蒂：蜷缩-0, 稍蜷-1, 硬挺-2
    敲声：浊响-0, 沉闷-1, 清脆-2
    纹理：清晰-0, 稍糊-1, 模糊-2
    脐部：凹陷-0, 稍凹-1, 平坦-2
    触感：硬滑-0, 软粘-1
    密度：<数值>
    含糖率：<数值>
    [y]
    好瓜：是-0, 否-1
    '''
    X = np.array([[1, 0, 0, 0, 0, 0, .697, .46],
                [0, 0, 1, 0, 0, 0, .774, .376],
                [0, 0, 0, 0, 0, 0, .634, .264],
                [1, 0, 1, 0, 0, 0, .608, .318],
                [2, 0, 0, 0, 0, 0, .556, .215],
                [1, 1, 0, 0, 1, 1, .403, .237],
                [0, 1, 0, 1, 1, 1, .481, .149],
                [0, 1, 0, 0, 1, 0, .437, .211],
                [0, 1, 1, 1, 1, 0, .666, .091],
                [1, 2, 2, 0, 2, 1, .243, .267],
                [2, 2, 2, 2, 2, 0, .245, .057],
                [2, 0, 0, 2, 2, 1, .343, .099],
                [1, 1, 0, 1, 0, 0, .639, .161],
                [2, 1, 1, 1, 0, 0, .657, .198],
                [0, 1, 0, 0, 1, 1, .36, .37],
                [2, 0, 0, 2, 2, 0, .593, .042],
                [1, 0, 1, 1, 1, 0, .719, .103]])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0])
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)  # 合并数据
    df = pd.DataFrame(data,
                      columns=['color', 'root', 'sound', 'texture',
                                'navel', 'tactility', 'density', 'sugar', 'label'])
    df = df[['color', 'root', 'sound', 'texture',
             'navel', 'tactility', 'label']]
    tree_classifier = DecisionTreeClassifier(df, alpha=0.05, epsilon=0.05)
    tree_classifier.train_tree()
    x = {'color': 0, 'root': 1, 'sound': 1,
         'texture': 0, 'nave': 0, 'tactility': 1}
    tree_classifier.predict(x, tree_classifier.tree)
