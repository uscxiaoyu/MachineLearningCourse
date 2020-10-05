class Tree:
    def __init__(self):
        self.nodes_dict = {}
        self.edges_dict = {}

    @property
    def nodes(self):
        """
        得到树的结点列表
        """
        self._nodes = sorted(list(self.nodes_dict))
        return self._nodes

    @property
    def edges(self):
        """
        得到树的边列表
        """
        self._edges = sorted(list(self.edges_dict))
        return self._edges

    def add_node(self, node_id):
        """
        添加1个结点
        """
        self.nodes_dict[node_id] = {}

    def add_edge(self, u, v, attr_dict=None):
        """
        添加1条边，及其对应的划分属性，操作符，以及对应的值。
        """
        if (u, v) not in self.edges_dict:
            if u not in self.nodes_dict:
                self.add_node(u)
            if v not in self.nodes_dict:
                self.add_node(v)
            if not attr_dict:
                self.edges_dict[(u, v)] = attr_dict
            else:
                self.edges_dict[(u, v)] = {}
        else:
            print(f"{(u, v)} has already existed!")

    def remove_node(self, node_id):
        """
        删除1个结点
        """
        try:
            self.nodes_dict.pop(node_id)
            for edge in self.edges_dict:
                if node_id in edge:
                    self.edges_dict.pop(edge)
        except Exception as e:
            print(e)

    def remove_edge(self, u, v):
        """
        删除一条边
        """
        edge = (u, v)
        try:
            self.edges_dict.pop(edge)
        except Exception as e:
            print(e)

    def get_successors(self, node_id):
        """
        得到结点的后继结点列表
        """
        return [e[1] for e in self.edges_dict if e[0] == node_id]

    def get_predecessors(self, node_id):
        """
        得到结点的前继结点列表
        """
        return [e[0] for e in self.edges_dict if e[1] == node_id]

    def get_subtree(self, node_id):
        """
        得到某个结点对应的子树对应的结点列表
        """
        node_list = []
        new_nodes = [node_id]
        while new_nodes:
            node = new_nodes.pop(0)
            successors = self.get_successors(node)
            node_list.append(node)
            new_nodes.extend(successors)
        return node_list

    def has_node(self, node_id):
        """
        检查是否有某个结点
        """
        if node_id in self.nodes_dict:
            return True
        else:
            return False

    def has_edge(self, u, v):
        """
        检查是否有某条边
        """
        if (u, v) in self.edges_dict:
            return True
        else:
            return False


if __name__ == "__main__":
    tree = Tree()
    # 添加3条有向边
    tree.add_edge(1, 2)  # 添加 1->2
    tree.add_edge(2, 3)
    tree.add_edge(2, 4)

    print(tree.nodes)  # 结点集合
    print(tree.edges)  # 有向边集合

    # 由2为根结点的子树
    print(tree.get_subtree(2))

    # 给结点添加属性
    tree.nodes_dict[4]["name"] = "leaf_node"
    tree.nodes_dict[4]["category"] = 1
    print(tree.nodes_dict[4])
