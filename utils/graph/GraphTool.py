import networkx as nx


class GraphTool:
    @classmethod
    def compute_page_rank(cls, similarity_relation_list):
        G = nx.Graph()  # 无向图

        for similarity_relation in similarity_relation_list:
            G.add_node(similarity_relation["n1"]["gid"], text=similarity_relation["n1"]["text"])
            G.add_node(similarity_relation["n2"]["gid"], text=similarity_relation["n2"]["text"])
            G.add_edge(
                similarity_relation["n1"]["gid"],
                similarity_relation["n2"]["gid"],
                weight=similarity_relation["r"]["score"]
            )

        # 默认阻尼系数d=0.85
        return nx.pagerank(G, weight='weight')
