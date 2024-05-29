# Poset_E.py  # 2021-20-10
# Base on Shaun Poset.py
# MIT LICENSE 2020 Ewerton R. Vieira

from pychomp.DirectedAcyclicGraph import *
import graphviz


class Poset_E:  # Represent a Poset

    def ListOfNivelSets(self):  # return the list of increasing Nivel sets of a given poset

        P0 = {v for v in self.vertices() if len(self.children(v)) == 0}

        def NivelSet(P_i):  # return the parent set of a given set
            P_iplus = set()
            for a in P_i:
                # print(str(a)+' -> '+ str(CMG.parents(a)))
                P_iplus = P_iplus.union(self.parents(a))
            return P_iplus - P_i

        P = [P0]
        j = 0
        while P[j] != set():
            P.append(NivelSet(P[j]))
            j += 1

        return P

    def __init__(self, graph, save_memory=False):
        """
        Create a Poset P from a DAG G such that x <= y in P iff there is a path from x to y in G
        """
        self.vertices_ = set(graph.vertices())

        if save_memory:  # to save memory it doesnt build descendants
            self.ancestors_ = graph.transpose()
            self.ancestors_ = self.ancestors_.transitive_closure()

        else:
            self.descendants_ = graph.transitive_closure()
            self.ancestors_ = self.descendants_.transpose()
            self.children_ = graph.transitive_reduction()
            self.parents_ = self.children_.transpose()

    def __iter__(self):
        """
        Allows for the semantics
          [v for v in poset]
        """
        return iter(self.vertices())

    def vertices(self):
        """
        Return the set of elements in the poset
        """
        return self.vertices_

    def parents(self, v):
        """
        Return the immediate predecessors of v in the poset
        """
        return self.parents_.adjacencies(v)

    def children(self, v):
        """
        Return the immediate successors of v in the poset
        """
        return self.children_.adjacencies(v)

    def ancestors(self, v):
        """
        Return the set { u : u > v }
        """
        return self.ancestors_.adjacencies(v)

    def upset(self, v):
        """
        Return the set { u : u >= v }
        """
        return self.ancestors(v).union({v})

    def descendants(self, v):
        """
        Return the set { u : u < v }
        """
        return self.descendants_.adjacencies(v)

    def downset(self, v):
        """
        Return the set { u : u <= v }
        """
        return self.descendants(v).union({v})

    def interval(self, p, q):
        """
        Return the minimal interval that has p and q
        """
        if p < q:
            return self.downset(q).intersection(self.upset(p))
        else:
            return self.downset(p).intersection(self.upset(q))

    def isConvexSet(self, I):  # return true if I is a convex subset in poset P
        for a in I:
            for b in I:
                if a < b:
                    if not self.interval(a, b).issubset(set(I)):
                        return False
        return True

    def less(self, u, v):
        """
        Return True if u < v, False otherwise
        """
        return v in self.ancestors(u)

    def maximal(self, subset):
        """
        Return the set of elements in "subset" which are maximal
        """
        return frozenset({v for v in subset if not any(self.less(v, u) for u in subset)})

    def decreasing_list(self):
        """
        Return a list of vertices [a,b, ...] such that a > b or a not comparable with b
        """
        decresing_list_ = []
        V_ = self.vertices_
        while V_ != set():
            Max_ = self.maximal(V_)
            decresing_list_ = decresing_list_ + list(Max_)
            V_ = V_ - Max_

        return decresing_list_

    def build_viz(self, graph, shape='circle'):
        """ Return a graphviz string describing the graph and its labels """
        gv = 'digraph {\n'

        gv += f'node [fontsize=12, shape={shape}]'

        for v in graph.vertices():
            gv += f"{v}[label={v}];\n"

        for v in graph.vertices():
            for u in graph.adjacencies(v):
                gv += f"{v}->{u};\n"
        return gv + '}\n'

    def _repr_svg_(self):
        """
        Return svg representation for visual display
        """
        return graphviz.Source(self.build_viz(self.children_))._repr_svg_()


def InducedPoset_E(G, predicate):
    result = DirectedAcyclicGraph()
    S = set([v for v in G.vertices() if predicate(v)])
    for v in S:
        result.add_vertex(v)
    for v in S:
        for u in G.descendants(v):
            if u in S and u != v:
                result.add_edge(v, u)
    return Poset_E(result)
