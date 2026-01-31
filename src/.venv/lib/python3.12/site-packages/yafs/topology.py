# -*- coding: utf-8 -*-
import logging
import warnings

import networkx as nx


class Topology:
    """
    Wrapper around a NetworkX graph used as the simulation topology.

    This class unifies the functions to deal with **Complex Networks** as
    a network topology within the simulator. In addition, it facilitates
    creation and assignment of attributes.
    """

    LINK_BW = "BW"
    "Link feature: Bandwidth"

    LINK_PR = "PR"
    "Link feature: Propagation delay"

    NODE_IPT = "IPT"
    "Node feature: IPT. Instructions per simulation time unit."

    def __init__(self, logger=None):
        # G is a NetworkX graph.
        self.G = None
        self.nodeAttributes = {}
        self.logger = logger or logging.getLogger(__name__)

    def __init_uptimes(self):
        for key in self.nodeAttributes:
            self.nodeAttributes[key]["uptime"] = (0, None)

    def get_edges(self):
        """
        Return a view of the graph edges, e.g. ``((1, 0), (0, 2), ...)``.
        """
        return self.G.edges

    def get_edge(self, key):
        """
        Return the attributes of a given edge.

        Parameters
        ----------
        key : tuple
            Edge identifier, e.g. ``(1, 9)``.
        """
        return self.G.edges[key]

    def get_nodes(self):
        """
        Return a view of all nodes in the graph.
        """
        return self.G.nodes

    def get_node(self, key):
        """
        Return the attributes of a given node.

        Parameters
        ----------
        key : hashable
            Node identifier.
        """
        return self.G.nodes[key]


    def get_info(self):
        return self.nodeAttributes

    def create_topology_from_graph(self, G):
        """
        Generate the topology from an existing NetworkX graph.

        Parameters
        ----------
        G : :class:`networkx.Graph`
            Graph instance to use as the topology.
        """
        if isinstance(G, nx.Graph):
            self.G = G
        else:
            raise TypeError("G must be a networkx.Graph instance")

    def create_random_topology(self, nxGraphGenerator, params):
        """
        Generate the topology from a NetworkX graph generator.

        Parameters
        ----------
        nxGraphGenerator : callable
            Graph generator function (e.g. from :mod:`networkx.generators`).
        params : list
            Positional parameters to pass to ``nxGraphGenerator``.
        """
        try:
            self.G = nxGraphGenerator(*params)
        except Exception as exc:
            raise Exception("Error generating topology from graph generator") from exc

    def load(self, data):
        warnings.warn(
            "The load function will be merged with load_all_node_attr in a future version",
                      FutureWarning,
                      stacklevel=8
                      )
        """
        Generate the topology from a JSON structure.

        See the ``Tutorial_JSONModelling`` example for the expected
        format.
        """
        self.G = nx.Graph()
        for edge in data["link"]:
            self.G.add_edge(edge["s"], edge["d"], BW=edge[self.LINK_BW],PR=edge[self.LINK_PR])


        # TODO This part can be removed in next versions.
        for node in data["entity"]:
            self.nodeAttributes[node["id"]] = node
        #end remove

        # Correct way to use custom and mandatory topology attributes.

        valuesIPT = {}
        # valuesRAM = {}
        for node in data["entity"]:
            try:
                valuesIPT[node["id"]] = node["IPT"]
            except KeyError:
                valuesIPT[node["id"]] = 0
            # try:
            #     valuesRAM[node["id"]] = node["RAM"]
            # except KeyError:
            #     valuesRAM[node["id"]] = 0


        nx.set_node_attributes(self.G, values=valuesIPT, name="IPT")
        # nx.set_node_attributes(self.G,values=valuesRAM,name="RAM")

        self.__init_uptimes()

    def load_all_node_attr(self, data):
        self.G = nx.Graph()
        for edge in data["link"]:
            self.G.add_edge(edge["s"], edge["d"], BW=edge[self.LINK_BW], PR=edge[self.LINK_PR])

        dc = {str(x): {} for x in data["entity"][0].keys()}
        for ent in data["entity"]:
            for key in ent.keys():
                dc[key][ent["id"]] = ent[key]
        for x in data["entity"][0].keys():
            nx.set_node_attributes(self.G, values=dc[x], name=str(x))

        for node in data["entity"]:
            self.nodeAttributes[node["id"]] = node

        self.__idNode = len(self.G.nodes)
        self.__init_uptimes()




    def load_graphml(self,filename):
        warnings.warn(
            "The load_graphml function is deprecated and will be removed "
            "in version 2.0.0. Use networkx.read_graphml instead.",
            FutureWarning,
            stacklevel=8,
        )

        self.G = nx.read_graphml(filename)
        attEdges = {}
        for k in self.G.edges():
            attEdges[k] = {"BW": 1, "PR": 1}
        nx.set_edge_attributes(self.G, values=attEdges)
        attNodes = {}
        for k in self.G.nodes():
            attNodes[k] = {"IPT": 1}
        nx.set_node_attributes(self.G, values=attNodes)
        for k in self.G.nodes():
            # It has an "id" attribute. TODO: improve this mapping.
            self.nodeAttributes[k] = self.G.nodes[k]


    def get_nodes_att(self):
        """
        Return a dictionary with the features of all nodes.
        """
        return self.nodeAttributes

    def find_IDs(self, value):
        """
        Search for nodes with matching attributes.

        Parameters
        ----------
        value : dict
            Example: ``{\"model\": \"m-\"}``. Only one key is admitted.

        Returns
        -------
        list
            Node IDs with the same attribute value.
        """
        keyS = list(value.keys())[0]

        result = []
        for key in self.nodeAttributes.keys():
            val = self.nodeAttributes[key]
            if keyS in val:
                if value[keyS] == val[keyS]:
                    result.append(key)
        return result


    def size(self):
        """
        Return the number of nodes in the topology.
        """
        return len(self.G.nodes)

    def add_node(self, nodes, edges=None):
        """
        Add a new node connected to the given list of nodes.

        Parameters
        ----------
        nodes : list
            List of existing node identifiers to which the new node will
            be connected.
        edges : list, optional
            Unused parameter kept for backwards compatibility.
        """
        self.__idNode = + 1
        self.G.add_node(self.__idNode)
        self.G.add_edges_from(zip(nodes, [self.__idNode] * len(nodes)))

        return self.__idNode

    def remove_node(self, id_node):
        """
        Remove a node from the topology.

        Parameters
        ----------
        id_node : int
            Node identifier.
        """
        self.G.remove_node(id_node)
        return self.size()


