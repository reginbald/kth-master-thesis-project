class Node:
    def __init__(self, id):
        """Node Constructor."""
        self.id = id
        self.forward_nodes = []
        self.forward_weights = []
        self.backward_nodes = []
        self.backward_weights = []
        self.partition_id = -1  # Partition id that this node is apart of
        self.max_weight = 0  # weight of node based on longest path
        self.num_sensors = 0

    def __str__(self):
        """Returns information about the node when it is printed."""
        return "Id: " + str(self.id) + \
            " Forward: " + str(self.forward_nodes) + \
            " Backward: " + str(self.backward_nodes) + \
            " Partition: " + str(self.partition_id)

    def add_forward(self, node, weight):
        """Adds node and weight to the forward lists"""
        self.forward_nodes.append(node)
        self.forward_weights.append(weight)

    def add_backward(self, node, weight):
        """Adds node and weight to the backward lists"""
        self.backward_nodes.append(node)
        self.backward_weights.append(weight)
