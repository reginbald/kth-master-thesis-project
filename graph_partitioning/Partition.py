class Partition:
    def __init__(self, identifier):
        """Partition Constructor."""
        self.identifier = identifier
        self.forward_partitions = list()
        self.backward_partitions = list()
        self.nodes = set()
        self.forward_nodes = set()
        self.backward_nodes = set()

    def __str__(self):
        """Returns information about the partition when it is printed."""
        return "Id: " + str(self.identifier) + \
            " Forward: " + str(self.forward_partitions) + \
            " Backward: " + str(self.backward_partitions)

    def add_forward_partition(self, partition):
        """Adds partition to the forward lists."""
        self.forward_partitions.append(partition)

    def add_backward_partition(self, partition):
        """Adds partition to the backward lists."""
        self.backward_partitions.append(partition)

    def add_nodes(self, nodes):
        """Adds center nodes to the partition."""
        self.nodes = nodes

    def add_forward_nodes(self, nodes):
        """Adds nodes that are in forward partitions."""
        self.forward_nodes = self.forward_nodes.union(nodes)

    def add_backward_nodes(self, nodes):
        """Adds nodes that are in backward partitions."""
        self.backward_nodes = self.backward_nodes.union(nodes)

    def generate_csv_list(self):
        """Generates a list used when exporting partition information into a CSV file."""
        csv = list()

        # node, partition, type
        csv += [[n, self.identifier, 'backward'] for n in self.backward_nodes]
        csv += [[n, self.identifier, 'critical'] for n in self.nodes]
        csv += [[n, self.identifier, 'forward'] for n in self.forward_nodes]

        return csv
