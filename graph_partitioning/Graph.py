from Node import Node
from Edge import Edge
from Partition import Partition


class Graph:
    def __init__(self, identifier):
        if not (isinstance(identifier, int)):
            raise TypeError("Please only call this function with an int as the argument")

        self.identifier = identifier
        self.edges = list()
        self.nodes = set()
        self.partitions = dict()
        self.partition_start_nodes = list()
        self.partition_center_nodes = list()
        self.partition_end_nodes = list()
        self.start_nodes = list()
        self.adjacency_dict = dict()
        self.total_weight = 0.0

        # Partition variables
        self.weight_criteria = -1.0
        self.weight_overlap = -1.0
        self.tmp_partition = set()
        self.tmp_key = 0
        self.reset_partition = set()
        self.visited_nodes = set()

        # Overlapping partition variables
        self.partition_graph = dict()

        # Verification variables
        self.too_long_paths = list()

        # Variables for visualization
        self.partitioning_record = list()

    def __str__(self):
        return "Id: " + str(self.identifier) + \
               ", total weight: " + str(self.total_weight) + \
               ", number of nodes: " + str(len(self.nodes))

    def add_edge(self, edge):
        """Method adds an edge and its nodes to the graph while constructing an adjacency dictionary"""
        if not (isinstance(edge, Edge)):
            raise TypeError("Please only call this function with an Edge object as the argument")

        self.edges.append(edge)
        self.nodes.add(edge.src)
        self.nodes.add(edge.dest)
        self.total_weight += edge.weight

        if edge.src not in self.adjacency_dict:
            self.adjacency_dict[edge.src] = Node(edge.src)
        if edge.dest not in self.adjacency_dict:
            self.adjacency_dict[edge.dest] = Node(edge.dest)

        self.adjacency_dict[edge.src].add_forward(edge.dest, edge.weight)
        self.adjacency_dict[edge.dest].add_backward(edge.src, edge.weight)

    def add_node_properties(self, node, num_sensors):
        """Method adds properties to node in the graph"""
        if not (isinstance(node, str) and isinstance(num_sensors, int)):
            raise TypeError("Please only call this function with a str and a int as the arguments")

        self.adjacency_dict[node].num_sensors = num_sensors

    def reset_partition_id(self, nodes, new_id = -1):
        """Method resets the partition_id in the adjacency dictionary for a set of nodes"""
        if not (isinstance(nodes, set) and isinstance(new_id, int)):
            raise TypeError("Please only call this function with a set and a int as the arguments")

        for node in nodes:
            self.adjacency_dict[node].partition_id = new_id

    def find_start_nodes(self, backwards=False):
        """Method finds nodes on the edge of the graph based"""

        end_nodes = self.nodes.copy()
        start_nodes = self.nodes.copy()
        for edge in self.edges:
            if edge.dest in start_nodes:
                start_nodes.remove(edge.dest)
            if edge.src in end_nodes:
                end_nodes.remove(edge.src)

        if backwards:
            self.start_nodes = list(end_nodes)
        else:
            self.start_nodes = list(start_nodes)

    def partition(self, weight_criteria, merge_criteria=0.0, backwards=True, record_steps=False):
        """
        Method partitions graph into partitions based on weight criteria given in minutes.
        Partitions with lower or equal size to merge_criteria will be merged into larger partition.
        """
        if not (isinstance(weight_criteria, float) and isinstance(merge_criteria, float) and isinstance(backwards, bool)):
            raise TypeError("Please only call this function with a str, a float and a boolean as the arguments")

        self.weight_criteria = weight_criteria

        if self.total_weight <= self.weight_criteria:
            self.partitions = [self.nodes.copy()]
            return

        self.find_start_nodes(backwards)

        self.partitions = dict()
        self.tmp_key = 0

        # While there are nodes to partition from
        while len(self.start_nodes) > 0:
            node = self.start_nodes.pop(0)

            self.tmp_partition = set()

            # start partitioning from start node.
            self.partition_helper(node, 0.0, backwards)

            if len(self.tmp_partition) == 0:
                continue

            # Update partition_id
            for n in self.tmp_partition:
                self.adjacency_dict[n].partition_id = self.tmp_key

            # Add to partitions
            self.partitions[self.tmp_key] = self.tmp_partition.copy()
            self.tmp_key += 1

            # Record of partitioning steps for visualization
            if record_steps:
                self.partitioning_record.append([x for x in self.partitions.values() if x])

        # CLEAN UP PHASE

        if merge_criteria > 0:
            partition_ids = [part_id for part_id in self.partitions]
            for part_id in partition_ids:
                if part_id not in self.partitions:
                    # partition has been merged
                    continue
                max_weight = self.partition_max_node_weight(self.partitions[part_id])
                if max_weight < merge_criteria:
                    neighbour_id = self.find_smallest_neighbour_partition(self.partitions[part_id])
                    if neighbour_id == -1:
                        continue
                    self.merge_partitions(part_id, neighbour_id)
                    # Todo: update node weightscd

        # remove empty partitions
        self.partitions = [x for x in self.partitions.values() if x]

    def partition_helper(self, node, current_weight, backwards=False):
        """Method runs through graph assigning nodes to partition"""
        if not (isinstance(node, str) and isinstance(current_weight, float)):
            raise TypeError("Please only call this function with a str and a float as the arguments")

        if self.adjacency_dict.get(node).partition_id != -1:
            # Stop if node has been assigned to a partition
            return

        if node in self.tmp_partition:
            # Prevent partition cycles
            return

        if current_weight > self.weight_criteria:
            # Weight criteria met
            if node not in self.start_nodes:
                self.start_nodes.append(node)
            return

        self.adjacency_dict[node].max_weight = current_weight
        self.tmp_partition.add(node)
        node_properties = self.adjacency_dict.get(node)

        next_nodes = node_properties.backward_nodes if backwards else node_properties.forward_nodes
        next_weights = node_properties.backward_weights if backwards else node_properties.forward_weights

        for i in range(0, len(next_nodes)):
            n_node = next_nodes[i]
            part_id = self.adjacency_dict[n_node].partition_id
            weight = next_weights[i]
            if self.adjacency_dict[n_node].partition_id == -1:
                # Node has not been assigned to a partition
                self.partition_helper(
                    node=n_node,
                    current_weight=current_weight + weight,
                    backwards=backwards
                )
            else:
                # Merge if intersecting with another partition
                if self.loop_checker(node):
                    # Skip if partition loops
                    return
                elif current_weight + weight <= self.adjacency_dict[n_node].max_weight:
                    # Partitions can be merged if weight does not effect previous partition
                    self.reset_partition_id(self.partitions[part_id])
                    self.tmp_partition = self.partitions[part_id].union(self.tmp_partition)
                    del self.partitions[part_id]

                elif current_weight + weight <= self.weight_criteria:
                    # check if new partition is greater than
                    # Special case where we will need to modify previous partition with new weights
                    self.reset_partition_id(self.partitions[part_id])
                    self.reset_partition = self.partitions[part_id].copy()
                    del self.partitions[part_id]

                    self.partition_reset(n_node, backwards)
                    self.tmp_partition = self.reset_partition.union(self.tmp_partition)
                    self.partition_helper(n_node, current_weight + weight, backwards)
                    self.remove_disconnected_nodes(backwards)

    def loop_checker(self, start_node):
        if not (isinstance(start_node, str)):
            raise TypeError("Please only call this function with a str as the argument")

        self.visited_nodes = {start_node}
        node_properties = self.adjacency_dict.get(start_node)
        for f in range(0, len(node_properties.forward_nodes)):
            f_node = node_properties.forward_nodes[f]
            f_weight = node_properties.forward_weights[f]
            if self.loop_checker_helper(f_node, start_node, f_weight):
                return True
        return False

    def loop_checker_helper(self, node, start_node, weight):
        if not (isinstance(node, str) and isinstance(start_node, str) and isinstance(weight, float)):
            raise TypeError("Please only call this function with two str and a float as the arguments")

        if node == start_node:
            # Path loops
            if weight <= self.weight_criteria:
                # Ok to merge
                return False
            return True

        if node in self.visited_nodes:
            # Path loops
            return False

        self.visited_nodes.add(node)
        node_properties = self.adjacency_dict.get(node)

        for f in range(0, len(node_properties.forward_nodes)):
            f_node = node_properties.forward_nodes[f]
            f_weight = node_properties.forward_weights[f]
            if self.loop_checker_helper(f_node, start_node, f_weight + weight):
                return True

        return False

    def partition_reset(self, node, backwards=False):
        if node in self.reset_partition:
            self.adjacency_dict[node].max_weight = 0
            self.reset_partition.remove(node)
            node_properties = self.adjacency_dict.get(node)
            next_nodes = node_properties.backward_nodes if backwards else node_properties.forward_nodes

            for n_node in next_nodes:
                self.partition_reset(n_node)

    def remove_disconnected_nodes(self, backwards=False):
        start_nodes = self.find_partition_edge_nodes(self.reset_partition, backwards)

        for node in start_nodes:
            connected = self.is_node_connected(node, backwards)
            if not connected:
                self.tmp_partition.remove(node)
                self.start_nodes.append(node)

    def is_node_connected(self, node, backwards=False):
        if node in self.tmp_partition and node not in self.reset_partition:
            return True

        node_props = self.adjacency_dict[node]
        next_nodes = node_props.backward_nodes if backwards else node_props.forward_nodes

        for n_node in next_nodes:
            if self.is_node_connected(n_node):
                return True

        return False

    # Clean up phase functions

    def partition_max_node_weight(self, partition):
        if not (isinstance(partition, set)):
            raise TypeError("Please only call this function with set as the argument")

        max_weight = 0
        for node in partition:
            node_weight = self.adjacency_dict[node].max_weight
            if node_weight > max_weight:
                max_weight = node_weight

        return max_weight

    def find_smallest_neighbour_partition(self, partition):
        if not (isinstance(partition, set)):
            raise TypeError("Please only call this function with set as the argument")

        start_nodes = self.find_partition_edge_nodes(partition, end=False)
        end_nodes = self.find_partition_edge_nodes(partition, end=True)

        partition_id = -1
        partition_min_weight = -1

        for e_node in end_nodes:
            for f_node in self.adjacency_dict[e_node].forward_nodes:
                part_id = self.adjacency_dict[f_node].partition_id
                part_weight = self.partition_max_node_weight(self.partitions[part_id])
                if partition_min_weight == -1 or partition_min_weight > part_weight:
                    partition_id = part_id
                    partition_min_weight = part_weight

        for s_node in start_nodes:
            for b_node in self.adjacency_dict[s_node].backward_nodes:
                part_id = self.adjacency_dict[b_node].partition_id
                part_weight = self.partition_max_node_weight(self.partitions[part_id])
                if partition_min_weight == -1 or partition_min_weight > part_weight:
                    partition_id = part_id
                    partition_min_weight = part_weight

        return partition_id

    def merge_partitions(self, a_id, b_id):
        if not (isinstance(a_id, int) and isinstance(b_id, int)):
            raise TypeError("Please only call this function with ints as the arguments")

        if a_id == b_id:
            # don't merge a partition with itself
            return

        self.partitions[a_id] = self.partitions[a_id].union(self.partitions[b_id])
        self.reset_partition_id(self.partitions[b_id], a_id)
        del self.partitions[b_id]

    def partition_with_overlap(self, base_partition_weight, forward_overlap, backward_overlap):
        """
        Method partitions graph into overlapping partitions based on weight criteria and overlap both given in minutes.
        Method starts by partitioning graph with partition method and then adds overlap.
        """
        if not (isinstance(base_partition_weight, float) and isinstance(forward_overlap, int) and isinstance(backward_overlap, int)):
            raise TypeError("Please only call this function with float, int and int as the arguments")

        self.partition(base_partition_weight)

        # Create partition dictionary
        self.partitions = dict(zip(range(len(self.partitions)), self.partitions))
        for i in self.partitions:
            self.reset_partition_id(self.partitions[i], i)

        # Generate partition graph
        self.partition_graph = dict()

        for index in self.partitions:
            part = Partition(index)
            part.add_nodes(self.partitions[index])
            end_nodes = self.find_partition_edge_nodes(self.partitions[index], end=True)
            start_nodes = self.find_partition_edge_nodes(self.partitions[index], end=False)

            for node in end_nodes:
                for f_node in self.adjacency_dict[node].forward_nodes:
                    part.add_forward_partition(self.adjacency_dict[f_node].partition_id)

            for node in start_nodes:
                for b_node in self.adjacency_dict[node].backward_nodes:
                    part.add_backward_partition(self.adjacency_dict[b_node].partition_id)

            self.partition_graph[index] = part

        # Generate overlapping partitions
        for index in self.partition_graph:
            for next_part in self.partition_graph[index].forward_partitions:
                self.forward_overlap_helper(index, next_part, 0, forward_overlap)

            for next_part in self.partition_graph[index].backward_partitions:
                self.backward_overlap_helper(index, next_part, 0, backward_overlap)

        # Clean up overlap between forward nodes and backward nodes
        for index in self.partition_graph:
            c_nodes = self.partition_graph[index].nodes
            f_nodes = self.partition_graph[index].forward_nodes
            b_nodes = self.partition_graph[index].backward_nodes

            self.partition_graph[index].forward_nodes = f_nodes - b_nodes - c_nodes
            self.partition_graph[index].backward_nodes = b_nodes - c_nodes

    def forward_overlap_helper(self, partition_id, next_id, current_overlap, forward_overlap):
        if current_overlap >= forward_overlap:
            return

        self.partition_graph[partition_id].add_forward_nodes(self.partition_graph[next_id].nodes)

        for next_part in self.partition_graph[next_id].forward_partitions:
            self.forward_overlap_helper(partition_id, next_part, current_overlap + 1, forward_overlap)

    def backward_overlap_helper(self, partition_id, next_id, current_overlap, backward_overlap):
        if current_overlap >= backward_overlap:
            return

        self.partition_graph[partition_id].add_backward_nodes(self.partition_graph[next_id].nodes)

        for next_part in self.partition_graph[next_id].backward_partitions:
            self.backward_overlap_helper(partition_id, next_part, current_overlap + 1, backward_overlap)

    def find_partition_edge_nodes(self, partition, end=False):
        """Method finds start nodes for a given partition"""
        if not (isinstance(partition, set)):
            raise TypeError("Please only call this function with a set as the argument")

        start_nodes = set()
        end_nodes = set()
        for node in partition:
            node_properties = self.adjacency_dict.get(node)

            if not set(node_properties.backward_nodes).intersection(partition):
                start_nodes.add(node)

            if not set(node_properties.forward_nodes).intersection(partition):
                end_nodes.add(node)

        if end:
            return end_nodes
        return start_nodes

    def assign_nodes_to_group(self):
        self.partition_start_nodes = list()
        self.partition_center_nodes = list()
        self.partition_end_nodes = list()

        for partition in self.partitions:
            self.partition_start_nodes += list(self.find_partition_edge_nodes(partition, end=False))
            self.partition_end_nodes += list(self.find_partition_edge_nodes(partition, end=True))

        # Ensure node is only assigned to one group
        self.partition_end_nodes = list(set(self.partition_end_nodes) - set(self.partition_start_nodes))

        self.partition_center_nodes = list(self.nodes - set(self.partition_start_nodes + self.partition_end_nodes))

    # ---------- Verification Methods ---------- #

    def graph_statistics(self, print_nodes=False, overlap=False):
        """Method prints out statistics about the graph"""
        if overlap:
            self.partitions = self.partitions.values()

        print("Number of partitions in graph: " + str(len(self.partitions)))
        print("Number of nodes in graph: " + str(len(self.nodes)))

        count = sum([len(part) for part in self.partitions])
        print("Number of nodes in partitions: " + str(count))
        print("Number of unique nodes in all partitions: " + str(len(set.union(*self.partitions))))
        self.assign_nodes_to_group()
        print("Number of partition start nodes: " + str(len(self.partition_start_nodes)))
        print("Number of partition center nodes: " + str(len(self.partition_center_nodes)))
        print("Number of partition end nodes: " + str(len(self.partition_end_nodes)))
        print(
            "Number of nodes in start, center and end: " +
            str(len(self.partition_start_nodes) + len(self.partition_center_nodes) + len(self.partition_end_nodes))
        )

        print("Partition sizes: " + str([len(part) for part in self.partitions]))

        if print_nodes:
            print("Partitions:" + str(self.partitions))

        if count > len(self.nodes):
            self.print_partition_intersection()

        self.print_total_weight()
        for i in range(0, len(self.partitions)):
            print("Partition: " + str(i))
            print("Max node weight:" + str(self.find_max_node_weight_in_partitions(i)))
            print("Total weight of the longest path: " + str(self.find_longest_path_in_partitions(i)))
        self.partition_load_balance_factor()

    def find_max_node_weight_in_partitions(self, partition_id):
        """Method finds largest weight that a node has in a partition"""
        max_weight = 0
        for node in self.partitions[partition_id]:
            node_weight = self.adjacency_dict[node].max_weight
            if node_weight > max_weight:
                max_weight = node_weight

        return max_weight

    def find_longest_path_in_partitions(self, partition_id):
        """Method finds largest weight that a node has in a partition"""
        max_weight = 0
        start_nodes = self.find_partition_edge_nodes(self.partitions[partition_id], end=False)
        self.visited_nodes = set()
        for s_node in start_nodes:
            weight = self.longest_path_helper(s_node, partition_id)
            if weight > max_weight:
                max_weight = weight

        return max_weight

    def longest_path_helper(self, node, part_id):
        if node not in self.partitions[part_id]:
            return -1
        if node in self.visited_nodes:
            return -1

        max_weight = 0
        self.visited_nodes.add(node)

        for i in range(0, len(self.adjacency_dict[node].forward_nodes)):
            f_node = self.adjacency_dict[node].forward_nodes[i]
            weight = self.adjacency_dict[node].forward_weights[i]
            path_weight = self.longest_path_helper(f_node, part_id)
            if path_weight == -1:
                continue
            if max_weight < weight + path_weight:
                max_weight = weight + path_weight
        self.visited_nodes.remove(node)

        return max_weight

    def print_total_weight(self):
        """Method prints out the total weight in the graph"""
        print("Sum of edge weights: " + str(self.total_weight))

    def print_partition_intersection(self):
        """Method prints intersection of partitions"""
        partition_intersection_indices = []
        partition_intersection_nodes = []
        for i in range(0, len(self.partitions)):
            for j in range(i + 1, len(self.partitions)):
                intersect = self.partitions[i].intersection(self.partitions[j])
                if len(intersect) > 0:
                    partition_intersection_indices.append("Partition indices: " + str(i) + str(j))
                    partition_intersection_nodes.append(intersect)

        if partition_intersection_indices:
            print("PARTITIONS INTERSECT!")
            for i in range(0, len(partition_intersection_indices)):
                print(partition_intersection_indices[i] + " nodes: " + str(partition_intersection_nodes[i]))

    def partition_load_balance_factor(self):
        """Method calculates the load balance factor for graph"""
        num_sensors_in_partition = [0.0 for _ in self.partitions]
        for i in range(0, len(self.partitions)):
            for node in self.partitions[i]:
                num_sensors_in_partition[i] += self.adjacency_dict[node].num_sensors

        max_num_sensors_in_partition = max(num_sensors_in_partition)
        total_num_sensors = sum(num_sensors_in_partition)

        print("Partition load balance factor: " + str(
            max_num_sensors_in_partition / (total_num_sensors / len(num_sensors_in_partition))))
