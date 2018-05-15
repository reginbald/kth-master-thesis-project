import glob
import csv
from Edge import Edge
from Graph import Graph


def import_data(path_to_edges, path_to_nodes):
    """Method extracts information about edges and nodes from csv files"""
    edges = []
    node_sensors = []

    for files in glob.glob(path_to_edges):
        with open(files) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            next(reader, None)  # skip the headers
            for row in reader:
                if row:
                    edges.append(Edge(row[0], row[1], row[2]))

    for files in glob.glob(path_to_nodes):
        with open(files) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            next(reader, None)  # skip the headers
            for row in reader:
                if row:
                    node_sensors.append((row[0], int(row[-1])))

    return edges, node_sensors


def construct_graph(edges, node_sensors):
    """Method constructs a graph based on edges and sensor information"""
    # Create the graph
    graph = Graph(identifier=0)

    for edge in edges:
        graph.add_edge(edge)

    for n in node_sensors:
        graph.add_node_properties(n[0], n[1])
    return graph


def partition(graph, weight_criteria, merge_criteria, backwards, record_steps):
    """Method partitions the graph based on the input parameters"""
    # Partition graph
    print("#######################################")
    graph.partition(weight_criteria, merge_criteria, backwards, record_steps)
    graph.assign_nodes_to_group()

    # Verify that partitions are correct
    graph.graph_statistics(print_nodes=False, overlap=False)

    # Export partitions to CSV file
    with open(export_path, 'wb') as csv_file:
        wr = csv.writer(csv_file, delimiter=";")
        wr.writerow(['node', 'partition', 'group'])
        for i, nodes in enumerate(graph.partitions):
            for node in nodes:
                if node in graph.partition_start_nodes:
                    wr.writerow([node, i, 'start'])
                elif node in graph.partition_end_nodes:
                    wr.writerow([node, i, 'end'])
                else:
                    wr.writerow([node, i, 'center'])

    if record_steps:
        with open("../data/partitions/partitioning_record.csv", 'wb') as csv_file:
            wr = csv.writer(csv_file, delimiter=";")
            wr.writerow(['iteration', 'node', 'partition'])
            for iteration in range(len(graph.partitioning_record)):
                for i, nodes in enumerate(graph.partitioning_record[iteration]):
                    for node in nodes:
                        wr.writerow([iteration, node, i])


def partition_with_overlap(graph, base_partition_weight, forward_overlap, backward_overlap):
    """Method partitions the graph with overlap based on the input parameters"""
    # Partition graph with overlap
    print("#######################################")
    graph.partition_with_overlap(base_partition_weight, forward_overlap, backward_overlap)

    # Verify that partitions are correct
    graph.graph_statistics(print_nodes=False, overlap=True)

    # Export partitions to CSV file
    with open(overlap_export_path, 'wb') as csv_file:
        wr = csv.writer(csv_file, delimiter=";")
        wr.writerow(['node', 'partition', 'type'])
        for partition_id in graph.partition_graph:
            for row in graph.partition_graph[partition_id].generate_csv_list():
                wr.writerow(row)


if __name__ == '__main__':
    # Regular Partitioning
    prediction_time_criteria = 3.0
    partition_merge_criteria = 0.0
    backwards_partitioning = True
    direction = "backward" if backwards_partitioning else "forward"
    export_path = "../data/partitions/" + direction + "_partitions-" + str(int(prediction_time_criteria)) + "min.csv"
    record_partitioning = True

    # Overlapping Partitioning
    overlap = False
    base_partition_prediction_time = 2.0
    partition_forward_overlap = 3
    partition_backward_overlap = 10

    overlap_export_path = "../data/partitions/overlapping_partitions-base_weight_" + \
                          str(int(base_partition_prediction_time)) + \
                          "_min-forward_" + str(partition_forward_overlap) + \
                          "-backward_" + str(partition_backward_overlap) + ".csv"

    graph_edges, graph_node_sensors = import_data("../data/edges_with_weight/*.csv", "../data/nodes/*.csv")
    traffic_graph = construct_graph(graph_edges, graph_node_sensors)

    if not overlap:
        print("Partitioning")
        partition(
            traffic_graph,
            prediction_time_criteria,
            partition_merge_criteria,
            backwards_partitioning,
            record_partitioning
        )
    else:
        print("Partitioning with overlap")
        partition_with_overlap(
            traffic_graph,
            base_partition_prediction_time,
            partition_forward_overlap,
            partition_backward_overlap
        )
