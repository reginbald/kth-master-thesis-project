import unittest
from Edge import Edge
from Graph import Graph


class OverlapTest(unittest.TestCase):
    def test_overlap_01(self):
        graph = Graph(1)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))

        graph.partition_with_overlap(base_partition_weight=1.0, forward_overlap=1, backward_overlap=1)

        self.assertEqual(len(graph.partitions), 3, "There should be 3 base partitions")

        self.assertEqual(
            2, len({'s2', 's3'}.intersection(graph.partition_graph[0].backward_nodes)),
            "Backward nodes {'s2', 's3'} are missing from overlapping partition 0"
        )

        self.assertEqual(
            2, len({'s4', 's5'}.intersection(graph.partition_graph[0].nodes)),
            "Critical nodes {'s4', 's5'} are missing from overlapping partition 0"
        )

        self.assertEqual(
            0, len(graph.partition_graph[0].forward_nodes),
            "There should be no forward nodes in overlapping partition 0"
        )

        # ---------------------------------------------------------------------------

        self.assertEqual(
            1, len({'s1'}.intersection(graph.partition_graph[1].backward_nodes)),
            "Backward nodes {'s1'} are missing from overlapping partition 1"
        )

        self.assertEqual(
            2, len({'s2', 's3'}.intersection(graph.partition_graph[1].nodes)),
            "Critical nodes {'s2', 's3'} are missing from overlapping partition 1"
        )

        self.assertEqual(
            2, len({'s4', 's5'}.intersection(graph.partition_graph[1].forward_nodes)),
            "Forward nodes {'s4', 's5'}  are missing from overlapping partition 1"
        )

        # ---------------------------------------------------------------------------

        self.assertEqual(
            0, len(graph.partition_graph[2].backward_nodes),
            "There should be no backward nodes in overlapping partition 2"
        )

        self.assertEqual(
            1, len({'s1'}.intersection(graph.partition_graph[2].nodes)),
            "Critical nodes {'s1'} are missing from overlapping partition 2"
        )

        self.assertEqual(
            2, len({'s2', 's3'}.intersection(graph.partition_graph[2].forward_nodes)),
            "Forward nodes {'s2', 's3'}  are missing from overlapping partition 2"
        )


if __name__ == '__main__':
    unittest.main()
