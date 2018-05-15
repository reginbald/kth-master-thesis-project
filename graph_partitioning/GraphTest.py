import unittest
from Edge import Edge
from Graph import Graph


class GraphTest(unittest.TestCase):
    def test_scenario_01(self):
        graph = Graph(1)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s6", "s3", 2))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 2, "There should be 2 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's4', 's6'} in graph.partitions,
            "{'s1', 's2', 's3', 's4', 's6'} missing"
        )
        self.assertEqual(
            True, {'s5'} in graph.partitions,
            "{'s5'} missing"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {0, 3.0}"
        )

    def test_scenario_02(self):
        graph = Graph(2)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s6", "s3", 4))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 3, "There should be 3 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's4'} in graph.partitions,
            "{'s1', 's2', 's3', 's4'} missing"
        )
        self.assertEqual(
            True, {'s5'} in graph.partitions,
            "{'s5'} missing"
        )
        self.assertEqual(
            True, {'s6'} in graph.partitions,
            "{'s6'} missing"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 3)])
        self.assertEqual(
            max_node_weights, {0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 3)])
        self.assertEqual(
            longest_paths, {0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {0, 3.0}"
        )

    def test_scenario_03(self):
        graph = Graph(3)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s3", "s6", 1))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 2, "There should be 2 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's4', 's6'} in graph.partitions,
            "{'s1', 's2', 's3', 's4', 's6'} missing"
        )
        self.assertEqual(
            True, {'s5'} in graph.partitions,
            "{'s5'} missing"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {0, 3.0}"
        )

    def test_scenario_04(self):
        graph = Graph(4)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s6", "s2", 2))
        graph.add_edge(Edge("s3", "s7", 1))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 3, "There should be 3 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's6'} in graph.partitions,
            "{'s1', 's2', 's3', 's6'} missing"
        )
        self.assertEqual(
            True, {'s4', 's5'} in graph.partitions,
            "{'s4', 's5'} missing"
        )
        self.assertEqual(
            True, {'s7'} in graph.partitions,
            "{'s7'} missing"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 3)])
        self.assertEqual(
            max_node_weights, {0, 1.0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {0, 1.0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 3)])
        self.assertEqual(
            longest_paths, {0, 1.0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {0, 1.0, 3.0}"
        )

    def test_scenario_05(self):
        graph = Graph(5)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s6", "s3", 3))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 2, "There should be 2 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's6'} in graph.partitions,
            "{'s1', 's2', 's3', 's6'} missing"
        )
        self.assertEqual(
            True, {'s4', 's5'} in graph.partitions,
            "{'s4', 's5'} missing"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {1.0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {1.0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {1.0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {1.0, 3.0}"
        )

    def test_scenario_06(self):
        graph = Graph(6)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s5", "s6", 1))
        graph.add_edge(Edge("s4", "s7", 1))
        graph.add_edge(Edge("s7", "s2", 1))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 3, "There should be 3 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's4'} in graph.partitions,
            "{'s1', 's2', 's3', 's4'} missing"
        )
        self.assertEqual(
            True, {'s5', 's6'} in graph.partitions,
            "{'s5', 's6'} missing"
        )
        self.assertEqual(
            True, {'s7'} in graph.partitions,
            "{'s7'} missing"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {0, 1.0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {0, 1.0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {0, 1.0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {0, 1.0, 3.0}"
        )

    def test_scenario_07(self):
        graph = Graph(7)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s5", "s6", 1))

        graph.add_edge(Edge("s8", "s7", 1))
        graph.add_edge(Edge("s7", "s2", 1))
        graph.add_edge(Edge("s4", "s7", 1))
        graph.add_edge(Edge("s5", "s8", 1))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 2, "There should be 2 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's4'} in graph.partitions,
            "{'s1', 's2', 's3', 's4'} missing"
        )
        self.assertEqual(
            True, {'s5', 's6', 's7', 's8'} in graph.partitions,
            "{'s5', 's6', 's7', 's8'} missing"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {2.0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {2.0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {2.0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {2.0, 3.0}"
        )

    def test_scenario_08(self):
        graph = Graph(8)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s5", "s6", 1))

        graph.add_edge(Edge("s9", "s8", 1))
        graph.add_edge(Edge("s8", "s7", 1))

        graph.add_edge(Edge("s7", "s2", 1))
        graph.add_edge(Edge("s4", "s7", 1))
        graph.add_edge(Edge("s5", "s8", 1))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 2, "There should be 2 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's4'} in graph.partitions or {'s1', 's2', 's7', 's8', 's9'} in graph.partitions,
            "First partition is wrong"
        )
        self.assertEqual(
            True, {'s5', 's6', 's7', 's8', 's9'} in graph.partitions or {'s3', 's4', 's5', 's6'} in graph.partitions,
            "Second partition is wrong"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {2.0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {2.0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {2.0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {2.0, 3.0}"
        )

    def test_scenario_09(self):
        graph = Graph(9)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s5", "s6", 1))

        graph.add_edge(Edge("s9", "s8", 1))
        graph.add_edge(Edge("s8", "s7", 1))

        graph.add_edge(Edge("s7", "s2", 1))
        graph.add_edge(Edge("s4", "s7", 1))
        graph.add_edge(Edge("s5", "s8", 1))
        graph.add_edge(Edge("s5", "s8", 1))

        graph.add_edge(Edge("s5", "s2", 1))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 2, "There should be 2 partitions")
        self.assertEqual(
            True, {'s1', 's2', 's3', 's4'} in graph.partitions or {'s1', 's2', 's7', 's8', 's9'} in graph.partitions,
            "First partition is wrong"
        )
        self.assertEqual(
            True, {'s5', 's6', 's7', 's8', 's9'} in graph.partitions or {'s3', 's4', 's5', 's6'} in graph.partitions,
            "Second partition is wrong"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {2.0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {2.0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {2.0, 3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {2.0, 3.0}"
        )

    def test_scenario_10(self):
        graph = Graph(10)
        graph.add_edge(Edge("s9", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s5", "s6", 1))
        graph.add_edge(Edge("s6", "s7", 1))

        graph.add_edge(Edge("s8", "s3", 1))
        graph.add_edge(Edge("s1", "s10", 1))
        graph.add_edge(Edge("s10", "s11", 1))
        graph.add_edge(Edge("s10", "s4", 1))
        graph.add_edge(Edge("s11", "s5", 1))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 2, "There should be 2 partitions")
        self.assertEqual(
            True,
            {'s1', 's2', 's3', 's4', 's5', 's8', 's9', 's10', 's11'} in graph.partitions or
            {'s1', 's2', 's3', 's4', 's8', 's9', 's10', 's11'} in graph.partitions,
            "First partition is wrong"
        )
        self.assertEqual(
            True,
            {'s6', 's7'} in graph.partitions or {'s6', 's7', 's5'} in graph.partitions,
            "Second partition is wrong"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {1.0, 3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {1.0, 3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {1.0, 4.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {1.0, 4.0}"
        )

    def test_scenario_11(self):
        graph = Graph(11)
        graph.add_edge(Edge("s1", "s2", 1))
        graph.add_edge(Edge("s2", "s3", 1))
        graph.add_edge(Edge("s3", "s4", 1))
        graph.add_edge(Edge("s4", "s5", 1))
        graph.add_edge(Edge("s5", "s6", 1))
        graph.add_edge(Edge("s6", "s7", 1))

        graph.add_edge(Edge("s8", "s4", 1))
        graph.add_edge(Edge("s9", "s5", 1))

        graph.partition(weight_criteria=3.0, merge_criteria=0.0, backwards=False)

        self.assertEqual(len(graph.partitions), 2, "There should be 2 partitions")
        self.assertEqual(
            True,
            {'s1', 's2', 's3', 's4', 's8'} in graph.partitions,
            "First partition is wrong"
        )
        self.assertEqual(
            True,
            {'s5', 's6', 's7', 's9'} in graph.partitions,
            "Second partition is wrong"
        )

        max_node_weights = set([graph.find_max_node_weight_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            max_node_weights, {3.0}.intersection(max_node_weights),
            "Partitions should have max node weight of {3.0}"
        )

        longest_paths = set([graph.find_longest_path_in_partitions(i) for i in range(0, 2)])
        self.assertEqual(
            longest_paths, {3.0}.intersection(longest_paths),
            "Partitions should have longest paths of size {3.0}"
        )


if __name__ == '__main__':
    unittest.main()
