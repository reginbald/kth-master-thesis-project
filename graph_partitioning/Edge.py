class Edge:
    def __init__(self, src, dest, weight):
        """Edge Constructor."""
        self.src = src
        self.dest = dest
        self.weight = float(weight)

    def __str__(self):
        """Returns information about the edge when it is printed."""
        return str(self.src) + " --> " + str(self.dest) + " weight: " + str(self.weight)
