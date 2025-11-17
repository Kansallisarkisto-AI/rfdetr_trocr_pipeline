from collections import defaultdict, deque

"""
================================================================================
DETAILED ALGORITHM STEPS
================================================================================

STEP 1: FEATURE EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each text line, extract:
  • Bounding box: [x_min, x_max, y_min, y_max] NOW x_min,y_min,x_max,y_max
  • Center point: ((x_min + x_max)/2, (y_min + y_max)/2)
  • Dimensions: width, height
  • (Optional) Additional features: font size, style, color, etc.

Example:
  Line A: [100, 300, 150, 180]
    → Center: (200, 165)
    → Width: 200px, Height: 30px


STEP 2: PAIRWISE COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For every pair of lines (i, j), decide: "Does i come before j?"

The decision logic:

  A. Calculate vertical overlap:
     overlap_y = min(i.y_max, j.y_max) - max(i.y_min, j.y_min)
     avg_height = (i.height + j.height) / 2
     
  B. If overlap_y > threshold * avg_height:
     → Lines are on the SAME ROW
     → Use HORIZONTAL ordering:
       - LTR: i before j if i.x_center < j.x_center
       - RTL: i before j if i.x_center > j.x_center
  
  C. Else (different rows):
     → Use VERTICAL ordering:
       - i before j if i.y_center < j.y_center
       
This creates a DIRECTED EDGE from i → j if i comes before j.


STEP 3: BUILD DIRECTED GRAPH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Graph G = (V, E) where:
  • V = set of all lines
  • E = set of precedence relationships
  
Example with 4 lines in 2x2 grid:
  
  [0]  [1]       Edges:
  [2]  [3]       0→1 (same row, 0 left of 1)
                 0→2 (0 above 2)
                 0→3 (0 left and above 3)
                 1→3 (1 above 3)
                 2→3 (same row, 2 left of 3)

  The graph encodes ALL precedence constraints!

STEP 4: TOPOLOGICAL SORT (Kahn's Algorithm)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Find a linear ordering that respects all edges:

  1. Calculate in-degree for each node:
     in_degree[i] = number of edges pointing TO node i
     
  2. Add all nodes with in_degree = 0 to queue
     (These have nothing before them, can be first)
     
  3. While queue is not empty:
     a. Remove node from queue → add to result
     b. For each neighbor of this node:
        - Decrease neighbor's in-degree by 1
        - If in-degree becomes 0, add to queue
        
  4. Result is the reading order
"""

class GraphBasedOrdering:
    """
    Graph-Based Reading Order Algorithm
    
    This class determines the reading order of text lines by:
    1. Comparing each pair of lines to determine precedence
    2. Building a directed graph of these relationships
    3. Finding a topological ordering of the graph
    
    Args:
        text_direction (str): Reading direction, either 'lr' (left-to-right) or 'rl' (right-to-left)
                             Default: 'lr'
    
    Example:
        >>> orderer = GraphBasedOrdering(text_direction='lr')
        >>> lines = [[50, 250, 100, 130], [300, 500, 100, 130]]  # Two lines side by side
        >>> reading_order = orderer.order(lines)
        >>> print(reading_order)  # [0, 1] - left line first, then right line
    """
    def __init__(self, text_direction='lr'):
        self.text_direction = text_direction
    
    def _get_features(self, line):
        """
        Extract spatial features from a text line's bounding box.
        
        Args:
            line (list or tuple): Bounding box coordinates in format:
                                 [x_min, x_max, y_min, y_max] NOW x_min,y_min,x_max,y_max
                                 where:
                                   - x_min: leftmost x-coordinate
                                   - x_max: rightmost x-coordinate  
                                   - y_min: topmost y-coordinate
                                   - y_max: bottommost y-coordinate
        
        Returns:
            dict: Dictionary containing extracted features
        """
        #x_min, x_max, y_min, y_max = line
        x_min,y_min,x_max,y_max = line
        return {
            'center': ((x_min + x_max) / 2, (y_min + y_max) / 2),
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
    def _should_precede(self, u_feat, v_feat):
        """"
        Determine if line u should come before line v in reading order.
        
        This is the core decision function that compares two lines spatially.
        The logic follows natural reading patterns:
        1. If lines are on the same row (vertical overlap) → use horizontal order
        2. If lines are on different rows → use vertical order (top to bottom)
        
        Args:
            u_feat (dict): Feature dictionary for line u (from _get_features)
                          Must contain: 'center', 'y_min', 'y_max', 'height'
            v_feat (dict): Feature dictionary for line v (from _get_features)
                          Must contain: 'center', 'y_min', 'y_max', 'height'
        
        Returns:
            bool: True if line u should come before line v in reading order
                  False otherwise
        """
        u_center = u_feat['center']
        v_center = v_feat['center']
        
        # Vertical overlap threshold
        v_overlap = min(u_feat['y_max'], v_feat['y_max']) - max(u_feat['y_min'], v_feat['y_min'])
        avg_height = (u_feat['height'] + v_feat['height']) / 2
        
        # If significant vertical overlap, use horizontal order
        if v_overlap > 0.5 * avg_height:
            if self.text_direction == 'lr':
                return u_center[0] < v_center[0]
            else:
                return u_center[0] > v_center[0]
        
        # Otherwise, use vertical order (top to bottom)
        return u_center[1] < v_center[1]
    
    def order(self, lines):
        """
        Compute the reading order of text lines using graph-based approach.
        
        Args:
            lines (list): List of bounding boxes, where each bounding box is:
                         [x_min, x_max, y_min, y_max] NOW x_min,y_min,x_max,y_max
                         
                         Example input:
                         [
                             [50, 250, 100, 130],    # Line 0
                             [300, 500, 100, 130],   # Line 1
                             [50, 250, 180, 210],    # Line 2
                             [300, 500, 180, 210]    # Line 3
                         ]
                         
                         This represents a 2x2 grid:
                         [Line 0]  [Line 1]
                         [Line 2]  [Line 3]
        
        Returns:
            list: Indices of lines in reading order
                  
                  Example output: [0, 1, 2, 3]
                  Meaning: Read line 0, then 1, then 2, then 3
                  
                  If input is empty, returns []
        
        Algorithm Steps:
            1. Handle empty input
            2. Extract features for all lines
            3. Build directed graph:
               - Nodes = line indices (0, 1, 2, ...)
               - Edges = precedence relationships (i→j means "i before j")
            4. Calculate in-degrees (number of predecessors for each node)
            5. Perform topological sort using Kahn's algorithm
            6. Return the sorted order
        """
        if not lines:
            return []
        
        n = len(lines)
        features = [self._get_features(line) for line in lines]
        
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i != j and self._should_precede(features[i], features[j]):
                    graph[i].append(j)
                    in_degree[j] += 1
        
        # Kahn's algorithm for topological sort
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result


import numpy as np

class OrderPolygons:
    def __init__(self, text_direction = 'lr'):
        self.text_direction = text_direction

    # Defines whether two lines overlap vertically
    def _y_overlaps(self, u, v):
        #u_y_min < v_y_max and u_y_max > v_y_min
        return u[3] < v[2] and u[2] > v[3]
    
    # Defines whether two lines overlap horizontally
    def _x_overlaps(self, u, v):
        #u_x_min < v_x_max and u_x_max > v_x_min
        return u[1] < v[0] and u[0] > v[1]
    
    # Defines whether one line (u) is above the other (v)
    def _above(self, u, v):
        #u_y_min < v_y_min
        return u[3] < v[3]

    # Defines whether one line (u) is left of the other (v)
    def _left_of(self, u, v):
        #u_x_max < v_x_min
        return u[0] < v[1]  
    
    # Defines whether one line (w) overlaps with two others (u,v)
    def _separates(self, w, u, v):
        if w == u or w == v:
            return 0
        #w_y_max < (min(u_y_min, v_y_min))
        if w[2] < min(u[3], v[3]):
            return 0
        #w_y_min > max(u_y_max, v_y_max)
        if w[3] > max(u[2], v[2]):
            return 0
        #w_x_min < u_x_max and w_x_max > v_x_min
        if w[1] < u[0] and w[0] > v[1]:
            return 1
        return 0

    # Slightly modified version of the Kraken implementation at
    # https://github.com/mittagessen/kraken/blob/master/kraken/lib/segmentation.py
    def reading_order(self, lines):
        """Given the list of lines, computes
        the partial reading order.  The output is a binary 2D array
        such that order[i,j] is true if line i comes before line j
        in reading order."""
        # Input lines are arrays with 4 polygon coordinates:
        # 0=x_right/x_max, 1=x_left/x_min, 2=y_down/y_max, 3=y_up/y_min
        
        # Array where the order of precedence between the lines is defined
        order = np.zeros((len(lines), len(lines)), 'B')

        # Defines reading direction: default is from left to right
        if self.text_direction == 'rl':
            def horizontal_order(u, v):
                return not self._left_of(u, v)
        else:
            horizontal_order = self._left_of

        for i, u in enumerate(lines):
            for j, v in enumerate(lines):
                if self._x_overlaps(u, v):
                    if self._above(u, v):
                        # line u is placed before line v in reading order
                        order[i, j] = 1
                else:
                        
                    if [w for w in lines if self._separates(w, u, v)] == []:
                        if horizontal_order(u, v):
                            order[i, j] = 1
                    elif self._y_overlaps(u, v) and horizontal_order(u, v):
                        order[i, j] = 1
                    
        return order
    
    # Taken from the Kraken implementation at 
    # https://github.com/mittagessen/kraken/blob/master/kraken/lib/segmentation.py
    def topsort(self, order):
        """Given a binary array defining a partial order (o[i,j]==True means i<j),
        compute a topological sort.  This is a quick and dirty implementation
        that works for up to a few thousand elements."""

        n = len(order)
        visited = np.zeros(n)
        L = []

        def _visit(k):
            if visited[k]:
                return
            visited[k] = 1
            a, = np.nonzero(np.ravel(order[:, k]))
            for line in a:
                _visit(line)
            L.append(k)

        for k in range(n):
            _visit(k)
        return L

    def order(self, lines):
        order = self.reading_order(lines)
        sorted = self.topsort(order)
        return sorted
