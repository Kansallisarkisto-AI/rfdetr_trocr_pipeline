from collections import defaultdict, deque

class GraphBasedOrdering:
    """
    Graph-Based Reading Order Algorithm
    
    Determines the reading order of polygons by:
    1. Comparing each pair of polygons to determine precedence
    2. Building a directed graph of these relationships
    3. Finding a topological ordering of the graph
    
    Args:
        text_direction (str): Reading direction, either 'lr' (left-to-right) or 'rl' (right-to-left). Default: 'lr'
    """
    def __init__(self, text_direction='lr'):
        self.text_direction = text_direction
    
    def _get_features(self, line):
        """
        Extract spatial features from a text line's bounding box.
        
        Args:
            line (list or tuple): Bounding box coordinates in format:
                                 [x_min, y_min, x_max, y_max]
                                 where:
                                   - x_min: leftmost x-coordinate
                                   - x_max: rightmost x-coordinate  
                                   - y_min: topmost y-coordinate
                                   - y_max: bottommost y-coordinate
        
        Returns:
            dict: Dictionary containing extracted features
        """
        x_min, y_min, x_max, y_max = line

        return {
            'center': ((x_min + x_max) / 2, (y_min + y_max) / 2),
            'x_min': x_min, 
            'x_max': x_max,
            'y_min': y_min, 
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
    def _should_precede(self, u_feat, v_feat):
        """"
        Determine if line u should come before line v in reading order.
        
        The logic follows natural reading patterns:
        1. If lines are on the same row (vertical overlap) -> use horizontal order
        2. If lines are on different rows -> use vertical order (top to bottom)
        
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
            for j in range(i + 1, n):
                if self._should_precede(features[i], features[j]):
                    graph[i].append(j)
                    in_degree[j] += 1
                else:  
                    graph[j].append(i)
                    in_degree[i] += 1
        
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
