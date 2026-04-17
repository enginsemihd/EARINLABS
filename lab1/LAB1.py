from collections import deque
 
#BFS (Breadth-First Search)
def bfs(maze, start, finish):
    """
    Breadth-first search
    Parameters:
    - maze: The 2D matrix that represents the maze with 0 represents empty space and 1 represents a wall
    - start: A tuple with the coordinates of starting position
    - finish: A tuple with the coordinates of finishing position
    Returns:
    - Number of steps from start to finish, equals -1 if the path is not found
    - viz: everything required for step-by-step visualization
    """
    # starts the queue with the starting position and its path
    queue = deque([(start, [start])])
    # keeps track of visited cells so we don't visit them twice
    visited_set = set([start])
    # keeps ordered list of visited cells for visualization
    visited_order = [start]
 
    while queue:
        # takes the next cell from the front of the queue (FIFO)
        (r, c), path = queue.popleft()
 
        #check if we reached the end
        if (r, c) == finish:
            viz = {'maze': maze, 'visited': visited_order,
                   'path': path, 'start': start, 'finish': finish}
            return len(path) - 1, viz
 
        #explores neighbours: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < len(maze)
                and 0 <= nc < len(maze[0])  # still inside the maze
                and (nr, nc) not in visited_set  # not visited yet
                and maze[nr][nc] != 1  # not a wall
            ):
                visited_set.add((nr, nc))
                visited_order.append((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
 
    #queue is empty and goal was never reached so no path exists
    viz = {'maze': maze, 'visited': visited_order,
           'path': None, 'start': start, 'finish': finish}
    return -1, viz
 
 
# DFS (Depth-First Search)
def dfs(maze, start, finish):
    """
    Depth-first search
    Parameters:
    - maze: The 2D matrix that represents the maze with 0 represents empty space and 1 represents a wall
    - start: A tuple with the coordinates of starting position
    - finish: A tuple with the coordinates of finishing position
    Returns:
    - Number of steps from start to finish, equals -1 if the path is not found
    - viz: everything required for step-by-step visualization
    """
    # stack stores the current position and the path taken so far
    dfs_stack = [(start, [start])]
    # keeps track of visited cells so we don't visit them twice
    visited_set = set([start])
    # keeps ordered list of visited cells for visualization
    visited_order = [start]
 
    while dfs_stack:
        # take the next cell from the top of the stack (LIFO)
        (r, c), path = dfs_stack.pop()
 
        #check if we reached the end
        if (r, c) == finish:
            viz = {'maze': maze, 'visited': visited_order,
                   'path': path, 'start': start, 'finish': finish}
            return len(path) - 1, viz
 
        #explore neighbours: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < len(maze)
                and 0 <= nc < len(maze[0])  # still inside the maze
                and (nr, nc) not in visited_set  # not visited yet
                and maze[nr][nc] != 1  # not a wall
            ):
                visited_set.add((nr, nc))
                visited_order.append((nr, nc))
                dfs_stack.append(((nr, nc), path + [(nr, nc)]))
 
    # stack is empty and goal was never reached
    viz = {'maze': maze, 'visited': visited_order,
           'path': None, 'start': start, 'finish': finish}
    return -1, viz
 
 
def visualize(viz):
    """
    Visualization function. Shows step by step the work of the search algorithm
    Parameters:
    - viz: dictionary containing maze, visited cells, path, start and finish positions
 
    Symbols used in output:
      # = wall (1)
      S = start
      E = finish
      * = cell on the final path
      . = explored cell (visited but not on final path)
        = unvisited empty cell (0)
    """
    maze   = viz['maze']
    start  = viz['start']
    finish = viz['finish']
    path_set    = set(viz['path']) if viz['path'] else set()
    visited_set = set(viz['visited'])
 
    # goes through every row and cell
    for r, row in enumerate(maze):
        line = ""
        for c, cell in enumerate(row):
            if (r, c) == start:
                line += 'S'
            elif (r, c) == finish:
                line += 'E'
            elif cell == 1:
                # wall
                line += '#'
            elif (r, c) in path_set:
                # marks final path cells with *
                line += '*'
            elif (r, c) in visited_set:
                # marks explored cells with .
                line += '.'
            else:
                # unvisited empty cell
                line += ' '
        print(f"  {line}")
    print()
 
# test case and print results
def run_test(maze, start, finish, test_name):
    """Run BFS and DFS on the given maze and print a comparison."""
    print(f"\n{'='*50}")
    print(f"test case: {test_name}")
    print()
 
    # BFS
    num_steps_bfs, viz_bfs = bfs(maze, start, finish)
    print(f"BFS — steps: {num_steps_bfs}, cells explored: {len(viz_bfs['visited'])}")
    visualize(viz_bfs)
    if num_steps_bfs != -1:
        print(f"Path from {start} to {finish} using BFS is {num_steps_bfs} steps.")
        print(f"\n{'-'*40}")
    else:
        print(f"No path from {start} to {finish} exists.")
        print(f"\n{'-'*40}")
 
    # DFS
    num_steps_dfs, viz_dfs = dfs(maze, start, finish)
    print(f"DFS — steps: {num_steps_dfs}, cells explored: {len(viz_dfs['visited'])}")
    visualize(viz_dfs)
    if num_steps_dfs != -1:
        print(f"Path from {start} to {finish} using DFS is {num_steps_dfs} steps.")
    else:
        print(f"No path from {start} to {finish} exists.")
 
# Maze format: 0 = empty space, 1 = wall
 
# Test 1: Simple small maze
MAZE_SIMPLE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
 
# Test 2: Straight corridor, corner case with only one possible route
MAZE_STRAIGHT = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]
 
# Test 3: Large open maze
MAZE_LARGE_OPEN = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
 
# Test 4: Corner case, no path exists (wall completely blocks the exit)
MAZE_NO_PATH = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]
 
# main
if __name__ == "__main__":
    run_test(MAZE_SIMPLE,     start=(1, 1), finish=(4, 5), test_name="simple maze")
    run_test(MAZE_STRAIGHT,   start=(1, 1), finish=(1, 5), test_name="straight corridor (corner case: only one route)")
    run_test(MAZE_LARGE_OPEN, start=(1, 1), finish=(5, 10), test_name="large open maze (BFS finds shortest path)")
    run_test(MAZE_NO_PATH,    start=(1, 1), finish=(3, 5), test_name="no path exists (corner case)")
