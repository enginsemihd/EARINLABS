from collections import deque

def find_position(maze, char):
    #goes through every cell in the maze to find where S or E is
    for r, row in enumerate(maze):
        for c, cell in enumerate(row):
            #Found it!Returs the position as (row, col)
            if cell == char:
                return (r, c)
    #character not found            
    return None

def print_maze(maze, visited, path=None, title=""):
    
#prints the maze with visited cells and the final path highlighted

    """
    Symbols used in the output:
      '#' = wall
      'S' = start
      'E' = end
      '.' = visited cell (explored during search)
      '*' = cell on the final path
      ' ' = unvisited empty cell
    """
    if title:
        #makes pretty frame
        print(f"\n{'='*40}")
        print(f"  {title}")
        print(f"{'='*40}")

    path_set = set(path) if path else set()

    #goes through every row and cell
    for r, row in enumerate(maze):
        line = ""
        for c, cell in enumerate(row):
            #keep walls, start and end as they are
            if cell in ('S', 'E', '#'):
                line += cell
                #mark final path cells with *
            elif (r, c) in path_set:
                line += '*'
                  #marks explored cells with .
            elif (r, c) in visited:
                line += '.'
            #unvisited empty cell
            else:
                line += ' '
        print(f"  {line}")
    print()

#BFS (Breadth-First Search)

def bfs(maze):
    """
 this function returns path fromstart to end, all cells that were checked while searching and number of steps in the final path (start position does not count)
    """
    
    start = find_position(maze, 'S')
    end   = find_position(maze, 'E')
    
    queue   = deque([(start, [start])])
    #starts the queue with the starting position and its path
    visited = set([start])
    #keeps track of visited cells so we don't visit them twice

    while queue:
        (r, c), path = queue.popleft()

        #check if we reached the end
        if (r, c) == end:
            return path, visited, len(path) - 1

        #explores neighbours: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < len(maze)
                and 0 <= nc < len(maze[0])
                and (nr, nc) not in visited
                and maze[nr][nc] != '#'
            ):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    #no path found
    return None, visited, -1

# DFS (Depth-First Search)

def dfs(maze):
    """
    it returns the path from S to E as a list of positions, all cells explored during the search and a step count
    """
    start = find_position(maze, 'S')
    end   = find_position(maze, 'E')

    #stack stores the current position and the path taken so far
    dfs_stack = [(start, [start])]
    visited   = set([start])

    while dfs_stack:
        (r, c), path = dfs_stack.pop()

        #check if we reached the end
        if (r, c) == end:
            return path, visited, len(path) - 1

        #explores neighbours: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < len(maze)
                and 0 <= nc < len(maze[0])
                and (nr, nc) not in visited
                and maze[nr][nc] != '#'
            ):
                visited.add((nr, nc))
                dfs_stack.append(((nr, nc), path + [(nr, nc)]))

    #no path found
    return None, visited, -1

#a test run, applies to both algorithms and prints results
def run_test(maze, test_name):
    """Apply BFS and DFS to the given maze and print a comparison."""
    print(f"\n{'#'*50}")
    print(f"  TEST CASE: {test_name}")
    print(f"{'#'*50}")

    #BFS
    bfs_path, bfs_visited, bfs_steps = bfs(maze)
    print_maze(maze, bfs_visited, bfs_path,
               title=f"BFS — steps: {bfs_steps}, cells explored: {len(bfs_visited)}")
    if bfs_path:
        print(f"  BFS path length : {bfs_steps} steps")
        print(f"  BFS cells visited: {len(bfs_visited)}")
    else:
        print("  BFS: No path found.")

    #DFS
    dfs_path, dfs_visited, dfs_steps = dfs(maze)
    print_maze(maze, dfs_visited, dfs_path,
               title=f"DFS — steps: {dfs_steps}, cells explored: {len(dfs_visited)}")
    if dfs_path:
        print(f"  DFS path length : {dfs_steps} steps")
        print(f"  DFS cells visited: {len(dfs_visited)}")
    else:
        print("  DFS: No path found.")

#Test 1: simple small maze
MAZE_SIMPLE = [
    "########",
    "#S     #",
    "###### #",
    "#      #",
    "#    E##",
    "########",
]

#Test 2: Straight corridor, corner case with only one possible route
MAZE_STRAIGHT = [
    "#######",
    "#S   E#",
    "#######",
]

#Test 3: Large open maze
MAZE_LARGE_OPEN = [
    "############",
    "#S         #",
    "#          #",
    "#          #",
    "#          #",
    "#         E#",
    "############",
]

#Test 4: Corner case, no path exists
MAZE_NO_PATH = [
    "#######",
    "#S    #",
    "#######",
    "#    E#",
    "#######",
]

#main
if __name__ == "__main__":
    run_test(MAZE_SIMPLE,     "Simple maze")
    run_test(MAZE_STRAIGHT,   "Straight corridor (corner case: only one route)")
    run_test(MAZE_LARGE_OPEN, "Large open maze (BFS finds shortest path)")
    run_test(MAZE_NO_PATH,    "No path exists (corner case)")
