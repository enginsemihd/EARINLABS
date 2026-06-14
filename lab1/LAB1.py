from collections import deque

# BFS (Breadth-First Search)
def bfs(maze, start, finish):
    queue = deque([(start, [start])])
    visited_set = set([start])
    visited_order = [start]

    while queue:
        (r, c), path = queue.popleft()

        if (r, c) == finish:
            viz = {'maze': maze, 'visited': visited_order,
                   'path': path, 'start': start, 'finish': finish}
            return len(path) - 1, viz

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < len(maze)
                and 0 <= nc < len(maze[0])
                and (nr, nc) not in visited_set
                and maze[nr][nc] != 1
            ):
                visited_set.add((nr, nc))
                visited_order.append((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    viz = {'maze': maze, 'visited': visited_order,
           'path': None, 'start': start, 'finish': finish}
    return -1, viz


# DFS (Depth-First Search)
def dfs(maze, start, finish):
    dfs_stack = [(start, [start])]
    visited_set = set([start])
    visited_order = [start]

    while dfs_stack:
        (r, c), path = dfs_stack.pop()

        if (r, c) == finish:
            viz = {'maze': maze, 'visited': visited_order,
                   'path': path, 'start': start, 'finish': finish}
            return len(path) - 1, viz

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < len(maze)
                and 0 <= nc < len(maze[0])
                and (nr, nc) not in visited_set
                and maze[nr][nc] != 1
            ):
                visited_set.add((nr, nc))
                visited_order.append((nr, nc))
                dfs_stack.append(((nr, nc), path + [(nr, nc)]))

    viz = {'maze': maze, 'visited': visited_order,
           'path': None, 'start': start, 'finish': finish}
    return -1, viz


def visualize(viz):
    maze   = viz['maze']
    start  = viz['start']
    finish = viz['finish']
    path_set    = set(viz['path']) if viz['path'] else set()
    visited_set = set(viz['visited'])

    for r, row in enumerate(maze):
        line = ""
        for c, cell in enumerate(row):
            if (r, c) == start:
                line += 'S'
            elif (r, c) == finish:
                line += 'E'
            elif cell == 1:
                line += '#'
            elif (r, c) in path_set:
                line += '*'
            elif (r, c) in visited_set:
                line += '.'
            else:
                line += ' '
        print(f"  {line}")
    print()

def run_test(maze, start, finish, test_name):
    print(f"\n{'='*50}")
    print(f"Test case: {test_name}")
    print()

    # BFS
    num_steps_bfs, viz_bfs = bfs(maze, start, finish)
    print(f"BFS — steps: {num_steps_bfs}, cells explored: {len(viz_bfs['visited'])}")
    visualize(viz_bfs)

    # DFS
    num_steps_dfs, viz_dfs = dfs(maze, start, finish)
    print(f"DFS — steps: {num_steps_dfs}, cells explored: {len(viz_dfs['visited'])}")
    visualize(viz_dfs)

# Test Cases
MAZE_SIMPLE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_STRAIGHT = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

MAZE_LARGE_OPEN = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_NO_PATH = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

if __name__ == "__main__":
    run_test(MAZE_SIMPLE,     start=(1, 1), finish=(4, 5), test_name="Simple Maze")
    run_test(MAZE_STRAIGHT,   start=(1, 1), finish=(1, 5), test_name="Straight Corridor")
    run_test(MAZE_LARGE_OPEN, start=(1, 1), finish=(5, 10), test_name="Large Open Maze")
    run_test(MAZE_NO_PATH,    start=(1, 1), finish=(3, 5), test_name="No Path Exists")