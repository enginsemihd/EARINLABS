import copy
import time

class CSP: 
    def __init__(self, variables, domains, constraints): 
        self.variables = variables 
        self.domains = domains 
        self.constraints = constraints 
        self.solution = None
        self.viz = [] # for visualization steps

    def print_sudoku(self, puzzle): 
        for i in range(9): 
            if i % 3 == 0 and i != 0: print("- - - - - - - - - - - ") 
            for j in range(9): 
                if j % 3 == 0 and j != 0: print(" | ", end="") 
                print(puzzle[i][j], end=" ") 
            print() 

    def visualize(self):
        print("\n--- Algorithm Visualization ---")
        step_interval = max(1, len(self.viz) // 10) 
        for step, state in enumerate(self.viz):
            if step % step_interval == 0 or step == len(self.viz) - 1:
                print(f"\nStep {step + 1}:")
                self.print_sudoku(state)
                time.sleep(0.1) # sped up slightly for multiple tests

    def solve(self): 
        assignment = {}
        # pre-fill knowns
        for var in self.variables:
            if len(self.domains[var]) == 1:
                assignment[var] = self.domains[var][0]
                
        self.solution = self.backtrack(assignment, self.domains) 
        return self.solution 
    
    def forward_checking(self, var, value, assignment, local_domains):
        # remove value from domains of peers
        for peer in self.constraints[var]:
            if peer not in assignment:
                if value in local_domains[peer]:
                    local_domains[peer].remove(value)
                    if len(local_domains[peer]) == 0:
                        return False # domain wipeout
        return True

    def backtrack(self, assignment, local_domains): 
        if len(assignment) == len(self.variables):
            return assignment
            
        # MRV heuristic
        unassigned = [v for v in self.variables if v not in assignment]
        var = min(unassigned, key=lambda v: len(local_domains[v]))
        
        for value in local_domains[var]:
            consistent = True
            for peer in self.constraints[var]:
                if peer in assignment and assignment[peer] == value:
                    consistent = False
                    break
            
            if consistent:
                assignment[var] = value
                
                # track state
                current_board = [[0 for _ in range(9)] for _ in range(9)]
                for (r, c), val in assignment.items():
                    current_board[r][c] = val
                self.viz.append(current_board)

                new_domains = copy.deepcopy(local_domains)
                
                if self.forward_checking(var, value, assignment, new_domains):
                    result = self.backtrack(assignment, new_domains)
                    if result is not None:
                        return result
                        
                del assignment[var]
                
        return None

def is_valid_input(puzzle):
    if len(puzzle) != 9 or any(len(row) != 9 for row in puzzle): return False
    for r in range(9):
        for c in range(9):
            if type(puzzle[r][c]) != int or not (0 <= puzzle[r][c] <= 9): return False
    return True

def setup_csp(puzzle):
    variables = [(r, c) for r in range(9) for c in range(9)]
    domains = {}
    constraints = {}

    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                domains[(r, c)] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                domains[(r, c)] = [puzzle[r][c]]
            
            # get peers
            peers = set()
            for i in range(9):
                if i != c: peers.add((r, i))
                if i != r: peers.add((i, c))
            
            box_r, box_c = (r // 3) * 3, (c // 3) * 3
            for i in range(3):
                for j in range(3):
                    if (box_r + i, box_c + j) != (r, c):
                        peers.add((box_r + i, box_c + j))
                        
            constraints[(r, c)] = list(peers)
            
    return variables, domains, constraints


# --- TEST CASES ---

# 1. Standard Case
standard_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0], 
    [0, 0, 0, 1, 0, 5, 0, 0, 0], 
    [0, 9, 8, 0, 0, 0, 0, 6, 0], 
    [0, 0, 0, 0, 0, 3, 0, 0, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 6], 
    [0, 0, 0, 0, 0, 0, 2, 8, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 8], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 4, 0, 0] 
]

# 2. Hard Case (sparse clues, tests efficiency of MRV + Forward Checking)
hard_puzzle = [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0]
]

# 3. Corner Case: Unsolvable (two 5s in the first row)
unsolvable_puzzle = [
    [5, 5, 0, 0, 7, 0, 0, 0, 0], 
    [0, 0, 0, 1, 0, 5, 0, 0, 0], 
    [0, 9, 8, 0, 0, 0, 0, 6, 0], 
    [0, 0, 0, 0, 0, 3, 0, 0, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 6], 
    [0, 0, 0, 0, 0, 0, 2, 8, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 8], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 4, 0, 0] 
]

# 4. Corner Case: Invalid Input (contains a string and a number > 9)
invalid_puzzle = [
    [5, "X", 0, 0, 7, 0, 0, 0, 0], 
    [0, 0, 0, 1, 0, 15, 0, 0, 0], 
    [0, 9, 8, 0, 0, 0, 0, 6, 0], 
    [0, 0, 0, 0, 0, 3, 0, 0, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 6], 
    [0, 0, 0, 0, 0, 0, 2, 8, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 8], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 4, 0, 0] 
]

test_suite = {
    "1. Standard Puzzle": standard_puzzle,
    "2. Hard Puzzle (Sparse Clues)": hard_puzzle,
    "3. Corner Case - Unsolvable (Constraint Violation)": unsolvable_puzzle,
    "4. Corner Case - Invalid Input Format": invalid_puzzle
}

# Run all tests
for name, puzzle in test_suite.items():
    print(f"\n{'='*40}")
    print(f"Running Test: {name}")
    print(f"{'='*40}")

    if not is_valid_input(puzzle):
        print(">> ERROR: Invalid puzzle dimensions or values detected.")
        continue

    variables, domains, constraints = setup_csp(puzzle)
    csp = CSP(variables, domains, constraints)
    
    print("Initial Board:")
    csp.print_sudoku(puzzle)

    start = time.time()
    sol = csp.solve()
    end = time.time()

    if sol:
        print(f"\n>> SUCCESS: Solution found in {end - start:.4f} seconds.")
        solution = [[0 for _ in range(9)] for _ in range(9)] 
        for (i, j), val in sol.items(): 
            solution[i][j] = val
        csp.print_sudoku(solution)
        
        # Only visualize the standard puzzle to avoid spamming the console
        if name == "1. Standard Puzzle":
            csp.visualize()
    else:
        print("\n>> FAILED: Solution does not exist (Domain wipeout detected).")

input("\nPress Enter to exit...")