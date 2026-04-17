import copy
import time

class CSP: 
    def __init__(self, variables, domains, constraints): 
        """
        Initialization of the CSP class
        """
        self.variables = variables 
        self.domains = domains 
        self.constraints = constraints 
        self.solution = None
        self.viz = [] # List to store board states for visualization

    def print_sudoku(self, puzzle): 
        for i in range(9): 
            if i % 3 == 0 and i != 0: 
                print("- - - - - - - - - - - ") 
            for j in range(9): 
                if j % 3 == 0 and j != 0: 
                    print(" | ", end="") 
                print(puzzle[i][j], end=" ") 
            print() 

    def visualize(self):
        print("\n--- Algorithm Visualization ---")
        # Visualizing every Nth step to avoid flooding the console
        step_interval = max(1, len(self.viz) // 10) 
        for step, state in enumerate(self.viz):
            if step % step_interval == 0 or step == len(self.viz) - 1:
                print(f"\nStep {step + 1}:")
                self.print_sudoku(state)
                time.sleep(0.2)

    def solve(self): 
        assignment = {}
        # Pre-assign variables that already have a known value from the puzzle
        for var in self.variables:
            if len(self.domains[var]) == 1:
                assignment[var] = self.domains[var][0]
                
        self.solution = self.backtrack(assignment, self.domains) 
        return self.solution 
    
    def forward_checking(self, var, value, assignment, local_domains):
        """
        Function that removes the value from the domains of free variables that are in the constraints of the var.
        Returns False if any domain becomes empty (domain wipeout).
        """
        for peer in self.constraints[var]:
            if peer not in assignment:
                if value in local_domains[peer]:
                    local_domains[peer].remove(value)
                    # If a peer has no legal moves left, forward checking fails
                    if len(local_domains[peer]) == 0:
                        return False 
        return True

    def backtrack(self, assignment, local_domains): 
        """
        Backtracking algorithm with forward checking
        """
        # Base case: If assignment is complete, return it
        if len(assignment) == len(self.variables):
            return assignment
            
        # Select an unassigned variable (using MRV - Minimum Remaining Values heuristic for efficiency)
        unassigned = [v for v in self.variables if v not in assignment]
        var = min(unassigned, key=lambda v: len(local_domains[v]))
        
        for value in local_domains[var]:
            # Check if value is consistent with current assignments
            consistent = True
            for peer in self.constraints[var]:
                if peer in assignment and assignment[peer] == value:
                    consistent = False
                    break
            
            if consistent:
                assignment[var] = value
                
                # Save state for visualization
                current_board = [[0 for _ in range(9)] for _ in range(9)]
                for (r, c), val in assignment.items():
                    current_board[r][c] = val
                self.viz.append(current_board)

                # Copy domains for forward checking
                new_domains = copy.deepcopy(local_domains)
                
                # Apply forward checking
                if self.forward_checking(var, value, assignment, new_domains):
                    result = self.backtrack(assignment, new_domains)
                    if result is not None:
                        return result
                        
                # Backtrack if failure occurs
                del assignment[var]
                
        return None

def is_valid_input(puzzle):
    """Check for invalid inputs (e.g., incorrect size or invalid numbers)"""
    if len(puzzle) != 9 or any(len(row) != 9 for row in puzzle):
        return False
    for r in range(9):
        for c in range(9):
            if not (0 <= puzzle[r][c] <= 9):
                return False
    return True

def setup_csp(puzzle):
    """Creates variables, domains, and constraints based on the puzzle"""
    variables = [(r, c) for r in range(9) for c in range(9)]
    domains = {}
    constraints = {}

    for r in range(9):
        for c in range(9):
            # Domains
            if puzzle[r][c] == 0:
                domains[(r, c)] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                domains[(r, c)] = [puzzle[r][c]]
            
            # Constraints (Peers in the same row, col, and 3x3 block)
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

# --- Test Cases ---

# 1. Standard Puzzle
puzzle_standard = [
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

# 2. Corner Case: Unsolvable Puzzle (violates rules immediately)
puzzle_unsolvable = [
    [5, 5, 0, 0, 7, 0, 0, 0, 0], # Two 5s in a row
    [0, 0, 0, 1, 0, 5, 0, 0, 0], 
    [0, 9, 8, 0, 0, 0, 0, 6, 0], 
    [0, 0, 0, 0, 0, 3, 0, 0, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 6], 
    [0, 0, 0, 0, 0, 0, 2, 8, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 8], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 4, 0, 0] 
]

# Select which puzzle to run
test_puzzle = puzzle_standard

if not is_valid_input(test_puzzle):
    print("Error: Invalid puzzle input provided.")
else:
    variables, domains, constraints = setup_csp(test_puzzle)

    print('*'*7,'Initial Board','*'*7) 
    csp = CSP(variables, domains, constraints) 
    csp.print_sudoku(test_puzzle)

    sol = csp.solve() 

    if sol is not None:
        print('\n' + '*'*7, 'Solution Found!', '*'*7) 
        solution = [[0 for i in range(9)] for i in range(9)] 
        for (i, j), val in sol.items(): 
            solution[i][j] = val
            
        csp.print_sudoku(solution)
        csp.visualize()
    else:
        print("\nSolution does not exist or the board is invalid.")