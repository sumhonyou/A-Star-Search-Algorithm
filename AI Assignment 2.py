# ----------------------------- Import Standard Libraries -----------------------------
import matplotlib.pyplot as plt                   # For plotting the maze and the path
import numpy as np                                # For numerical operations
from matplotlib.patches import RegularPolygon     # For drawing hexagons
from matplotlib.lines import Line2D               # For customizing plot legends
import heapq                                      # For the A* priority queue (min-heap)
import itertools                                  # For unique counters (tie-breakers in heapq)

# --------------------------------------------------
# Special terms we use in the code to explain:
# tiles: individual cells in the maze ; The coordinate (r,q) uniquely identifies each tile.
# axes: plural of axis; refers to the complete plotting area within a figure where data is visualized; 
#       includes both x and y axes, ticks, labels, and the plotted data.
# --------------------------------------------------

# -----------------------------------------------------------
# Assumptions:
# a)	Step = x, energy = y, Movement Cost = x*y
# b)	Steps may be x, x/2, or 2x, while energy may be y, y/2, or 2y, depending on the effects of rewards and traps.
# c)	Once the algorithm takes all 4 treasures, it will immediately leave the virtual world through any one of the edges of the maze.
# d)	Reward 1: Once taken, the halved energy will be applied to every next move.
# e)	Reward 2: Once taken, the halved step will be applied to every next move.
# f)	Once the rewards (Reward 1 and Reward 2) are taken, the rest of the moves will be affected, and it is acceptable to apply multiple rewards on every move. 
# g)	Trap 1: Once triggered, the double energy will be applied to every next move.
# h)	Trap 2: Once triggered, the double steps will only be applied to every next move.
# i)	Trap 3: Once triggered, it will be moved two steps away based on the last movement direction, but the movement will only affect the energy consumed; the steps are not counted, because it was not a voluntary move.
# j)	Trap 4: Once triggered, the program will be terminated immediately. 

# --------------------------------------------------

# ------------------ Node class: represents each state in the search ------------------ 
class Node:
    def __init__(self, state, parent=None, cost=0, steps=0, treasures=None, rewards=None, step_mul=1.0, energy_mul=1.0,
                 trap2=False, trap3=False, trap4=False):
        self.state = state                     # (row, col) tuple representing the position in the maze
        self.parent = parent                   # Link to the parent Node for path reconstruction
        self.cost = cost                       # Total cost incurred to reach this Node (accounts for traps / reward)
        self.treasures = treasures or set()    # Set of treasures collected so far
        self.rewards = rewards or set()        # Set of rewards collected so far
        self.step_mul = step_mul               # Step multiplier (affected by Reward2)
        self.energy_mul = energy_mul           # Energy multiplier (affected by traps and Reward1)
        self.trap2 = trap2                     # If currently under Trap 2 effect → all move afterward cause double step
        self.trap3 = trap3                     # If currently under Trap 3 effect → triggers forced move
        self.trap4 = trap4                     # If currently under Trap 4 effect → this path will be ignored else this trap will cause game over
        self.h = 0                             # Set initial heuristic value to 0
        self.f = 0                             # Set initial total cost value to 0
        self.steps = steps                     # Number of "steps" taken, not same with path length

# ------------------ HexMaze class: encapsulates maze logic and rules ------------------
class HexMaze:
    
    def __init__(self, rows, cols, special_labels):
        # Stored as instance variables (with self.) so the maze object remembers them.
        self.rows = rows               # Number of rows in hex grid (maze)
        self.cols = cols               # Number of maze in hex grid (columns
        self.labels = special_labels   # Dictionary marking special cells with (row, col): (label, color) pairs for special tiles
        
    # -- Determine the type of a given tile --
    # Given a cell at row r, column q, determines what kind of tile it is.
    def get_type(self, r, q):  
        # Check if (r,q) appear in self.labels
        if (r, q) in self.labels: 
            # If yes, gets the first part (the label string), makes it lowercase for consistent comparison.
            label = self.labels[(r, q)][0].lower()
            
            # Standardize the returned type string
            # By mapping the label to a standardized string
            if 'obstacle' in label: 
                return 'obstacle'
            elif 'trap 1' in label: 
                return 'trap1'
            elif 'trap 2' in label: 
                return 'trap2'
            elif 'trap 3' in label: 
                return 'trap3'
            elif 'trap 4' in label: 
                return 'trap4'
            elif 'reward 1' in label: 
                return 'reward1'
            elif 'reward 2' in label: 
                return 'reward2'
            elif 'treasure' in label: 
                return 'treasure'
        return 'empty'   # If no match, returns 'empty' (regular cell, not special).
    
    # Returns all valid (non-obstacle) neighboring positions from a given tile, considering hex grid geometry
    def neighbors(self, state):
        # Given the current cell (state is a tuple (r, q)), computes all valid neighboring cells you can move to.
        r, q = state
        # In hex grids, neighboring cells differ depending on whether the column is even or odd (q % 2)
        # So, got two different directions lists.
        if q % 2 == 0:  
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
        else:
            directions = [(-1, -1), (-1, 0), (0, 1), (1, 0), (1, -1), (0, -1)]
        result = []  # collect all valid neighbors
        
        #Loops through each possible direction.
        for dr, dq in directions:
            nr, nq = r + dr, q + dq   # Computes nr (new row), nq (new column).
            # Checks that the new position is within the grid bounds & is not an obstacle.
            if 0 <= nr < self.rows and 0 <= nq < self.cols and self.get_type(nr, nq) != 'obstacle':
                result.append((nr, nq)) # If valid, appends it to the result.
        return result # Returns the list of all valid neighboring cell positions as (row, col) tuples.
    
    # Compute direction vector from one tile to another
    # Returns the direction (as row/col differences) needed to go from "from_pos" to "to_pos".
    def get_direction_vector(self, from_pos, to_pos):
        r1, q1 = from_pos           # Unpack the starting tile's (row, col)
        r2, q2 = to_pos             # Unpack the target tile's (row, col)
        return (r2 - r1, q2 - q1)   # Compute the difference in rows and cols as a tuple
    
    # Move in a given direction for a given number of steps, unless blocked
    # Moves from pos in the given direction (dr, dq) for up to steps steps.
    # Stops if it hits the edge of the maze or an obstacle.
    # Returns the last valid position reached.
    def move_in_direction(self, pos, direction, steps=1):
        r, q = pos                 # Unpack the starting position
        dr, dq = direction         # Unpack the direction vector (row/col deltas)
        
        # Repeat for the number of steps requested
        for _ in range(steps):
            new_r, new_q = r + dr, q + dq   # Calculate the next position in the direction
            if (0 <= new_r < self.rows and 0 <= new_q < self.cols and 
                self.get_type(new_r, new_q) != 'obstacle'):   # Only move if not blocked or out of bounds
                r, q = new_r, new_q                           # Update to new position
            else:
                break  # Stop moving if blocked 
        return (r, q)  # Return the final position reached

    # --- Heuristic for A* : Hex grid distance applying cube coordinate system ---
    # Mahatthan for hex
    # Computes the shortest path ("hex Manhattan distance") between two hex tiles, regardless of obstacles.
    # Converts each tile from (row, col) to cube coordinates (a standard way to calculate distance in hex grids).
    # Returns the largest difference among x, y, or z coordinates.
    def hex_distance(self, a, b):
        def offset_to_cube(r, q):   # Helper: convert (row, col) "offset" to cube coords
            x = q
            z = r - (q - (q&1)) // 2
            y = -x - z
            return x, y, z         # Cube coordinates (x, y, z)
        x1, y1, z1 = offset_to_cube(*a)    # Convert position a to cube coords
        x2, y2, z2 = offset_to_cube(*b)    # Convert position b to cube coords
        return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))   # Hex distance = max coordinate delta (difference between coordinates of two hexagon tiles)

# ------------------ A* Solver class: searches for the optimal path ------------------
class AStarSolver:
    
    # Constructor, which sets up self.maze, self.tresure, self.counter
    def __init__(self, maze, treasures): 
        self.maze = maze                     # Store the maze object as references to the maze (contains map logic, rules, etc.)
        self.treasures = treasures           # Keeps the set of all treasure locations to check when all have been collected.
        self.counter = itertools.count()     # Used as a tie-breaker when inserting into the heap queue (priority queue).
    
    # Expands a node: generates all possible child nodes for one move
    # Generates all valid moves from the current node’s position (i.e., all neighboring tiles).
    def expand(self, node):
        children = []     # List to store the resulting child nodes
        
        # For each neighbor nbr:
        # Make copies of treasures and rewards (so changes in one child do not affect others).
        # Carry over step and energy multipliers.
        # Carry over all trap effect flags for this possible path.
        for nbr in self.maze.neighbors(node.state):    
            treasures = node.treasures.copy()    # Copy the collected treasures set (so each child can update safely)
            rewards = node.rewards.copy()        # Copy the collected rewards set (same reason)
            step_mul = node.step_mul             # Step multiplier (affected by rewards/penalties)
            energy_mul = node.energy_mul         # Energy multiplier (affected by rewards/penalties)
            trap2 = node.trap2                   # Trap2 flag: True if all move afterward steps costs double
            trap3 = node.trap3                   # Trap3 flag: True if trap3 effect is active
            trap4 = node.trap4                   # Trap4 flag: True if trap4 effect is active
            
            # --- STEP & COST LOGIC ---
            # If under Trap 2 effect, all the next moves costs double
            move_cost = (2.0 if trap2 else 1.0) * step_mul * energy_mul  # Multiply also by step and energy multipliers (which can be affected by rewards/other traps).
            
            direction = self.maze.get_direction_vector(node.state, nbr)   # Calculate the direction vector from current position to neighbor (nbr).
            final_position = nbr                                          # The neighbor you will land on

            cell_type = self.maze.get_type(*nbr)  # Identify the type of the cell (e.g., empty, trap, reward, treasure).
                        
            # -------- TRAP 3 (Forced Move) SPECIAL CASE --------
            # Assumption: Entering Trap 3 forces you to move 2 more steps in the same direction.
            # Energy IS charged for the forced move, so cost is increased also
            # "Steps" is NOT incremented for the forced move (since it's not a voluntary action)
            if cell_type == 'trap3':
                # If stepping onto Trap 3:
                # Special effect: Must "jump" two more cells in the same direction (forced movement).
                # Cost: Energy is consumed for the jump, but step count is not increased for the forced movement
                # Copy the current state for treasures, rewards, and multipliers for this new forced-move branch.
                trap_treasures = treasures.copy()
                trap_rewards = rewards.copy()
                trap_step_mul = step_mul
                trap_energy_mul = energy_mul
                trap_trap2 = trap2
                trap_trap4 = trap4
                # If the cell is also a treasure, add it to the collection.
                if cell_type == 'treasure':
                    trap_treasures.add(nbr)
                                        
                # 1. Enter the Trap 3 cell (one step, one cost)       
                # Create a new node for the action of stepping onto the Trap 3 cell:
                trap_node = Node(nbr, node, node.cost + move_cost, node.steps + 1, 
                                 trap_treasures, trap_rewards, trap_step_mul, trap_energy_mul, 
                                 trap_trap2, trap3, trap_trap4)
                
                # 2. Forced move: move two more in the same direction (energy is consumed, but steps do NOT increment)
                final_position = self.maze.move_in_direction(nbr, direction, 2)     # Move two tiles further in the same direction, from the Trap 3 cell (total of 2 steps).
                intermediate_pos = self.maze.move_in_direction(nbr, direction, 1)   # The tile you land on after 1 extra step (for checking any rewards/traps on the way).
                
                # Apply any trap/reward effects for tiles landed on during the forced move
                # For both intermediate and final forced-move tiles:
                # Reward 1: Add to rewards, halve energy cost for every future movement
                # Reward 2: Add to rewards, halve step increment for every future movement
                # Treasure: Add to treasures.
                # Trap 1: Double future energy cost.
                # Trap 2: Set flag for all next move to cost double.
                # Trap 4: Set flag to fail the path.
                for check_pos in [intermediate_pos, final_position]:
                    check_type = self.maze.get_type(*check_pos)
                    if check_type == 'reward1' and check_pos not in trap_rewards:
                        trap_rewards.add(check_pos)
                        trap_energy_mul *= 0.5          # Halves energy cost for future moves
                    elif check_type == 'reward2' and check_pos not in trap_rewards:
                        trap_rewards.add(check_pos)
                        trap_step_mul *= 0.5            # Halves step increment for future moves
                    elif check_type == 'treasure':
                        trap_treasures.add(check_pos)   # Collect treasure if landed on
                    elif check_type == 'trap1':
                        trap_energy_mul *= 2.0          # Doubles energy cost for future moves
                    elif check_type == 'trap2':
                        trap_trap2 = True               # All next move will cost double
                    elif check_type == 'trap4':
                        trap_trap4 = True               # Triggers instant fail for this path

                # Forced move cost: 2 steps worth of cost, but no step increment.
                forced_move_cost = 2 * (2.0 if trap2 else 1.0) * trap_step_mul * trap_energy_mul
                # Creates a new node after forced move:              
                # Position is after 2 forced steps.                
                # Cost is increased by the energy spent for forced movement.             
                # Step count is not incremented.             
                # All updated state variables passed on.
                final_child = Node(final_position, trap_node, trap_node.cost + forced_move_cost, trap_node.steps, 
                                   trap_treasures, trap_rewards, trap_step_mul, trap_energy_mul, 
                                   trap_trap2, trap3, trap_trap4)

                # Add this new forced-move node as a possible child for expansion.
                children.append(final_child)
            
            # ----------- NORMAL MOVE -------------
            else:
                # Collect rewards and apply effects.
                # For each reward/trap/treasure:
                # Update the relevant multipliers, reward/treasure collections, and trap flags.
                if cell_type == 'reward1' and nbr not in rewards:
                    rewards.add(nbr)
                    energy_mul *= 0.5   # Halves energy cost from now on
                elif cell_type == 'reward2' and nbr not in rewards:
                    rewards.add(nbr)
                    step_mul *= 0.5     # Halves step increment from now on (so each move is "half a step")
                elif cell_type == 'trap1':
                    energy_mul *= 2.0   # Doubles energy cost from now on
                elif cell_type == 'trap2':
                    trap2 = True        # All move will cost double from now
                elif cell_type == 'trap4':
                    trap4 = True        # Triggers instant fail for this path
                elif cell_type == 'treasure':
                    treasures.add(nbr)
                    
                # Step increment depends on step_mul (default 1, after Reward 2 is 0.5 per move)
                step_increment = 1.0 if step_mul == 1.0 else 0.5
                
                # Create a new node for a regular move:           
                # Increments both cost and step count (possibly by 0.5).               
                # Passes on all updated states.             
                # Add to children for further expansion.
                child = Node(final_position, node, node.cost + move_cost, node.steps + step_increment, treasures, rewards,
                             step_mul, energy_mul, trap2, trap3, trap4)
                children.append(child)
        # Returns all the possible next moves (children nodes).
        return children
    
    # Main search routine
    def solve(self, start):
        start_node = Node(start)             # Make the start node
        frontier = [(0, next(self.counter), start_node)]  # Priority queue: (priority, tie-breaker, node)
        visited = set()   # Set to track visited (state, treasures) pairs

        # While there are nodes to explore... :
        while frontier:
            _, _, node = heapq.heappop(frontier)   # Get the node with lowest cost + heuristic
            if node.trap4: continue          # Skip if path hits instant-fail Trap 4 ;  if hit Trap 4 (game over tile)
            
            # Goal check: If this node has collected all 4 treasures, reconstruct and return the path, total cost, and steps.
            if len(node.treasures) == 4:     # Success: all treasures collected
                return self.reconstruct_path(node), node.cost, node.steps
            
            # Make a unique key based on position and treasures collected (order doesn’t matter, so use frozenset).
            key = (node.state, frozenset(node.treasures))
            if key in visited: continue      # Avoid revisiting the same state + treasure set; Skip if already visited this state + treasures combination
            visited.add(key)    # otherwise, mark as visited.

            # For every possible next move (child node):
            for child in self.expand(node): 
                remaining = [t for t in self.treasures if t not in child.treasures]           # remaining: Which treasures are left to collect?
                h = min((self.maze.hex_distance(child.state, t) for t in remaining), default=0)  # h: Heuristic = minimum hex distance from this child to any remaining treasure (optimistic guess to goal).
                child.h = h  # child.h: Store the heuristic.
                child.f = child.cost + h     # Set total priority for A*;  f = g + h
                heapq.heappush(frontier, (child.f, next(self.counter), child))  # Push this new node into the frontier (priority queue).
        return None, float('inf'), 0         # If while loop ends, no path found
    
    # Backtrack from goal node to reconstruct the solution path
    # Starting from the final (goal) node
    # keep following the parent links backward and add each state to the front of the path (so the path is start → goal).
    def reconstruct_path(self, node):
        path = []
        current = node
        while current:
            path.insert(0, current.state)  # Insert each parent state at the front of the path
            current = current.parent
        return path # Return full path

# ----------------------------- MAIN BLOCK -----------------------------

# -- Setup the maze and special tiles (obstacles, traps, rewards, treasures) --
special_labels = {
    (2, 0): ("Obstacle", "dimgray"),
    (4, 1): ("Trap 2", "orchid"),
    (2, 1): ("Reward 1", "mediumturquoise"),
    (3, 2): ("Obstacle", "dimgray"),
    (1, 2): ("Trap 2", "orchid"),
    (4, 3): ("Trap 4", "orchid"),
    (2, 3): ("Obstacle", "dimgray"),
    (1, 3): ("Treasure", "orange"),
    (5, 4): ("Reward 1", "mediumturquoise"),
    (4, 4): ("Treasure", "orange"),
    (3, 4): ("Obstacle", "dimgray"),
    (1, 4): ("Obstacle", "dimgray"),
    (2, 5): ("Trap 3", "orchid"),
    (0, 5): ("Reward 2", "mediumturquoise"),
    (4, 6): ("Trap 3", "orchid"),
    (2, 6): ("Obstacle", "dimgray"),
    (1, 6): ("Obstacle", "dimgray"),
    (3, 7): ("Reward 2", "mediumturquoise"),
    (2, 7): ("Treasure", "orange"),
    (1, 7): ("Obstacle", "dimgray"),
    (4, 8): ("Obstacle", "dimgray"),
    (3, 8): ("Trap 1", "orchid"),
    (2, 9): ("Treasure", "orange")
}

maze = HexMaze(6, 10, special_labels)         # Create a maze object with 6 rows, 10 cols, and special labels
treasures = {(1, 3), (4, 4), (2, 7), (2, 9)}  # Coordinates of all treasures to collect
solver = AStarSolver(maze, treasures)         # Initialize A* solver with the maze and treasures
start_position = (5, 0)                       # Starting coordinate

# -- Solve the maze --
# Calls the solver to find the optimal path, total energy cost, and number of steps from start to collect all treasure
path, cost, total_steps = solver.solve(start_position)

# --- Output the results ---
print("\n========== Best Path ==========")
print(f"{path}")
print("="*31)

# ---- Visualization (matplotlib) ----
hex_radius = 1                       # Set the radius of each hexagon tile
dx = 3/2 * hex_radius                # Horizontal spacing between hex centers
dy = np.sqrt(3) * hex_radius / 2     # Vertical offset between even/odd columns

fig, ax = plt.subplots(figsize=(14, 10))  # Create a matplotlib figure and axes with a specific size
ax.set_aspect('equal')                    # Ensure hexagons are not distorted

# --- Draw the grid and label special tiles ---
for r in range(6):        # For every row in the grid
    for q in range(10):   # For every column in the grid
        x = q * dx        # Compute x-position for this hex center
        y = r * np.sqrt(3) * hex_radius + (q % 2) * dy    # Compute y-position, offset for even/odd columns
        
        # Choose the fill color based on special_labels; default to white if not special
        facecolor = special_labels.get((r, q), ('', 'white'))[1]
        # Create the hexagon patch for this tile
        hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius * 0.95,
                                 orientation=np.radians(30), facecolor=facecolor, edgecolor='black')
        ax.add_patch(hexagon)  # Add the hexagon to the plot
        label_text = f'({r},{q})'  # Start with coordinates
        
        if (r, q) in special_labels:  # If special tile, show label name too
            label_text += f'\n{special_labels[(r, q)][0]}'
        ax.text(x, y, label_text, ha='center', va='center', fontsize=7, color='black')   # Draw text in the center

# --- Draw the solution path and mark jumps for Trap 3 ---
if path:  # Only run if there is a path to display
    x_coords, y_coords = [], []  # Lists to store all x and y coordinates along the path
    for i, (r, q) in enumerate(path):   # Loop over every position (r, q) in the path
        x = q * dx  # Calculate x position in plot coordinates
        y = r * np.sqrt(3) * hex_radius + (q % 2) * dy  # Calculate y position (with vertical offset)
        x_coords.append(x)  # Save x position
        y_coords.append(y)  # Save y position
        
        # Mark start and end positions with different symbols/colors
        if i == 0:
            ax.plot(x, y, 'go', markersize=12, label='Start')  # Green for start
        elif i == len(path) - 1:
            ax.plot(x, y, 'bs', markersize=12, label='End')    # Blue square for end
        else:
            ax.plot(x, y, 'ro', markersize=10)                 # Red circles for path
        
    # Draw path lines for each segment, highlighting trap 3 jumps specially
    for i in range(len(path)-1):
        r1, q1 = path[i]      # Current position
        r2, q2 = path[i+1]    # Next position in the path
        
        # Compute the actual Cartesian coordinates
        # Convert (row, col) to (x, y) plot coordinates for both ends
        x1 = q1 * dx
        y1 = r1 * np.sqrt(3)*hex_radius + (q1 % 2)*dy
        x2 = q2 * dx
        y2 = r2 * np.sqrt(3)*hex_radius + (q2 % 2)*dy
    
        # --- Highlight trap 3 jumps with arrows ---
        # Detect if the step starts on a Trap 3 cell
        if maze.get_type(r1, q1) == 'trap3':
            # draw an arrow for the forced two-cell jump
            ax.annotate(
                '',
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', ls="dotted", lw=3)
            )  # Draw a black dotted arrow for the forced two-cell jump
        else:
            # Normal path segment
            ax.plot([x1, x2], [y1, y2],
                    '-', color='red', linewidth=3, alpha=0.7)

# --- Set plot limits and remove axis ticks for a cleaner look ---
ax.set_xlim(-1, dx * 10 + 1)   # Set horizontal axis range
ax.set_ylim(-1, np.sqrt(3) * hex_radius * 6 + 1)  # Set vertical axis range
ax.set_xticks([]) #Remove the tick values (but keep the hex grid visible)
ax.set_yticks([])# Remove y tick labels but keep grid visible

# Build custom legend for the plot
legend_elements = [  # Each entry here will appear in the plot legend
    Line2D([0], [0], marker='o', color='w',  # 'o' means circle, color 'w' means background white
           markerfacecolor='green', markersize=12, label='Start'),  # Green circle for Start
    
    Line2D([0], [0], marker='s', color='w',  # 's' means square, blue for End
           markerfacecolor='blue', markersize=12, label='End'),
    
    Line2D([0], [0], color='red', lw=3,      # Solid red line for normal path segments
           label='Normal move'),
    
    Line2D([0], [0], color='black', lw=3, linestyle=':',  # Dotted black line for Trap-3 jumps
           label='Trap-3 jump'),
]

ax.legend(
    handles=legend_elements,          # Use only these hand-picked legend entries
    loc='upper left',                 # Position legend in upper left of plot area
    bbox_to_anchor=(1.02, 1),         # Place legend just outside the plot to the right
    borderaxespad=0                   # No extra border padding
)

# Prepare aligned summary for annotation (the info box)
labels = ["Start Position", "Treasures", "Nodes Visited", "Steps Taken", "Total Cost"]  # Info fields
values = [
    start_position,                   # Starting cell coordinate
    f"{len(treasures)}/4",            # Number of treasures collected out of 4
    len(path),                        # Total nodes visited along the path
    total_steps,                      # Total physical steps taken (could be float if half-steps)
    f"{cost:.1f}"                     # Total energy/cost, 1 decimal
]

# Build aligned summary lines (monospaced and left-aligned)
summary = [
    f"{lab:<15} : {val}"              # "<15" means pad labels to width 15 for alignment
    for lab, val in zip(labels, values)
]
summary_text = "\n".join(summary)     # Multi-line string for annotation

# Box style for the annotation text (summary info box)
bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7)

# Draw the summary text block outside the main plot
ax.text(
    1.02, 0.5, summary_text,                  # x, y location in axes coordinates (outside right)
    transform=ax.transAxes,                   # Use axes-relative coordinates
    fontsize=10,
    fontfamily="monospace",                   # Use monospace font for clean alignment
    ha="left",                                # Left-align the text
    va="center",                              # Vertically center the text block
    bbox=bbox_props                           # Put it inside a white rounded box
)

plt.title("Treasure Hunt Using A*")           # Set main title of the plot
plt.tight_layout()                            # Improve spacing/fit of plot elements
plt.show()                                    # Show the plot window

# ------------------------ Console Output ---------------------------

# Detailed Path Analysis
print("\n===== Full Path Sequence:=====")
for i, pos in enumerate(path):
    cell_type = maze.get_type(*pos)
    print(f"Step {i} : {pos}  [Type: {cell_type}]")
print("="*31 + "\n")

# Final Summary
print("\n======== FINAL SUMMARY ========")
print(f"Start Position                : {start_position}")
print(f"Number of Treasures Collected : {len(treasures)} / 4")
print(f"Number of Nodes Visited       : {len(path)} positions")  # includes start
print(f"Total Physical Steps Taken    : {total_steps} steps")
print(f"Total Cost                    : {cost:.1f}")
print("="*31 + "\n")