# ----------
# User Instructions:
#
# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space
import numpy as np

grid = [[0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid,init,goal,cost):
    # ----------------------------------------
    # insert code here
    # ----------------------------------------
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid[0])-1)]
    closed[init[0]][init[1]] = 1
    x = init[0]
    y = init[1]
    g = 0

    open = [[g, x, y]]

    found = False # flag that is set when search complete
    resign = False # flag set if we can't expand

    while found is False and resign is False:

        # check if we still have elements on the open list
        if len(open) == 0:
            resign = True
            print('Fail')
        else:
            # remove node from the list
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]

        # check if the goal is reached
        if x == goal[0] and y == goal[1]:
            found = True
            print(next)

        else:
            for i in range(len(delta)):
                x2 = x + delta[i][0]
                y2 = y + delta[i][1]

                if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 <= len(grid):
                    if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                        g2 = g + cost
                        open.append([g2, x2, y2])
                        closed[x2][y2] = 1

search(grid, init, goal, cost)
