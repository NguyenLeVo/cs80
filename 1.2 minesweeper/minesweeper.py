import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    # Anytime the number of cells is equal to the count, we know that all of that sentence's cells must be mines.
    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        cell_count = 0
        for cell in self.cells:
            cell_count += 1
        if cell_count == self.count:
            if cell_count != 0:
                return self.cells
        return None

    # Anytime we have a sentence whose count is 0, we know that all of that sentence's cells must be safe
    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        cell_count = 0
        for cell in self.cells:
            cell_count += 1
        if self.count == 0:
            if cell_count != 0:
                return self.cells
        return None

    # If it's a mine, remove the cell from unknown potential cells with mine and remove a mine from potential mines.
    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    # If it's safe, remove the cell. The mines are still there so cannot remove.
    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1. mark the cell as a move that has been made
        self.moves_made.add(cell)

        # 2. mark the cell as safe
        self.mark_safe(cell)

        # 3. add a new sentence to the AI's knowledge base based on the value of 'cell' and 'count'
        # Same as nearby mines
        cells = []
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if (i, j) in self.mines:
                        count -= 1
                    elif (i,j) not in self.safes:
                        cells.append((i, j))

        self.knowledge.append(Sentence(cells, count))

        # 4. mark any additional cells as safe or as mines if it can be concluded based on the AI's knowledge base
        for sentence in self.knowledge:
            safe = sentence.known_safes()
            mine = sentence.known_mines()
            if safe != None:
                self.safes = self.safes.union(safe)
            if mine != None:
                self.mines = self.mines.union(mine)

        # 5. add any new sentences to the AI's knowledge base if they can be inferred from existing knowledge
        """
        More generally, any time we have two sentences set1 = count1 and 
        set2 = count2 where set1 is a subset of set2, then we can construct 
        the new sentence set2 - set1 = count2 - count1.
        """
        n = len(self.knowledge)
        for i in range(n):
            for j in range(i + 1, n):
                set1 = self.knowledge[i]
                set2 = self.knowledge[j]
                if set1.cells.issubset(set2.cells):
                    cells = set2.cells - set1.cells
                    subset = Sentence(cells, set2.count - set1.count)
                    if subset not in self.knowledge:
                        self.knowledge.append(subset)
                elif set2.cells.issubset(set1.cells):
                    cells = set1.cells - set2.cells
                    subset = Sentence(cells, set1.count - set2.count)
                    if subset not in self.knowledge:
                        self.knowledge.append(subset)



    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for cell in self.safes:
            if cell not in self.moves_made:
                return cell

        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        for i in range(self.height):
            for j in range(self.width):
                cell = (i, j)
                if cell not in self.moves_made:
                    if cell not in self.safes:
                        if cell not in self.mines:
                            return cell

        return None