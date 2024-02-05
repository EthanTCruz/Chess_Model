import random
import chess

class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)

    def select_child(self):
        # Select a child node with the highest UCB1 (Upper Confidence Bound 1)
        return sorted(self.children, key=lambda c: c.wins / c.visits + (2 ** 0.5) * ((2 * math.log(self.visits)) / c.visits) ** 0.5)[-1]

    def add_child(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, self)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result


def monte_carlo_tree_search(root, iterations=1000):
    for _ in range(iterations):
        node = root
        # Selection
        while node.untried_moves == [] and node.children != []:  # node is fully expanded and non-terminal
            node = node.select_child()

        # Expansion
        if node.untried_moves != []:
            m = random.choice(node.untried_moves) 
            node = node.add_child(m)

        # Simulation
        simulation_board = node.board.copy()
        while not simulation_board.is_game_over():
            simulation_board.push(random.choice(list(simulation_board.legal_moves)))

        # Backpropagation
        result = 1 if simulation_board.result() == '1-0' else 0
        while node != None:
            node.update(result)
            node = node.parent

    return sorted(root.children, key=lambda c: c.visits)[-1].board.move_stack[-1]


# Example usage
board = chess.Board()
root = MCTSNode(board)
best_move = monte_carlo_tree_search(root)
print(f"Best move: {best_move}")
