import chess


class boardEval:
    def __init__(self,fen,player):
        self.fen = fen
        self.fen_components = fen.split(" ") 
        self.board = chess.Board(fen=fen)
        self.player = player

    def get_board_scores(self):
        results = self.get_piece_amounts()
        results += self.number_of_moves()
        results += [self.is_checkmate()]
        results += [self.get_game_time()]
        #for w/l
        results += [0]
        return results


    def get_piece_amounts(self):
        pieces = ['P','N','B','R','Q']
        opp_pieces = ['p','n','b','r','q']
        results = []
        fen = self.fen_components[0]
        if self.player == 'b':
            opp_pieces = ['P','N','B','R','Q']
            pieces = ['p','n','b','r','q']
        for i in range(0,5):
            results.append((fen.count(pieces[i])-fen.count(opp_pieces[i])))
        return results

    def get_game_time(self):
        #could be split up into two features for each side?
        board = self.fen_components[0]
        count = len(board)
        count = count - board.count("/")
        for i in range(1,9):
            count = count - board.count(str(i))
        return (round(count/32))

    def is_checkmate(self):
        turn = self.fen.split(" ")[1]
        checkmate = self.board.is_checkmate()
        if not checkmate:
            return 0
        if turn == self.player:
            return -1
        else:
            return 1

    def possible_moves(self,player):
        temp_fen = self.fen_components
        temp_fen[1] = player
        fen = ""
        for value in temp_fen:
            fen = fen + " "+value
        board = chess.Board(fen=fen)
        moves = list(board.legal_moves)
        return len(moves)

    def number_of_moves(self):
        black = self.possible_moves(player='b')
        white = self.possible_moves(player='w')
        data =[white,black] 
        if self.player == 'w':
            data = [black,white]
        return data