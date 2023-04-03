import chess


class boardEval:
    def __init__(self,fen,player):
        self.fen = fen
        self.fen_components = fen.split(" ") 
        self.board = chess.Board(fen=fen)
        self.player = player

    def get_board_scores(self,victor="NA"):
        if victor != "NA":
            self.player = victor
        results = self.get_piece_amounts()
        results += self.number_of_moves()
        results += self.middle_square_attacks()
        results += [self.is_checkmate()]
        results += [self.get_game_time()]
        #for w/l
        results += [victor]
        return results

    def middle_square_attacks(self):
        middle_squares =[chess.E4,chess.E5,chess.D4,chess.D5]
        square_possesion = []

        for square in middle_squares:
            
            if self.player == 'w':
                attackers = str(self.board.attackers(chess.WHITE,square=square)).count("1") - str(self.board.attackers(chess.BLACK,square=square)).count("1")
                square_possesion.append(attackers)
            else:
                attackers = str(self.board.attackers(chess.BLACK,square=square)).count("1") - str(self.board.attackers(chess.WHITE,square=square)).count("1")
                square_possesion.append(attackers)
        return square_possesion


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
        value = 3
        if (count/32) > (1/3):
            value =  2
        if (count/32) > (2/3):
            value =  1
        return (value)

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
        #max moves from one side is 187?
        return (len(moves)/187)

    def number_of_moves(self):
        black = self.possible_moves(player='b')
        white = self.possible_moves(player='w')
        data =[white,black] 
        if self.player == 'w':
            data = [black,white]
        return data