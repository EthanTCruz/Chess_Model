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
        
        dict_results = {}
        
        piece_amounts = self.get_piece_amounts()
        dict_results["pawns"] = piece_amounts[0]
        dict_results["knights"] = piece_amounts[1]
        dict_results["bishops"] = piece_amounts[2]
        dict_results["rooks"] = piece_amounts[3]
        dict_results["queens"] = piece_amounts[4]    
        
        moves = self.number_of_moves()
        dict_results["my moves"] = moves[0]
        dict_results["opp moves"] = moves[1]

        middle_square_possesion = self.middle_square_attacks()
        dict_results["e4 possesion"] = middle_square_possesion[0]
        dict_results["e5 possesion"] = middle_square_possesion[1]
        dict_results["d4 possesion"] = middle_square_possesion[2]
        dict_results["d5 possesion"] = middle_square_possesion[3]
        
        time = self.get_game_time()
        dict_results["Beginning Game"] = time[0]
        dict_results["End Game"] = time[1]
        
        dict_results["checkmate"] = self.is_checkmate()
        dict_results["w/b"] = victor

        return dict_results

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
            divider = 10
            if i == 0:
                divider = 8
            elif i == -1:
                divider = 9
            results.append(((fen.count(pieces[i])/divider)-(fen.count(opp_pieces[i]))/divider))
        return results

    def get_game_time(self):
        #could be split up into two features for each side?
        board = self.fen_components[0]
        count = len(board)
        count = count - board.count("/")
        bgame = 0
        egame = 0

        for i in range(1,9):
            count = count - board.count(str(i))
        value = 1
        if (count) > (16):
            bgame=1
        else:
            egame = 1
        return ([bgame,egame])

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
    
    def get_king_pressure(self):
        return 0
    
    def get_opp_king_pressure(self):
        return 0
    
    def get_king_xray(self):
        
        return 0
    
    def get_opp_king_xray(self):
        return 0