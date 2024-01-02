import chess


class boardEval:
    def __init__(self,player="NA",fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        self.fen = fen
        self.fen_components = fen.split(" ") 
        self.board = chess.Board(fen=fen)
        self.player = player
        self.opponent = 'w'
        if player == 'w':
            self.opponent = 'b'

    def get_features(self):
        scores = self.get_board_scores()
        features = ["moves(id)"] + list(scores.keys())
        return features

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

        piece_pairs = self.knight_bishop_pairs()
        dict_results["player has bishop pair"] = piece_pairs["player has bishop pair"]
        dict_results["opponent has bishop pair"] = piece_pairs["opponent has bishop pair"]
        dict_results["player has knight bishop pair"] = piece_pairs["player has knight bishop pair"]
        dict_results["opponent has knight bishop pair"] = piece_pairs["opponent has knight bishop pair"]                
        dict_results["player has knight pair"] = piece_pairs["player has knight pair"]
        dict_results["opponent has knight pair"] = piece_pairs["opponent has knight pair"]

        moves = self.number_of_moves()
        dict_results["my moves"] = moves["player"]
        dict_results["opp moves"] = moves["opponent"]
        total_moves = moves["player"] + moves["opponent"]
        if total_moves == 0:
            dict_results["moves score"] = 0
        else:
            dict_results["moves score"] = (moves["player"]/(moves["player"]+moves["opponent"]))

        dict_results["knight moves"] = moves["player N"]
        dict_results["bishop moves"] = moves["player B"]
        dict_results["rook moves"] = moves["player R"]
        dict_results["queen moves"] = moves["player Q"]

        dict_results["opponent knight moves"] = moves["opponent N"]
        dict_results["opponent bishop moves"] = moves["opponent B"]
        dict_results["opponent rook moves"] = moves["opponent R"]
        dict_results["opponent queen moves"] = moves["opponent Q"]

        '''
        dict_results["knight ratio"] = (dict_results["knight moves"]+1)/(dict_results["opponent knight moves"]+1)
        dict_results["bishop ratio"] = (dict_results["bishop moves"]+1)/(dict_results["opponent bishop moves"]+1)
        dict_results["rook ratio"] = (dict_results["rook moves"]+1)/(dict_results["opponent rook moves"]+1)
        dict_results["queen ratio"] = (dict_results["queen moves"]+1)/(dict_results["opponent queen moves"]+1)
        '''

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
            divider = 2
            
            if i == 0:
                divider = 8
            elif i == 4:
                divider = 1
            results.append(((fen.count(pieces[i])/divider)-(fen.count(opp_pieces[i]))/divider))
        return results
    
    def knight_bishop_pairs(self):
        results = {}
        fen = self.fen_components[0]
        
        player_bishop = 'B'
        opp_bishop = 'b'
        player_knight  = 'N'
        opp_knight = 'n'
        if self.player == 'b':
            opp_bishop = 'B'
            player_bishop = 'b'
            player_knight  = 'n'
            opp_knight = 'N'

        opp_bishop_count = fen.count(opp_bishop)
        player_bishop_count = fen.count(player_bishop)
        opp_knight_count = fen.count(opp_knight)
        player_knight_count = fen.count(player_knight)

        results["opponent knight"] = opp_knight_count
        results["player knight"] = player_knight_count
        results["opponent bishops"] = opp_bishop_count
        results["player bishops"] = player_bishop_count
        
        player_bishop_pair = 0
        player_knight_pair = 0
        player_knight_bishop_pair = 0
        opp_bishop_pair = 0
        opp_knight_pair = 0
        opp_knight_bishop_pair = 0
        
        if player_bishop_count >= 2:
            player_bishop_pair = 1
        if player_knight_count >= 2:
            player_knight_pair = 1
        if player_knight_count >= 1 and player_bishop_count >= 1:
            player_knight_bishop_pair = 1

        if opp_bishop_count >= 2:
            opp_bishop_pair = 1
        if opp_knight_count >= 2:
            opp_knight_pair = 1
        if opp_knight_count >= 1 and opp_bishop_count >= 1:
            opp_knight_bishop_pair = 1

        results["opponent has knight pair"] = opp_knight_pair
        results["opponent has knight bishop pair"] = opp_knight_bishop_pair
        results["opponent has bishop pair"] = opp_bishop_pair

        results["player has knight pair"] = player_knight_pair
        results["player has knight bishop pair"] = player_knight_bishop_pair
        results["player has bishop pair"] = player_bishop_pair
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
        #max moves from one side is 187?
        moves = list(board.legal_moves)
        moves = [board.san(chess.Move.from_uci(str(move))) for move in moves]
        return (moves)

    def number_of_moves(self):
        black_moves = self.possible_moves(player='b')
        white_moves = self.possible_moves(player='w')
        black = len(black_moves)/30
        white = len(white_moves)/30
        data = {}
        if self.player == 'w':
            pieces = ['P','N','B','R','Q']
            for piece in pieces:
                data[f'player {piece}'] = 0
                data[f'opponent {piece}'] = 0
                new_moves = []
                for i in range(len(white_moves)):
                    if white_moves[i][0] == piece:
                        data[f'player {piece}'] = data[f'player {piece}'] + 1
                    else:
                        new_moves.append(white_moves[i])
                white_moves=new_moves
                new_moves = []
                for i in range(len(black_moves)):
                    if black_moves[i][0] == piece:
                        data[f'opponent {piece}'] = data[f'opponent {piece}'] + 1
                    else:
                        new_moves.append(black_moves[i])
                black_moves=new_moves                

            data["player"] = white
            data["opponent"] = black
        else:
            pieces = ['P','N','B','R','Q']
            for piece in pieces:
                data[f'player {piece}'] = 0
                data[f'opponent {piece}'] = 0
                new_moves = []
                for i in range(len(white_moves)):
                    if white_moves[i][0] == piece:
                        data[f'opponent {piece}'] = data[f'opponent {piece}'] + 1
                    else:
                        new_moves.append(white_moves[i])
                white_moves=new_moves

                new_moves = []
                for i in range(len(black_moves)):
                    if black_moves[i][0] == piece:
                        data[f'player {piece}'] = data[f'player {piece}'] + 1
                    else:
                        new_moves.append(black_moves[i])
                black_moves=new_moves
            data["player"] = black
            data["opponent"] = white
        return data
    
    def get_king_pressure(self):
        return 0
    
    def get_opp_king_pressure(self):
        return 0
    
    def get_king_xray(self):
        
        return 0
    
    def get_opp_king_xray(self):
        return 0