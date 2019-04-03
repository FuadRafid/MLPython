class TicTacToe(object):

    def __init__(self):
        self.game = [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]

    def play_game(self, player=0, row=0, column=0, just_display=False):
        try:
            if self.game[row][column] != 0 and not just_display:
                print("This position is occupied!")
                return False
            if not just_display:
                self.game[row][column] = player
                return True
            row_nums = "   " + "  ".join([str(i) for i in range(len(self.game))])
            print(row_nums)
            for index, row in enumerate(self.game):
                print(index, row)
        except IndexError as e:
            print("Out of bounds!, entry must be 0,1 or 2.", e)
            return False
        except Exception as e:
            print("Something went very wrong!", e)
            return False

    # x = play_game
    # x(1, 0, 0, just_display=False)

    def is_game_won(self):
        is_won = self.horizontal_win() or self.vertical_win() \
                 or self.diagonal_win_ltr() or self.diagonal_win_rtl()
        if is_won:
            is_won, player = self.horizontal_win() or self.vertical_win() \
                             or self.diagonal_win_ltr() or self.diagonal_win_rtl()
            print(f"Winner is player {player}")
        return is_won

    def horizontal_win(self):
        for row in self.game:
            if row.count(row[0]) == len(row) and row[0] != 0:
                return True, row[0]
        return False

    def vertical_win(self):
        for col in range(len(self.game)):
            check = []
            for row in self.game:
                check.append(row[col])
            if check.count(check[0]) == len(check) and check[0] != 0:
                return True, check[0]
        return False

    def diagonal_win_ltr(self):
        check = []
        for i in range(len(self.game)):
            check.append(self.game[i][i])
        if check.count(check[0]) == len(check) and check[0] != 0:
            return True, check[0]
        return False

    def diagonal_win_rtl(self):
        rows = range(len(self.game))
        cols = list(reversed(rows))
        check = []
        for col, row in zip(cols, rows):
            check.append(self.game[row][col])
        if check.count(check[0]) == len(check) and check[0] != 0:
            return True, check[0]
        return False

    def is_game_finished(self):
        for row in self.game:
            for val in row:
                if val == 0:
                    return False
        return True
