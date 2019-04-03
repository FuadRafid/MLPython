from Basics.TicTacToe import TicTacToe

# a comment
'''  multi line
comment '''
# tuples
programming_languages = ("Python", "Java", "C++", "C#")

print(programming_languages)

for count, language in enumerate(programming_languages):
    print(count, language)


# functions

def addition(x, y):
    return x + y


def addition_int(x: int, y: int):
    return x + y


print(addition("Hey", " there"))
print(addition_int(13, 12))

# lists

players = [1, 2]

game = TicTacToe()
while True:
    curr_player = players[0]
    game.play_game(just_display=True)
    print(f"Current player {curr_player}")
    column_choice = int(input("Enter column (0,1,2) : "))
    row_choice = int(input("Enter row (0,1,2) : "))
    change_player = game.play_game(curr_player, row_choice, column_choice, just_display=False)
    if change_player:
        players = list(reversed(players))
    if game.is_game_won():
        break
    if game.is_game_finished():
        break
