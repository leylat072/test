from engine1 import TetrisEngine 
engine = TetrisEngine(10,20)
engine.serialize_board(engine.board)
i = 0 
while(True):
    engine.step1(2)
    print(engine.serialize_board(engine.board))
    i = i +1 
    if i == 6 :
        i =0


