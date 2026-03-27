import mcts

mcts = mcts(timeLimit=1000)
bestAction = mcts.search(initialState=initialState)
