# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Información valiosa es extraída (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Inicializar el puntaje
        score = childGameState.getScore()

        # Encontrar la distancia al alimento más cercano y agregar la inversa de esta distancia al puntaje
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += 10.0 / min(foodDistances)  # Agregar la inversa de la distancia al alimento más cercano al puntaje e incrementar el puntaje

        # Para cada fantasma, si el fantasma está asustado y el tiempo asustado es mayor que la distancia al fantasma,
        # agregar un número grande al puntaje. De lo contrario, si la distancia al fantasma es menor que un cierto umbral,
        # restar un número grande al puntaje
        for i, ghostState in enumerate(newGhostStates):
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            if newScaredTimes[i] > ghostDistance:
                score += 100
            elif ghostDistance < 2:
                score -= 100  # Decrementar el puntaje por estar cerca de un fantasma
            elif action == Directions.STOP:
                score -= 500  # Decrementar el puntaje por quedarse quieto

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        """ def minimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
                result = self.evaluationFunction(gameState)
                return result
            nextAgent = agent + 1 if agent + 1 < gameState.getNumAgents() else 0
            actions = gameState.getLegalActions(agent)
            if not actions:  # No legal actions, return a default value
                result = self.evaluationFunction(gameState)
                return result
            
            if depth%2 == 0:  # Pacman
                result = (minimax(nextAgent, depth+1, gameState.getNextState(agent, action)) 
                            for action in actions)
                max_result = max(result)
                return max_result
            elif depth%2 != 0 or agent != 0:  # Ghosts
                result = (minimax(nextAgent, depth+1, gameState.getNextState(agent, action)) 
                            for action in actions)
                min_result = min(result)
                return min_result
        moves = gameState.getLegalActions(0)
        minimax_results = [(move, minimax(0, 1, gameState.getNextState(0, move))) for move in moves]
        move = max(minimax_results, key=lambda x: x[1])[0]
        return move """
        
        def minimize(state, depth, agent): # Fantasmas
            bestCase = float('inf') # El mejor caso posible para los fantasmas es infinito, que también es el peor para el PacMan
            minimum = bestCase # Inicializar el mínimo con el mejor caso
            if state.isWin() or state.isLose() or depth == self.depth: # Si el estado es una victoria o derrota o si se alcanza la profundidad máxima
                return self.evaluationFunction(state) # Devolver la evaluación del estado
            
            # De lo contrario, obtener el siguiente agente y recorrer las acciones legales para este
            next_agent = agent + 1 if agent + 1 < state.getNumAgents() else 0 # Recorrer los agentes y reiniciar si se alcanza el último (Pacman)
            for action in state.getLegalActions(agent): # Por cada acción legal
                if next_agent == 0 and depth + 1 == self.depth: # Si el siguiente agente es Pacman y se alcanza la profundidad máxima
                    minimum = self.evaluationFunction(state.getNextState(agent, action)) # Evaluar el estado
                elif next_agent == 0: # Si el siguiente agente es solo Pacman
                    minimum = maximize(state.getNextState(agent, action), depth + 1) # Maximizar el estado
                else: # Si el siguiente agente es un fantasma
                    minimum = minimize(state.getNextState(agent, action), depth, next_agent) # Minimizar el estado
                bestCase = min(bestCase, minimum) # Obtener el mínimo entre el mejor caso y el mínimo
            return bestCase # Devolver el mejor caso
        
        def maximize(state, depth): # Pacman
            worstCase = float('-inf') # El peor caso posible para Pacman es infinito negativo, se asume el peor de todos los casos
            toGo = Directions.STOP # Inicializar la dirección a seguir con STOP, por si no hay acciones legales
            if state.isWin() or state.isLose() or depth == self.depth: # Si el estado es una victoria o derrota o si se alcanza la profundidad máxima
                return self.evaluationFunction(state) # Devolver la evaluación del estado
            
            # De lo contrario, recorrer las acciones legales para Pacman
            if state.getLegalActions(0):
                for action in state.getLegalActions(0):
                    
                    # Si la minimización del siguiente estado del PacMan es mayor que el peor caso, se maximiza actualizándolo con el peor (mejor) caso y se guarda la acción
                    if minimize(state.getNextState(0, action), depth, 1) > worstCase:
                        worstCase = minimize(state.getNextState(0, action), depth, 1)
                        toGo = action
                    # Se maximiza de esta forma debido a que es necesario obtener la acción, no solo el valor del minimax
            
            # Si la profundidad es 0, devolver la acción a seguir, de lo contrario, devolver el peor caso
            if depth == 0:
                return toGo
            else: 
                return worstCase
            
        return maximize(gameState, 0)
            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def minimize(state, depth, agent, alpha, beta): # Fantasmas
            bestCase = float('inf') # El mejor caso posible para los fantasmas es infinito, que también es el peor para el PacMan
            minimum = bestCase # Inicializar el mínimo con el mejor caso
            if state.isWin() or state.isLose(): # Si el estado es una victoria o derrota
                return self.evaluationFunction(state) # Devolver la evaluación del estado
            
            # De lo contrario, obtener el siguiente agente y recorrer las acciones legales para este
            next_agent = agent + 1 if agent + 1 < state.getNumAgents() else 0 # Recorrer los agentes y reiniciar si se alcanza el último (Pacman)
            
            
            for action in state.getLegalActions(agent): # Por cada acción legal
                if next_agent == 0 and depth + 1 == self.depth: # Si el siguiente agente es Pacman y el siguiente nivel es la profundidad máxima
                    minimum = self.evaluationFunction(state.getNextState(agent, action)) # Evaluar el estado
                elif next_agent == 0: # Si el siguiente agente es solo Pacman
                    minimum = maximize(state.getNextState(agent, action), depth + 1, alpha, beta) # Maximizar el estado
                else: # Si el siguiente agente es un fantasma
                    minimum = minimize(state.getNextState(agent, action), depth, next_agent, alpha, beta) # Minimizar el estado
               

                bestCase = min(bestCase, minimum) # Obtener el mínimo entre el mejor caso y el mínimo
                
                # Actualizar el beta con el mejor caso (el peor caso para PacMan)
                beta = min(beta, bestCase)
                
                # Si el mejor caso es menor que el alpha, se devuelve el mejor caso (pruning)
                if bestCase < alpha:
                    return bestCase
                
            return bestCase # Devolver el mejor caso
        
        def maximize(state, depth, alpha, beta): # Pacman
            worstCase = float('-inf') # El peor caso posible para Pacman es infinito negativo, se asume el peor de todos los casos
            toGo = Directions.STOP # Inicializar la dirección a seguir con STOP, por si no hay acciones legales
            if state.isWin() or state.isLose(): # Si el estado es una victoria o derrota
                return self.evaluationFunction(state) # Devolver la evaluación del estado
            
            # De lo contrario, recorrer las acciones legales para Pacman
            if state.getLegalActions(0):
                for action in state.getLegalActions(0):
                    
                    # Si la minimización del siguiente estado del PacMan es mayor que el peor caso, se maximiza actualizándolo con el peor (mejor) caso y se guarda la acción
                    minimized = minimize(state.getNextState(0, action), depth, 1, alpha, beta)
                    if minimized > worstCase:
                        worstCase = minimized
                        toGo = action
                    # Se maximiza de esta forma debido a que es necesario obtener la acción, no solo el valor del minimax
                    
                    # Se actualiza el alpha con el peor (mejor) caso
                    alpha = max(alpha, worstCase)
                    
                    # Si el peor caso es mayor que el beta, se devuelve el peor caso (pruning)
                    if worstCase > beta:
                        return worstCase
            
            # Si la profundidad es 0, devolver la acción a seguir, de lo contrario, devolver el peor caso
            if depth == 0:
                return toGo
            else: 
                return worstCase
            
        return maximize(gameState, 0, float('-inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def minimize(state, depth, agent): # Fantasmas
            if state.isWin() or state.isLose(): # Si el estado es una victoria o derrota
                return self.evaluationFunction(state) # Devolver la evaluación del estado

            next_agent = agent + 1 if agent + 1 < state.getNumAgents() else 0 # Recorrer los agentes y reiniciar si se alcanza el último (Pacman)
            actions = state.getLegalActions(agent) # Obtener las acciones legales
            uniform = 1.0 / len(actions) # Probabilidad uniforme

            expected_value = 0 # Inicializar el valor esperado
            for action in actions: # Por cada acción legal
                if next_agent == 0 and depth + 1 == self.depth: # Si el siguiente agente es Pacman y se alcanza la profundidad máxima
                    expected_value += uniform * self.evaluationFunction(state.getNextState(agent, action))
                elif next_agent == 0: # Si el siguiente agente es solo Pacman
                    expected_value += uniform * maximize(state.getNextState(agent, action), depth + 1) # Maximizar el estado
                else: # Si el siguiente agente es un fantasma
                    expected_value += uniform * minimize(state.getNextState(agent, action), depth, next_agent) # Minimizar el estado

            return expected_value # Devolver el valor esperado
        
        def maximize(state, depth): # Pacman
            worstCase = float('-inf') # El peor caso posible para Pacman es infinito negativo, se asume el peor de todos los casos
            toGo = Directions.STOP # Inicializar la dirección a seguir con STOP, por si no hay acciones legales
            if state.isWin() or state.isLose(): # Si el estado es una victoria o derrota
                return self.evaluationFunction(state) # Devolver la evaluación del estado
            
            # De lo contrario, recorrer las acciones legales para Pacman
            if state.getLegalActions(0):
                for action in state.getLegalActions(0):
                    
                    # Si la minimización del siguiente estado del PacMan es mayor que el peor caso, se maximiza actualizándolo con el peor (mejor) caso y se guarda la acción
                    minimized = minimize(state.getNextState(0, action), depth, 1)
                    if minimized > worstCase:
                        worstCase = minimized
                        toGo = action
                    # Se maximiza de esta forma debido a que es necesario obtener la acción, no solo el valor del minimax
            
            # Si la profundidad es 0, devolver la acción a seguir, de lo contrario, devolver el peor caso
            if depth == 0:
                return toGo
            else: 
                return worstCase
            
        return maximize(gameState, 0)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()    

# Abbreviation
better = betterEvaluationFunction
