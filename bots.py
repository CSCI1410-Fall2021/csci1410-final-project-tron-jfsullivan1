#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""


    def heuristic_func(self, state):
        player_symbol = str(state.player_to_move() + 1)
        opp_symbol = '0'
        if player_symbol == '1':
            opp_symbol = '2'
        else:
            opp_symbol = '1'
        player_score = 0 # number of squares for the player
        board = state.board
        board_arr = np.array(board)
        index_player = np.where(board_arr == player_symbol)
        index_opp = np.where(board_arr == opp_symbol)
        shape = np.shape(board_arr)
        rows = shape[0]
        cols = shape[1]
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                current_square = board_arr[row, col]
                if current_square == ' ':
                    dist_to_player = abs(row - index_player[0]) + abs(col - index_player[1])
                    dist_to_opp = abs(row - index_opp[0]) + abs(col - index_opp[1])
                    if dist_to_player < dist_to_opp:
                        player_score += 1

        return player_score

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        #locs = state.player_locs
        #board = state.board
        #ptm = state.ptm
        #loc = locs[ptm]
        #possibilities = list(TronProblem.get_safe_actions(board, loc))
        #if not possibilities:
         #   return "L"
        #best_move = possibilities[0]
        #most_moves = -1
        #for move in possibilities:
           # next_loc = TronProblem.move(loc, move)
           # if len(TronProblem.get_safe_actions(board, next_loc)) <= 2:
              #  if len(TronProblem.get_safe_actions(board, next_loc)) > most_moves:
               #     best_move = move
                #    most_moves = len(TronProblem.get_safe_actions(board, next_loc))

        best_move = self.alpha_beta_cutoff(asp, 10, self.heuristic_func)
        if (asp.is_terminal_state(asp.transition(asp.get_start_state(), best_move))):
            if (asp.get_start_state().get_safe_actions()):
                best_move = asp.get_start_state().get_safe_actions()[0]
        return best_move

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """

        pass

    def alpha_beta_cutoff(self, asp, cutoff_ply, heuristic_func):
        '''
        heuristic_func - a function that takes in a GameState and outputs a
            real number indicating how good that state is for the player who is
            using alpha_beta_cutoff to choose their action. You do not need to
            implement this function, as it should be provided by whomever is
            calling alpha_beta_cutoff, however you are welcome to write
            evaluation functions to test your implemention. The heuristic_func
            we provide does not handle terminal states, so evaluate terminal
            states the same way you evaluated them in the previous algorithms.
        '''

        start = asp.get_start_state()
        _, move = self.max_ab_cutoff(asp, start, -math.inf, math.inf, cutoff_ply, heuristic_func)
        return move

    def max_ab_cutoff(self, asp, state, alpha, beta, ply, heuristic_func):
        if asp.is_terminal_state(state):
            maximizing_player = asp.get_start_state().player_to_move()
            terminal_tuple = asp.evaluate_state(state)
            if(maximizing_player == 0):
                return terminal_tuple[0], None
            else:
                return terminal_tuple[1], None
        if ply <= 0:
            return heuristic_func(state), None
        value = -math.inf
        move = None
        for action in asp.get_available_actions(state):
            max_value, _ = self.min_ab_cutoff(asp, asp.transition(state, action), alpha, beta, ply-1, heuristic_func)
            if max_value > value:
                value, move = max_value, action
                alpha = max(alpha, value)
            if value >= beta:
                return value, move

        return value, move

    def min_ab_cutoff(self, asp, state, alpha, beta, ply, heuristic_func):
        if asp.is_terminal_state(state):
            maximizing_player = asp.get_start_state().player_to_move()
            terminal_tuple = asp.evaluate_state(state)
            if(maximizing_player == 0):
                return terminal_tuple[0], None
            else:
                return terminal_tuple[1], None
        if ply <= 0:
            return heuristic_func(state), None
        value = math.inf
        move = None
        for action in asp.get_available_actions(state):
            min_value, _ = self.max_ab_cutoff(asp, asp.transition(state, action), alpha, beta, ply-1, heuristic_func)
            if min_value < value:
                value, move = min_value, action
                beta = min(beta, value)
            if value <= alpha:
                return value, move
        return value, move

class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
