import logging
from copy import deepcopy
import numpy as np
from ..utils import probability_of_successful_attack
from ..utils import possible_attacks
from .utils import eval_heuristic

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class MixedHeuristicAI:
    """ This agent implements several heuristics that rate the position
        strength. The mixed heuristic is calculated using a weighted sum of
        various heuristic function outputs, where the weights were obtained
        experimentally and through random search.
        Uses a shallow Expectiminimax to search for the best move in
        a reasonably short time.
    """

    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """ MixedHeuristicAI agent's turn. Every move's aim is to improve the
            board strength. If there is no move that improves it, the turn is
            ended, except endgame situations (1v1) where it's generally better
            to play aggressively even for the cost of temporarily losing board
            evaluation score.
        """
        self.board = board
        current_eval, hf = eval_heuristic(self.board, self.player_name)


        best_move = self.get_one_move(current_eval)

        if best_move:
            best_move_eval = best_move[2]

            if best_move_eval > current_eval:
                return BattleCommand(best_move[0], best_move[1])

        # There is no reasonable move right now, end this turn and wait
        # for new round that should open new possibilities.
        return EndTurnCommand()

    def get_one_move(self, current_eval):
        """ Find the best attack move.
        """
        nbp_alive = self.board.nb_players_alive()
        attacks = []
        for source, target in possible_attacks(self.board, self.player_name):
            success_prob = probability_of_successful_attack(self.board,
                                                            source.get_name(),
                                                            target.get_name())

            # Only analyze moves that have a decent chance to succeed.
            # Play safer moves when there are more players in the game.
            if (nbp_alive == 2 and success_prob > 0.35) \
                    or (nbp_alive == 3 and success_prob > 0.50) \
                    or (nbp_alive >= 4 and success_prob > 0.65):
                expected_eval = self.fork_board_eval(self.board, source,
                                                     target, success_prob)

                attacks.append([source.get_name(),
                                target.get_name(),
                                expected_eval])
        if attacks:
            return sorted(attacks, key=lambda atk: atk[2], reverse=True)[0]
        else:
            return None

    def fork_board_eval(self, sim_board, source, target, success_prob):
        """ Simulate both possible outcomes of an attack move and find the
            average expected value of this move given its probability
            and board scores.
        """

        # Simulate new board for successful attack
        success_board = deepcopy(sim_board)
        source_area = success_board.get_area(source.get_name())
        target_area = success_board.get_area(target.get_name())
        target_area.set_owner(source.get_owner_name())
        target_area.set_dice(source.get_dice() - 1)
        source_area.set_dice(1)

        # Simulate new board for failed attack
        fail_board = deepcopy(sim_board)
        source_area = fail_board.get_area(source.get_name())
        source_area.set_dice(1)

        # Evaluate new boards
        success_eval, success_hf = eval_heuristic(success_board,
                                                  self.player_name)
        fail_eval, fail_hf = eval_heuristic(fail_board, self.player_name)

        expected_eval = np.average([success_eval, fail_eval],
                                   weights=[success_prob, 1 - success_prob])

        return expected_eval
