import logging

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand    
from ..utils import *
from copy import deepcopy
import numpy as np
from queue import PriorityQueue
from collections import namedtuple

NUM_AREAS = 29

def board_to_numpy(board):
    state = np.array([
        (area.get_owner_name(), area.get_dice()) 
        for area in board.areas.values()
    ]).T
    return state

def make_adjacent_area_masks(board):
    masks = np.array([
        [
            area_.get_name() in area.get_adjacent_areas() 
            for area_ in board.areas.values()
        ]
        for area in board.areas.values()
    ])
    return masks

def get_possible_attacks(state, adjacent_areas, player_name):
    opponent_areas_mask = state[0] != player_name
    # print(f"Opponent area mask={opponent_areas_mask}")
    able_to_attack_mask = ~opponent_areas_mask & (state[1] > 1) 
    # print(f"Able to attack mask={able_to_attack_mask}")
    possible_attacks = []
    for source in np.nonzero(able_to_attack_mask)[0]:
        # print(f"Source={source}")
        for target in np.nonzero(adjacent_areas[source] & opponent_areas_mask)[0]:
            # f"Target={target}")
            possible_attacks.append((source, target))
    return possible_attacks

def emulate_attack(state, attack):
    source, target = attack
    success_state = state.copy()
    fail_state = state.copy()
    success_state[1, target] = success_state[1, source]
    success_state[1, source] = 1
    success_state[0, target] = success_state[0, source]
    fail_state[1, source] = 1
    return success_state, fail_state

def get_next_player(player_name, player_order):
    i = player_order.index(player_name) + 1
    return player_order[i] if i < len(player_order) else player_order[0]

def get_attack_success_prob(state, attack):
    source, target = attack
    prob = attack_succcess_probability(state[1, source], state[1, target])
    return prob

class Node(namedtuple("Node", "prob actions state")):
    def __lt__(self, other):
        return self.prob >= other.prob

def get_possible_actions(state, adjacent_areas, player_name):
    nodes = PriorityQueue()
    nodes.put(Node(1.0, [], state))
    while not nodes.empty():
        node = nodes.get()
        if node.prob < MINIMAX_PROB_THRESH:
            break
        yield node
        for attack in get_possible_attacks(node.state, adjacent_areas, player_name):
            success_prob = get_attack_success_prob(node.state, attack)
            success_state, fail_state = emulate_attack(node.state, attack)
            success_node = Node(success_prob * node.prob, node.actions + [attack], success_state)
            fail_node = Node((1.0 - success_prob) * node.prob, node.actions + [attack], success_state)
            nodes.put(success_node)
            nodes.put(fail_node)

def evaluate_state(state, player_name):
    own_areas_mask = (state[0] == player_name)
    dice_sum = state[1][own_areas_mask].sum() 
    return dice_sum

MINIMAX_MAX_DEPTH = 4
MINIMAX_MAX_CHILDREN = 10
MINIMAX_PROB_THRESH = 0.3

def minimax(state, adjacent_areas, player_name, player_order, depth=0):
    if depth == MINIMAX_MAX_DEPTH:
        return state, []

    best_score, best_state, best_action = -100000, None, None
    next_player_name = get_next_player(player_name, player_order)
    n_children = 0
    for node in get_possible_actions(state, adjacent_areas, player_name):
        state, _ = minimax(node.state, adjacent_areas, next_player_name, player_order, depth + 1)
        score = evaluate_state(state, player_name) * node.prob
        if score > best_score:
            best_score = score
            best_state = state
            best_action = node.actions
        #print(node.actions, node.prob)
        n_children += 1
        if n_children == MINIMAX_MAX_CHILDREN:
            break
    return best_state, best_action

class AI:
    def __init__(self, player_name, board, player_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        print("Hello world!")
        self.adjacent_areas = make_adjacent_area_masks(board)
        self.player_order = player_order

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        print("It's my turn!")
        state = board_to_numpy(board)
        attacks_ = [(a.get_name(), b.get_name()) for a, b in possible_attacks(board, self.player_name)]
        attacks = get_possible_attacks(state, self.adjacent_areas, self.player_name)
        #emulate_attacks(state, attacks)
        _, action = minimax(state, self.adjacent_areas, self.player_name, self.player_order)
        if not action:
            print(f"AI: end turn")
            return EndTurnCommand()
        else:
            source, target = action[0]
            source = int(source) + 1
            target = int(target) + 1
            print(f"AI: {source} attack {target}")
            return BattleCommand(source, target)