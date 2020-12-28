
# from .ai import AI
import os
import time
import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from ..utils import possible_attacks
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


MEMORY_MAX_CAPACITY = 10000
MEMORY_SAVE_FILE = "memory.pt"
POLICY_MODEL_SAVE_FILE = "policy_model.pt"
TARGET_MODEL_SAVE_FILE = "target_model.pt"
VAR_SAVE_FILE = "vars.pt"
NUM_AREAS = 29
NUM_INPUTS = NUM_AREAS * 2
NUM_ACTIONS = NUM_AREAS * NUM_AREAS * 2 + 1
END_TURN_OFFSET = NUM_AREAS * NUM_AREAS
ACTION_END_TURN = NUM_ACTIONS - 1
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 10
WIN_REWARD = 100
LOSE_REWARD = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])
Action = namedtuple("Action", ["src", "dst", "end_turn"])

def dprint(*args):
    print(*args)

class ReplayMemory:
    def __init__(self, max_capacity):
        self.data = []
        self.max_capacity = max_capacity
    
    def push(self, *args):
        transition = Transition(*args)
        self.data.append(transition)
        if len(self.data) > self.max_capacity:
            self.data = self.data[1:]

    def sample(self, num_samples):
        return random.sample(self.data, num_samples)

    def save(self, filename):
        torch.save(self.data, filename)
        #dprint(f"ReplayMemory: Saved {len(self.data)} transitions")
    
    def load(self, filename):
        self.data = torch.load(filename)
        #dprint(f"ReplayMemory: Loaded {len(self.data)} transitions")
    
    def __len__(self):
        return len(self.data)

def calc_state_value(state):
    own_mask = (state[:, 0] == 1)
    n_areas = sum(own_mask)
    n_dice = sum(state[own_mask, 1])
    value = n_areas + n_dice
    return value

def encode_action(action):
    if not action.src and not action.dst:
        return ACTION_END_TURN
    action_num = ((action.src - 1) * NUM_AREAS) + (action.dst - 1)
    if action.end_turn:
        action_num += END_TURN_OFFSET
    return action_num
    
def decode_action(action_num):
    if action_num == ACTION_END_TURN:
        return Action(None, None, True)
    end_turn = False
    if action_num >= END_TURN_OFFSET:
        end_turn = True
        action_num -= END_TURN_OFFSET
    src = action_num // NUM_AREAS + 1
    dst = action_num % NUM_AREAS + 1
    return Action(src, dst, end_turn)

def is_end_turn(action):
    return not action.src

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(NUM_INPUTS, 1024)
        self.hidden_layer = nn.Linear(1024, 1024)
        self.output_layer = nn.Linear(1024, NUM_ACTIONS)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = F.relu(self.output_layer(x))
        return x

loss = None

def optimize_model():
    global loss
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    #print(transitions[0])
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                  dtype=torch.bool, device=device)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device=device)
    state_batch = torch.stack(batch.state).to(device=device)
    #print(batch.action)
    #print(batch.reward)
    action_batch = torch.stack(batch.action).to(device=device)
    reward_batch = torch.stack(batch.reward).to(device=device)
    # values by the policy net of the actions taken
    #print(state_batch.shape)
    state_action_values = policy_net(state_batch)
    #print(state_action_values)
    state_action_values = state_action_values.gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # expected values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #print("LOSS", loss)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        # ???
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.state = None
        self.action = None
        self.end_turn_queued = False
        self.players = players_order
        self.n_moves = 0
    
    def get_state(self, board):
        return torch.tensor([
            (int(area.get_owner_name() == self.player_name), area.dice) 
            for area in board.areas.values()
        ])
    
    def get_possible_actions(self, board):
        actions = [ACTION_END_TURN]
        for src, dst in possible_attacks(board, self.player_name):
            actions.append(encode_action(Action(src.name, dst.name, False)))
            actions.append(encode_action(Action(src.name, dst.name, True)))
        return actions
    
    def choose_action(self, board):
        possible_actions = self.get_possible_actions(board)
        action = random.choice(possible_actions)
        action = decode_action(action)

        possible_actions = self.get_possible_actions(board)

        var.steps_done += 1        
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * var.steps_done / EPS_DECAY)
        use_model = random.random() > eps_threshold
        action = None
        if use_model:
            #print("using model")
            with torch.no_grad():
                state = self.get_state(board).flatten().float()
                t = policy_net(state)
                inv_actions_mask = torch.ones(NUM_ACTIONS, device=device, dtype=torch.bool)
                inv_actions_mask[possible_actions] = 0
                t[inv_actions_mask] = -1000
                action = t.max(0)[1].item()
        else:
            #print("using random")
            action = random.choice(possible_actions)
        #print(f"possible={possible_actions}, chosen={action}")
        optimize_model()
        return decode_action(action)
    
    def push_transition(self, state, action, next_state, reward):
        memory.push(
            state.flatten().float(),
            torch.tensor([encode_action(action)]),
            next_state.flatten().float() if next_state is not None else None,
            torch.tensor([reward])
        )

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """
            print("AI TURN")
            print(" Area counts: ", end="")
            for player in self.players:
                n_areas = sum(area.get_owner_name() == player for area in board.areas.values())
                print(f"{player}:{n_areas}  ", end="")
            print("")
        """
        self.n_moves += 1
        if self.end_turn_queued:
            self.end_turn_queued = False
            # print(" End turn")
            return EndTurnCommand()

        next_state = self.get_state(board)
        if self.state is not None and not self.end_turn_queued:
            reward = calc_state_value(next_state) - calc_state_value(self.state)
            self.push_transition(self.state, self.action, next_state, reward)
        self.state = next_state

        self.action = self.choose_action(board)
        if is_end_turn(self.action):
            # print(" End turn")
            return EndTurnCommand()
        if self.action.end_turn:
            self.end_turn_queued = True
        # print(f" Attack {self.action.src} -> {self.action.dst}")
        return BattleCommand(self.action.src, self.action.dst)

    def game_end(self, winner_name):
        win_lose = "WIN" if self.player_name == winner_name else "LOSE"
        print(f"Episode {var.episode_num} ({var.steps_done} steps): {win_lose} in {self.n_moves} (last training loss {loss})")
        reward = WIN_REWARD if self.player_name == winner_name else LOSE_REWARD
        self.push_transition(self.state, self.action, None, reward)
    
    def finalize(self):
        #print("Saving!")
        if os.path.exists(MEMORY_SAVE_FILE):
            os.rename(MEMORY_SAVE_FILE, MEMORY_SAVE_FILE + ".bak")
        memory.save(MEMORY_SAVE_FILE)
        if os.path.exists(VAR_SAVE_FILE):
            os.rename(VAR_SAVE_FILE, VAR_SAVE_FILE + ".bak")
        var.save(VAR_SAVE_FILE)
        if os.path.exists(POLICY_MODEL_SAVE_FILE):
            os.rename(POLICY_MODEL_SAVE_FILE, POLICY_MODEL_SAVE_FILE + ".bak")
            torch.save(policy_net.state_dict(), POLICY_MODEL_SAVE_FILE)

class TrainingVars:
    def __init__(self):
        self.steps_done = 0
        self.episode_num = 0

    def save(self, filename):
        torch.save(self, filename)
    
    @staticmethod
    def load(filename):
        return torch.load(filename)

memory = ReplayMemory(MEMORY_MAX_CAPACITY)
if os.path.exists(MEMORY_SAVE_FILE):
    memory.load(MEMORY_SAVE_FILE)
policy_net = QNet()
if os.path.exists(POLICY_MODEL_SAVE_FILE):
    policy_net.load_state_dict(torch.load(POLICY_MODEL_SAVE_FILE))
target_net = QNet()
if os.path.exists(TARGET_MODEL_SAVE_FILE):
    target_net.load_state_dict(torch.load(TARGET_MODEL_SAVE_FILE))
optimizer = optim.RMSprop(policy_net.parameters())
var = TrainingVars()
if os.path.exists(VAR_SAVE_FILE):
    var = TrainingVars.load(VAR_SAVE_FILE)

var.episode_num += 1
if var.episode_num % TARGET_UPDATE == 0:
    print("Updating target net")
    target_net.load_state_dict(policy_net.state_dict())
    if os.path.exists(TARGET_MODEL_SAVE_FILE):
        os.rename(TARGET_MODEL_SAVE_FILE, TARGET_MODEL_SAVE_FILE + ".bak")
    torch.save(target_net.state_dict(), TARGET_MODEL_SAVE_FILE)

