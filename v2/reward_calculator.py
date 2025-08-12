import numpy as np
from global_map import local_to_global, GLOBAL_MAP_SHAPE

class RewardCalculator:
    def __init__(self, env):
        self.env = env
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.total_healing_rew = 0
        self.died_count = 0
        self.last_health = 1
        self.max_map_progress = 0
        self.base_event_flags = sum([
            self.env.bit_count(self.env.read_m(i))
            for i in range(self.env.event_flags_start, self.env.event_flags_end)
        ])

    def get_game_state_reward(self):
        state_scores = {
            "event": self.env.reward_scale * self.update_max_event_rew() * 4,
            "heal": self.env.reward_scale * self.total_healing_rew * 10,
            "badge": self.env.reward_scale * self.get_badges() * 10,
            "explore": self.env.reward_scale * self.env.explore_weight * len(self.env.seen_coords) * 0.1,
            "stuck": self.env.reward_scale * self.get_current_coord_count_reward() * -0.05
        }
        return state_scores

    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
            max([
                self.env.read_m(a)
                for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
            ])
            - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health and self.env.read_m(0xD163) == self.env.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1
        self.last_health = cur_health

    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.env.read_m(start) + self.env.read_m(start + 1)

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.env.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_badges(self):
        return self.env.bit_count(self.env.read_m(0xD356))

    def get_all_events_reward(self):
        return max(
            sum([
                self.env.bit_count(self.env.read_m(i))
                for i in range(self.env.event_flags_start, self.env.event_flags_end)
            ])
            - self.base_event_flags
            - int(self.env.read_bit(self.env.museum_ticket[0], self.env.museum_ticket[1])),
            0,
        )

    def get_current_coord_count_reward(self):
        x_pos, y_pos, map_n = self.env.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.env.seen_coords.keys():
            count = self.env.seen_coords[coord_string]
        else:
            count = 0
        return 0 if count < 600 else 1

    def update_map_progress(self):
        map_idx = self.env.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.env.essential_map_locations.keys():
            return self.env.essential_map_locations[map_idx]
        else:
            return -1
