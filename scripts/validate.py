"""Standalone validation script — runs core tests using Python's unittest.

Requires only pettingzoo + numpy (already in the venv).
Run with: uv run python scripts/validate.py
"""

import sys
import unittest

sys.path.insert(0, "/root/Scouter")

from scouter.env.card import (
    Card, build_deck, build_2p_deck, hand_to_array, array_to_hand,
    _REMOVED_2P_CARD,
)
from scouter.env.game_logic import (
    classify_set, is_stronger, compute_round_scores,
    encode_show, decode_show, encode_scout, decode_scout,
    valid_show_slices, valid_scout_insertions,
    MAX_ACTIONS, N_SHOW, N_SCOUT,
)
from scouter.env.scout_env import ScoutEnv, AGENTS

import numpy as np


# ─────────────────────────────────────────────────────────────
# Card tests
# ─────────────────────────────────────────────────────────────

class TestCard(unittest.TestCase):
    def test_deck_size(self):
        self.assertEqual(len(build_deck()), 45)

    def test_2p_deck_size(self):
        self.assertEqual(len(build_2p_deck()), 44)

    def test_2p_removed_card_absent(self):
        ra, rb = _REMOVED_2P_CARD
        for c in build_2p_deck():
            self.assertFalse(c.a == ra and c.b == rb)

    def test_card_value_unflipped(self):
        c = Card(3, 7)
        self.assertEqual(c.value, 3)

    def test_card_value_flipped(self):
        c = Card(3, 7, flipped=True)
        self.assertEqual(c.value, 7)

    def test_card_flipped_copy(self):
        c = Card(2, 9)
        fc = c.flipped_copy()
        self.assertEqual(fc.value, 9)
        self.assertEqual(c.value, 2)  # original unchanged

    def test_card_invalid_same_values(self):
        with self.assertRaises(ValueError):
            Card(5, 5)

    def test_card_invalid_out_of_order(self):
        with self.assertRaises(ValueError):
            Card(8, 3)

    def test_array_roundtrip(self):
        c = Card(4, 10, flipped=True)
        self.assertEqual(Card.from_array(c.to_array()), c)

    def test_hand_to_array_padding(self):
        hand = [Card(1, 2), Card(3, 8, flipped=True)]
        arr = hand_to_array(hand, max_size=5)
        self.assertEqual(arr.shape, (5, 3))
        self.assertEqual(list(arr[2]), [0, 0, 0])


# ─────────────────────────────────────────────────────────────
# Game logic tests
# ─────────────────────────────────────────────────────────────

class TestGameLogic(unittest.TestCase):
    def test_classify_empty(self):
        self.assertEqual(classify_set([]), "invalid")

    def test_classify_single(self):
        self.assertEqual(classify_set([5]), "match")

    def test_classify_match(self):
        self.assertEqual(classify_set([4, 4, 4]), "match")

    def test_classify_run_asc(self):
        self.assertEqual(classify_set([3, 4, 5]), "run")

    def test_classify_run_desc(self):
        self.assertEqual(classify_set([7, 6, 5]), "run")

    def test_classify_invalid(self):
        self.assertEqual(classify_set([1, 3, 5]), "invalid")

    def test_stronger_empty_active(self):
        self.assertTrue(is_stronger([5], []))

    def test_stronger_more_cards_wins(self):
        self.assertTrue(is_stronger([1, 2, 3], [4, 5]))

    def test_stronger_match_beats_run(self):
        self.assertTrue(is_stronger([3, 3], [4, 5]))

    def test_stronger_higher_min_wins(self):
        self.assertTrue(is_stronger([5, 6, 7], [3, 4, 5]))

    def test_stronger_tie_false(self):
        self.assertFalse(is_stronger([4, 5, 6], [4, 5, 6]))

    def test_encode_decode_show(self):
        for start in range(11):
            for end in range(start, 11):
                a = encode_show(start, end)
                s, e = decode_show(a)
                self.assertEqual((s, e), (start, end))

    def test_encode_decode_scout(self):
        for side in (0, 1):
            for flip in (0, 1):
                for ins in range(12):
                    a = encode_scout(side, flip, ins)
                    s, f, i = decode_scout(a)
                    self.assertEqual((s, f, i), (side, flip, ins))

    def test_max_actions_constant(self):
        self.assertEqual(MAX_ACTIONS, N_SHOW + N_SCOUT)
        # MAX_HAND=14 (11 dealt + up to 3 scouted), so:
        self.assertEqual(N_SHOW, 196)   # 14*14
        self.assertEqual(N_SCOUT, 60)   # 2*2*15
        self.assertEqual(MAX_ACTIONS, 256)

    def test_score_basic(self):
        scores = compute_round_scores(
            collected={"p0": 5, "p1": 3},
            hands={"p0": [Card(1,2), Card(3,4)], "p1": [Card(5,6)]*4},
            chips={"p0": 2, "p1": 1},
            round_ender=None,
        )
        self.assertEqual(scores["p0"], 5)   # 5 - 2 + 2
        self.assertEqual(scores["p1"], 0)   # 3 - 4 + 1

    def test_score_ender_exempt(self):
        scores = compute_round_scores(
            collected={"p0": 4, "p1": 6},
            hands={"p0": [], "p1": [Card(1,2)]*3},
            chips={"p0": 1, "p1": 2},
            round_ender="p0",
        )
        self.assertEqual(scores["p0"], 5)   # 4 + 1 (no penalty)
        self.assertEqual(scores["p1"], 5)   # 6 - 3 + 2


# ─────────────────────────────────────────────────────────────
# AEC environment tests
# ─────────────────────────────────────────────────────────────

class TestScoutEnv(unittest.TestCase):
    def setUp(self):
        self.env = ScoutEnv()
        self.env.reset(seed=42)

    def test_agents(self):
        self.assertEqual(set(self.env.agents), set(AGENTS))

    def test_hand_sizes(self):
        for a in AGENTS:
            self.assertEqual(len(self.env._hands[a]), 11)

    def test_obs_in_space(self):
        for a in AGENTS:
            obs = self.env.observe(a)
            self.assertTrue(
                self.env.observation_space(a).contains(obs),
                f"Obs for {a} not in obs space"
            )

    def test_first_move_show_only(self):
        agent = self.env.agent_selection
        obs = self.env.observe(agent)
        mask = obs["action_mask"]
        self.assertFalse(mask[N_SHOW:].any(), "Scout actions should be masked at start")

    def test_invalid_action_raises(self):
        agent = self.env.agent_selection
        obs = self.env.observe(agent)
        mask = obs["action_mask"]
        invalid = np.where(mask == 0)[0]
        if len(invalid) == 0:
            self.skipTest("No invalid actions")
        with self.assertRaises(ValueError):
            self.env.step(int(invalid[0]))

    def test_obs_has_action_mask(self):
        obs, rew, term, trunc, info = self.env.last()
        self.assertIn("action_mask", obs)
        self.assertEqual(obs["action_mask"].shape, (MAX_ACTIONS,))

    def test_state_method_stable_shape(self):
        shape1 = self.env.state().shape
        agent = self.env.agent_selection
        obs = self.env.observe(agent)
        valid = np.where(obs["action_mask"])[0]
        self.env.step(int(valid[0]))
        shape2 = self.env.state().shape
        self.assertEqual(shape1, shape2)

    def test_pettingzoo_api(self):
        from pettingzoo.test import api_test
        env = ScoutEnv()
        api_test(env, num_cycles=50, verbose_progress=False)

    def test_50_random_games(self):
        for seed in range(50):
            env = ScoutEnv()
            env.reset(seed=seed)
            steps = 0
            while env.agents:
                agent = env.agent_selection
                if env.terminations[agent] or env.truncations[agent]:
                    env.step(None)
                else:
                    obs = env.observe(agent)
                    mask = obs["action_mask"]
                    valid = np.where(mask)[0]
                    self.assertGreater(len(valid), 0, f"No valid actions (seed={seed}, step={steps})")
                    env.step(int(np.random.choice(valid)))
                steps += 1
                self.assertLess(steps, 2000, f"Game {seed} did not end")

    def test_terminations_set_at_game_end(self):
        env = ScoutEnv()
        env.reset(seed=0)
        last_infos = {}
        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                # Capture info before dead step removes this agent
                last_infos[agent] = dict(env.infos.get(agent, {}))
                env.step(None)
            else:
                obs = env.observe(agent)
                valid = np.where(obs["action_mask"])[0]
                env.step(int(np.random.choice(valid)))
        # After all dead steps, agents is empty
        self.assertEqual(env.agents, [])
        # Final scores must have been captured in last_infos
        for a in AGENTS:
            self.assertIn("final_scores", last_infos[a])
            self.assertIn("winner", last_infos[a])


# ─────────────────────────────────────────────────────────────
# RL interface tests (no real LLM)
# ─────────────────────────────────────────────────────────────

class TestRLInterfaces(unittest.TestCase):
    def _get_obs(self, seed=0):
        env = ScoutEnv()
        env.reset(seed=seed)
        agent = env.agent_selection
        return env.observe(agent), agent, env

    def test_format_observation_returns_string(self):
        from scouter.rl.obs_formatter import format_observation
        obs, agent, env = self._get_obs()
        prompt = format_observation(obs, agent)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)
        self.assertIn(agent, prompt)

    def test_format_observation_legal_actions_listed(self):
        from scouter.rl.obs_formatter import format_observation
        obs, agent, env = self._get_obs()
        prompt = format_observation(obs, agent)
        self.assertIn("SHOW", prompt)

    def test_parse_show_json(self):
        from scouter.rl.action_parser import parse_action
        import json
        text = json.dumps({"action": "show", "start": 1, "end": 3})
        action = parse_action(text, hand_size=11)
        self.assertEqual(action, encode_show(0, 2))

    def test_parse_scout_json(self):
        from scouter.rl.action_parser import parse_action
        import json
        text = json.dumps({"action": "scout", "side": "left", "flip": False, "insert": 3})
        action = parse_action(text, hand_size=11)
        self.assertEqual(action, encode_scout(0, 0, 2))

    def test_parse_invalid_returns_minus_one(self):
        from scouter.rl.action_parser import parse_action
        action = parse_action("I have no idea", hand_size=11)
        self.assertEqual(action, -1)

    def test_parse_regex_show(self):
        from scouter.rl.action_parser import parse_action
        action = parse_action("I will show cards 2-4", hand_size=11)
        self.assertEqual(action, encode_show(1, 3))

    def test_grpo_dataset_structure(self):
        from scouter.rl.grpo_rollout import collect_grpo_dataset
        def dummy_policy(prompt):
            return '{"action": "show", "start": 1, "end": 1}'
        records = collect_grpo_dataset(num_episodes=2, policy_fn=dummy_policy, seed=0)
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)
        for r in records:
            self.assertIn("prompt", r)
            self.assertIn("completion", r)
            self.assertIn("reward", r)
            self.assertIsInstance(r["reward"], float)

    def test_grpo_custom_reward(self):
        from scouter.rl.grpo_rollout import collect_grpo_dataset
        def policy(p): return '{"action": "show", "start": 1, "end": 1}'
        def reward_fn(info): return 99.0
        records = collect_grpo_dataset(1, policy, reward_fn=reward_fn, seed=1)
        self.assertTrue(all(r["reward"] == 99.0 for r in records))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestCard, TestGameLogic, TestScoutEnv, TestRLInterfaces]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
