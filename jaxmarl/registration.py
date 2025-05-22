from .environments import (
    SimpleMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
    SimpleSpreadMPE,
    SimpleCryptoMPE,
    SimpleSpeakerListenerMPE,
    SimpleFacmacMPE,
    SimpleFacmacMPE3a,
    SimpleFacmacMPE6a,
    SimpleFacmacMPE9a,
    SimplePushMPE,
    SimpleAdversaryMPE,
    SimpleReferenceMPE,
    SMAX,
    HeuristicEnemySMAX,
    LearnedPolicyEnemySMAX,
    SwitchRiddle,
    Ant,
    Humanoid,
    Hopper,
    Walker2d,
    HalfCheetah,
    InTheGrid,
    InTheGrid_2p,
    HanabiGame,
    Overcooked,
    CoinGame,
)



def make(env_id: str, **ENV_KWARGS):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    # 1. MPE PettingZoo Environments
    if env_id == "MPE_simple_v3":
        env = SimpleMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_tag_v3":
        env = SimpleTagMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_world_comm_v3":
        env = SimpleWorldCommMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_spread_v3":
        env = SimpleSpreadMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_crypto_v3":
        env = SimpleCryptoMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_speaker_listener_v4":
        env = SimpleSpeakerListenerMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_push_v3":
        env = SimplePushMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_adversary_v3":
        env = SimpleAdversaryMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_reference_v3":
        env = SimpleReferenceMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_facmac_v1":
        env = SimpleFacmacMPE(**ENV_KWARGS)
    elif env_id == "MPE_simple_facmac_3a_v1":
        env = SimpleFacmacMPE3a(**ENV_KWARGS)
    elif env_id == "MPE_simple_facmac_6a_v1":
        env = SimpleFacmacMPE6a(**ENV_KWARGS)
    elif env_id == "MPE_simple_facmac_9a_v1":
        env = SimpleFacmacMPE9a(**ENV_KWARGS)

    # 2. Switch Riddle
    elif env_id == "switch_riddle":
        env = SwitchRiddle(**ENV_KWARGS)

    # 3. SMAX
    elif env_id == "SMAX":
        env = SMAX(**ENV_KWARGS)
    elif env_id == "HeuristicEnemySMAX":
        env = HeuristicEnemySMAX(**ENV_KWARGS)
    elif env_id == "LearnedPolicyEnemySMAX":
        env = LearnedPolicyEnemySMAX(**ENV_KWARGS)

    # 4. MABrax
    elif env_id == "ant_4x2":
        env = Ant(**ENV_KWARGS)
    elif env_id == "halfcheetah_6x1":
        env = HalfCheetah(**ENV_KWARGS)
    elif env_id == "hopper_3x1":
        env = Hopper(**ENV_KWARGS)
    elif env_id == "humanoid_9|8":
        env = Humanoid(**ENV_KWARGS)
    elif env_id == "walker2d_2x3":
        env = Walker2d(**ENV_KWARGS)

    # 5. InTheGrid
    elif env_id == "storm":
        env = InTheGrid(**ENV_KWARGS)
    # 5. InTheGrid
    elif env_id == "storm_2p":
        env = InTheGrid_2p(**ENV_KWARGS)
    
    # 6. Hanabi
    elif env_id == "hanabi":
        env = HanabiGame(**ENV_KWARGS)

    # 7. Overcooked
    elif env_id == "overcooked":
        env = Overcooked(**ENV_KWARGS)

    # 8. Coin Game
    elif env_id == "coin_game":
        env = CoinGame(**ENV_KWARGS)


    # New envs
    elif env_id == "simple_spread_v3":
        from pettingzoo.mpe import simple_spread_v3
        env = simple_spread_v3.parallel_env(**ENV_KWARGS)
        env.agents = env.possible_agents

    elif "rware" in env_id:

        import rware
        import gym
        env = gym.make(env_id)
        env.num_agents = env.n_agents

    return env

registered_envs = [
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "switch_riddle",
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
    "storm",
    "storm_2p",
    "hanabi",
    "overcooked",
    "coin_game",
    # new
    "simple_spread_v3",
    "rware-small-4ag-v1"
]
