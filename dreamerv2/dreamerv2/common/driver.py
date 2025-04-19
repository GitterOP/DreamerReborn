import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._act_spaces = [env.act_space for env in envs]
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      obs = {
          i: self._envs[i].reset()
          for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
      for i, ob in obs.items():
        self._obs[i] = ob() if callable(ob) else ob
        act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
        self._eps[i] = [tran]
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      actions, self._state = policy(obs, self._state, **self._kwargs)
      #print(f"[DEBUG Driver.__call__] Policy output 'actions' type: {type(actions)}")
      if isinstance(actions, dict):
          #print(f"  Keys: {list(actions.keys())}")
          first_key = list(actions.keys())[0]
          action_example = actions[first_key]
          is_batched = hasattr(action_example, 'shape') and len(action_example.shape) > len(self._act_spaces[0][first_key].shape)
          #print(f"  Inferred is_batched: {is_batched} (based on shape {getattr(action_example, 'shape', 'N/A')})")

          #for k, v in actions.items():
              #print(f"  Key '{k}': type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")
      else:
          #print(f"  Actions is not a dict, type: {type(actions)}")
          is_batched = hasattr(actions, 'shape') and len(actions.shape) > len(self._act_spaces[0]['action'].shape)
          action_example = actions

      if isinstance(actions, tuple):
        actions, self._state = actions

      processed_actions = []
      num_envs = len(self._envs)
      for i in range(num_envs):
          env_action = {}
          for k, v in actions.items():
              if is_batched:
                  action_for_env = v[i]
              else:
                  assert num_envs == 1, "Policy returned unbatched action but num_envs > 1"
                  action_for_env = v
              env_action[k] = np.array(action_for_env)
          processed_actions.append(env_action)

      actions_per_env = processed_actions
      #print(f"[DEBUG Driver.__call__] Processed 'actions_per_env' (first env): type={type(actions_per_env[0])}")
      if isinstance(actions_per_env[0], dict):
         pass
          #print(f"  Keys: {list(actions_per_env[0].keys())}")
          #for k, v in actions_per_env[0].items():
              #print(f"  Key '{k}': type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")

      assert len(actions_per_env) == len(self._envs)
      obs = [e.step(a) for e, a in zip(self._envs, actions_per_env)]
      obs = [ob() if callable(ob) else ob for ob in obs]
      for i, (act, ob) in enumerate(zip(actions_per_env, obs)):
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
        self._eps[i].append(tran)
        step += 1
        if ob['is_last']:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          [fn(ep, **self._kwargs) for fn in self._on_episodes]
          episode += 1
      self._obs = obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value
