import atexit
import os
import sys
import threading
import traceback
import glob  # Added for template loading
import logging  # Added for detector logging

import cloudpickle
import gym
import numpy as np
import cv2  # Import OpenCV


class GymWrapper:

  def __init__(self, env, obs_key='image', act_key='action'):
    self._env = env
    #print(f"[DEBUG GymWrapper.__init__] Checking spaces on env type: {type(env)}")
    try:
        # Access the properties to ensure they exist and trigger __getattr__ if needed
        inner_obs_space = self._env.obs_space
        inner_act_space = self._env.act_space
        # Check if the *inner* spaces are dicts (or gym.spaces.Dict)
        self._obs_is_dict = isinstance(inner_obs_space, (dict, gym.spaces.Dict))
        self._act_is_dict = isinstance(inner_act_space, (dict, gym.spaces.Dict))
        #print(f"[DEBUG GymWrapper.__init__] inner_obs_space type: {type(inner_obs_space)}, inner_act_space type: {type(inner_act_space)}")
        #print(f"[DEBUG GymWrapper.__init__] _obs_is_dict={self._obs_is_dict}, _act_is_dict={self._act_is_dict}")
    except Exception as e:
        #print(f"[DEBUG GymWrapper.__init__] ERROR checking spaces: {e}")
        # Set defaults or re-raise depending on desired robustness
        self._obs_is_dict = False
        self._act_is_dict = False
        #print(f"[DEBUG GymWrapper.__init__] Falling back: _obs_is_dict={self._obs_is_dict}, _act_is_dict={self._act_is_dict}")
        # Potentially re-raise the error if it's critical: raise e
    self._obs_key = obs_key
    self._act_key = act_key
    #print(f"[DEBUG GymWrapper.__init__] Init complete for wrapper around {type(env)}")

  def __getattr__(self, name):
    #print(f"[DEBUG GymWrapper.__getattr__] Trying to get attribute '{name}' from {type(self)}")
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      #print(f"[DEBUG GymWrapper.__getattr__] Accessing '{name}' on inner env of type: {type(self._env)}")
      return getattr(self._env, name)
    except AttributeError:
      #print(f"[DEBUG GymWrapper.__getattr__] AttributeError: Inner env {type(self._env)} has no attribute '{name}'")
      raise ValueError(name)

  @property
  def obs_space(self):
    #print(f"[DEBUG GymWrapper.obs_space] Property called on {type(self)}. _obs_is_dict={self._obs_is_dict}")
    try:
        env_obs_space = self._env.obs_space
        #print(f"[DEBUG GymWrapper.obs_space] Accessed self._env.obs_space: type={type(env_obs_space)}")
        # Handle both gym.spaces.Dict and regular dict
        if isinstance(env_obs_space, gym.spaces.Dict):
            spaces = env_obs_space.spaces.copy()
            #print(f"[DEBUG GymWrapper.obs_space] Copied spaces from gym.spaces.Dict: keys={list(spaces.keys())}")
        elif isinstance(env_obs_space, dict):
             spaces = env_obs_space.copy()
             #print(f"[DEBUG GymWrapper.obs_space] Copied spaces from dict: keys={list(spaces.keys())}")
        else:
            spaces = {self._obs_key: env_obs_space}
            #print(f"[DEBUG GymWrapper.obs_space] Created space dict with key '{self._obs_key}'")

        # Add standard wrapper keys if they don't exist
        spaces.setdefault('reward', gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32))
        spaces.setdefault('is_first', gym.spaces.Box(0, 1, (), dtype=np.bool_)) # Use np.bool_
        spaces.setdefault('is_last', gym.spaces.Box(0, 1, (), dtype=np.bool_)) # Use np.bool_
        spaces.setdefault('is_terminal', gym.spaces.Box(0, 1, (), dtype=np.bool_)) # Use np.bool_
        #print(f"[DEBUG GymWrapper.obs_space] Returning final gym.spaces.Dict: keys={list(spaces.keys())}")
        # Always return a gym.spaces.Dict
        return gym.spaces.Dict(spaces)
    except Exception as e:
        #print(f"[DEBUG GymWrapper.obs_space] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e

  @property
  def act_space(self):
    #print(f"[DEBUG GymWrapper.act_space] Property called on {type(self)}. _act_is_dict={self._act_is_dict}")
    try:
        env_act_space = self._env.act_space
        #print(f"[DEBUG GymWrapper.act_space] Accessed self._env.act_space: type={type(env_act_space)}")
        # If the inner env returns a dict (like OneHotAction does), return it directly.
        # Also handle if it returns a gym.spaces.Dict
        if isinstance(env_act_space, gym.spaces.Dict):
            spaces = env_act_space.spaces.copy()
            #print(f"[DEBUG GymWrapper.act_space] Returning spaces copied from inner gym.spaces.Dict: keys={list(spaces.keys())}")
            return spaces
        elif isinstance(env_act_space, dict):
            spaces = env_act_space.copy()
            #print(f"[DEBUG GymWrapper.act_space] Returning spaces copied from inner dict: keys={list(spaces.keys())}")
            return spaces # Return the dict directly
        else:
            # If the inner env returns a single space, wrap it in a dict
            space_dict = {self._act_key: env_act_space}
            #print(f"[DEBUG GymWrapper.act_space] Created and returning space dict with key '{self._act_key}'")
            return space_dict
    except Exception as e:
        #print(f"[DEBUG GymWrapper.act_space] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e

  def step(self, action):
    #print(f"[DEBUG GymWrapper.step] Received action type: {type(action)}")
    original_action = action # Keep original for debugging
    if isinstance(action, dict):
       pass
        #print(f"  Keys: {list(action.keys())}")
        #for k, v in action.items():
            #print(f"  Key '{k}': type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")

    action_to_pass = action # Default to passing the received action
    if not self._act_is_dict:
      #print(f"[DEBUG GymWrapper.step] Action is NOT dict type for inner env. Extracting key '{self._act_key}'")
      if self._act_key not in action:
          #print(f"[ERROR GymWrapper.step] Action key '{self._act_key}' not found in action dict!")
          raise KeyError(f"Action key '{self._act_key}' not found in action dict: {action.keys()}")
      action_to_pass = action[self._act_key] # Use a different variable
      #print(f"[DEBUG GymWrapper.step] Extracted action type: {type(action_to_pass)}, shape: {getattr(action_to_pass, 'shape', 'N/A')}")
    else:
       pass
       #print(f"[DEBUG GymWrapper.step] Action IS dict type for inner env. Passing as is.")

    # Pass the potentially modified action to the inner env
    #print(f"[DEBUG GymWrapper.step] Calling self._env.step() on {type(self._env)} with action type: {type(action_to_pass)}")
    # --- CORRECTED MODIFICATION START ---
    inner_step_return = self._env.step(action_to_pass)
    #print(f"[DEBUG GymWrapper.step] Inner env step returned type: {type(inner_step_return)}")

    # Check if the inner env returned a dict (like Atari) or a tuple (standard Gym)
    if isinstance(inner_step_return, dict):
        #print(f"[DEBUG GymWrapper.step] Inner env returned dict. Extracting info.")
        # The whole dict is the observation, potentially containing reward, done, etc.
        obs = inner_step_return
        # Extract reward, done, info from the dict, using defaults if keys are missing
        reward = obs.get('reward')
        # Use 'is_last' as the primary indicator for done, fallback to False
        done = obs.get('is_last')
        # Create an info dict, including 'is_terminal' if available
        info = {'is_terminal': obs.get('is_terminal', done)}
        #print(f"[DEBUG GymWrapper.step] Extracted from dict: reward={reward}, done={done}, info={info}")
    elif isinstance(inner_step_return, tuple) and len(inner_step_return) == 4:
        #print(f"[DEBUG GymWrapper.step] Inner env returned tuple. Unpacking.")
        obs, reward, done, info = inner_step_return
        # Ensure info is a dict, create one if it's None
        if info is None:
            info = {}
        #print(f"[DEBUG GymWrapper.step] Unpacked tuple: reward={reward}, done={done}, info={info}")
    else:
        # Add more details to the error message
        #print(f"[ERROR GymWrapper.step] Inner env {type(self._env)} returned unexpected value from step():")
        #print(f"  Type: {type(inner_step_return)}")
        #print(f"  Value: {inner_step_return}")
        raise TypeError(f"Inner env {type(self._env)} returned unexpected type or structure from step()")
    # --- CORRECTED MODIFICATION END ---


    # Process observation (obs is now either the dict from inner_step_return or the first element of the tuple)
    #print(f"[DEBUG GymWrapper.step] Processing obs: type={type(obs)}")
    if not self._obs_is_dict:
      # This case happens if the inner env returned a tuple AND its obs part wasn't a dict
      #print(f"[DEBUG GymWrapper.step] Inner obs is NOT dict. Wrapping with key '{self._obs_key}'")
      obs = {self._obs_key: obs}
    else:
       # This case happens if the inner env returned a dict OR returned a tuple where the obs part was a dict
       # Ensure it's a mutable dict for adding keys
       if isinstance(obs, gym.spaces.Dict): # Convert Gym Dict to regular dict
           obs = obs.spaces.copy()
           #print(f"[DEBUG GymWrapper.step] Converted inner gym.spaces.Dict obs to regular dict.")
       elif isinstance(obs, dict):
           obs = obs.copy() # Ensure we have a mutable copy
           #print(f"[DEBUG GymWrapper.step] Copied inner dict obs.")
       else:
           # This should not happen if _obs_is_dict is True, but handle defensively
            #print(f"[WARNING GymWrapper.step] _obs_is_dict is True, but obs is type {type(obs)}. Wrapping with key '{self._obs_key}'.")
            obs = {self._obs_key: obs}


    # Add standard keys, potentially overwriting if they came from the inner dict
    obs['reward'] = float(reward)
    obs['is_first'] = False
    obs['is_last'] = bool(done) # Ensure boolean type
    obs['is_terminal'] = bool(info.get('is_terminal', done)) # Ensure boolean type
    #print(f"[DEBUG GymWrapper.step] Returning final obs dict with keys: {list(obs.keys())}")
    return obs

  def reset(self):
    obs = self._env.reset()
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = 0.0
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    return obs


class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    os.environ['MUJOCO_GL'] = 'egl'
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if domain == 'manip':
      from dm_control import manipulation
      self._env = manipulation.load(task + '_vision')
    elif domain == 'locom':
      from dm_control.locomotion.examples import basic_rodent_2020
      self._env = getattr(basic_rodent_2020, task)()
    else:
      from dm_control import suite
      self._env = suite.load(domain, task)
    self._action_repeat = action_repeat
    self._size = size
    if camera in (-1, None):
      camera = dict(
          quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
          quadruped_fetch=2, locom_rodent_maze_forage=1,
          locom_rodent_two_touch=1,
      ).get(name, 0)
    self._camera = camera
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      if value.shape == (0,):
        #print(f"Ignoring empty observation key '{key}'.")
        self._ignored_keys.append(key)

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
    }
    for key, value in self._env.observation_spec().items():
      if key in self._ignored_keys:
        continue
      if value.dtype == np.float64:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
      elif value.dtype == np.uint8:
        spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
      else:
        raise NotImplementedError(value.dtype)
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return {'action': action}

  def step(self, action):
    assert np.isfinite(action['action']).all(), action['action']
    reward = 0.0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action['action'])
      reward += time_step.reward or 0.0
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    obs = {
        'reward': reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    self._env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._size = size
    self._grayscale = grayscale

  @property
  def obs_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    return {
        'image': gym.spaces.Box(0, 255, shape, np.uint8),
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
    }

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def step(self, action):
    image, reward, done, info = self._env.step(action['action'])
    if self._grayscale:
      image = image[..., None]
    return {
        'image': image,
        'ram': self._env.env._get_ram(),
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': done,
    }

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    return {
        'image': image,
        'ram': self._env.env._get_ram(),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }

  def close(self):
    return self._env.close()


class Crafter:

  def __init__(self, outdir=None, reward=True, seed=None):
    import crafter
    self._env = crafter.Env(reward=reward, seed=seed)
    self._env = crafter.Recorder(
        self._env, outdir,
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    self._achievements = crafter.constants.achievements.copy()

  @property
  def obs_space(self):
    spaces = {
        'image': self._env.observation_space,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        'log_reward': gym.spaces.Box(-np.inf, np.inf, (), np.float32),
    }
    spaces.update({
        f'log_achievement_{k}': gym.spaces.Box(0, 2 ** 31 - 1, (), np.int32)
        for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def step(self, action):
    image, reward, done, info = self._env.step(action['action'])
    obs = {
        'image': image,
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': info['discount'] == 0,
        'log_reward': info['reward'],
    }
    obs.update({
        f'log_achievement_{k}': v
        for k, v in info['achievements'].items()})
    return obs

  def reset(self):
    obs = {
        'image': self._env.reset(),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'log_reward': 0.0,
    }
    obs.update({
        f'log_achievement_{k}': 0
        for k in self._achievements})
    return obs


class Dummy:

  def __init__(self):
    pass

  @property
  def obs_space(self):
    return {
        'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    return {'action': gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

  def step(self, action):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': False,
        'is_last': False,
        'is_terminal': False,
    }

  def reset(self):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    #print(f"[DEBUG OneHotAction.__init__] Initializing with env type: {type(env)}")
    try:
        # Access the property to ensure it exists and is structured as expected
        inner_act_space = env.act_space
        #print(f"[DEBUG OneHotAction.__init__] Accessed inner env act_space: type={type(inner_act_space)}")
        if not isinstance(inner_act_space, (dict, gym.spaces.Dict)):
             raise TypeError(f"Expected inner action space to be a dict or gym.spaces.Dict, got {type(inner_act_space)}")
        if key not in inner_act_space:
             raise KeyError(f"Action key '{key}' not found in inner action space keys: {list(inner_act_space.keys())}")
        space_to_convert = inner_act_space[key]
        if not hasattr(space_to_convert, 'n'):
             raise AttributeError(f"Inner env's action space under key '{key}' (type: {type(space_to_convert)}) does not have 'n' attribute")
        #print(f"[DEBUG OneHotAction.__init__] Inner space '{key}' is Discrete with n={space_to_convert.n}")
    except Exception as e:
        #print(f"[DEBUG OneHotAction.__init__] ERROR checking action space: {e}")
        raise e
    self._env = env
    self._key = key
    self._random = np.random.RandomState()
    #print(f"[DEBUG OneHotAction.__init__] Init complete.")

  def __getattr__(self, name):
    #print(f"[DEBUG OneHotAction.__getattr__] Trying to get attribute '{name}' from {type(self)}")
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      #print(f"[DEBUG OneHotAction.__getattr__] Accessing '{name}' on inner env of type: {type(self._env)}")
      return getattr(self._env, name)
    except AttributeError:
      #print(f"[DEBUG OneHotAction.__getattr__] AttributeError: Inner env {type(self._env)} has no attribute '{name}'")
      raise ValueError(name)

  @property
  def act_space(self):
    #print(f"[DEBUG OneHotAction.act_space] Property called.")
    try:
        inner_act_space = self._env.act_space
        #print(f"[DEBUG OneHotAction.act_space] Inner act_space type: {type(inner_act_space)}")
        if isinstance(inner_act_space, gym.spaces.Dict):
            inner_act_space_dict = inner_act_space.spaces.copy()
            #print(f"[DEBUG OneHotAction.act_space] Copied spaces from inner gym.spaces.Dict. Keys: {list(inner_act_space_dict.keys())}")
        elif isinstance(inner_act_space, dict):
            inner_act_space_dict = inner_act_space.copy()
            #print(f"[DEBUG OneHotAction.act_space] Copied spaces from inner dict. Keys: {list(inner_act_space_dict.keys())}")
        else:
             raise TypeError(f"Unexpected inner action space type: {type(inner_act_space)}")
        original_discrete_space = inner_act_space_dict[self._key]
        shape = (original_discrete_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        new_act_space_dict = inner_act_space_dict.copy()
        new_act_space_dict[self._key] = space
        #print(f"[DEBUG OneHotAction.act_space] Returning modified dict: keys={list(new_act_space_dict.keys())}")
        return new_act_space_dict
    except Exception as e:
        #print(f"[DEBUG OneHotAction.act_space] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e

  def step(self, action):
    #print(f"[DEBUG OneHotAction.step] Received action type: {type(action)}")
    if isinstance(action, dict):
        #print(f"  Keys: {list(action.keys())}")
        #for k, v in action.items():
            #print(f"  Key '{k}': type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")
        action_tensor = action[self._key]
    else:
        # This case should ideally not happen if GymWrapper logic is correct
        #print(f"[WARNING OneHotAction.step] Received non-dict action: type={type(action)}. Assuming it's the tensor itself.")
        action_tensor = action

    #print(f"[DEBUG OneHotAction.step] action_tensor type: {type(action_tensor)}, shape: {getattr(action_tensor, 'shape', 'N/A')}, dtype: {getattr(action_tensor, 'dtype', 'N/A')}")

    # Ensure action_tensor is a numpy array for np.argmax
    if not isinstance(action_tensor, np.ndarray):
        try:
            action_tensor_np = np.array(action_tensor)
            #print(f"[DEBUG OneHotAction.step] Converted action_tensor to numpy array. New shape: {action_tensor_np.shape}")
        except Exception as e:
            #print(f"[ERROR OneHotAction.step] Failed to convert action_tensor to numpy array: {e}")
            raise TypeError(f"Could not convert action_tensor to numpy array. Original type: {type(action_tensor)}")
    else:
        action_tensor_np = action_tensor

    try:
        index = np.argmax(action_tensor_np).astype(int)
        #print(f"[DEBUG OneHotAction.step] Calculated index: {index}")
    except Exception as e:
        #print(f"[ERROR OneHotAction.step] Error during np.argmax: {e}")
        #print(f"  action_tensor_np: {action_tensor_np}")
        raise e

    try:
        reference = np.zeros_like(action_tensor_np)
        #print(f"[DEBUG OneHotAction.step] Created reference array shape: {reference.shape}, dtype: {reference.dtype}")
    except Exception as e:
        #print(f"[ERROR OneHotAction.step] Error during np.zeros_like: {e}")
        #print(f"  action_tensor_np: {action_tensor_np}")
        raise e

    # --- Failing line ---
    try:
        #print(f"[DEBUG OneHotAction.step] Attempting reference[index] = 1 with index={index}")
        reference[index] = 1
    except IndexError as e:
        #print(f"[FATAL OneHotAction.step] IndexError: {e}")
        #print(f"  reference shape: {reference.shape}")
        #print(f"  index: {index}")
        #print(f"  action_tensor_np shape: {action_tensor_np.shape}")
        raise e
    except Exception as e:
        #print(f"[ERROR OneHotAction.step] Unexpected error setting reference[index]: {e}")
        raise e
    # --- End Failing line ---

    #print(f"[DEBUG OneHotAction.step] Checking np.allclose...")
    if not np.allclose(reference, action_tensor_np):
      #print(f"[ERROR OneHotAction.step] Invalid one-hot action detected!")
      #print(f"  Original: {action_tensor_np}")
      #print(f"  Reference: {reference}")
      raise ValueError(f'Invalid one-hot action:\n{action}') # Keep original action dict in error

    # Pass discrete index to inner env
    inner_action = index
    #print(f"[DEBUG OneHotAction.step] Calling self._env.step() on {type(self._env)} with discrete action: {inner_action}")
    # If the inner env expects a dict, we need to reconstruct it
    # Assuming Atari expects {'action': index}
    inner_action_dict = {self._key: inner_action}
    return self._env.step(inner_action_dict)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.act_space[self._key].n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class PacmanDetectionAndResizeWrapper(GymWrapper):
    """
    Detects Pacman using template matching on input frames (assumed 128x128),
    generates a binary mask, and resizes both the original image and the mask
    to the final processing size (e.g., 64x64).
    """
    def __init__(self, env, template_path, threshold, process_size=(64, 64), image_key='image', mask_key='pacman_mask'):
        #print(f"[DEBUG PacmanWrapper.__init__] Initializing with env type: {type(env)}")
        #print(f"[DEBUG PacmanWrapper.__init__] image_key='{image_key}', mask_key='{mask_key}'")
        inner_obs_key = None
        try:
            #print(f"[DEBUG PacmanWrapper.__init__] Attempting getattr(env, '_obs_key', '{image_key}') on env type: {type(env)}")
            inner_obs_key = getattr(env, '_obs_key', image_key)  # Get inner obs key if possible
            #print(f"[DEBUG PacmanWrapper.__init__] getattr succeeded. inner_obs_key = '{inner_obs_key}'")
        except Exception as e:
            #print(f"[DEBUG PacmanWrapper.__init__] getattr FAILED. Error: {e}")
            # Fallback if getattr itself fails unexpectedly, though it shouldn't with a default
            inner_obs_key = image_key
            #print(f"[DEBUG PacmanWrapper.__init__] Falling back to inner_obs_key = '{inner_obs_key}'")

        #print(f"[DEBUG PacmanWrapper.__init__] Calling super().__init__ with obs_key='{inner_obs_key}'")
        # Initialize the parent GymWrapper. This sets self._env, self._obs_key, self._act_key,
        # self._obs_is_dict, self._act_is_dict based on the *wrapped* env (OneHotAction).
        super().__init__(env, obs_key=inner_obs_key)
        #print(f"[DEBUG PacmanWrapper.__init__] super().__init__ finished. self._obs_key is now '{self._obs_key}'")

        self._process_size = tuple(process_size)
        # Use the obs_key determined/passed during super().__init__
        self._image_key = self._obs_key # This should be 'image' based on fallback/default
        self._mask_key = mask_key
        self._threshold = threshold
        self.templates = []
        self.logger = logging.getLogger('PacmanDetector')
        self.logger.setLevel(logging.INFO)

        #print(f"[DEBUG PacmanWrapper.__init__] Determining original shape for detection...")
        try:
            parent_obs_space_dict = super().obs_space
            #print(f"[DEBUG PacmanWrapper.__init__] Accessed super().obs_space for original shape. Type: {type(parent_obs_space_dict)}, Keys: {list(parent_obs_space_dict.spaces.keys())}")

            if self._image_key not in parent_obs_space_dict.spaces:
                 raise KeyError(f"Image key '{self._image_key}' not found in parent observation space keys: {list(parent_obs_space_dict.spaces.keys())}")

            self._load_templates(template_path)
            if not self.templates:
                self.logger.warning(f"No templates loaded from {template_path}. Pacman mask will always be zeros.")
            #print(f"[DEBUG PacmanWrapper.__init__] Finished initialization.")
        except Exception as e:
            #print(f"[DEBUG PacmanWrapper.__init__] ERROR during detection size determination or template loading: {e}")
            import traceback
            traceback.print_exc()
            raise e

    @property
    def obs_space(self):
        #print(f"[DEBUG PacmanWrapper.obs_space] Property called on {type(self)}. self._image_key='{self._image_key}'")
        try:
            #print(f"[DEBUG PacmanWrapper.obs_space] Calling super().obs_space...")
            base_space_dict = super().obs_space
            #print(f"[DEBUG PacmanWrapper.obs_space] Received gym.spaces.Dict from super(): type={type(base_space_dict)}, keys={list(base_space_dict.spaces.keys())}")

            spaces = base_space_dict.spaces

            if self._image_key not in spaces:
                 raise KeyError(f"Image key '{self._image_key}' not found in base observation space keys: {list(spaces.keys())}")

            original_img_space = spaces[self._image_key]
            #print(f"[DEBUG PacmanWrapper.obs_space] Original image space from base dict: type={type(original_img_space)}, shape={original_img_space.shape}")
            img_shape = self._process_size + original_img_space.shape[2:]
            spaces[self._image_key] = gym.spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )
            #print(f"[DEBUG PacmanWrapper.obs_space] Modified image space shape: {img_shape}")

            mask_shape = self._process_size
            spaces[self._mask_key] = gym.spaces.Box(
                low=0, high=1, shape=mask_shape, dtype=np.uint8
            )
            #print(f"[DEBUG PacmanWrapper.obs_space] Added mask space '{self._mask_key}' shape: {mask_shape}")

            #print(f"[DEBUG PacmanWrapper.obs_space] Returning final gym.spaces.Dict: keys={list(spaces.keys())}")
            return gym.spaces.Dict(spaces)
        except Exception as e:
            #print(f"[DEBUG PacmanWrapper.obs_space] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _load_templates(self, template_path):
        if not template_path:
            self.logger.error("Template path is empty.")
            return

        template_path = os.path.expanduser(template_path)

        if not os.path.isabs(template_path):
            script_dir = os.path.dirname(__file__)
            base_dir = os.path.dirname(os.path.dirname(script_dir))
            potential_path = os.path.join(base_dir, template_path)
            if not os.path.exists(potential_path):
                potential_path_cwd = os.path.abspath(template_path)
                if os.path.exists(potential_path_cwd):
                    template_path = potential_path_cwd
                else:
                    self.logger.error(f"Cannot find template path: {template_path} (tried relative to project root and CWD)")
                    template_path = None
            else:
                template_path = potential_path

        if not template_path:
            return

        self.logger.info(f"Attempting to load templates from resolved path: {template_path}")

        if os.path.isdir(template_path):
            template_files = glob.glob(os.path.join(template_path, "*.png"))
            if not template_files:
                self.logger.warning(f"No *.png template files found in directory: {template_path}")
                return
            self.logger.info(f"Found {len(template_files)} potential template files in directory.")
            for file_path in sorted(template_files):
                self._load_single_template(file_path)
        elif os.path.isfile(template_path):
            self._load_single_template(template_path)
        else:
            self.logger.error(f"Template path exists but is neither a file nor a directory: {template_path}")

    def _load_single_template(self, file_path):
        """Helper method to load a single template file."""
        try:
            template = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                self.logger.warning(f"Failed to load template (imread returned None): {file_path}")
                return False
            self.templates.append(template)
            self.logger.info(f"Successfully loaded template: {os.path.basename(file_path)} with shape {template.shape}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading template {file_path}: {e}")
            return False

    def _detect_pacman(self, frame_detect_size):
        if not self.templates:
            return np.zeros(self._process_size, dtype=np.uint8)

        if len(frame_detect_size.shape) == 3 and frame_detect_size.shape[2] == 1:
            gray_frame = frame_detect_size[:, :, 0] # Remove channel dim
        elif len(frame_detect_size.shape) == 2:
            gray_frame = frame_detect_size # Already grayscale 2D
        else:
            # Attempt grayscale conversion if multiple channels (e.g., RGB input somehow)
            try:
                gray_frame = cv2.cvtColor(frame_detect_size, cv2.COLOR_RGB2GRAY) # Assume RGB
                self.logger.warning(f"Frame had unexpected shape {frame_detect_size.shape}, converted to grayscale.")
            except cv2.error:
                self.logger.error(f"Unsupported frame shape for detection: {frame_detect_size.shape}")
                return np.zeros(self._detection_size, dtype=np.uint8)

        if gray_frame.shape[0] > 103:
            gray_frame_cropped = gray_frame.copy() # Work on a copy to not modify original if needed elsewhere
            gray_frame_cropped[103:, :] = 0 # Set area below 103 to 0 (black)
            # Use the cropped frame for matching
            match_frame = gray_frame_cropped
        else:
             self.logger.warning(f"Frame height ({gray_frame.shape[0]}) is not > 103, cannot apply crop before detection.")
             match_frame = gray_frame # Use original grayscale frame if too short to crop
        # --- End Moved Crop Logic ---

        binary_mask = np.zeros_like(gray_frame, dtype=np.uint8) # Mask should still be the original size
        detection_found = False

        # Try each template until a match is found using the potentially cropped frame
        for template in self.templates:
            # Basic check if template fits within the frame
            if template.shape[0] > match_frame.shape[0] or template.shape[1] > match_frame.shape[1]:
                self.logger.warning(f"Skipping template {template.shape} larger than frame {match_frame.shape}")
                continue

            try:
                # Perform matching on the potentially cropped frame
                res = cv2.matchTemplate(match_frame, template, cv2.TM_CCOEFF_NORMED)
                _, max_confidence, _, max_loc = cv2.minMaxLoc(res)

                if max_confidence >= self._threshold:
                    # Create mask for this detection based on location found
                    w, h = template.shape[1], template.shape[0]
                    x, y = max_loc
                    # Reset mask and draw detection at the found location in the *original sized* mask
                    binary_mask = np.zeros_like(gray_frame, dtype=np.uint8)
                    binary_mask[y:y+h, x:x+w] = 1
                    detection_found = True
                    # Stop searching once we find a match
                    break
            except cv2.error as e:
                 self.logger.error(f"OpenCV error during matchTemplate: {e}. Frame shape: {match_frame.shape}, Template shape: {template.shape}")
                 continue # Skip this template if error occurs

        return binary_mask

    def step(self, action):
        obs = super().step(action)
        return self._process_obs(obs)

    def reset(self):
        obs = super().reset()
        return self._process_obs(obs)

    def _process_obs(self, obs):
        image_detect_size = obs[self._image_key]

        mask_detect_size = self._detect_pacman(image_detect_size)

        image_process_size = cv2.resize(
            image_detect_size,
            (self._process_size[1], self._process_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        if len(image_detect_size.shape) == 3 and len(image_process_size.shape) == 2:
            image_process_size = np.expand_dims(image_process_size, axis=-1)

        mask_process_size = cv2.resize(
            mask_detect_size,
            (self._process_size[1], self._process_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        obs[self._image_key] = image_process_size
        obs[self._mask_key] = mask_process_size

        return obs


class RenderImage:

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._shape = self._env.render().shape

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
    return spaces

  def step(self, action):
    obs = self._env.step(action)
    obs[self._key] = self._env.render('rgb_array')
    return obs

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._env.render('rgb_array')
    return obs


class Async:

  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _CLOSE = 4
  _EXCEPTION = 5

  def __init__(self, constructor, strategy='thread'):
    self._pickled_ctor = cloudpickle.dumps(constructor)
    if strategy == 'process':
      import multiprocessing as mp
      context = mp.get_context('spawn')
    elif strategy == 'thread':
      import multiprocessing.dummy as context
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = context.Pipe()
    self._process = context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    self._receive()
    self._obs_space = None
    self._act_space = None

  def access(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      pass
    self._process.join(5)

  @property
  def obs_space(self):
    if not self._obs_space:
      self._obs_space = self.access('obs_space')()
    return self._obs_space

  @property
  def act_space(self):
    if not self._act_space:
      self._act_space = self.access('act_space')()
    return self._act_space

  def step(self, action, blocking=False):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=False):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    try:
      ctor = cloudpickle.loads(self._pickled_ctor)
      env = ctor()
      conn.send((self._RESULT, None))
      while True:
        try:
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      #print('Error in environment process: {}'.format(stacktrace))
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass
