import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np
import cv2  # Import OpenCV

try:
    # Use relative import since mspacman is in the same directory (common)
    from . import mspacman
    from .mspacman import _init_objects_ram, _detect_objects_ram
except ImportError as e:
    print(f"Error during relative import from common: {e}") # Print the specific import error
    print("Please ensure mspacman.py, game_objects.py, _helper_methods.py, and __init__.py exist in the 'common' directory.")
    # Attempt absolute import as a fallback (might work depending on sys.path)
    try:
        print("Attempting absolute import as fallback...")
        import mspacman # Keep original attempt as fallback
        from mspacman import _init_objects_ram, _detect_objects_ram
        print("Absolute import fallback succeeded.")
    except ImportError as e_abs:
        print(f"Absolute import fallback failed: {e_abs}")
        exit()
except Exception as e_other:
     print(f"An unexpected error occurred during import: {e_other}")
     exit()


class GymWrapper:

  def __init__(self, env, obs_key='image', act_key='action'):
    self._env = env
    try:
        inner_obs_space = self._env.obs_space
        inner_act_space = self._env.act_space
        self._obs_is_dict = isinstance(inner_obs_space, (dict, gym.spaces.Dict))
        self._act_is_dict = isinstance(inner_act_space, (dict, gym.spaces.Dict))
    except Exception as e:
        self._obs_is_dict = False
        self._act_is_dict = False
    self._obs_key = obs_key
    self._act_key = act_key

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    try:
        env_obs_space = self._env.obs_space
        if isinstance(env_obs_space, gym.spaces.Dict):
            spaces = env_obs_space.spaces.copy()
        elif isinstance(env_obs_space, dict):
             spaces = env_obs_space.copy()
        else:
            spaces = {self._obs_key: env_obs_space}

        spaces.setdefault('reward', gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32))
        spaces.setdefault('is_first', gym.spaces.Box(0, 1, (), dtype=np.bool_))
        spaces.setdefault('is_last', gym.spaces.Box(0, 1, (), dtype=np.bool_))
        spaces.setdefault('is_terminal', gym.spaces.Box(0, 1, (), dtype=np.bool_))
        return gym.spaces.Dict(spaces)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

  @property
  def act_space(self):
    try:
        env_act_space = self._env.act_space
        if isinstance(env_act_space, gym.spaces.Dict):
            spaces = env_act_space.spaces.copy()
            return spaces
        elif isinstance(env_act_space, dict):
            spaces = env_act_space.copy()
            return spaces
        else:
            space_dict = {self._act_key: env_act_space}
            return space_dict
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

  def step(self, action):
    original_action = action
    action_to_pass = action
    if not self._act_is_dict:
      if self._act_key not in action:
          raise KeyError(f"Action key '{self._act_key}' not found in action dict: {action.keys()}")
      action_to_pass = action[self._act_key]

    inner_step_return = self._env.step(action_to_pass)

    if isinstance(inner_step_return, dict):
        obs = inner_step_return
        reward = obs.get('reward')
        done = obs.get('is_last')
        info = {'is_terminal': obs.get('is_terminal', done)}
    elif isinstance(inner_step_return, tuple) and len(inner_step_return) == 4:
        obs, reward, done, info = inner_step_return
        if info is None:
            info = {}
    elif isinstance(inner_step_return, tuple) and len(inner_step_return) == 5:
        obs, reward, terminated, truncated, info = inner_step_return
        done = terminated or truncated
        if info is None:
            info = {}
        info['is_terminal'] = terminated
    else:
        raise TypeError(f"Inner env {type(self._env)} returned unexpected type or structure from step()")

    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    else:
       if isinstance(obs, gym.spaces.Dict):
           obs = obs.spaces.copy()
       elif isinstance(obs, dict):
           obs = obs.copy()
       else:
            obs = {self._obs_key: obs}

    obs['reward'] = float(reward)
    obs['is_first'] = False
    obs['is_last'] = bool(done)
    obs['is_terminal'] = bool(info.get('is_terminal', done))
    return obs

  def reset(self):
    reset_return = self._env.reset()
    if isinstance(reset_return, tuple) and len(reset_return) == 2:
        obs, info = reset_return
    else:
        obs = reset_return
        info = {}

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
    Detects Pacman using RAM state on 64x64 input frames,
    generates a 64x64 binary mask with a 7x7 patch around Pacman's scaled coordinates.
    """
    def __init__(self, env, process_size=(64, 64), image_key='image', mask_key='pacman_mask'):
        inner_obs_key = None
        try:
            inner_obs_key = getattr(env, '_obs_key', image_key)
        except Exception as e:
            inner_obs_key = image_key

        super().__init__(env, obs_key=inner_obs_key)

        self._process_size = tuple(process_size)
        self._image_key = self._obs_key
        self._mask_key = mask_key
        self._patch_radius = 3

        try:
            self.objects = _init_objects_ram(hud=False)
        except Exception as e:
            raise e

        self._SCALE_X = self._process_size[1] / 160
        self._SCALE_Y = self._process_size[0] / 194
        self._SHIFT_Y = 8

    def _scale_coords(self, x, y):
        x_scaled = int(x * self._SCALE_X)
        y_scaled = int((y - self._SHIFT_Y) * self._SCALE_Y)
        max_x = self._process_size[1] - 1
        max_y = self._process_size[0] - 1
        return np.clip(x_scaled, 0, max_x), np.clip(y_scaled, 0, max_y)

    @property
    def obs_space(self):
        try:
            base_space_dict = super().obs_space
            spaces = base_space_dict.spaces

            if self._image_key not in spaces:
                 raise KeyError(f"Image key '{self._image_key}' not found in base observation space keys: {list(spaces.keys())}")

            original_img_space = spaces[self._image_key]

            img_shape = self._process_size + original_img_space.shape[2:]
            spaces[self._image_key] = gym.spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )

            mask_shape = self._process_size
            spaces[self._mask_key] = gym.spaces.Box(
                low=0, high=1, shape=mask_shape, dtype=np.uint8
            )

            if 'pacman_coords' in spaces: # Keep removing old key just in case
                del spaces['pacman_coords']

            return gym.spaces.Dict(spaces)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    def step(self, action):
        obs = super().step(action)
        return self._process_obs(obs)

    def reset(self):
        obs = super().reset()
        return self._process_obs(obs)

    def _process_obs(self, obs):
        image_process_size = obs[self._image_key]
        if image_process_size.shape[:2] != self._process_size:
             image_process_size = cv2.resize(
                 image_process_size,
                 (self._process_size[1], self._process_size[0]),
                 interpolation=cv2.INTER_NEAREST
             )
             if len(obs[self._image_key].shape) == 3 and obs[self._image_key].shape[-1] == 1 and len(image_process_size.shape) == 2:
                 image_process_size = np.expand_dims(image_process_size, axis=-1)
             obs[self._image_key] = image_process_size

        ram_state = obs.get('ram')
        mask = np.zeros(self._process_size, dtype=np.uint8)

        if ram_state is not None:
            try:
                _detect_objects_ram(self.objects, ram_state, hud=False)

                player = self.objects[0] if self.objects else None
                if player and player.visible and hasattr(player, 'xy'):
                    raw_coords = player.xy
                    scaled_x, scaled_y = self._scale_coords(*raw_coords)
                    #print(f"[Pacman Coords] Step: Scaled=({scaled_x}, {scaled_y}) Raw={raw_coords}")

                    y1 = max(0, scaled_y - self._patch_radius)
                    y2 = min(self._process_size[0], scaled_y + self._patch_radius + 1)
                    x1 = max(0, scaled_x - self._patch_radius)
                    x2 = min(self._process_size[1], scaled_x + self._patch_radius + 1)

                    mask[y1:y2, x1:x2] = 1
            except Exception as e:
                pass

        obs[self._mask_key] = mask

        if 'pacman_coords' in obs: # Keep removing old key just in case
            del obs['pacman_coords']

        if 'pacman_coords_scaled' in obs:
            del obs['pacman_coords_scaled']

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
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass
