import collections #estructuras de datos especializadas (listas, diccionarios, tuplas, etc)
import functools #manejo de funciones
import logging #mensajes de estado, errores, advertencias, etc
import os
import pathlib
import re
import sys #Variables mantenidas por el interprete
import warnings
import time #manejo de tiempo

try:
  import rich.traceback #mejora la salida de los errores
  rich.traceback.install()
except ImportError:
  pass

#os.environ es un diccionario que contiene todas las variables de entorno del sistema
#TF_CPP_MIN_LOG_LEVEL es una variable de entorno que establece el nivel de registro de TensorFlow (3: solo errores críticos)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#logging.getLogger() devuelve el logger raíz
#setLevel() establece el nivel de registro del logger raíz, en este caso, ERROR, solo se mostrarán mensajes de error
logging.getLogger().setLevel('ERROR')
#Se ignoran los warnings que tengan el mensaje 'box bound precision lowered'
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

#sys.path es una lista de directorios en los que el intérprete buscará los módulos
#pathlib.Path(__file__).parent devuelve el directorio padre del archivo actual
#pathlib.Path(__file__).parent.parent devuelve el directorio padre del directorio padre del archivo actual
#En pocas palabras, se añaden al path los directorios dreamerv2/dreamerv2 y dreamerv2 para que los módulos puedan ser importados
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
#import ruamel.yaml as yaml
#ruamel.yaml es una librería que permite leer y escribir archivos YAML (para cargar la configuración)
from ruamel.yaml import YAML

#El módulo agent.py contiene la clase Agent que define un agente de DreamerV2
import agent
#El módulo common.py contiene funciones y clases comunes que se utilizan en DreamerV2
import common


def main():
  start_time = time.time() # Record start time
  print("--- Starting main function ---")
  print("Version de dv2:",1.2,"-"*50)
  #configs = yaml.safe_load(
      #pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  
  #Se carga la configuración por defecto
  yaml = YAML()
  #configs = yaml.load((pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  configs = yaml.load((pathlib.Path(sys.argv[0]).parent / 'full_configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  # --- Force sequential execution for debugging prints ---
  print("!!! Forcing sequential environment execution for debug prints !!!")
  #config = config.update({'envs_parallel': 'none'})
  # --- End forcing sequential execution ---

  #Se configura el directorio de logs
  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  #Se configura TensorFlow
  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  #Se configuran los buffers de replay para el entrenamiento y la evaluación
  #Se configura el contador de pasos y el logger para registrar métricas
  print("--- Setting up replay buffers and logger ---")
  train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  #Definición de funciones que determinan cuándo se debe entrenar, registrar métricas y grabar videos
  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until // config.action_repeat)

  #Función que crea el entorno de entrenamiento 
  def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      # Use atari_render_size for initial Atari wrapper size
      env = common.Atari(
          task, config.action_repeat, config.atari_render_size,
          config.atari_grayscale)
      print(f"Creando entorno de Atari: {task} con render size {config.atari_render_size}")
      env = common.OneHotAction(env)
      # Add Pacman detection and resizing wrapper
      env = common.PacmanDetectionAndResizeWrapper(
          env,
          process_size=config.atari_process_size # Final size (e.g., 64x64)
      )
      print(f"Aplicado PacmanDetectionAndResizeWrapper, procesando a {config.atari_process_size}")
    elif suite == 'crafter':
      assert config.action_repeat == 1
      outdir = logdir / 'crafter' if mode == 'train' else None
      reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
      env = common.Crafter(outdir, reward)
      env = common.OneHotAction(env)
    else:
      raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env

  #Función que registra las métricas de un episodio
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      # Log Pacman mask info if present (e.g., average activation)
      if key == 'pacman_mask':
          logger.scalar(f'mean_{mode}_pacman_mask_active', value.mean()) # Avg activation (should be small)

      # --- Removed coordinate logging ---

      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        # Avoid double logging if already handled above
        if key not in ['pacman_mask', 'pacman_coords_scaled']:
             logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        # Ensure the key exists before logging video
        if key in ep:
             # Log the mask as a video too if requested
             if key == 'pacman_mask':
                 # Convert mask to 3 channels for video logging if needed
                 mask_video = ep[key]
                 if mask_video.ndim == 3 and mask_video.shape[-1] == 1: # (T, H, W, 1)
                     mask_video = np.repeat(mask_video, 3, axis=-1) # (T, H, W, 3)
                 elif mask_video.ndim == 2: # (T, H, W) -> Add channel and repeat
                     mask_video = np.expand_dims(mask_video, axis=-1)
                     mask_video = np.repeat(mask_video, 3, axis=-1)

                 # Scale mask values (0, 1) to (0, 255) for video
                 logger.video(f'{mode}_policy_{key}', (mask_video * 255).astype(np.uint8))
             # Don't log coordinates as video
             elif key != 'pacman_coords_scaled':
                 logger.video(f'{mode}_policy_{key}', ep[key])
        else:
             print(f"Warning: Video log key '{key}' not found in episode data.")
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  print('Create envs.')
  print("--- Creating environments ---")
  #Se crean los entornos de entrenamiento y evaluación
  num_eval_envs = min(config.envs, config.eval_eps)
  # --- Ensure sequential execution based on updated config ---
  if config.envs_parallel == 'none':
    train_envs = [make_env('train') for _ in range(config.envs)]
    print(f"Entornos de entrenamiento creados: {config.task}")
    eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
    print(f"Entornos de evaluación creados: {config.task}")
  else:
    print(f"WARNING: envs_parallel is '{config.envs_parallel}', debug prints might not show.")
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    train_envs = [make_async_env('train') for _ in range(config.envs)]
    eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
  
  #Se configura el driver de entrenamiento y evaluación, define los callbacks para registrar métricas y guardar datos
  print("--- Configuring drivers ---")
  _train_envs_for_space = [make_env('train') for _ in range(1)]
  act_space = _train_envs_for_space[0].act_space
  obs_space = _train_envs_for_space[0].obs_space
  print("Final Observation Space:", obs_space) # Log final obs space
  print(f"[DEBUG train.py] obs_space type: {type(obs_space)}, content: {obs_space}")
  print(f"[DEBUG train.py] act_space type: {type(act_space)}, content: {act_space}")
  _train_envs_for_space[0].close() # Close the temporary env

  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  #Se pre-entrena el agente con un agente aleatorio para llenar el buffer de replay
  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'--- Starting prefill ({prefill} steps) ---')
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(act_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  #Se crea el agente y se entrena, si se ha guardado un checkpoint previo, se carga
  print('Create agent.')
  print("--- Creating agent ---")
  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  print(f"[DEBUG train.py] Passing to Agent: obs_space type={type(obs_space)}, act_space type={type(act_space)}")
  agnt = agent.Agent(config, obs_space, act_space, step)
  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  #Función que define el paso de entrenamiento
  def train_step(tran, worker):
    if should_train(step):
      print(f'--- Training step {step.value} ---')
      for _ in range(config.train_steps):
        mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      print(f'--- Logging step {step.value} ---')
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)
  train_driver.on_step(train_step)

  #Función que define el paso de evaluación, se evalúa el agente y se guarda el checkpoint
  #Esto se realiza hasta cumplir con el número de pasos especificado en la configuración
  print("--- Starting main training loop ---")
  while step < config.steps:
    logger.write()
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    print('Start training.')
    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')
  print("--- Training finished ---")
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass
  end_time = time.time() # Record end time
  elapsed_time = end_time - start_time
  print(f"--- Total execution time: {elapsed_time:.2f} seconds ---")


if __name__ == '__main__':
  main()
