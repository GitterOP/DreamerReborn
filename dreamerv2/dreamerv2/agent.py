import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import tensorflow_probability as tfp # Import the main package
tfd = tfp.distributions # Define the alias
import numpy as np # Import numpy for parameter counting

#Importa el modulo common (herraientas comunes)
import common
#Importa el modulo expl (exploración), define como se comporta la exploración aleatoria y la exploración de Plan2Explore (actor-critic)
import expl 

# Helper function to count parameters
def count_params(module):
    if hasattr(module, 'variables') and module.variables:
        return np.sum([np.prod(v.shape.as_list()) for v in module.variables])
    return 0


class Agent(common.Module):

  def __init__(self, config, obs_space, act_space, step): #Inicializa el agente y world_model (lo gestiona)
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    self.wm = WorldModel(config, obs_space, self.tfstep)
    self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm, self.tfstep,
          lambda seq: self.wm.heads['reward'](seq['feat'])['reward'].mode())

  @tf.function
  def policy(self, obs, state=None, mode='train'): #Define la política del agente
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action
    latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], sample)
    feat = self.wm.rssm.get_feat(latent)
    if mode == 'eval':
      # Access 'action' key before calling mode()
      actor = self._task_behavior.actor(feat)['action']
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      # Access 'action' key before calling sample()
      actor = self._expl_behavior.actor(feat)['action']
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      # Access 'action' key before calling sample()
      actor = self._task_behavior.actor(feat)['action']
      action = actor.sample()
      noise = self.config.expl_noise
    action = common.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None): #Entrena el agente
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    start = outputs['post']
    reward = lambda seq: self.wm.heads['reward'](seq['feat'])['reward'].mode()
    metrics.update(self._task_behavior.train(
        self.wm, start, data['is_terminal'], reward))
    if self.config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    return state, metrics

  @tf.function
  def report(self, data): #Reporta el estado del agente
    report = {}
    data = self.wm.preprocess(data)
    decoder_head = self.wm.heads['decoder']
    # Check if the decoder has CNN shapes defined
    if hasattr(decoder_head, '_cnn_shapes') and decoder_head._cnn_shapes:
        # Iterate over the keys the CNN decoder was built for
        for key in decoder_head._cnn_shapes.keys(): # Use keys() from _cnn_shapes
            # Ensure the key is actually expected in the video prediction keys
            if key in self.config.log_keys_video:
                 name = key.replace('/', '_')
    return report


class WorldModel(common.Module):

  def __init__(self, config, obs_space, tfstep): #Inicializa el modelo del mundo (RSSM, encoder, decoder, reward, discount)
    super().__init__() # Call super constructor
    if hasattr(obs_space, 'spaces') and isinstance(obs_space.spaces, dict):
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
    elif isinstance(obs_space, dict):
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    else:
        raise TypeError(f"obs_space must be a dict or have a .spaces attribute, got {type(obs_space)}")
    self.config = config
    self.tfstep = tfstep

    self.encoder = common.Encoder(shapes, **config.encoder)
    encoder_params = count_params(self.encoder)

    self.rssm = common.EnsembleRSSM(**config.rssm)
    _ = self.rssm.initial(1)
    rssm_params = count_params(self.rssm)

    self.heads = {}
    self.heads['decoder'] = common.Decoder(shapes, **config.decoder)
    decoder_params = count_params(self.heads['decoder'])

    self.heads['reward'] = common.MLP({'reward': []}, **config.reward_head)
    reward_params = count_params(self.heads['reward'])

    discount_params = 0
    if config.pred_discount:
      self.heads['discount'] = common.MLP({'discount': []}, **config.discount_head)
      discount_params = count_params(self.heads['discount'])

    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)


  @tf.function
  def train(self, data, state=None): #Entrena el modelo del mundo, calcula la pérdida del modelo y ¿actualiza las métricas?
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  @tf.function
  def loss(self, data, state=None): #Procesa los datos y calcula la pérdida del modelo
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        if key not in data:
             continue # Skip loss calculation for this key if data is missing

        target = data[key]
        target_shape = tf.shape(target) # e.g., (B, T) or (B, T, H, W, C)
        dist_event_rank = len(dist.event_shape_tensor())

        is_batch_reshaped = isinstance(dist, tfd.BatchReshape)
        is_scalar_event = (dist_event_rank == 0)

        if is_batch_reshaped and is_scalar_event:
            reshaped_target = target
        else:
            batch_size = target_shape[0]
            time_length = target_shape[1]
            event_shape = target_shape[len(target_shape)-dist_event_rank:] # Get event dims dynamically
            new_shape = tf.concat([[batch_size * time_length], event_shape], axis=0)
            reshaped_target = tf.reshape(target, new_shape)

        like = tf.cast(dist.log_prob(reshaped_target), tf.float32)
        likes[key] = like
        losses[key] = -tf.reduce_mean(like)
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon): #Calcula la secuencia completa de estados imaginados
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    # Access the 'action' key from the policy output dict before calling mode()
    start['action'] = tf.zeros_like(policy(start['feat'])['action'].mode())
    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
      # Access the 'action' key from the policy output dict before calling sample()
      action = policy(tf.stop_gradient(seq['feat'][-1]))['action'].sample()
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      # Access the 'discount' key from the head output dict before calling mean()
      disc = self.heads['discount'](seq['feat'])['discount'].mean()
      if is_terminal is not None:
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= self.config.discount
        disc = tf.concat([true_first[None], disc[1:]], 0)
    else:
      disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
    seq['discount'] = disc
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

  @tf.function
  def preprocess(self, obs): #Ajusta la observación para que sea compatible con el modelo del mundo (tipo de datos, normalización, etc.)
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        if key == 'pacman_mask':
             value = value.astype(dtype)
        else:
             value = value.astype(dtype) / 255.0 - 0.5
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data, key): #Predice el video para calcular el error
    decoder = self.heads['decoder']
    truth = data[key][:6] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
    recon = decoder(self.rssm.get_feat(states))[key].mode()[:6] # Shape (6, 5, H, W, C)
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = decoder(self.rssm.get_feat(prior))[key].mode() # Shape (L-5, 6, H, W, C)

    # Transpose openl to (6, L-5, H, W, C) before concatenation
    openl_transposed = tf.transpose(openl, [1, 0, 2, 3, 4])

    # Concatenate along axis 1 (time)
    model = tf.concat([recon[:, :5] + 0.5, openl_transposed + 0.5], 1) # Use openl_transposed
    error = (model - truth + 1) / 2
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class ActorCritic(common.Module):

  def __init__(self, config, act_space, tfstep): #Inicializa el actor-critic
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})
    self.actor = common.MLP({'action': act_space.shape}, **self.config.actor)
    self.critic = common.MLP({'critic': []}, **self.config.critic)
    if self.config.slow_target:
      self._target_critic = common.MLP({'critic': []}, **self.config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def train(self, world_model, start, is_terminal, reward_fn): #Entrena el actor-critic
    metrics = {}
    hor = self.config.imag_horizon
    with tf.GradientTape() as actor_tape:
      seq = world_model.imagine(self.actor, start, is_terminal, hor)
      reward = reward_fn(seq)
      seq['reward'], mets1 = self.rewnorm(reward)
      mets1 = {f'reward_{k}': v for k, v in mets1.items()}
      target, mets2 = self.target(seq)
      actor_loss, mets3 = self.actor_loss(seq, target)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets4 = self.critic_loss(seq, target)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()
    return metrics

  def actor_loss(self, seq, target): #Calcula la pérdida del actor
    metrics = {}
    # Access the 'action' key from the policy output dict
    policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))['action']
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      # Access the 'critic' key from the critic output dict
      baseline = self._target_critic(seq['feat'][:-2])['critic'].mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      action = tf.stop_gradient(seq['action'][1:-1])
      objective = policy.log_prob(action) * advantage
    elif self.config.actor_grad == 'both':
      # Access the 'critic' key from the critic output dict
      baseline = self._target_critic(seq['feat'][:-2])['critic'].mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target): #Calcula la pérdida del crítico
    # Access the 'critic' key from the critic output dict
    dist = self.critic(seq['feat'][:-1])['critic']
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, seq): #Esta función calcula el objetivo del crítico
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    # Access the 'critic' key from the critic output dict
    value = self._target_critic(seq['feat'])['critic'].mode()
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self): #Actualiza el objetivo lento del crítico para estailizar el entrenamiento
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)
