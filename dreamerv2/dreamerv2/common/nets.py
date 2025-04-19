import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class EnsembleRSSM(common.Module):

  def __init__(
      self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
      act='elu', norm='none', std_act='softplus', min_std=0.1):
    super().__init__()
    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = get_act(act)
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std
    self._cell = GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  @tf.function
  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (swap(action), swap(embed), swap(is_first)), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, ensemble=False):
    if ensemble:
      state = self._suff_stats_ensemble(state['deter'])
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
    prev_state, prev_action = tf.nest.map_structure(
        lambda x: tf.einsum(
            'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
        (prev_state, prev_action))
    prior = self.img_step(prev_state, prev_action, sample)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
    x = self.get('obs_out_norm', NormLayer, self._norm)(x)
    x = self._act(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', NormLayer, self._norm)(x)
    x = self._act(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    stats = self._suff_stats_ensemble(x)
    index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_ensemble(self, inp):
    bs = list(inp.shape[:-1])
    inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
      x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
      x = self._act(x)
      stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
    stats = {
        k: tf.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    stats = {
        k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
        for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class Encoder(common.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
    super().__init__()
    self._shapes = shapes
    if isinstance(cnn_keys, str) and ',' in cnn_keys and not any(c in cnn_keys for c in r'.*+?$^[]{}()|'):
        allowed_cnn_keys = cnn_keys.split(',')
        self._cnn_keys = [
            k for k, v in shapes.items() if k in allowed_cnn_keys and len(v) >= 2]
    else:
        self._cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) >= 2]

    if isinstance(mlp_keys, str) and ',' in mlp_keys and not any(c in mlp_keys for c in r'.*+?$^[]{}()|'):
        allowed_mlp_keys = mlp_keys.split(',')
        self._mlp_keys = [
            k for k, v in shapes.items() if k in allowed_mlp_keys and len(v) < 2]
    else:
        self._mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) < 2]

    if self._cnn_keys:
      print('Encoder CNN inputs:', self._cnn_keys)
      self._conv_encoder = ConvEncoder(cnn_depth, cnn_kernels, act, norm)
    if self._mlp_keys:
      print('Encoder MLP inputs:', self._mlp_keys)
      self._mlp_encoder = MLP(None, mlp_layers, act, norm)

  def __call__(self, data):
    found_key = None
    for k in self._shapes:
        if k in data:
            found_key = k
            break
    if found_key is None:
        raise ValueError("Encoder.__call__: None of the expected keys found in input data.")

    shape = self._shapes[found_key]
    batch_dims = data[found_key].shape[:-len(shape)]

    data = {
        k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
        for k, v in data.items() if k in self._shapes
    }
    outputs = []
    if self._cnn_keys:
      cnn_data = {k: data[k] for k in self._cnn_keys if k in data}
      if cnn_data:
           outputs.append(self._conv_encoder(cnn_data))
    if self._mlp_keys:
      mlp_data = {k: data[k] for k in self._mlp_keys if k in data}
      if mlp_data:
           outputs.append(self._mlp_encoder(mlp_data))

    if not outputs:
        raise ValueError("Encoder.__call__: No CNN or MLP features generated. Check key matching and input data.")

    output = tf.concat(outputs, -1)
    return output.reshape(batch_dims + output.shape[1:])

  def _cnn(self, data):
    x = tf.concat(list(data.values()), -1)
    x = x.astype(prec.global_policy().compute_dtype)
    for i, kernel in enumerate(self._cnn_kernels):
      depth = 2 ** i * self._cnn_depth
      x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
      x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x.reshape(tuple(x.shape[:-3]) + (-1,))

  def _mlp(self, data):
    x = tf.concat(list(data.values()), -1)
    x = x.astype(prec.global_policy().compute_dtype)
    for i, width in enumerate(self._mlp_layers):
      x = self.get(f'dense{i}', tfkl.Dense, width)(x)
      x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x


class Decoder(common.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=4, mlp_units=400,
      **kwargs):
    super().__init__()
    excluded = ('is_first', 'is_last', 'is_terminal')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self._shapes = shapes

    cnn_match_fn = None
    if isinstance(cnn_keys, str) and ',' in cnn_keys and not any(c in cnn_keys for c in r'.*+?$^[]{}()|'):
        allowed_cnn_keys = cnn_keys.split(',')
        cnn_match_fn = lambda k: k in allowed_cnn_keys
    else:
        cnn_keys_regex = re.compile(cnn_keys)
        cnn_match_fn = lambda k: cnn_keys_regex.match(k)

    mlp_match_fn = None
    if isinstance(mlp_keys, str) and ',' in mlp_keys and not any(c in mlp_keys for c in r'.*+?$^[]{}()|'):
        allowed_mlp_keys = mlp_keys.split(',')
        mlp_match_fn = lambda k: k in allowed_mlp_keys
    else:
        mlp_keys_regex = re.compile(mlp_keys)
        mlp_match_fn = lambda k: mlp_keys_regex.match(k)

    self._cnn_shapes = {k: v for k, v in shapes.items() if cnn_match_fn(k) and len(v) >= 2}
    self._mlp_shapes = {k: v for k, v in shapes.items() if mlp_match_fn(k) and len(v) < 2}

    self._conv_decoder = None
    if self._cnn_shapes:
        print(f"Decoder CNN outputs: {list(self._cnn_shapes.keys())}")
        self._conv_decoder = ConvDecoder(cnn_depth, cnn_kernels, act, norm, shapes=self._cnn_shapes)

    self._mlp_decoder = None
    if self._mlp_shapes:
        print(f"Decoder MLP outputs: {list(self._mlp_shapes.keys())}")
        self._mlp_decoder = MLP(shape=self._mlp_shapes, layers=mlp_layers, units=mlp_units, act=act, norm=norm, **kwargs)

    if not self._conv_decoder and not self._mlp_decoder:
        print("[WARNING Decoder.__init__] No CNN or MLP keys matched. Decoder will produce empty output.")

  def __call__(self, features):
    features = tf.cast(features, prec.global_policy().compute_dtype)
    outputs = {}
    if self._conv_decoder:
      outputs.update(self._conv_decoder(features))
    if self._mlp_decoder:
      outputs.update(self._mlp_decoder(features))
    return outputs


class ConvEncoder(common.Module):
    def __init__(self, depth=48, kernels=(4, 4, 4, 4), act='elu', norm='none'):
        super().__init__()
        self._act = get_act(act)
        self._norm = norm
        self._depth = depth
        self._kernels = kernels

    def __call__(self, data):
        processed_values = []
        for key, value in data.items():
            if len(value.shape) == 3:  # Assuming (batch, H, W)
                processed_values.append(tf.expand_dims(value, -1))
            elif len(value.shape) == 4:  # Assuming (batch, H, W, C)
                processed_values.append(value)
            else:
                processed_values.append(value)

        x = tf.concat(processed_values, -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, kernel in enumerate(self._kernels):
            depth = 2 ** i * self._depth
            x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
            x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
            x = self._act(x)
        x = x.reshape((-1, np.prod(x.shape[1:])))
        return x


class ConvDecoder(common.Module):
    def __init__(self, depth=48, kernels=(4, 4, 4, 4), act='elu', norm='none', shapes=None):
        super().__init__()
        if shapes is None or not isinstance(shapes, dict):
             raise ValueError("ConvDecoder requires a 'shapes' dictionary.")
        self._shapes = shapes
        self._depth = depth
        self._kernels = kernels
        self._act = get_act(act)
        self._norm = norm

    def __call__(self, features):
        ConvT = tfkl.Conv2DTranspose
        x = self.get('convin', tfkl.Dense, 32 * self._depth)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        for i, kernel in enumerate(self._kernels):
            depth = 2 ** (len(self._kernels) - i - 2) * self._depth
            act = self._act
            norm = self._norm
            if i == len(self._kernels) - 1:
                depth = 0
                for shape in self._shapes.values():
                    if len(shape) == 2:
                        depth += 1
                    elif len(shape) == 3:
                        depth += shape[-1]
                act = tf.identity
                norm = 'none'
            x = self.get(f'conv{i}', ConvT, depth, kernel, 2)(x)
            x = self.get(f'convnorm{i}', NormLayer, norm)(x)
            x = act(x)

        split_sizes = []
        for shape in self._shapes.values():
             if len(shape) == 2:
                 split_sizes.append(1)
             elif len(shape) == 3:
                 split_sizes.append(shape[-1])
        means = tf.split(x, split_sizes, axis=-1)

        dists = {}
        idx = 0
        for key, shape in self._shapes.items():
            mean = means[idx]
            event_ndims = len(shape)
            if len(shape) == 2:
                mean = tf.squeeze(mean, axis=-1)
            dists[key] = tfd.Independent(tfd.Normal(mean, 1), event_ndims)
            idx += 1
        return dists


class MLP(common.Module):

  def __init__(self, shape, layers, units, act='elu', norm='none', **out):
    super().__init__()
    self._shape = shape
    self._layers = layers
    self._units = units
    self._norm = norm
    self._act = get_act(act)
    self._out = out

  def __call__(self, features):
    if isinstance(features, dict):
        first_key = next(iter(features))
        original_batch_dims = features[first_key].shape[:-1]
        x = tf.concat(list(features.values()), -1)
    else:
        x = features
        original_batch_dims = x.shape[:-1]

    x = tf.cast(x, prec.global_policy().compute_dtype)

    needs_flatten = len(original_batch_dims) > 0
    if needs_flatten:
        x = tf.reshape(x, [-1, x.shape[-1]])

    for index in range(self._layers):
      x = self.get(f'dense{index}', tfkl.Dense, self._units)(x)
      x = self.get(f'norm{index}', NormLayer, self._norm)(x)
      x = self._act(x)

    if self._shape is None:
        if needs_flatten:
             x = tf.reshape(x, tf.concat([original_batch_dims, [tf.shape(x)[-1]]], axis=0))
        return x
    else:
        dists = {}
        output_shapes = self._shape
        for key, shape in output_shapes.items():
            base_dist = self.get(f'out_{key}', DistLayer, shape, **self._out)(x)

            needs_batch_reshape = len(original_batch_dims) > 1
            if needs_batch_reshape:
                 new_batch_shape = tf.concat([original_batch_dims, base_dist.batch_shape_tensor()[len(original_batch_dims):]], axis=0)
                 dists[key] = tfd.BatchReshape(base_dist, batch_shape=new_batch_shape)
            else:
                 dists[key] = base_dist

        return dists


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = get_act(act)
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  @tf.function
  def call(self, inputs, state):
    state = state[0]
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class DistLayer(common.Module):

  def __init__(
      self, shape, dist='mse', min_std=0.1, init_std=0.0):
    super().__init__()
    self._shape = shape
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std

  def __call__(self, inputs):
    out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
    out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
    out = tf.cast(out, tf.float32)
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
      std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
      std = tf.cast(std, tf.float32)
    if self._dist == 'mse':
      dist = tfd.Normal(out, 1.0)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * tf.tanh(out / 5)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, common.TanhBijector())
      dist = tfd.Independent(dist, len(self._shape))
      return common.SampleDist(dist)
    if self._dist == 'trunc_normal':
      std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
      return tfd.Independent(dist, 1)
    if self._dist == 'onehot':
      return common.OneHotDist(out)
    raise NotImplementedError(self._dist)


class NormLayer(common.Module):

  def __init__(self, name):
    super().__init__()
    if name == 'none':
      self._layer = None
    elif name == 'layer':
      self._layer = tfkl.LayerNormalization()
    else:
      raise NotImplementedError(name)

  def __call__(self, features):
    if not self._layer:
      return features
    return self._layer(features)


def get_act(name):
  if name == 'none':
    return tf.identity
  if name == 'mish':
    return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
  elif hasattr(tf.nn, name):
    return getattr(tf.nn, name)
  elif hasattr(tf, name):
    return getattr(tf, name)
  else:
    raise NotImplementedError(name)
