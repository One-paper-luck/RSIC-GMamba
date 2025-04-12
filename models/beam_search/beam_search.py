# import torch
# import utils
# from models.transformer.mamba import MambaConfig
# from models.transformer.mamba_lm import MambaLMConfig
#
#
# class BeamSearch(object):
#     def __init__(self, model, max_len: int, eos_idx: int, beam_size: int, config=None):
#         self.model = model
#         self.max_len = max_len
#         self.eos_idx = eos_idx
#         self.beam_size = beam_size
#         self.b_s = None
#         self.device = None
#         self.seq_mask = None
#         self.seq_logprob = None
#         self.outputs = None
#         self.log_probs = None
#         self.selected_words = None
#         self.all_log_probs = None
#
#         self.config = MambaConfig()
#
#     def _expand_state(self, selected_beam, cur_beam_size):
#         def fn(s):
#             shape = [int(sh) for sh in s.shape]
#             beam = selected_beam
#             for _ in shape[1:]:
#                 beam = beam.unsqueeze(-1)
#             s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
#                              beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
#             s = s.view(*([-1, ] + shape[1:]))
#             return s
#
#         return fn
#
#     def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
#         if isinstance(visual, torch.Tensor):
#             visual_shape = visual.shape  # 10,49,2048
#             visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]  # 10,1,49,2048
#             visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]  # 50,49,2048
#             selected_beam_red_size = (self.b_s, self.beam_size) + tuple(
#                 1 for _ in range(len(visual_exp_shape) - 2))  # 10，5，1，1
#             selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]  # 10，5，49，2048
#             visual_exp = visual.view(visual_exp_shape)  # 10,49,2048 -> 10,1,49,2048
#             selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(
#                 selected_beam_exp_size)  # 10,5 -> 10,5,1,1 -> 10，5，49，2048
#             visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)  # 50，49，2048
#         else:
#             new_visual = []
#             for im in visual:
#                 visual_shape = im.shape
#                 visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
#                 visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
#                 selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
#                 selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
#                 visual_exp = im.view(visual_exp_shape)
#                 selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
#                 new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
#                 new_visual.append(new_im)
#             visual = tuple(new_visual)
#         return visual
#
#     def _expand_catche(self, cache: list, cur_beam_size: int, selected_beam: torch.Tensor):
#         new_catche = []
#         for tup in cache:
#             tensor_tuple = ()
#             for tensor in tup:
#                 # 获取原始张量的形状
#                 original_shape = tensor.shape
#                 visual_exp_shape = (self.b_s, cur_beam_size) + original_shape[1:]
#                 visual_red_shape = (self.b_s * self.beam_size,) + original_shape[1:]
#                 selected_beam_red_size = (self.b_s, self.beam_size) + tuple(
#                     1 for _ in range(len(visual_exp_shape) - 2))  # 10，5，1，1
#
#                 selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]  # 10，5，49，2048
#                 visual_exp = tensor.view(visual_exp_shape)  # 10,49,2048 -> 10,1,49,2048
#                 selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(
#                     selected_beam_exp_size)  # 10,5 -> 10,5,1,1 -> 10，5，49，2048
#                 tensor_tem = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)  # 50，1024，16
#                 tensor_tuple += (tensor_tem,)
#
#             new_catche.append(tensor_tuple)
#
#         return new_catche
#
#     def apply(self, visual: utils.TensorOrSequence, out_size=1, return_probs=False, **kwargs):
#         self.b_s = utils.get_batch_size(visual)
#         self.device = utils.get_device(visual)
#         self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
#         self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
#         self.log_probs = []
#         self.selected_words = None
#         if return_probs:
#             self.all_log_probs = []
#
#         outputs = []
#
#         # caches is a list of cache, one per layer
#         # cache is composed of : the hidden state, and the last d_conv-1 inputs
#         # the hidden state because the update is like an RNN
#         # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
#
#         caches = [(None, torch.zeros(self.b_s, self.config.d_inner, self.config.d_conv - 1, device=self.device))
#                   for _ in range(self.config.n_layers)]
#
#         with self.model.statefulness(self.b_s):
#             for t in range(self.max_len):
#                 # visual, outputs = self.iter(t, visual, outputs, return_probs, **kwargs)
#                 visual, outputs, caches = self.iter(t, visual, outputs, return_probs, caches, **kwargs)
#
#         # Sort result
#         seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
#         outputs = torch.cat(outputs, -1)
#         outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
#         log_probs = torch.cat(self.log_probs, -1)
#         log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
#         if return_probs:
#             all_log_probs = torch.cat(self.all_log_probs, 2)
#             all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
#                                                                                           self.max_len,
#                                                                                           all_log_probs.shape[-1]))
#
#         outputs = outputs.contiguous()[:, :out_size]
#         log_probs = log_probs.contiguous()[:, :out_size]
#         if out_size == 1:
#             outputs = outputs.squeeze(1)
#             log_probs = log_probs.squeeze(1)
#
#         if return_probs:
#             return outputs, log_probs, all_log_probs
#         else:
#             return outputs, log_probs
#
#     def select(self, t, candidate_logprob, **kwargs):
#         selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
#         selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
#         return selected_idx, selected_logprob
#
#     def iter(self, t: int, visual: utils.TensorOrSequence, outputs, return_probs, caches=None, **kwargs):
#         cur_beam_size = 1 if t == 0 else self.beam_size
#
#         # word_logprob = self.model.step(t, self.selected_words, visual, None, mode='feedback', **kwargs)
#
#         word_logprob, caches = self.model.step(t, self.selected_words, visual, None, caches, mode='feedback', **kwargs)
#
#         word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)  # 10,1,167
#         candidate_logprob = self.seq_logprob + word_logprob
#
#         # Mask sequence if it reaches EOS
#         if t > 0:
#             mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
#             self.seq_mask = self.seq_mask * mask
#             word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
#             old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
#             old_seq_logprob[:, :, 1:] = -999
#             candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)
#
#         selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs) #10,5;10,5
#         selected_beam = selected_idx // candidate_logprob.shape[-1]
#         selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
#
#         self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
#         visual = self._expand_visual(visual, cur_beam_size, selected_beam)
#
#         caches = self._expand_catche(caches, cur_beam_size, selected_beam)
#
#         self.seq_logprob = selected_logprob.unsqueeze(-1) #t=0;10,5,1;
#         self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
#         outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
#         outputs.append(selected_words.unsqueeze(-1))
#
#         if return_probs:
#             if t == 0:
#                 self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
#             else:
#                 self.all_log_probs.append(word_logprob.unsqueeze(2))
#
#         this_word_logprob = torch.gather(word_logprob, 1,
#                                          selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
#                                                                             word_logprob.shape[-1]))
#         this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
#         self.log_probs = list(
#             torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
#         self.log_probs.append(this_word_logprob)
#         self.selected_words = selected_words.view(-1, 1)
#
#         # return visual, outputs
#         return visual, outputs, caches

#######################################################################################################################
# import torch
# import utils
# from models.transformer.mamba import MambaConfig
# from models.transformer.mamba_lm import MambaLMConfig
# import torch.nn.functional as F
#
# class BeamSearch(object):
#     def __init__(self, model, max_len: int, eos_idx: int, beam_size: int, config=None):
#         self.model = model
#         self.max_len = max_len
#         self.eos_idx = eos_idx
#         self.beam_size = beam_size
#         self.b_s = None
#         self.device = None
#         self.seq_mask = None
#         self.seq_logprob = None
#         self.outputs = None
#         self.log_probs = None
#         self.selected_words = None
#         self.all_log_probs = None
#
#         self.config = MambaConfig()
#
#     def _expand_state(self, selected_beam, cur_beam_size):
#         def fn(s):
#             shape = [int(sh) for sh in s.shape]
#             beam = selected_beam
#             for _ in shape[1:]:
#                 beam = beam.unsqueeze(-1)
#             s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
#                              beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
#             s = s.view(*([-1, ] + shape[1:]))
#             return s
#
#         return fn
#
#     def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
#         if isinstance(visual, torch.Tensor):
#             visual_shape = visual.shape  # 10,49,2048
#             visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]  # 10,1,49,2048
#             visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]  # 50,49,2048
#             selected_beam_red_size = (self.b_s, self.beam_size) + tuple(
#                 1 for _ in range(len(visual_exp_shape) - 2))  # 10，5，1，1
#             selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]  # 10，5，49，2048
#             visual_exp = visual.view(visual_exp_shape)  # 10,49,2048 -> 10,1,49,2048
#             selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(
#                 selected_beam_exp_size)  # 10,5 -> 10,5,1,1 -> 10，5，49，2048
#             visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)  # 50，49，2048
#         else:
#             new_visual = []
#             for im in visual:
#                 visual_shape = im.shape
#                 visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
#                 visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
#                 selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
#                 selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
#                 visual_exp = im.view(visual_exp_shape)
#                 selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
#                 new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
#                 new_visual.append(new_im)
#             visual = tuple(new_visual)
#         return visual
#
#     def _expand_catche(self, cache: list, cur_beam_size: int, selected_beam: torch.Tensor):
#         new_catche = []
#         for tup in cache:
#             tensor_tuple = ()
#             for tensor in tup:
#                 # 获取原始张量的形状
#                 original_shape = tensor.shape
#                 visual_exp_shape = (self.b_s, cur_beam_size) + original_shape[1:]
#                 visual_red_shape = (self.b_s * self.beam_size,) + original_shape[1:]
#                 selected_beam_red_size = (self.b_s, self.beam_size) + tuple(
#                     1 for _ in range(len(visual_exp_shape) - 2))  # 10，5，1，1
#
#                 selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]  # 10，5，49，2048
#                 visual_exp = tensor.view(visual_exp_shape)  # 10,49,2048 -> 10,1,49,2048
#                 selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(
#                     selected_beam_exp_size)  # 10,5 -> 10,5,1,1 -> 10，5，49，2048
#                 tensor_tem = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)  # 50，1024，16
#                 tensor_tuple += (tensor_tem,)
#
#             new_catche.append(tensor_tuple)
#
#         return new_catche
#
#     def apply(self, visual: utils.TensorOrSequence, out_size=1, return_probs=False, **kwargs):
#         self.b_s = utils.get_batch_size(visual)
#         self.device = utils.get_device(visual)
#         self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
#         self.seq_logprob = torch.zeros((self.b_s, 1), device=self.device)
#         self.log_probs = []
#         self.selected_words = None
#         if return_probs:
#             self.all_log_probs = []
#
#         # outputs = []
#         outputs = torch.zeros((self.b_s, 1),dtype=torch.int64,device=self.device)
#
#         # caches is a list of cache, one per layer
#         # cache is composed of : the hidden state, and the last d_conv-1 inputs
#         # the hidden state because the update is like an RNN
#         # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
#
#         caches = [(None, torch.zeros(self.b_s, self.config.d_inner, self.config.d_conv - 1, device=self.device))
#                   for _ in range(self.config.n_layers)]
#
#         with self.model.statefulness(self.b_s):
#             for t in range(self.max_len):
#                 # visual, outputs = self.iter(t, visual, outputs, return_probs, **kwargs)
#                 visual, outputs, caches = self.iter(t, visual, outputs, return_probs, caches, **kwargs)
#         outputs = outputs.contiguous()
#         log_probs =self.seq_logprob.contiguous()
#         return outputs, log_probs
#
#     def select(self, t, candidate_logprob, **kwargs):
#         selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
#         selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
#         return selected_idx, selected_logprob
#
#     def iter(self, t: int, visual: utils.TensorOrSequence, outputs, return_probs, caches=None,
#              sample=True,temperature: float = 1.0, **kwargs):
#
#         next_token_logits, caches = self.model.step(t, self.selected_words, visual, None, caches, mode='feedback', **kwargs)
#         # sample (no sampling when the prompt is being processed)
#
#         probs = F.softmax(next_token_logits.clone() / temperature, dim=-1)  # (batch_size, vocab_size)
#         if self.beam_size is not None:
#             values, _ = torch.topk(probs, k=self.beam_size)  # (batch_size, k) ordered from lowest to biggest
#             probs[probs < values[:, -1, None]] = 0
#             probs = probs / probs.sum(axis=1, keepdims=True)  # 10,167
#
#         if sample:
#             next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch_size)
#         else:
#             next_token = torch.argmax(probs, dim=-1)  # (batch_size)
#
#         next_token_probs = torch.gather(probs, dim=1, index=next_token.unsqueeze(1))
#         # self.seq_logprob = torch.cat([self.seq_logprob, next_token_probs], dim=1)
#
#         if t==0:
#             self.seq_logprob=self.seq_logprob+next_token_probs
#             outputs=outputs+next_token.unsqueeze(-1)
#         elif t>0:
#             self.seq_logprob = torch.cat([self.seq_logprob, next_token_probs], dim=1)
#             outputs = torch.cat([outputs, next_token.unsqueeze(1)], dim=1)
#
#         self.selected_words=next_token.unsqueeze(1)
#
#         return visual, outputs, caches
#####################################################################################################################
import torch
import utils
from models.transformer.mamba import MambaConfig
from models.transformer.mamba_lm import MambaLMConfig


class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int, config=None):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

        self.config = MambaConfig()

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape  # 10,49,2048
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]  # 10,1,49,2048
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]  # 50,49,2048
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(
                1 for _ in range(len(visual_exp_shape) - 2))  # 10，5，1，1
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]  # 10，5，49，2048
            visual_exp = visual.view(visual_exp_shape)  # 10,49,2048 -> 10,1,49,2048
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(
                selected_beam_exp_size)  # 10,5 -> 10,5,1,1 -> 10，5，49，2048
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)  # 50，49，2048
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def _expand_catche(self, cache: list, cur_beam_size: int, selected_beam: torch.Tensor):
        new_catche = []
        for tup in cache:
            tensor_tuple = ()
            for tensor in tup:
                # 获取原始张量的形状
                original_shape = tensor.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + original_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + original_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(
                    1 for _ in range(len(visual_exp_shape) - 2))  # 10，5，1，1

                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]  # 10，5，49，2048
                visual_exp = tensor.view(visual_exp_shape)  # 10,49,2048 -> 10,1,49,2048
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(
                    selected_beam_exp_size)  # 10,5 -> 10,5,1,1 -> 10，5，49，2048
                tensor_tem = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)  # 50，1024，16
                tensor_tuple += (tensor_tem,)

            new_catche.append(tensor_tuple)

        return new_catche

    def apply(self, visual: utils.TensorOrSequence, out_size=1, return_probs=False, **kwargs):
        self.b_s = utils.get_batch_size(visual)
        self.device = utils.get_device(visual)
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        outputs = []

        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)

        caches = [(None, torch.zeros(self.b_s, self.config.d_inner, self.config.d_conv - 1, device=self.device))
                  for _ in range(self.config.n_layers)]

        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                # visual, outputs = self.iter(t, visual, outputs, return_probs, **kwargs)
                visual, outputs, caches = self.iter(t, visual, outputs, return_probs, caches, **kwargs)

        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                                          self.max_len,
                                                                                          all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):  # candidate_logprob 10,1,167
        # # top k
        # probs=candidate_logprob.view(self.b_s, -1)
        # values, _ = torch.topk(probs, k=self.beam_size)  # (batch_size, k) ordered from lowest to biggest
        # probs[probs < values[:, -1, None]] = 0
        # probs = probs / probs.sum(axis=1, keepdims=True)  # 10,167
        # # probs = torch.softmax(probs, dim=-1)
        # next_token = torch.multinomial(probs, num_samples=1)  # (batch_size,1)
        # next_token_probs = torch.gather(probs, dim=1, index=next_token)  # (batch_size,1)
        # next_token=next_token.expand(self.b_s,self.beam_size)
        # next_token_probs=next_token_probs.expand(self.b_s,self.beam_size)
        #
        # return next_token, next_token_probs
        # #############################################beam search
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob



    def iter(self, t: int, visual: utils.TensorOrSequence, outputs, return_probs, caches=None, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size

        # word_logprob = self.model.step(t, self.selected_words, visual, None, mode='feedback', **kwargs)

        word_logprob, caches = self.model.step(t, self.selected_words, visual, None, caches, mode='feedback', **kwargs)
        # torch.nn.functional.log_softmax(word_logprob, dim=-1)
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)  # 10,1,167
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)

        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs) #10,5;10,5
        selected_beam = selected_idx // candidate_logprob.shape[-1]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
        visual = self._expand_visual(visual, cur_beam_size, selected_beam)

        caches = self._expand_catche(caches, cur_beam_size, selected_beam)

        self.seq_logprob = selected_logprob.unsqueeze(-1) #t=0;10,5,1;
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
        outputs.append(selected_words.unsqueeze(-1))

        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)

        # return visual, outputs
        return visual, outputs, caches

