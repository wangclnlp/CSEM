# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from fairseq.sequence_generator import SequenceGenerator
import fairseq.search as search
import sacrebleu

import warnings
warnings.filterwarnings('ignore')

from fairseq.detoken import my_detokenizer

import logging

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    # Parameters for ECT
    reward_lambda_loss: float = field(
        default=0.7,
        metadata={"help": "set lambda in `loss = lambda_loss * rl_loss + (1 - lambda_loss) * mle_loss`"}
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


@register_criterion(
    "reinforcement_learning_training_with_efficient_sampling", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        reward_lambda_loss,
        ignore_prefix_size=0,
        report_accuracy=False
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.generator = None
        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary #self.src_dict.string(torch.LongTensor([2111,32,434]), bpe_symbol="subword_nmt", escape_unk=True, extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
        self.lambda_loss = reward_lambda_loss

    def padding_tensor(self, sequences, pad):
        """
        :param sequences: list of tensors
        pad: item of padding
        :return:
        """
        num = len(sequences)
        max_len = max([s.shape[0] for s in sequences])
        out_dims = (num, max_len, *sequences[0].shape[1:])
        out_tensor = sequences[0].data.new(*out_dims).fill_(pad)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
        return out_tensor, mask

    def forward(self, model, sample, reduce=True, if_valid=False,
                rl_reward_function=None, sampling_task=None,
                reward_model=None, comet_model_path=None, comet_need_reference=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:  
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if not if_valid:
            # use MLE loss only
            if rl_reward_function == None:
                sample_id = sample["id"]
                target_rewards = rewards[sample_id].type(torch.HalfTensor).cuda()
                net_output = model(**sample["net_input"], if_reward_model=True) 
                
                #mse loss
                mse_loss_func = torch.nn.MSELoss()
                pred = torch.sigmoid(net_output[0]).view(1, -1).squeeze()*10
                loss = mse_loss_func(pred, target_rewards)
                nll_loss = loss
                acc = loss.item()
                
                sample_size = (
                    sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
                )
                logging_output = {
                    "accuracy": acc,
                    "loss": loss.data,
                    "nll_loss": nll_loss.data,
                    "ntokens": sample["ntokens"],
                    "nsentences": sample["target"].size(0),
                    "sample_size": sample_size,
                }

                return loss, sample_size, logging_output
            
            # use reward model
            else:
                lambda_loss = self.lambda_loss
                if_add_reference = comet_need_reference

                if lambda_loss == 1:
                    mle_loss = 0
                else:
                    net_output = model(**sample["net_input"])
                    mle_loss, mle_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

                sampling_intput = {}
                sampling_intput['net_input'] = {}
                sampling_intput['net_input']['src_tokens'] = sample['net_input']['src_tokens']
                sampling_intput['net_input']['src_lengths'] = sample['net_input']['src_lengths']

                with torch.no_grad():
                    if sampling_task == 'translation':
                        '''translation'''
                        SAMPLING_TOPK = 50
                        search_strategy = search.Sampling(self.tgt_dict, sampling_topk=SAMPLING_TOPK)
                        generator = SequenceGenerator([model], self.tgt_dict, beam_size=5, match_source_len=False,
                                                    max_len_a=1.2, max_len_b=20, search_strategy=search_strategy)
                        out = generator.generate(sample = sampling_intput, noise=True)

                    if sampling_task == 'summarization':
                        '''summariztion'''
                        SAMPLING_TOPK = 50
                        search_strategy = search.Sampling(self.tgt_dict, sampling_topk=SAMPLING_TOPK)
                        generator = SequenceGenerator([model], self.tgt_dict, beam_size=5, match_source_len=False,
                                                    min_len=55, max_len=140, search_strategy=search_strategy, no_repeat_ngram_size=3)
                        out = generator.generate(sample = sampling_intput, noise=True)

                    if sampling_task == 'style_transfer':
                        '''text style transfer'''
                        SAMPLING_TOPK = 50
                        search_strategy = search.Sampling(self.tgt_dict, sampling_topk=SAMPLING_TOPK)
                        generator = SequenceGenerator([model], self.tgt_dict, beam_size=5, match_source_len=False,
                                                    max_len_a=1.2, max_len_b=20, search_strategy=search_strategy, no_repeat_ngram_size=3)
                        out = generator.generate(sample = sampling_intput, noise=True)

                # extract all sampled generations
                sample_num = len(out[0])
                out_len = len(out)
                all_sampled_generations = []
                preset_prev_output_tokens = []
                for j in range(sample_num):
                    for i in range(out_len):
                        if out[i][j]["tokens"][-1] == self.tgt_dict.eos_index:
                            all_sampled_generations.append(out[i][j]["tokens"])
                            preset_prev_output_tokens.append(torch.cat((torch.LongTensor([self.tgt_dict.eos_index]).cuda(), out[i][j]["tokens"][:-1])))
                        else:
                            all_sampled_generations.append(torch.cat((out[i][j]["tokens"], torch.LongTensor([self.tgt_dict.eos_index]).cuda())))
                            preset_prev_output_tokens.append(torch.cat((torch.LongTensor([self.tgt_dict.eos_index]).cuda(), out[i][j]["tokens"])))

                # mask these generations to form a tensor
                all_sampled_generations, all_sampled_generations_mask = self.padding_tensor(all_sampled_generations, self.padding_idx)

                preset_prev_output_tokens, _ = self.padding_tensor(preset_prev_output_tokens, self.padding_idx)

                # record human references
                if if_add_reference:
                    human_references = sample['target'].clone()
                    human_references = human_references.repeat(sample_num, 1)

                # batch reconsitution
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].repeat(sample_num, 1)
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].repeat(sample_num, 1).view(-1, 1).squeeze()
                sample['net_input']['prev_output_tokens'] = preset_prev_output_tokens
                sample['target'] = all_sampled_generations
                

                # compute rewards by using the trained reward model. 
                reward_model_net_input = {}
                reward_model_net_input['reward_model_net_input'] = {}
                reward_model_net_input['reward_model_net_input']['src_tokens'] = sample['net_input']['src_tokens']
                reward_model_net_input['reward_model_net_input']['src_lengths'] = sample['net_input']['src_lengths']
                reward_model_net_input['reward_model_net_input']['prev_output_tokens'] = preset_prev_output_tokens

                ignore_tokens = get_symbols_to_strip_from_output(generator)
                ignore_tokens.add(1)
                if rl_reward_function == "comet_model":
                    src_str = self.src_dict.string(reward_model_net_input['reward_model_net_input']['src_tokens'], 
                                                bpe_symbol="subword_nmt", 
                                                escape_unk=True, 
                                                extra_symbols_to_ignore=ignore_tokens).split('\n')
                    
                    tgt_str = self.tgt_dict.string(sample['target'],
                                                bpe_symbol="subword_nmt", 
                                                escape_unk=True, 
                                                extra_symbols_to_ignore=ignore_tokens).split('\n')
                    if if_add_reference:
                        ref_str = self.src_dict.string(human_references, 
                                                bpe_symbol="subword_nmt", 
                                                escape_unk=True, 
                                                extra_symbols_to_ignore=ignore_tokens).split('\n')

                    src_str = my_detokenizer(src_str)
                    tgt_str = my_detokenizer(tgt_str)
                    # with reference
                    if if_add_reference:
                        ref_str = my_detokenizer(ref_str)

                    data = []
                    # len_tokens = 0
                    if if_add_reference:
                        for s, r, t in zip(src_str, ref_str, tgt_str):
                            data.append({"src": s, "ref":r, "mt": t})
                            # len_tokens += (len(s.split())+len(r.split())+len(t.split()))
                    else:
                        for s, t in zip(src_str, tgt_str):
                            data.append({"src": s, "mt": t})
                            # len_tokens += (len(s.split())+len(t.split()))
                    # print(len_tokens)
                    # import os
                    # gpus = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
                    # print(out_len)
                    output = reward_model.predict(data, batch_size=8, gpus=1, progress_bar = False, num_workers=0, devices=[sample["id"].device.index])
                    reward_model_output = torch.tensor(output.scores).to(next(model.parameters()).device)
                    # from fairseq import pdb ; pdb.set_trace()
                    # print(reward_model_output)
                    reward_model_output = [reward_model_output]
                    computed_rewards = torch.sigmoid(reward_model_output[0])*10
                elif rl_reward_function == "bleu":
                    '''use the bleu reward'''
                    all_rewards = []
                    src_str = self.src_dict.string(reward_model_net_input['reward_model_net_input']['src_tokens'], 
                                                bpe_symbol="subword_nmt", 
                                                escape_unk=True, 
                                                extra_symbols_to_ignore=ignore_tokens).split('\n')
                    
                    tgt_str = self.tgt_dict.string(sample['target'],
                                                bpe_symbol="subword_nmt", 
                                                escape_unk=True, 
                                                extra_symbols_to_ignore=ignore_tokens).split('\n')
                    ref_str = self.src_dict.string(human_references, 
                                            bpe_symbol="subword_nmt", 
                                            escape_unk=True, 
                                            extra_symbols_to_ignore=ignore_tokens).split('\n')

                    src_str = my_detokenizer(src_str)
                    tgt_str = my_detokenizer(tgt_str)
                    # with reference
                    ref_str = my_detokenizer(ref_str)

                    for r, t in zip(ref_str, tgt_str):
                        all_rewards.append(sacrebleu.corpus_bleu([t], [[r]], tokenize="13a").score)

                    reward_model_output = torch.tensor(all_rewards).to(next(model.parameters()).device)
                    computed_rewards = torch.sigmoid(reward_model_output)/10
                else:
                    raise Exception(f"Unknown --rl-reward-function {rl_reward_function}")

                # rl finetune
                net_output = model(**sample['net_input'])
                lprobs = F.log_softmax(net_output[0], dim=-1)
                sentence_probs = lprobs.gather(dim=-1, index=sample['target'].unsqueeze(-1))
                sentence_probs = torch.sum(sentence_probs, dim=1).view(sample_num, out_len)
                Q_ = torch.softmax(sentence_probs*0.005, dim=0)
                rewards = computed_rewards.view(sample_num, out_len)
                loss = Q_*rewards
                if reduce:
                    loss = -loss.sum()

                loss = lambda_loss*loss + (1-lambda_loss)*mle_loss

                sample_size = (
                    sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
                )

                logging_output = {
                    "loss": loss.data,
                    "ntokens": sample["ntokens"],
                    "nsentences": sample["target"].size(0),
                    "sample_size": sample_size,
                } 
                if self.report_accuracy:
                    n_correct, total = self.compute_accuracy(model, net_output, sample)
                    logging_output["n_correct"] = utils.item(n_correct.data)
                    logging_output["total"] = utils.item(total.data)

                return loss, sample_size, logging_output
            
        # valid step
        else:
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)
            return loss, sample_size, logging_output
        

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
