from abc import abstractmethod
from typing import List, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from tamer.datamodule import vocab
from tamer.utils.utils import Hypothesis, ce_loss, to_tgt_output, to_struct_output
from einops import rearrange
from einops.einops import repeat
from torch import FloatTensor, LongTensor

from .beam_search import BeamSearchScorer

# modified from
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_utils.py#L1843


class DecodeModel(pl.LightningModule):
    @abstractmethod
    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        """decode one step

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids : LongTensor
            [b, l]

        Returns
        -------
        Tuple[FloatTensor, FloatTensor]
            [b, l, vocab_size], [b, l, l]: out, sim
        """
        raise NotImplementedError("This is an abstract method.")

    def beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
    ) -> List[Hypothesis]:
        """run beam search to decode

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        beam_size : int
        max_len : int
        alpha : float
        early_stopping : bool

        Returns
        -------
        List[Hypothesis]: [batch_size,]
        """
        print("DEBUG: Starting beam search with beam_size:", beam_size, "max_len:", max_len)
        batch_size = src[0].shape[0] * 2  # mul 2 for bi-direction
        batch_beam_size = batch_size * beam_size
        half_bb_size = batch_beam_size // 2

        for i in range(len(src)):
            # [2 * b, t, d], [l2r l2r, r2l r2l]
            src[i] = torch.cat((src[i], src[i]), dim=0)
            src_mask[i] = torch.cat((src_mask[i], src_mask[i]), dim=0)

        l2r = torch.full(
            (batch_size // 2, 1),
            fill_value=vocab.SOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        r2l = torch.full(
            (batch_size // 2, 1),
            fill_value=vocab.EOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = torch.cat((l2r, r2l), dim=0)
        print(f"DEBUG: Initial input_ids shape: {input_ids.shape}, values: {input_ids}")

        beam_scorer = BeamSearchScorer(
            batch_size, beam_size, alpha, early_stopping, self.device
        )

        # first beam search
        hyps, scores = self._beam_search(
            src=src,
            src_mask=src_mask,
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            beam_size=beam_size,
            max_len=max_len,
            temperature=temperature,
        )
        
        print(f"DEBUG: After beam search, got {len(hyps)} hypotheses")
        for i, (h, s) in enumerate(zip(hyps[:5], scores[:5])):  # Chỉ hiện 5 đầu tiên để tránh quá nhiều output
            # Chuyển h thành danh sách Python nếu là tensor
            h_list = h.tolist() if hasattr(h, 'tolist') else h
            print(f"DEBUG: Hyp {i}: score={s}, tokens={h}")
            if len(h_list) > 0:
                try:
                    print(f"DEBUG: Hyp {i} words: {vocab.indices2words(h_list)}")
                except Exception as e:
                    print(f"DEBUG: Error converting tokens to words: {e}")

        # reverse half last
        for i in range(half_bb_size, batch_beam_size):
            hyps[i] = torch.flip(hyps[i], dims=[0])

        lens = [len(h) + 1 for h in hyps]  # plus to append start token
        r2l_tgt, r2l_out = to_tgt_output(
            hyps[:half_bb_size], "r2l", self.device, pad_to_len=max(lens)
        )
        l2r_tgt, l2r_out = to_tgt_output(
            hyps[half_bb_size:], "l2r", self.device, pad_to_len=max(lens)
        )
        tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
        out = torch.cat((l2r_out, r2l_out), dim=0)

        # calculate final score
        rev_scores = self._rate(src, src_mask, tgt, out, alpha, temperature)
        rev_scores = torch.cat(
            (rev_scores[half_bb_size:], rev_scores[:half_bb_size]), dim=0
        )
        # TODO clean the code
        l2r_tgt, _ = to_tgt_output(hyps, "l2r", self.device)
        r2l_tgt, _ = to_tgt_output(hyps, "r2l", self.device)
        tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
        _, sim = self.transform(
            [src[0].repeat(2, 1, 1, 1)], [src_mask[0].repeat(2, 1, 1)], tgt
        )

        struct_out, illegal = to_struct_output(hyps, self.device)
        l = struct_out.shape[1]
        struct_loss = ce_loss(sim, struct_out, ignore_idx=-1, reduction="none")
        struct_loss = rearrange(struct_loss, "(b l) -> b l", l=l)
        mask = struct_out == -1
        n = (~mask).sum(dim=1)
        struct_loss = -torch.sum(struct_loss, dim=-1) / n
        struct_loss = rearrange(struct_loss, "(n b) -> n b", n=2).mean(dim=0)

        scores = scores + rev_scores + struct_loss
        scores = scores.masked_fill(illegal, float("-inf"))

        # [2 * b, beam_size]
        scores = rearrange(scores, "(b m) -> b m", b=batch_size)
        l2r_scores, r2l_scores = torch.chunk(scores, 2, dim=0)
        # [b, 2 * beam_size]
        scores = torch.cat((l2r_scores, r2l_scores), dim=1)
        # [batch_size, ]
        best_scores, best_indices = torch.max(scores, dim=1)
        best_split = best_indices // beam_size
        best_indices = best_indices % beam_size
        batch_indices = torch.arange(
            0, batch_size // 2, dtype=torch.long, device=self.device
        )
        best_indices = (
            best_split * half_bb_size + batch_indices * beam_size + best_indices
        )

        print("DEBUG: Final best indices:", best_indices)
        print("DEBUG: Final best scores:", best_scores)

        ret: List[Hypothesis] = []
        for idx, score in zip(best_indices, best_scores):
            hpy = Hypothesis(hyps[idx], score, "l2r")
            ret.append(hpy)
            
        print("DEBUG: Final return hypotheses:")
        for i, h in enumerate(ret):
            print(f"DEBUG: Return {i}: score={h.score}, seq={h.seq}")
            if h.seq:
                print(f"DEBUG: Return {i} words: {vocab.indices2words(h.seq)}")
                
        return ret

    def _beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        input_ids: LongTensor,
        beam_scorer: BeamSearchScorer,
        beam_size: int,
        max_len: int,
        temperature: float,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        """inner beam search

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids: LongTensor
            [b, 1]
        beam_size : int
        max_len : int

        Returns
        _______
        Tuple[List[LongTensor], FloatTensor]
            List[LongTensor]: [b * beam_size] without SOS or EOS token
            FloatTensor: [b * beam_size] corresponding scores
        """
        batch_size, cur_len = input_ids.shape
        print(f"DEBUG: _beam_search - input_ids shape: {input_ids.shape}")

        beam_scores = torch.zeros(
            batch_size, dtype=torch.float, device=self.device)

        # Kiểm tra độ dài tối đa để đảm bảo không quá dài
        max_len = min(max_len, 200)  # Giới hạn tối đa là 200 token
        
        # Biến để theo dõi số bước không cải thiện
        no_improvement_steps = 0
        last_best_score = float('-inf')

        while cur_len < max_len and not beam_scorer.is_done():
            print(f"DEBUG: Beam search step {cur_len}/{max_len}")
            next_token_logits = (
                self.transform(src, src_mask, input_ids)[0][
                    :, -1, :] / temperature
            )
            # [b *, l, v]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )
            # [batch_size, beam_size * vocab_size]
            reshape_size = next_token_scores.shape[0] // batch_size
            next_token_scores = rearrange(
                next_token_scores,
                "(b m) v -> b (m v)",
                m=reshape_size,
            )

            # [b, 2 * beam_size]
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1
            )
            
            if cur_len == 1:  # Show first step in detail
                print(f"DEBUG: First step token scores: {next_token_scores}")
                print(f"DEBUG: First step tokens: {next_tokens}")
                top_tokens = next_tokens % len(vocab)
                for i in range(min(5, batch_size)):  # Show first few batches
                    print(f"DEBUG: Batch {i} top tokens: {top_tokens[i][:5]}")
                    try:
                        words = [vocab.indices2words([t.item()])[0] for t in top_tokens[i][:5]]
                        print(f"DEBUG: Batch {i} top words: {words}")
                    except Exception as e:
                        print(f"DEBUG: Error converting tokens to words: {e}")

            next_indices = next_tokens // len(vocab)
            next_tokens = next_tokens % len(vocab)

            if cur_len == 1:
                input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
                for i in range(len(src)):
                    src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
                    src_mask[i] = repeat(
                        src_mask[i], "b ... -> (b m) ...", m=beam_size)
                print(f"DEBUG: After expansion, input_ids shape: {input_ids.shape}")

            beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
            )

            input_ids = torch.cat(
                (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1
            )
            
            if cur_len <= 3 or cur_len % 5 == 0:  # Show every 5 steps after the first 3
                print(f"DEBUG: Step {cur_len} - input_ids shape: {input_ids.shape}")
                # Show some examples of current sequences
                for i in range(min(3, input_ids.shape[0])):
                    seq = input_ids[i].tolist()
                    print(f"DEBUG: Sequence {i} (len {len(seq)}): {seq}")
                    try:
                        words = vocab.indices2words(seq)
                        print(f"DEBUG: Words {i}: {words}")
                    except Exception as e:
                        print(f"DEBUG: Error converting seq {i} to words: {e}")
            
            # Kiểm tra xem có cải thiện không
            current_best_score = beam_scores.max().item()
            if current_best_score <= last_best_score:
                no_improvement_steps += 1
                if no_improvement_steps >= 10 and cur_len > 50:  # Nếu không cải thiện sau 10 bước và đã có ít nhất 50 token
                    print(f"DEBUG: Stopping early at step {cur_len} due to no improvement")
                    break
            else:
                no_improvement_steps = 0
                last_best_score = current_best_score

            cur_len += 1

        hyps, scores = beam_scorer.finalize(input_ids, beam_scores)
        print(f"DEBUG: Finalized beam search with {len(hyps)} hypotheses")
        return hyps, scores

    def _rate(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        tgt: LongTensor,
        out: LongTensor,
        alpha: float,
        temperature: float,
    ) -> FloatTensor:
        """rate tgt and output

        Parameters
        ----------
        src : List[FloatTensor]
            [b * beam_size, t, d]
        src_mask : List[LongTensor]
            [b * beam_size, t]
        tgt : LongTensor
            [b * beam_size, l]
        out : LongTensor
            [b * beam_size, l]
        alpha : float
        temperature : float

        Returns
        -------
        FloatTensor
            [b * beam_size]
        """
        b = tgt.shape[0]
        out_hat = self.transform(src, src_mask, tgt)[0] / temperature

        loss = ce_loss(out_hat, out, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)

        mask = tgt == vocab.PAD_IDX
        penalty = (~mask).sum(dim=1) ** alpha
        loss = -torch.sum(loss, dim=1) / penalty

        return loss
