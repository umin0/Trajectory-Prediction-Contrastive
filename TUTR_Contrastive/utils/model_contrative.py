import torch
import torch.nn as nn

from utils.transformer_encoder import Encoder
from utils.transformer_decoder import Decoder


class TrajectoryModel(nn.Module):

    def __init__(self, in_size, obs_len, pred_len, embed_size,
                 enc_num_layers, int_num_layers_list, heads, forward_expansion):
        super(TrajectoryModel, self).__init__()

        # 관측 + 미래 모드를 합친 시퀀스 임베딩
        self.embedding = nn.Linear(in_size * (obs_len + pred_len), embed_size)

        # 모션 모드 인코더 (논문 기준 bottleneck feature 위치)
        self.mode_encoder = Encoder(
            embed_size, enc_num_layers, heads, forward_expansion, islinear=True
        )
        self.cls_head = nn.Linear(embed_size, 1)  # 모션 모드 score

        # 이웃(사회적) 상호작용을 위한 임베딩/디코더
        self.nei_embedding = nn.Linear(in_size * obs_len, embed_size)
        self.social_decoder = Decoder(
            embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False
        )
        self.reg_head = nn.Linear(embed_size, in_size * pred_len)  # 궤적 회귀

    def spatial_interaction(self, ped, neis, mask):
        """
        ped:  [B, K, embed_size]
        neis: [B, N, obs_len, 2]  (N: 최대 agent 수)
        mask: [B, N, N]           (invalid agent에 대한 attention mask)
        """
        # 이웃 궤적 임베딩
        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B, N, obs_len*2]
        nei_embeddings = self.nei_embedding(neis)              # [B, N, embed_size]

        # ped의 각 모드에 대해 N개의 이웃을 보도록 mask 확장
        mask_expanded = mask[:, 0:1].repeat(1, ped.shape[1], 1)  # [B, K, N]

        # social interaction 디코더
        int_feat = self.social_decoder(ped, nei_embeddings, mask_expanded)  # [B, K, embed_size]

        return int_feat  # [B, K, embed_size]

    def forward(self,
                ped_obs,
                neis_obs,
                motion_modes,
                mask,
                closest_mode_indices,
                test: bool = False,
                num_k: int = 20,
                return_feat: bool = False):
        """
        ped_obs:            [B, obs_len, 2]
        neis_obs:           [B, N, obs_len, 2]
        motion_modes:       [K, pred_len, 2]
        mask:               [B, N, N]
        closest_mode_indices: [B] (train일 때만 사용, test일 때는 None)
        test:               True면 top-k 샘플 생성 모드
        num_k:              test 시 top-k 개수
        return_feat:        True면 train 모드에서 bottleneck feature(z)를 함께 반환
                            (Makansi contrastive용)
        """

        B = ped_obs.shape[0]
        K = motion_modes.shape[0]

        # --------------------------------------------------
        # 1) 모션 모드 + 관측 궤적을 함께 인코딩 (mode encoder 입력)
        # --------------------------------------------------
        # ped_obs: [B, 1, obs_len, 2] → [B, K, obs_len, 2]
        ped_obs_rep = ped_obs.unsqueeze(1).repeat(1, K, 1, 1)        # [B, K, obs_len, 2]
        motion_modes_rep = motion_modes.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, K, pred_len, 2]

        # 관측 + 모션 모드 concat
        ped_seq = torch.cat((ped_obs_rep, motion_modes_rep), dim=-2)     # [B, K, obs_len+pred_len, 2]
        ped_seq = ped_seq.reshape(B, K, -1)                              # [B, K, (obs_len+pred_len)*2]

        # 임베딩 + encoder
        ped_embedding = self.embedding(ped_seq)                          # [B, K, embed_size]
        ped_feat = self.mode_encoder(ped_embedding)                      # [B, K, embed_size]

        # 모션 모드 score
        scores = self.cls_head(ped_feat).squeeze(-1)                     # [B, K]

        # --------------------------------------------------
        # 2) Train 모드: 가장 가까운 모드 하나를 골라 social interaction + regression
        #    + bottleneck feature 반환 (return_feat=True일 때)
        # --------------------------------------------------
        if not test:
            # closest_mode_indices: [B]
            # 각 배치별 가장 가까운 모드의 feature 선택
            batch_index = torch.arange(B, device=ped_feat.device, dtype=torch.long)  # [B]
            mode_index = closest_mode_indices.long()                                  # [B]

            # bottleneck feature z: [B, embed_size]
            bottleneck = ped_feat[batch_index, mode_index]                            # [B, E]

            # social interaction을 위해 [B, 1, E]로 확장
            closest_feat = bottleneck.unsqueeze(1)                                    # [B, 1, E]
            int_feat = self.spatial_interaction(closest_feat, neis_obs, mask)         # [B, 1, E]

            # 회귀: [B, pred_len*2]
            pred_traj = self.reg_head(int_feat.squeeze(1))                            # [B, pred_len*2]

            # Makansi contrastive를 위해 bottleneck feature를 함께 반환
            if return_feat:
                return pred_traj, scores, bottleneck  # [B, pred_len*2], [B, K], [B, E]
            else:
                return pred_traj, scores              # 기존 인터페이스 유지

        # --------------------------------------------------
        # 3) Test 모드: top-k 모드에 대해 궤적 샘플 생성
        # --------------------------------------------------
        if test:
            # top-k 모드 인덱스 선택
            top_k_indices = torch.topk(scores, k=num_k, dim=-1).indices  # [B, num_k]

            # batch index 확장
            batch_index = torch.arange(B, device=ped_feat.device, dtype=torch.long)   # [B]
            batch_index = batch_index.unsqueeze(1).repeat(1, num_k).reshape(-1)       # [B*num_k]
            flat_top_k_indices = top_k_indices.reshape(-1)                            # [B*num_k]

            # top-k 모드 feature 추출
            top_k_feat = ped_feat[batch_index, flat_top_k_indices]                    # [B*num_k, E]
            top_k_feat = top_k_feat.view(B, num_k, -1)                                # [B, num_k, E]

            # social interaction
            int_feats = self.spatial_interaction(top_k_feat, neis_obs, mask)          # [B, num_k, E]

            # 회귀: [B, num_k, pred_len*2]
            pred_trajs = self.reg_head(int_feats)                                     # [B, num_k, pred_len*2]

            return pred_trajs, scores

