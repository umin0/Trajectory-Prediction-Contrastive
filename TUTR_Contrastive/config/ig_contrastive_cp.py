# ==========================
# 기본 설정 (변경 없음)
# ==========================
OB_RADIUS = 50
OB_HORIZON = 10
PRED_HORIZON = 30
INCLUSIVE_GROUPS = []
model_hidden_dim = 128
n_clusters = 200
smooth_size = 3
random_rotation = True
traj_seg = False

lr = 1e-4
batch_size = 1024
dist_threshold = 50
epoch = 1000
EPOCH_BATCHES = 100
TEST_SINCE = 500
PRED_SAMPLES = 5
WORLD_SCALE = 1

# ==========================
# Contrastive Learning Params (default)
# ==========================
lambda_contr = 0.5      # λ
contr_tau = 0.5      # τ
contr_pos_delta = 0.33   # θ_p
contr_neg_delta = 0.33   # θ_n


# # ==========================
# # 하이퍼파라미터 탐색 범위
# # ==========================
search_space = {
    "n_clusters": [200],
    "lr": [1e-4],
    "batch_size": [1024],
    "lambda_contr": [0.5],        # contrastive loss weight
    "contr_tau": [0.4],                 # temperature
    "contr_pos_delta": [0.5],          # θ_p
    "contr_neg_delta": [0.33],          # θ_n
}
