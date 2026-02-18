# model
OB_RADIUS = 50       # observe radius, neighborhood radius
OB_HORIZON = 10      # number of observation frames
PRED_HORIZON = 30   # number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []
model_hidden_dim = 128
n_clusters=200
smooth_size = 3
random_rotation = True
traj_seg = False

# training
lr = 1e-4
batch_size = 1024
dist_threshold = 50
epoch = 1000       # total number of epochs for training
EPOCH_BATCHES = 100 # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 500    # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 5   # best of N samples

# evaluation
WORLD_SCALE = 1