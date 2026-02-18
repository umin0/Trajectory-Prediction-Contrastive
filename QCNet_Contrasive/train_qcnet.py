# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from datamodules import ArgoverseV2DataModule_ACL
from predictors import QCNet
import os 

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    QCNet.add_model_specific_args(parser)
    # args = parser.parse_args()
    args = parser.parse_args([
        '--root',                   '/home/user/Algorithm/QCNet_cum/Code/QCNet_Contrasive/Data/LIS',
        '--train_batch_size',       '16',
        '--val_batch_size',         '16',
        '--test_batch_size',        '16',
        '--devices',                '6',
        '--dataset',                'argoverse_v2_ACL',
        '--num_historical_steps',   '10',
        '--num_future_steps',       '30',
        '--num_recurrent_steps',    '3',
        '--pl2pl_radius',           '150',
        '--time_span',              '10',
        '--pl2a_radius',            '50',
        '--a2a_radius',             '50',
        '--num_t2m_steps',          '30',
        '--pl2m_radius',            '150',
        '--a2m_radius',             '150'
    ])
    model = QCNet(**vars(args))
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
        'argoverse_v2_ACL' : ArgoverseV2DataModule_ACL
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=2, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)
