import segmentation_models_pytorch as smp
from utils.Models import MLP, timeUnetPlusPlus
import torch.nn.functional as F
from utils.bridge.models import UNetModel


def get_model_before_after(args):
    if args.task == 'gaussian2mnist':
        if args.model == 'tunet++':
            model = timeUnetPlusPlus(
                encoder_name="efficientnet-b4",
                model_channels=64,
                in_channels=2,
                encoder_depth=3, 
                decoder_channels=(64, 32, 16),
                classes=1,
                )
            before_train = lambda x: F.pad(x, (2, 2, 2, 2, 0, 0, 0, 0), 'constant', 0)
            after_train = lambda x: x[:, :, 2:-2, 2:-2]
        elif args.model == 'unet++':
            model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b5",
                model_channels=64,
                in_channels=2,
                encoder_depth=3, 
                decoder_channels=(64, 32, 16),
                classes=1,
                )
            before_train = lambda x: F.pad(x, (2, 2, 2, 2, 0, 0, 0, 0), 'constant', 0)
            after_train = lambda x: x[:, :, 2:-2, 2:-2]
        elif args.model == 'unet':
            model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b0",
                model_channels=32,
                in_channels=2,
                encoder_depth=3, 
                decoder_channels=(64, 32, 16),
                classes=1,
                )
            before_train = lambda x: F.pad(x, (2, 2, 2, 2, 0, 0, 0, 0), 'constant', 0)
            after_train = lambda x: x[:, :, 2:-2, 2:-2]
        elif args.model == 'aunet':
            # kwargs = {'in_channels': 2, 
            #           'model_channels': 64, 
            #           'out_channels': 1, 
            #           'num_res_blocks': 2, 
            #           'attention_resolutions': (0,), 
            #           'dropout': 0.0, 
            #           'channel_mult': (1, 2, 2), 
            #           'num_classes': None, 
            #           'use_checkpoint': False, 
            #           'num_heads': 4, 
            #           'num_heads_upsample': -1, 
            #           'use_scale_shift_norm': True
            #           }
            kwargs = {'in_channels': 2, 
                    'model_channels': 64, 
                    'out_channels': 1, 
                    'num_res_blocks': 4, 
                    'attention_resolutions': (0,), 
                    'dropout': 0.0, 
                    'channel_mult': (1, 2, 4), 
                    'num_classes': None, 
                    'use_checkpoint': False, 
                    'num_heads': 8, 
                    'num_heads_upsample': -1, 
                    'use_scale_shift_norm': True
                    }

            model = UNetModel(**kwargs)
            before_train = None
            after_train = None
    else:
        model = MLP(
            input_dim=3, 
            output_dim=1, 
            hidden_layers=4, 
            hidden_dim=256
            )
        before_train = None
        after_train = None
    return model, before_train, after_train