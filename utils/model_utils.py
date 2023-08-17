import segmentation_models_pytorch as smp
from utils.Models import MLP


def get_model_before_after(args):
    if args.task == 'gaussian2minst':
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            in_channels=3,
            encoder_depth=3, 
            decoder_channels=(64, 32, 16),
            classes=1,
            )
        before_train = lambda x: F.pad(x, (2, 2, 2, 2, 0, 0, 0, 0), 'constant', 0)
        after_train = lambda x: x[:, :, 2:-2, 2:-2]
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