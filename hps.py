import numpy as np

HPARAMS_REGISTRY = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


vmpo_none = Hyperparams()
# model
vmpo_none.model = 'vmpo'
vmpo_none.state_rep = 'none'
# env
vmpo_none.env_name = 'rooms_watermaze'
vmpo_none.action_dim = 9
vmpo_none.action_list = np.array([
        [0, 0, 0, 1, 0, 0, 0],    # Forward
        [0, 0, 0, -1, 0, 0, 0],   # Backward
        [0, 0, -1, 0, 0, 0, 0],   # Strafe Left
        [0, 0, 1, 0, 0, 0, 0],    # Strafe Right
        [-20, 0, 0, 0, 0, 0, 0],  # Look Left
        [20, 0, 0, 0, 0, 0, 0],   # Look Right
        [-20, 0, 0, 1, 0, 0, 0],  # Look Left + Forward
        [20, 0, 0, 1, 0, 0, 0],   # Look Right + Forward
        [0, 0, 0, 0, 1, 0, 0],    # Fire.
    ])
HPARAMS_REGISTRY['vmpo_none'] = vmpo_none


ppo_none = Hyperparams()
ppo_none.update(vmpo_none)
# model
ppo_none.model = 'ppo'
HPARAMS_REGISTRY['ppo_none'] = ppo_none



def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        for k in hps:
            if k not in valid_args:
                raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
    H.update(parser.parse_args(s).__dict__)


def add_arguments(parser):

    # utils
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--hparam_sets', '--hps', type=str)
    parser.add_argument('--gpu', type=str, default='0')

    # model
    parser.add_argument('--model', type=str, default='vmpo', help='{vmpo|ppo}')
    parser.add_argument('--state_rep', type=str, default='none', help='{none|lstm|trxl|gtrxl}')
    parser.add_argument('--n_latent_var', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--mem_len', type=int, default=10)

    # env
    parser.add_argument('--env_name', type=str, default='rooms_watermaze')
    parser.add_argument('--action_dim', type=int, default=9)
    parser.add_argument('--log_interval', type=int, default=40)
    parser.add_argument('--max_episodes', type=int, default=50000)
    parser.add_argument('--max_timesteps', type=int, default=300)
    parser.add_argument('--update_timestep', type=int, default=1200)
    parser.add_argument('--action_list', type=list, default=[])

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--K_epochs', type=int, default=4)
    parser.add_argument('--eps_clip', type=float, default=0.2)

    return parser
