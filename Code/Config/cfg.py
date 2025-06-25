import argparse
import socket


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GAVE')
parser.add_argument('--num_iterations', type=int, default=5)
parser.add_argument('--criterion', type=str, default='RRLoss')
parser.add_argument('--base_criterion', type=str, default='BCE3Loss')
parser.add_argument('--model', type=str, default='GAVENet')
parser.add_argument('--num_folds', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=1e-04)
parser.add_argument('--num_epochs', type=int, default=None)
parser.add_argument('--base_channels', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--n_proc', type=int, default=1)
parser.add_argument('--data_folder', type=str, default='../Data/')
parser.add_argument('--version', type=str, default='test0625')
parser.add_argument('--seed', type=int, default=3)
args = parser.parse_args()


num_folds = args.num_folds
active_folds = range(num_folds)

learning_rate = args.learning_rate
num_epochs = args.num_epochs

dataset = args.dataset

model = args.model
in_channels = args.in_channels
out_channels = args.out_channels
if socket.gethostname() == 'hemingway':
    args.base_channels = 16
base_channels = args.base_channels
num_iterations = args.num_iterations

criterion = args.criterion
base_criterion = args.base_criterion

n_proc = args.n_proc
gpu_id = args.gpu_id


training_folder = f'../Log/'
version = args.version

seed = args.seed

if dataset == 'GAVE':
    images = [
        'g_001', 'g_002', 'g_003', 'g_004', 'g_005', 'g_006', 'g_007', 'g_008', 'g_009', 'g_010',
        'g_011', 'g_012', 'g_013', 'g_014', 'g_015', 'g_016', 'g_017', 'g_018', 'g_019', 'g_020',
        'g_021', 'g_022', 'g_023', 'g_024', 'g_025', 'g_026', 'g_027', 'g_028', 'g_029', 'g_030',
        'g_031', 'g_032', 'g_033', 'g_034', 'g_035', 'g_036', 'g_037', 'g_038', 'g_039', 'g_040',
        'g_041', 'g_042', 'g_043', 'g_044', 'g_045', 'g_046', 'g_047', 'g_048', 'g_049', 'g_050',
        'g_051', 'g_052', 'g_053', 'g_054', 'g_055', 'g_056', 'g_057', 'g_058', 'g_059', 'g_060',
        'g_061', 'g_062', 'g_063', 'g_064', 'g_065', 'g_066', 'g_067', 'g_068', 'g_069', 'g_070',
        'g_071', 'g_072', 'g_073', 'g_074', 'g_075', 'g_076', 'g_077', 'g_078', 'g_079', 'g_080',
        'g_081', 'g_082', 'g_083', 'g_084', 'g_085', 'g_086', 'g_087', 'g_088', 'g_089', 'g_090',
        'g_091', 'g_092', 'g_093', 'g_094', 'g_095', 'g_096', 'g_097', 'g_098', 'g_099', 'g_100',
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': f'GAVE/training/av',
            'pattern': r'^g_\d{3}\.png$'
        },
        'original': {
            'path': f'GAVE/training/images',
            'pattern': r'^g_\d{3}\.png$'
        },
        'mask': {
            'path': f'GAVE/training/masks',
            'pattern': r'^g_\d{3}\.png$'
        }
    }

else:
    raise ValueError('dataset not supported')

