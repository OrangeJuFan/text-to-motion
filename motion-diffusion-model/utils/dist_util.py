"""
Helpers for distributed training.
"""

import socket
import platform

import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

used_device = 0

def setup_dist(device=0):
    """
    Setup a distributed process group.
    
    参数:
        device: 设备标识，可以是：
            - 整数: CUDA 设备 ID (如 0, 1, 2)
            - 字符串: 'cuda', 'mps', 'cpu', 或 'cuda:0', 'cuda:1' 等
    """
    global used_device
    used_device = device
    if dist.is_initialized():
        return
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device) # f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    # comm = MPI.COMM_WORLD
    # backend = "gloo" if not th.cuda.is_available() else "nccl"

    # if backend == "gloo":
    #     hostname = "localhost"
    # else:
    #     hostname = socket.gethostbyname(socket.getfqdn())
    # os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    # os.environ["RANK"] = str(comm.rank)
    # os.environ["WORLD_SIZE"] = str(comm.size)

    # port = comm.bcast(_find_free_port(), root=used_device)
    # os.environ["MASTER_PORT"] = str(port)
    # dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    
    自动检测并返回最佳可用设备：
    - 如果指定了 CUDA 设备 ID，且 CUDA 可用，返回 cuda:ID
    - 如果指定了 'mps' 或 'mps:0'，且 MPS 可用，返回 mps
    - 如果指定了 'cuda' 或 'cuda:0'，且 CUDA 可用，返回 cuda:0
    - 否则返回 cpu
    
    返回:
        torch.device: 设备对象
    """
    global used_device
    
    # 处理字符串类型的设备
    if isinstance(used_device, str):
        device_str = used_device.lower()
        
        # MPS 设备（macOS）
        if device_str in ['mps', 'mps:0']:
            if hasattr(th.backends, 'mps') and th.backends.mps.is_available():
                return th.device("mps")
            else:
                print("警告: MPS 不可用，回退到 CPU")
                return th.device("cpu")
        
        # CUDA 设备
        elif device_str.startswith('cuda'):
            if ':' in device_str:
                # 格式: 'cuda:0', 'cuda:1' 等
                cuda_id = int(device_str.split(':')[1])
                if th.cuda.is_available():
                    return th.device(f"cuda:{cuda_id}")
                else:
                    print(f"警告: CUDA 不可用，回退到 CPU")
                    return th.device("cpu")
            else:
                # 格式: 'cuda'
                if th.cuda.is_available():
                    return th.device("cuda:0")
                else:
                    print("警告: CUDA 不可用，回退到 CPU")
                    return th.device("cpu")
        
        # CPU 设备
        elif device_str == 'cpu':
            return th.device("cpu")
        
        else:
            print(f"警告: 未知的设备类型 '{used_device}'，回退到 CPU")
            return th.device("cpu")
    
    # 处理整数类型的设备（向后兼容）
    elif isinstance(used_device, int):
        if used_device >= 0 and th.cuda.is_available():
            return th.device(f"cuda:{used_device}")
        elif used_device >= 0 and hasattr(th.backends, 'mps') and th.backends.mps.is_available():
            # 如果 CUDA 不可用但 MPS 可用，使用 MPS
            print(f"提示: CUDA 不可用，使用 MPS 设备")
            return th.device("mps")
        else:
            return th.device("cpu")
    
    # 默认情况
    else:
        # 自动检测最佳设备
        if th.cuda.is_available():
            return th.device("cuda:0")
        elif hasattr(th.backends, 'mps') and th.backends.mps.is_available():
            return th.device("mps")
        else:
            return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
