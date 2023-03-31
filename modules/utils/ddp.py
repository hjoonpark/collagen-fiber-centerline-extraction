import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class DDPWraper():
    # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    def __init__(self, rank, world_size):
        """
        rank: index of current machine this script is running on (e.g., master=0, slaves=1,2,...)
        world_size: number of gpus
        """
        self.rank = rank
        self.world_size = world_size

        # 1. Setup the process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    def setup_dataloader(self, dataset, batch_size):
        # 2. Split the dataloader
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size, sampler=sampler, pin_memory=False)
        
        return dataloader

    def setup_model(self, model):
        # 3. Wrap the model with DDP
        model = model.to(self.rank)

        # wrap the model with DDP
        # device_ids tell DDP where your model is
        # output_device tells DDP where to output, in our case, it is rank
        # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
        model = DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)

        return model