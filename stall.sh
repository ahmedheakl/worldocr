# this script stalls the gpu for 3 days, make slight use of the gpus. like use some memory and compute (gpu utilization)
#!/bin/bash
while true; do
    # Allocate some GPU memory
    python -c "import torch; a = torch.randn((1024, 1024, 10)).cuda(); torch.cuda.synchronize(); del a; torch.cuda.empty_cache()"
    
    # Perform some light computation
    python -c "import torch; x = torch.randn((1000, 1000)).cuda(); y = torch.matmul(x, x); torch.cuda.synchronize(); del x, y; torch.cuda.empty_cache()"
done
