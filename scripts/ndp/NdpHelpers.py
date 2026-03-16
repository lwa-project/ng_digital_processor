
import os
import glob

from .NdpConfig import parse_config_file
from .NdpCommon import compute_constants


def cores_to_numa_binding(cores):
    """
    Given a list of core numbers, return a string suitable for passing to
    `numactl` for CPU node and memory binding.
    """

    nodes = []
    for c in cores:
        node = glob.glob(f"/sys/devices/system/cpu/cpu{c}/node*")[0]
        node = os.path.basename(node)
        node = int(node.replace('node', ''), 10)
        if node not in nodes:
            nodes.append(node)
    nodes.sort()

    node_sets = [[nodes[0],nodes[0]]]
    for i in range(1, len(nodes)):
        if nodes[i] == node_sets[-1][1]+1:
            node_sets[-1][1] = nodes[i]
        else:
            node_sets.append([nodes[i],nodes[i]])

    binding = ''
    for ns in node_sets:
        if ns[0] == ns[1]:
            binding += f"{ns[0]},"
        else:
            binding += f"{ns[0]}-{ns[1]},"
    binding = binding[:-1]
    return binding


def pipeline_setup(config_file, server, pipeline):
    """
    Compute and print shell-evaluable pipeline configuration for use
    in systemd service files.
    """

    config = parse_config_file(config_file)
    computed = compute_constants(config)
    npipe = computed['NPIPE_PER_SERVER']

    print(f"npipe={npipe}")
    if pipeline < npipe:
        tuning = (server - 1) * npipe + pipeline
        binding = cores_to_numa_binding(config['drx'][tuning]['cpus'])
        print(f"tuning={tuning}")
        print(f"binding={binding}")
