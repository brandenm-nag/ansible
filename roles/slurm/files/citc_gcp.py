import re
from collections import defaultdict
import subprocess
import time
from typing import Dict, Optional, Tuple
from google.oauth2 import service_account  # type: ignore
import googleapiclient.discovery  # type: ignore
import logging
import yaml
import os
from pathlib import Path
import asyncio

__all__ = ["get_nodespace", "start_node", "start_nodes"]


def load_yaml(filename: str) -> dict:
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def get_nodespace(file="/etc/citc/startnode.yaml") -> Dict[str, str]:
    """
    Get the information about the space into which we were creating nodes
    This will be static for all nodes in this cluster
    """
    return load_yaml(file)


def wait_for_operation(gce_compute, project, operation):
    result = None
    while True:
        if 'zone' in operation:
            result = gce_compute.zoneOperations().get(
                        project=project,
                        zone=operation['zone'].split('/')[-1],
                        operation=operation['name']).execute()
        elif 'region' in operation:
            result = gce_compute.regionOperations().get(
                        project=project,
                        region=operation['region'].split('/')[-1],
                        operation=operation['name']).execute()
        else:
            result = gce_compute.globalOperations().get(
                        project=project,
                        operation=operation['name']).execute()
        if result['status'] == 'DONE':
            if 'error' in result:
                raise Exception(result['error'])
            return result
        time.sleep(1)


def create_placement_group(log, gce_compute, vmCount, shape, nodespace):
    name=f'pg-{nodespace["cluster_id"]}-{shape}-{vmCount}-{int(time.time()*100000)}'
    log.info(f"Creating placement group '{name}' of size {vmCount}")
    operation = gce_compute.resourcePolicies().insert(
                region=nodespace["region"],
                project=nodespace["compartment_id"],
                body={
                    'name': name,
                    'groupPlacementPolicy': {
                        # Max size, via doc: 
                        # https://cloud.google.com/compute/docs/instances/define-instance-placement#restrictions
                        "vmCount": vmCount,
                        "collocation": "CLUSTERED"
                    }
                }).execute()

    url = operation['targetLink'] if 'targetLink' in operation else None
    wait_for_operation(gce_compute, nodespace["compartment_id"], operation)
    return url

def delete_placement_group(log, gce_compute, region, project, name):
    log.info(f"Attempting delete placement group '{name}'")
    try:
        operation = gce_compute.resourcePolicies().delete(
                region=region,
                project=project,
                resourcePolicy=name).execute()
    except googleapiclient.errors.HttpError as e:
        log.info(f"Unable to delete placement group '{name}': {e}")
        # Note, for deleting placement groups, we're not going to care if it has an error or not
        # We can't trivially check in advance to see how many instances may be connected to it
        pass

def list_placement_groups(log, gce_compute, region, project, cluster_id):
    op = gce_compute.resourcePolicies().list(
            project=project, region=region,
            filter=f'name = "pg-{cluster_id}-*"')
    results = op.execute()
    if 'items' in results:
        groups = [x['name'] for x in results['items']]
        log.info(f"Found Groups: {groups}")
        return groups
    return []


def get_node(gce_compute, log, compartment_id: str, zone: str, hostname: str, cluster_id: str) -> Dict:
    filter_clause = f"name={hostname} AND labels.cluster={cluster_id}"

    result = gce_compute.instances().list(project=compartment_id, zone=zone, filter=filter_clause).execute()
    item = result['items'][0] if 'items' in result else None
    log.debug(f'get items {item}')
    return item


def get_node_state(gce_compute, log, compartment_id: str, zone: str, hostname: str, cluster_id: str) -> Optional[str]:
    """
    Get the current node state of the VM for the given hostname
    If there is no such VM, return "TERMINATED"
    """

    item = get_node(gce_compute, log, compartment_id, zone, hostname, cluster_id)

    if item:
        return item['status']
    return None


def get_ip_for_vm(gce_compute, log, compartment_id: str, zone: str, hostname: str, cluster_id: str) -> str:
    item = get_node(gce_compute, log, compartment_id, zone, hostname, cluster_id)
    if not item:
        return None

    nics = item.get('networkInterfaces', [])
    if not nics:
        return None
    network = nics[0]
    log.debug(f'network {network}')
    ip = network.get('networkIP', None)
    return ip

def get_node_features(hostname):
    features = subprocess.run(
        ["sinfo", "--Format=features:200", "--noheader", f"--nodes={hostname}"],
        stdout=subprocess.PIPE
    ).stdout.decode().strip().split(',')
    features = {f.split("=")[0]: f.split("=")[1] for f in features}
    return features

def should_use_tier_1_networking(machine_type):
    mach_info = machine_type.split('-')
    if mach_info[0] == "c2" and int(mach_info[2]) >= 30:
        return True
    if mach_info[0] == "n2" and int(mach_info[2]) >= 32:
        return True
    return False
    


def create_node_config(gce_compute, hostname: str, ip: Optional[str], nodespace: Dict[str, str], ssh_keys: str):
    """
    Create the configuration needed to create ``hostname`` in ``nodespace`` with ``ssh_keys``
    """
    features = get_node_features(hostname)
    shape = features["shape"]
    subnet = nodespace["subnet"]
    zone = nodespace["zone"]
    image_family = f'citc-slurm-compute-{nodespace["cluster_id"]}'

    with open("/home/slurm/bootstrap.sh", "rb") as f:
        user_data = f.read().decode()

    machine_type = f"zones/{zone}/machineTypes/{shape}"

    image_response = gce_compute.images().getFromFamily(project=nodespace["compartment_id"], family=image_family).execute()
    source_disk_image = image_response['selfLink']

    config = {
        'name': hostname,
        'machineType': machine_type,

        'disks': [
            {
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': source_disk_image,
                }
            }
        ],
        'networkInterfaces': [
            {
                'subnetwork': subnet,
                'nicType': "GVNIC",
                'addressType': 'INTERNAL',  # Can't find this in the docs...
                'networkIP': ip,
                'accessConfigs': [
                    {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                ]
            }
        ],
        'metadata': {
            'items': [{
                'key': 'startup-script',
                'value': user_data
            }]
        },
        'tags': {
            "items": [
                f"compute-{nodespace['cluster_id']}",
            ],
        },
        'labels': {
            "cluster": nodespace['cluster_id']
        }
    }
    # Only 'n1' instances really should specify minimum CPU type
    # Other instances only have a single CPU type, and require you pick
    # the correct one (if set)
    if shape.startswith('n1-') or shape.startswith('e2-'):
        config['minCpuPlatform'] = 'Intel Skylake'

    if should_use_tier_1_networking(shape):
        config["networkPerformanceConfigs"] = {
            "totalEgressBandwidthTier": "TIER_1"
        }

    return config


def get_ip(hostname: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    host_dns_match = re.match(r"(\d+\.){3}\d+", subprocess.run(["host", hostname], stdout=subprocess.PIPE).stdout.decode().split()[-1])
    dns_ip = host_dns_match.group(0) if host_dns_match else None

    slurm_dns_match = re.search(r"NodeAddr=((\d+\.){3}\d+)", subprocess.run(["scontrol", "show", "node", hostname], stdout=subprocess.PIPE).stdout.decode())
    slurm_ip = slurm_dns_match.group(1) if slurm_dns_match else None

    ip = dns_ip or slurm_ip

    return ip, dns_ip, slurm_ip


def get_credentials():
    service_account_file = Path(os.environ.get('SA_LOCATION', '/home/slurm/mgmt-sa-credentials.json'))
    if service_account_file.exists():
        return service_account.Credentials.from_service_account_file(service_account_file)
    return None


def get_build():
    credentials = get_credentials()

    compute = googleapiclient.discovery.build('compute', 'beta', credentials=credentials, cache_discovery=False)

    return compute


async def start_node(log, host: str, nodespace: Dict[str, str], ssh_keys: str) -> None:
    project = nodespace["compartment_id"]
    zone = nodespace["zone"]
    cluster_id = nodespace["cluster_id"]

    log.info(f"Starting {host} in {project} {zone} cluster {cluster_id}")

    gce_compute = get_build()

    while get_node_state(gce_compute, log, project, zone, host, nodespace["cluster_id"]) in ["STOPPING", "TERMINATED"]:
        log.info(" host is currently being deleted. Waiting...")
        await asyncio.sleep(5)

    node_state = get_node_state(gce_compute, log, project, zone, host, nodespace["cluster_id"])
    if node_state is not None:
        log.warning(f" host is already running with state {node_state}")
        return

    ip, _dns_ip, slurm_ip = get_ip(host)

    instance_details = create_node_config(gce_compute, host, ip, nodespace, ssh_keys)
    log.debug(f"Host Config: {instance_details}")

    loop = asyncio.get_event_loop()

    try:
        inserter = gce_compute.instances().insert(project=project, zone=zone, body=instance_details)
        instance = await loop.run_in_executor(None, inserter.execute)
        watcher = gce_compute.zoneOperations().get(project=project, zone=zone, operation=instance['name'])
        while True:
            r = await loop.run_in_executor(None, watcher.execute)
            if r['status'] == 'DONE':
                if r.get('error', None):
                    err = r['error']['errors'][0]
                    raise Exception(err['message'])
                break
            else:
                log.info(f"{host}:  Waiting for start operation to complete...")
                await asyncio.sleep(5)
    except Exception as e:
        log.error(f" problem launching instance: {e}")
        return

    if not slurm_ip:
        vm_ip = get_ip_for_vm(gce_compute, log, project, zone, host, nodespace["cluster_id"])
        while not vm_ip:
            log.info(f"{host}:  No VNIC attachment yet. Waiting...")
            await asyncio.sleep(5)
            vm_ip = get_ip_for_vm(gce_compute, log, project, zone, host, nodespace["cluster_id"])

        log.info(f"  Private IP {vm_ip}")

        subprocess.run(["scontrol", "update", f"NodeName={host}", f"NodeAddr={vm_ip}"])

    log.info(f" Started {host}")

    return instance


def add_instance_cb(request_id, response, exception):
    if exception is not None:
        log.error(f"Exception while starting node {request_id}: {exception}")



async def start_node_group(log, hosts, nodespace: Dict[str, str], ssh_keys: str) -> None:
    project = nodespace["compartment_id"]
    zone = nodespace["zone"]
    region = nodespace["region"]
    cluster_id = nodespace["cluster_id"]
    gce_compute = get_build()

    if len(hosts) == 1:
        await start_node(log, hosts[0], nodespace, ssh_keys)
        return

    log.info(f"Starting {len(hosts)} hosts in a group {hosts}")
    features = get_node_features(hosts[0])

    BATCH_SIZE = 100 # Currently, same as max placement group size


    batches = []
    curr_batch_size = 0
    placement_group = None
    for host in hosts:
        if curr_batch_size == 0:
            num_processed = BATCH_SIZE * len(batches)
            placement_group = create_placement_group(log, gce_compute, min(len(hosts)-num_processed, BATCH_SIZE), features["shape"], nodespace) if features["pg"] == 'True' else None
            batches.append(gce_compute.new_batch_http_request(callback=add_instance_cb))

        ip, _dns_ip, slurm_ip = get_ip(host)
        instance_details = create_node_config(gce_compute, host, ip, nodespace, ssh_keys)

        if placement_group:
            instance_details["resourcePolicies"] = [ placement_group ]
            # Required when using compact placement
            instance_details["scheduling"] = {
                "onHostMaintenance": "TERMINATE",
                "automaticRestart": False
            }


        batches[-1].add(gce_compute.instances().insert(project=project, zone=zone, body=instance_details), request_id=host)
        curr_batch_size += 1
        if curr_batch_size >= BATCH_SIZE:
            curr_batch_size = 0 # Will cause a new batch to be added next loop

    # Start up the nodes
    first_batch = True
    for batch in batches:
        if not first_batch:
            await asyncio.sleep(30) # Delay ... is this necessary?

        try:
            batch.execute()
        except Exception as e:
            log.error(f" problem launching batched instances {e}")
            return

    if not slurm_ip:
        for host in hosts:
            while not get_node(gce_compute, log, project, zone, host, nodespace["cluster_id"])['networkInterfaces'][0].get("networkIP"):
                log.info(f"{host}:  No VNIC attachment yet. Waiting...")
                await asyncio.sleep(5)
            vm_ip = get_ip_for_vm(gce_compute, log, project, zone, host, nodespace["cluster_id"])

            log.info(f"  Private IP {vm_ip}")

            subprocess.run(["scontrol", "update", f"NodeName={host}", f"NodeAddr={vm_ip}"])



async def start_nodes(log, hosts, nodespace: Dict[str, str], ssh_keys: str) -> None:
    project = nodespace["compartment_id"]
    zone = nodespace["zone"]
    region = nodespace["region"]
    cluster_id = nodespace["cluster_id"]
    gce_compute = get_build()

    # Wait for nodes that may currently be deleting
    for host in hosts:
        while get_node_state(gce_compute, log, project, zone, host, nodespace["cluster_id"]) in ["STOPPING", "TERMINATED"]:
            log.info(" host is currently being deleted. Waiting...")
            await asyncio.sleep(5)

    # Filter out nodes that are in bringup or running already
    hosts = [h for h in hosts if get_node_state(gce_compute, log, project, zone, h, nodespace["cluster_id"]) is None]

    # group nodes based off of shape
    groups = defaultdict(lambda: [])
    for host in hosts:
        shape = get_node_features(host)["shape"]
        groups[shape].append(host)

    await asyncio.gather(*(
        start_node_group( log, group, nodespace, ssh_keys)
        for group in groups.values()
    ))


def terminate_instance(log, hosts, nodespace=None):
    gce_compute = get_build()

    if not nodespace:
        nodespace = get_nodespace()

    project = nodespace["compartment_id"]
    zone = nodespace["zone"]

    clean_placement_groups = False
    requests = []
    for host in hosts:
        log.info(f"Stopping {host}")
        features = get_node_features(host)
        if features["pg"] == 'True':
            clean_placement_groups = True

        try:
            requests.append(gce_compute.instances() \
                .delete(project=project,
                        zone=zone,
                        instance=host) \
                .execute())
        except Exception as e:
            log.error(f" problem while stopping: {e}")
            continue


    if clean_placement_groups:
        # Wait for nodes to terminate
        activePGs = list_placement_groups(log, gce_compute, nodespace["region"], project, nodespace["cluster_id"])
        if len(activePGs):
            for host in hosts:
                while get_node_state(gce_compute, log, project, zone, host, nodespace["cluster_id"]) in ["STOPPING", "TERMINATED"]:
                    log.info(f"    {host} is currently being deleted. Waiting...")
                    time.sleep(2)
                log.info(f"    {host} has stopped")
        for pg in activePGs:
            delete_placement_group(log, gce_compute, nodespace["region"], project, pg)


# [START run]
async def do_create_instance():
    os.environ['SA_LOCATION'] = '/home/davidy/secrets/ex-eccoe-university-bristol-52b726c8a1f3.json'
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    log = logging.getLogger("startnode")

    hosts = ['dy-test-node1']

    log.info('Creating instance.')

    await asyncio.gather(*(
        start_node(log, host, get_nodespace('test_nodespace.yaml'), "")
        for host in hosts
    ))

    log.info(f'Instances in project done')

    log.info(f'Terminating')
    terminate_instance(log, hosts, nodespace=get_nodespace('test_nodespace.yaml'))


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(do_create_instance())
    finally:
        loop.close()
