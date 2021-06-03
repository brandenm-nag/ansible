import functools
import subprocess
from typing import Dict, Optional
from collections import defaultdict
import yaml
import asyncio, time

import boto3
import citc.aws
import citc.cloud
import citc.utils
#from mypy_boto3 import ec2, route53

def create_placement_group(client, shape, nodespace):
    name=f'pg-{nodespace["cluster_id"]}-{shape}'
    res = client.create_placement_group(
            GroupName=name,
            Strategy='cluster',
            TagSpecifications=[
                {
                    'ResourceType': 'placement-group',
                    'Tags': [
                        {
                            'Key': 'cluster',
                            'Value': nodespace["cluster_id"]
                        },
                        {
                            'Key': 'shape',
                            'Value': shape
                        }
                    ]
                }
            ])
    return name


def delete_placement_group(client, shape, nodespace):
    res = client.delete_placement_group(
            GroupName=f'pg-{nodespace["cluster_id"]}-{shape}')


def get_placement_group(client, shape, nodespace):
    groups = client.describe_placement_groups(
                    Filters=[
                        {
                            'Name': 'tag:cluster',
                            'Values': [nodespace["cluster_id"]]
                        },
                        {
                            'Name': 'tag:shape',
                            'Values': [shape]
                        }
                    ]
                    )
    validPGs = groups["PlacementGroups"]
    if len(validPGs) > 0:
        if validPGs[0]["State"] in ['pending', 'available']:
            return validPGs[0]["GroupName"]

    return None




def get_node_state(client, hostname: str, nodespace: Dict[str, str]) -> citc.cloud.NodeState:
    """
    Get the current node state of the VM for the given hostname
    If there is no such VM, return TERMINATED
    """

    try:
        return citc.aws.AwsNode.from_name(hostname, client, nodespace).state
    except citc.aws.NodeNotFound:
        return citc.cloud.NodeState.TERMINATED


def get_node_features(hostname):
    features = subprocess.run(
        ["sinfo", "--Format=features:200", "--noheader", f"--nodes={hostname}"],
        stdout=subprocess.PIPE
    ).stdout.decode().strip().split(',')
    features = {f.split("=")[0]: f.split("=")[1] for f in features}
    return features


def create_node_config(client, hostname: str, nodespace: Dict[str, str], ssh_keys: str):
    """
    Create the configuration needed to create ``hostname`` in ``nodespace`` with ``ssh_keys``
    """
    with open("/home/slurm/bootstrap.sh", "rb") as f:
        user_data = f.read().decode()

    features = get_node_features(hostname)
    shape = features["shape"]
    if features["arch"] == "x86_64":
        arch = "x86_64"
    elif features["arch"] in {"aarch64", "arm64"}:
        arch = "arm64"  # This is what AWS calls aarch64
    else:
        raise ValueError(f"'{shape}' architecture ({features['arch']}) not recognised")
    images = client.describe_images(
        Filters=[
            {'Name': 'name', 'Values': ['citc-slurm-compute-*']},
            {'Name': 'tag:cluster', 'Values': [nodespace['cluster_id']]},
            {'Name': 'architecture', 'Values': [arch]},
        ],
        Owners=['self'],
    )
    images = sorted(images['Images'], key=lambda x: x['CreationDate'], reverse=True)
    if not images:
        raise RuntimeError(f"No matching image found")
    image = images[0]['ImageId']

    config = {
        "ImageId": image,
        "InstanceType": shape,
        "KeyName": f"ec2-user-{nodespace['cluster_id']}",
        "MinCount": 1,
        "MaxCount": 1,
        "UserData": user_data,
        "IamInstanceProfile": {
            'Name': f"describe_tags-{nodespace['cluster_id']}",
        },
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {
                        "Key": "Name",
                        "Value": hostname
                    },
                    {
                        "Key": "cluster",
                        "Value": nodespace["cluster_id"]
                    },
                    {
                        "Key": "type",
                        "Value": "compute"
                    },
                ]
            }
        ],
        "NetworkInterfaces": [
            {
                "AssociatePublicIpAddress": True,
                "DeviceIndex": 0,
                "SubnetId": nodespace["subnet"],
                "Groups": [nodespace["compute_security_group"]],
            },
        ],
    }

    return config


def add_dns_record(
    client,
    zone_id: str,
    rrname: str,
    rrtype,
    value: str,
    ttl: int,
    action="UPSERT"
) -> None:
    client.change_resource_record_sets(
        HostedZoneId=zone_id,
        ChangeBatch={
            'Comment': f'add {rrname} -> {value}',
            'Changes': [
                {
                    'Action': action,
                    'ResourceRecordSet': {
                        'Name': rrname,
                        'Type': rrtype,
                        'TTL': ttl,
                        'ResourceRecords': [{'Value': value}]
                    }
                }
            ]
        }
    )


def delete_dns_record(
    client,
    zone_id: str,
    rrname: str,
    rrtype: str,
    value: str,
    ttl: int
) -> None:
    client.change_resource_record_sets(
        HostedZoneId=zone_id,
        ChangeBatch={
            'Comment': f'delete {rrname}: {value}',
            'Changes': [
                {
                    'Action': "DELETE",
                    'ResourceRecordSet': {
                        'Name': rrname,
                        'Type': rrtype,
                        'TTL': ttl,
                        'ResourceRecords': [{'Value': value}]
                    }
                }
            ]
        }
    )


def ec2_client(region: str):
    import configparser
    config = configparser.ConfigParser()
    config.read('/home/slurm/aws-credentials.csv')
    client = boto3.client(
        "ec2",
        region_name=region,
        aws_access_key_id=config["default"]["aws_access_key_id"],
        aws_secret_access_key=config["default"]["aws_secret_access_key"]
    )
    return client


def route53_client():
    import configparser
    config = configparser.ConfigParser()
    config.read('/home/slurm/aws-credentials.csv')
    client = boto3.client(
        "route53",
        aws_access_key_id=config["default"]["aws_access_key_id"],
        aws_secret_access_key=config["default"]["aws_secret_access_key"]
    )
    return client


async def start_node(log, host: str, nodespace: Dict[str, str], ssh_keys: str) -> None:
    region = nodespace["region"]

    log.info(f"Starting {host}")

    client = ec2_client(region)

    while get_node_state(client, host, nodespace) in [citc.cloud.NodeState.TERMINATING, citc.cloud.NodeState.STOPPING]:
        log.info(" host is currently being deleted. Waiting...")
        await asyncio.sleep(5)

    node_state = get_node_state(client, host, nodespace)
    if node_state in [citc.cloud.NodeState.PENDING, citc.cloud.NodeState.RUNNING]:
        log.warning(f" host already exists with state {node_state}")
        return

    instance_details = create_node_config(client, host, nodespace, ssh_keys)

    loop = asyncio.get_event_loop()

    try:
        start_instance = functools.partial(client.run_instances, **instance_details)
        instance_result = await loop.run_in_executor(None, start_instance)
        instance = instance_result["Instances"][0]
    except Exception as e:
        log.error(f" problem launching instance: {e}")
        return

    vm_ip = instance["PrivateIpAddress"]

    log.info(f"  Private IP {vm_ip}")

    r53_client = route53_client()

    fqdn = f"{host}.{nodespace['dns_zone']}"
    add_dns_record(r53_client, nodespace["dns_zone_id"], fqdn, "A", vm_ip, 300)

    subprocess.run(["scontrol", "update", f"NodeName={host}", f"NodeAddr={vm_ip}"])

    log.info(f" Started {host}")

async def start_node_group(log, hosts, nodespace: Dict[str, str], ssh_keys: str) -> None:
    region = nodespace["region"]
    client = ec2_client(region)
    if len(hosts) == 1:
        await start_node(log, hosts[0], nodespace, ssh_keys)
        return

    log.info(f"Starting {len(hosts)} hosts in a group")

    instance_details = create_node_config(client, hosts[0], nodespace, ssh_keys)
    features = get_node_features(hosts[0])
    

    instance_details["MaxCount"] = len(hosts)
    instance_details["MinCount"] = len(hosts)
    # Remove the 'Name' tag - will assign later
    del instance_details["TagSpecifications"][0]["Tags"][0]
    if features["pg"] == 'True':
        PG = get_placement_group(client, features["shape"], nodespace)
        if not PG:
            PG = create_placement_group(client, features["shape"], nodespace)
        instance_details["Placement"] = { 'GroupName' : PG }

    loop = asyncio.get_event_loop()
    try:
        start_instance = functools.partial(client.run_instances, **instance_details)
        instance_result = await loop.run_in_executor(None, start_instance)
        instances = instance_result["Instances"]
    except Exception as e:
        log.error(f" problem launching instances: {e}")
        return

    r53_client = route53_client()
    for host, instance in zip(hosts, instances):
        client.create_tags(Resources=[instance["InstanceId"]], Tags=[{'Key': 'Name', 'Value': host}])
        vm_ip = instance["PrivateIpAddress"]
        fqdn = f"{host}.{nodespace['dns_zone']}"
        add_dns_record(r53_client, nodespace["dns_zone_id"], fqdn, "A", vm_ip, 300)
        subprocess.run(["scontrol", "update", f"NodeName={host}", f"NodeAddr={vm_ip}"])
        log.info(f" Started {host}")


async def start_nodes(log, hosts, nodespace: Dict[str, str], ssh_keys: str) -> None:
    region = nodespace["region"]
    client = ec2_client(region)

    # Wait for nodes terminating
    for host in hosts:
        while get_node_state(client, host, nodespace) in [citc.cloud.NodeState.TERMINATING, citc.cloud.NodeState.STOPPING]:
            log.info(" host is currently being deleted. Waiting...")
            await asyncio.sleep(5)

    # Filter out nodes that are in bringup or running already
    hosts = [h for h in hosts if get_node_state(client, h, nodespace) not in [citc.cloud.NodeState.PENDING, citc.cloud.NodeState.RUNNING]]

    # Group nodes based off of shape
    groups = defaultdict(lambda: [])
    for host in hosts:
        shape = get_node_features(host)["shape"]
        groups[shape].append(host)
    await asyncio.gather(*(
        start_node_group(log, group, nodespace, ssh_keys)
        for group in groups.values()
    ))


def terminate_instance(log, hosts, nodespace=None):
    if not nodespace:
        nodespace = citc.utils.get_nodespace()

    region = nodespace["region"]

    client = ec2_client(region)

    for host in hosts:
        log.info(f"Stopping {host}")

        try:
            instance = citc.aws.AwsNode.from_name(host, client, nodespace)
            fqdn = f"{host}.{nodespace['dns_zone']}"
            client.terminate_instances(InstanceIds=[instance.id])
            r53_client = route53_client()
            delete_dns_record(r53_client, nodespace["dns_zone_id"], fqdn, "A", instance.ip, 300)
        except Exception as e:
            log.error(f" problem while stopping: {e}")
            continue

        log.info(f" Stopped {host}")
