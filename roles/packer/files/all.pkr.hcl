variable "google_account_file" {}
variable "google_destination_image_family" {}
variable "google_project_id" {}
variable "google_zone" {}
variable "google_network" {}
variable "google_subnetwork" {}

variable "aws_region" {}

variable "oracle_availability_domain" {}
variable "oracle_base_image_ocid" {}
variable "oracle_base_image_ocid_gpu" {}
variable "oracle_compartment_ocid" {}
variable "oracle_subnet_ocid" {}
variable "oracle_shape" {}
variable "oracle_shape_gpu" {}
variable "oracle_access_cfg_file" {}
variable "oracle_key_file" {}

variable "destination_image_name" {}
variable "cluster" {}
variable "ca_cert" {}

source "googlecompute" "google" {
    account_file = var.google_account_file
    source_image = "${var.destination_image_name}-${var.cluster}-v1"
    ssh_username = "centos"
    project_id = var.google_project_id
    zone = var.google_zone
    network = var.google_network
    subnetwork = var.google_subnetwork
    use_internal_ip = true
    tags = ["compute-${var.cluster}"]
    image_name = "${var.destination_image_name}-${var.cluster}-v{{timestamp}}"
    image_family = "${var.google_destination_image_family}-${var.cluster}"
    machine_type = "n1-standard-16"
    labels = {
        cluster = var.cluster
    }
    image_labels = {
        cluster = var.cluster
    }
}

source "amazon-ebs" "aws" {
    run_volume_tags = {
        cluster = var.cluster
    }
    tags = {
        cluster = var.cluster
    }
    snapshot_tags = {
        cluster = var.cluster
    }
    run_tags = {
        cluster = var.cluster
    }
    force_deregister = true
    force_delete_snapshot = true
    region = var.aws_region
    ssh_username = "centos"
    vpc_filter {
        filter {
            name = "tag:cluster"
            value = var.cluster
        }
    }
    subnet_filter {
        filter {
            name = "tag:cluster"
            value = var.cluster
        }
    }
    associate_public_ip_address = true

    launch_block_device_mappings {
        device_name = "/dev/sda1"
        volume_size =  20
        delete_on_termination = true
    }
}

source "oracle-oci" "oracle" {
    image_name = "${var.destination_image_name}-${var.cluster}-v{{timestamp}}"
    availability_domain = var.oracle_availability_domain
    base_image_ocid = var.oracle_base_image_ocid
    compartment_ocid = var.oracle_compartment_ocid
    shape = var.oracle_shape
    subnet_ocid = var.oracle_subnet_ocid
    access_cfg_file = var.oracle_access_cfg_file
    key_file = var.oracle_key_file
    tags = {
        cluster = var.cluster
    }
    ssh_username = "opc"
}

source "oracle-oci" "oracle-gpu" {
    image_name = "${var.destination_image_name}-${var.cluster}-GPU-v{{timestamp}}"
    oracle_availability_domain = var.oracle_availability_domain
    oracle_base_image_ocid = var.oracle_base_image_ocid_gpu
    oracle_compartment_ocid = var.oracle_compartment_ocid
    oracle_shape = var.oracle_shape_gpu
    oracle_subnet_ocid = var.oracle_subnet_ocid
    oracle_access_cfg_file = var.oracle_access_cfg_file
    oracle_key_file = var.oracle_key_file
    tags = {
        cluster = var.cluster
    }
    ssh_username = "opc"
}

build {

    source "source.amazon-ebs.aws" {
        ami_name = "${var.destination_image_name}-${var.cluster}-x86_64-v{{timestamp}}"
        instance_type = "c5.9xlarge"
        source_ami_filter {
            filters = {
                name = "CentOS 8.*"
                architecture = "x86_64"
            }
            owners = ["125523088429"]
            most_recent = true
        }
    }
    source "source.amazon-ebs.aws" {
        ami_name = "${var.destination_image_name}-${var.cluster}-aarch64-v{{timestamp}}"
        instance_type = "c6g.8xlarge"
        source_ami_filter {
            filters = {
                name = "CentOS 8.*"
                architecture = "arm64"
            }
            owners = ["125523088429"]
            most_recent = true
        }
    }


    sources = [
        "source.googlecompute.google",
        "source.oracle-oci.oracle",
        "source.oracle-oci.oracle-gpu",
    ]

    provisioner "shell" {
        script = "/etc/citc/packer/update_kernel.sh"
        expect_disconnect = true
    }
    provisioner "shell" {
        pause_before = "20s"
        script = "/etc/citc/packer/install_gvnic.sh"
        environment_vars = [
            "PACKER_SOURCE_NAME=${source.name}",
            ]
    }

    provisioner "file" {
        source = "/home/citc/.ssh/authorized_keys"
        destination = "/tmp/citc_authorized_keys"
    }

    provisioner "file" {
        source = var.ca_cert
        destination = "/tmp/CA.crt"
    }

    provisioner "file" {
        source = "/etc/munge/munge.key"
        destination = "/tmp/munge.key"
    }

    provisioner "shell" {
        script = "/etc/citc/packer/run_ansible.sh"
    }

    provisioner "shell" {
        script = "/home/citc/compute_image_extra.sh"
    }
}
