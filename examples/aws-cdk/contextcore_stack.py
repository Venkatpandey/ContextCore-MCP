"""
ContextCore-MCP — AWS CDK Infrastructure Stack
===============================================
Deploys a fully containerized, production-ready ContextCore-MCP architecture
on AWS using ECS Fargate, persistent Amazon EFS, and an Application Load Balancer.

Architecture:
  - Amazon VPC (2 AZs, public & private subnets, NAT Gateway)
  - Amazon EFS:
      * Volume for Qdrant storage (/qdrant/storage)
      * Volume for Hugging Face model cache (/root/.cache/huggingface)
  - Amazon ECS Cluster with AWS Cloud Map Service Discovery (contextcore.local)
  - Qdrant Service (ECS Fargate + EFS storage + Private DNS)
  - ContextCore FastMCP Server (ECS Fargate + Docker Asset + EFS cache)
  - Internet-Facing Application Load Balancer (ALB) with /healthz healthchecks
  - CloudWatch Logs & Metrics
"""

from __future__ import annotations

import os
from constructs import Construct
from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_efs as efs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_logs as logs,
    aws_servicediscovery as sd,
)


class ContextCoreStack(Stack):
    """Production AWS ECS Fargate + EFS stack for ContextCore-MCP."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------------
        # 1. Networking (VPC)
        # ---------------------------------------------------------------------
        vpc = ec2.Vpc(
            self,
            "ContextCoreVpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # ---------------------------------------------------------------------
        # 2. Persistent Storage (Amazon EFS)
        # ---------------------------------------------------------------------
        # Guarantees that vector embeddings and HuggingFace models survive
        # container restarts, ECS task deployments, or crashes.
        efs_security_group = ec2.SecurityGroup(
            self,
            "EfsSecurityGroup",
            vpc=vpc,
            description="Allow NFS traffic from ECS tasks",
            allow_all_outbound=True,
        )

        file_system = efs.FileSystem(
            self,
            "ContextCoreEfs",
            vpc=vpc,
            security_group=efs_security_group,
            performance_mode=efs.PerformanceMode.GENERAL_PURPOSE,
            throughput_mode=efs.ThroughputMode.BURSTING,
            encrypted=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_policy=efs.LifecyclePolicy.AFTER_30_DAYS,
        )

        # EFS Access Point for Qdrant vector storage
        qdrant_ap = file_system.add_access_point(
            "QdrantStorageAccessPoint",
            path="/qdrant_storage",
            create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="755"),
            posix_user=efs.PosixUser(uid="1000", gid="1000"),
        )

        # EFS Access Point for Hugging Face SentenceTransformers cache
        hf_cache_ap = file_system.add_access_point(
            "HuggingFaceCacheAccessPoint",
            path="/hf_cache",
            create_acl=efs.Acl(owner_uid="0", owner_gid="0", permissions="755"),
            posix_user=efs.PosixUser(uid="0", gid="0"),
        )

        # ---------------------------------------------------------------------
        # 3. ECS Cluster & Cloud Map Service Discovery
        # ---------------------------------------------------------------------
        namespace = sd.PrivateDnsNamespace(
            self,
            "ContextCoreDnsNamespace",
            name="contextcore.local",
            vpc=vpc,
        )

        cluster = ecs.Cluster(
            self,
            "ContextCoreCluster",
            vpc=vpc,
            default_cloud_map_namespace=ecs.CloudMapNamespaceOptions(
                name="contextcore.local",
                vpc=vpc,
            ),
        )

        # ---------------------------------------------------------------------
        # 4. Security Groups for Tasks
        # ---------------------------------------------------------------------
        qdrant_sg = ec2.SecurityGroup(
            self,
            "QdrantSecurityGroup",
            vpc=vpc,
            description="Security group for Qdrant vector database container",
            allow_all_outbound=True,
        )

        mcp_sg = ec2.SecurityGroup(
            self,
            "McpSecurityGroup",
            vpc=vpc,
            description="Security group for ContextCore FastMCP server container",
            allow_all_outbound=True,
        )

        # FastMCP can reach Qdrant on HTTP (6333) and gRPC (6334)
        qdrant_sg.add_ingress_rule(
            mcp_sg,
            ec2.Port.tcp(6333),
            "Allow FastMCP to query Qdrant HTTP API",
        )
        qdrant_sg.add_ingress_rule(
            mcp_sg,
            ec2.Port.tcp(6334),
            "Allow FastMCP to query Qdrant gRPC API",
        )

        # Allow ECS tasks to access EFS
        efs_security_group.add_ingress_rule(
            qdrant_sg,
            ec2.Port.tcp(2049),
            "Allow Qdrant task to access EFS volume",
        )
        efs_security_group.add_ingress_rule(
            mcp_sg,
            ec2.Port.tcp(2049),
            "Allow FastMCP task to access EFS cache",
        )

        # ---------------------------------------------------------------------
        # 5. Service 1: Qdrant Vector Database (ECS Fargate)
        # ---------------------------------------------------------------------
        qdrant_task_def = ecs.FargateTaskDefinition(
            self,
            "QdrantTaskDef",
            cpu=1024,
            memory_limit_mib=2048,
        )

        qdrant_log_group = logs.LogGroup(
            self,
            "QdrantLogGroup",
            log_group_name="/ecs/contextcore-qdrant",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        qdrant_volume = ecs.Volume(
            name="qdrant_persistent_storage",
            efs_volume_configuration=ecs.EfsVolumeConfiguration(
                file_system_id=file_system.file_system_id,
                transit_encryption="ENABLED",
                authorization_config=ecs.AuthorizationConfig(
                    access_point_id=qdrant_ap.access_point_id,
                    iam="ENABLED",
                ),
            ),
        )
        qdrant_task_def.add_volume(qdrant_volume)

        file_system.grant(
            qdrant_task_def.task_role,
            "elasticfilesystem:ClientMount",
            "elasticfilesystem:ClientWrite",
        )

        qdrant_container = qdrant_task_def.add_container(
            "QdrantContainer",
            image=ecs.ContainerImage.from_registry("qdrant/qdrant:latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="qdrant",
                log_group=qdrant_log_group,
            ),
            environment={
                "QDRANT__TELEMETRY_DISABLED": "true",
            },
        )
        qdrant_container.add_port_mappings(
            ecs.PortMapping(container_port=6333, protocol=ecs.Protocol.TCP),
            ecs.PortMapping(container_port=6334, protocol=ecs.Protocol.TCP),
        )
        qdrant_container.add_mount_points(
            ecs.MountPoint(
                container_path="/qdrant/storage",
                read_only=False,
                source_volume="qdrant_persistent_storage",
            ),
        )

        qdrant_service = ecs.FargateService(
            self,
            "QdrantService",
            cluster=cluster,
            task_definition=qdrant_task_def,
            desired_count=1,
            security_groups=[qdrant_sg],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            cloud_map_options=ecs.CloudMapOptions(
                name="qdrant",
                dns_record_type=sd.DnsRecordType.A,
                dns_ttl=Duration.seconds(10),
            ),
        )

        # ---------------------------------------------------------------------
        # 6. Service 2: ContextCore FastMCP Server (ECS Fargate)
        # ---------------------------------------------------------------------
        mcp_task_def = ecs.FargateTaskDefinition(
            self,
            "McpTaskDef",
            cpu=1024,
            memory_limit_mib=2048,
        )

        mcp_log_group = logs.LogGroup(
            self,
            "McpLogGroup",
            log_group_name="/ecs/contextcore-mcp",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        hf_volume = ecs.Volume(
            name="hf_cache_storage",
            efs_volume_configuration=ecs.EfsVolumeConfiguration(
                file_system_id=file_system.file_system_id,
                transit_encryption="ENABLED",
                authorization_config=ecs.AuthorizationConfig(
                    access_point_id=hf_cache_ap.access_point_id,
                    iam="ENABLED",
                ),
            ),
        )
        mcp_task_def.add_volume(hf_volume)

        file_system.grant(
            mcp_task_def.task_role,
            "elasticfilesystem:ClientMount",
            "elasticfilesystem:ClientWrite",
        )

        # Build Docker container asset from the repository root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        mcp_image = ecs.ContainerImage.from_asset(project_root)

        mcp_container = mcp_task_def.add_container(
            "McpContainer",
            image=mcp_image,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="mcp-server",
                log_group=mcp_log_group,
            ),
            environment={
                "PYTHONUNBUFFERED": "1",
                "QDRANT_HOST": "qdrant.contextcore.local",
                "QDRANT_PORT": "6333",
                "MCP_HOST": "0.0.0.0",
                "MCP_PORT": "8000",
                "MCP_TRANSPORT": "sse",
            },
        )
        mcp_container.add_port_mappings(
            ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP),
        )
        mcp_container.add_mount_points(
            ecs.MountPoint(
                container_path="/root/.cache/huggingface",
                read_only=False,
                source_volume="hf_cache_storage",
            ),
        )

        mcp_service = ecs.FargateService(
            self,
            "McpService",
            cluster=cluster,
            task_definition=mcp_task_def,
            desired_count=1,
            security_groups=[mcp_sg],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
        )
        mcp_service.node.add_dependency(qdrant_service)

        # ---------------------------------------------------------------------
        # 7. Application Load Balancer (ALB)
        # ---------------------------------------------------------------------
        alb_sg = ec2.SecurityGroup(
            self,
            "AlbSecurityGroup",
            vpc=vpc,
            description="Security group for internet-facing ALB",
            allow_all_outbound=True,
        )
        alb_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(80),
            "Allow HTTP traffic from allowed clients",
        )

        mcp_sg.add_ingress_rule(
            alb_sg,
            ec2.Port.tcp(8000),
            "Allow ALB to reach FastMCP container on port 8000",
        )

        alb = elbv2.ApplicationLoadBalancer(
            self,
            "ContextCoreAlb",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        http_listener = alb.add_listener(
            "HttpListener",
            port=80,
            open=True,
        )

        target_group = http_listener.add_targets(
            "McpTargets",
            port=8000,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[mcp_service],
            health_check=elbv2.HealthCheck(
                path="/healthz",
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )

        # ---------------------------------------------------------------------
        # 8. Stack Outputs
        # ---------------------------------------------------------------------
        CfnOutput(
            self,
            "McpEndpointUrl",
            description="FastMCP Server SSE Endpoint URL",
            value=f"http://{alb.load_balancer_dns_name}/sse",
        )

        CfnOutput(
            self,
            "HealthCheckUrl",
            description="ContextCore Health Check URL",
            value=f"http://{alb.load_balancer_dns_name}/healthz",
        )

        CfnOutput(
            self,
            "PrometheusMetricsUrl",
            description="Prometheus Telemetry Metrics URL",
            value=f"http://{alb.load_balancer_dns_name}/metrics",
        )

        CfnOutput(
            self,
            "EfsFileSystemId",
            description="EFS File System ID (Persistent vector storage)",
            value=file_system.file_system_id,
        )
