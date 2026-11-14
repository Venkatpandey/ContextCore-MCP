#!/usr/bin/env python3
import os
import aws_cdk as cdk
from contextcore_stack import ContextCoreStack

app = cdk.App()

account = os.getenv("CDK_DEFAULT_ACCOUNT")
region = os.getenv("CDK_DEFAULT_REGION", "us-east-1")

ContextCoreStack(
    app,
    "ContextCoreStack",
    env=cdk.Environment(account=account, region=region),
    description="ContextCore-MCP: FastMCP + Qdrant on AWS ECS Fargate & EFS",
)

app.synth()
