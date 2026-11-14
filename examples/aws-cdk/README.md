# ContextCore-MCP — AWS CDK Package

This directory contains the AWS Cloud Development Kit (CDK) infrastructure-as-code package to deploy ContextCore-MCP on AWS ECS Fargate, persistent Amazon EFS, and an Application Load Balancer.

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Bootstrap CDK (once per account/region)
cdk bootstrap

# 3. View synthesized CloudFormation
cdk synth

# 4. Deploy to AWS
cdk deploy
```

For the complete architectural breakdown, security hardening, and HTTPS certificate configuration, see the [AWS Deployment Guide](../../docs/aws-deployment-guide.md).
