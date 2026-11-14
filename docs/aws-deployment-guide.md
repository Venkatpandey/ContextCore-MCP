# ContextCore-MCP — AWS Deployment Guide

This guide covers deploying **ContextCore-MCP** to AWS in a production-ready, highly available architecture with persistent storage for Qdrant vectors and Hugging Face model caches.

---

## 1. Architecture Overview

```
                      Internet / AI Agents / GitHub CI
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │   Application Load Balancer   │
                     │   (HTTPS:443 / HTTP:80)       │
                     └───────────────┬───────────────┘
                                     │
                         /sse, /metrics, /healthz
                                     ▼
┌─ VPC (2 AZs) ────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌─ Private Subnet ────────────────────────────────────────────────┐ │
│  │                                                                 │ │
│  │   ┌────────────────────────┐      ┌─────────────────────────┐   │ │
│  │   │  ContextCore FastMCP   │─────▶│     Qdrant Database     │   │ │
│  │   │  ECS Fargate (:8000)   │      │  ECS Fargate (:6333)    │   │ │
│  │   └───────────┬────────────┘      └────────────┬────────────┘   │ │
│  │               │                                │                │ │
│  │               │ Mount: /hf_cache               │ Mount: /storage│ │
│  │               ▼                                ▼                │ │
│  │   ┌─────────────────────────────────────────────────────────┐   │ │
│  │   │                   Amazon EFS Storage                    │   │ │
│  │   │        (Survives task restarts, crashes, deploys)       │   │ │
│  │   └─────────────────────────────────────────────────────────┘   │ │
│  │                                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Cloud Map Private DNS: `qdrant.contextcore.local:6333`              │
│  CloudWatch Container Insights & Metrics                             │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Characteristics
- **Zero Vector Data Loss**: Qdrant stores its vectors in `/qdrant/storage` backed by an encrypted **Amazon EFS** volume. If an ECS task crashes, is replaced, or updated, vectors remain 100% intact.
- **Fast Cold Starts**: Hugging Face sentence-transformers weights (`all-MiniLM-L6-v2`) are cached in an EFS volume mounted at `/root/.cache/huggingface`, preventing repeated model downloads on container spin-up.
- **Internal Service Discovery**: Qdrant communicates with FastMCP over an internal AWS Cloud Map private DNS (`qdrant.contextcore.local`), isolated from the public internet.
- **Health Checks & Telemetry**: ALB monitors `/healthz` on port 8000. Prometheus metrics are exposed on `/metrics`.

---

## 2. Automated Deployment via AWS CDK (Example)

An infrastructure-as-code reference implementation is provided in [`examples/aws-cdk/`](../examples/aws-cdk/).

### Prerequisites
1. [AWS CLI](https://aws.amazon.com/cli/) installed and configured (`aws configure`).
2. Node.js (v18+) and [AWS CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/cli.html):
   ```bash
   npm install -g aws-cdk
   ```
3. Python 3.11+ and `uv` (or `pip`):
   ```bash
   cd examples/aws-cdk
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

### Step 1 — Bootstrap AWS Environment (First Time Only)
```bash
cdk bootstrap aws://<ACCOUNT_ID>/<REGION>
```

### Step 2 — Deploy ContextCore Stack
```bash
cdk deploy
```

CDK synthesizes the CloudFormation template, creates the VPC, EFS file systems, ECS cluster, builds the Docker image from your root directory, and deploys both services behind the ALB.

### Step 3 — Outputs
Upon successful deployment, CDK outputs:
- **`McpEndpointUrl`**: `http://<alb-dns-name>/sse` (Connect AI agents here)
- **`HealthCheckUrl`**: `http://<alb-dns-name>/healthz`
- **`PrometheusMetricsUrl`**: `http://<alb-dns-name>/metrics`
- **`EfsFileSystemId`**: `fs-xxxxxxxx`

---

## 3. Production Hardening & Customization

### Adding HTTPS / TLS Certificate (ACM)
To secure the ALB with a custom domain and HTTPS:
1. Request or import a certificate in AWS Certificate Manager (ACM).
2. Update the ALB listener in [`contextcore_stack.py`](../examples/aws-cdk/contextcore_stack.py):
   ```python
   certificate = acm.Certificate.from_certificate_arn(self, "Cert", "arn:aws:acm:region:account:certificate/xxx")
   https_listener = alb.add_listener("HttpsListener", port=443, certificates=[certificate])
   https_listener.add_targets("McpTargets", port=8000, targets=[mcp_service], ...)
   ```

### Restricting Access
To restrict access to your team or VPN:
```python
alb_sg.add_ingress_rule(
    ec2.Peer.ipv4("203.0.113.0/24"), # Your corporate IP / VPN CIDR
    ec2.Port.tcp(443),
    "Allow authorized corporate traffic",
)
```

### Ingestion from GitHub Actions into AWS
When ingesting repositories via GitHub Actions to an AWS-hosted ContextCore:
1. Add the ALB endpoint and Qdrant credentials to your repository secrets:
   - `QDRANT_URL`: `https://qdrant.yourdomain.com:6333` (or via private runner)
   - `QDRANT_API_KEY`: `<your-qdrant-api-key>`
2. Run the workflow provided in [`examples/github-workflow-ingest.yml`](../examples/github-workflow-ingest.yml).

---

## 4. Observability: CloudWatch & Amazon Managed Prometheus

1. **CloudWatch Metrics**: ECS CPU and Memory utilization are automatically reported to CloudWatch under `AWS/ECS`.
2. **Prometheus Metrics**: Scrape `http://<alb-dns-name>/metrics` directly into **Amazon Managed Service for Prometheus (AMP)** or any external Prometheus instance.
3. **Grafana**: Import the provided [`monitoring/contextcore-mcp-grafana-dashboard.json`](../monitoring/contextcore-mcp-grafana-dashboard.json) into **Amazon Managed Grafana (AMG)** for instantaneous visualization.

---

## 5. Teardown
To destroy all deployed AWS resources and stop billing:
```bash
cd examples/aws-cdk
cdk destroy
```
*(Note: EFS volumes are configured with `RemovalPolicy.RETAIN` by default to prevent accidental vector loss. Delete the EFS file system manually in the AWS Console if you want to purge all data).*
