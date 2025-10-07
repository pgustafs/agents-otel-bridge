# Agents OpenTelemetry Bridge

OpenTelemetry tracing integration for the OpenAI Agents SDK.

## Features

- 🔍 **Complete Visibility**: Trace LLM calls, tool executions, and agent workflows
- 📊 **Token Tracking**: Monitor token usage and estimate costs
- 🚀 **Zero Code Changes**: Works with existing Agents SDK code
- 🔌 **Standard Protocol**: Uses OpenTelemetry for compatibility with Jaeger, Grafana, DataDog, etc.
- ⚡ **Async Export**: Non-blocking trace export for production use

## Installation
```bash
pip install git+https://github.com/pgustafs/agents-otel-bridge.git
```

## Quick Start
```python
from agents import Agent, Runner
from agents_otel_bridge import setup_otel_tracing

# Enable tracing (one line!)
setup_otel_tracing(
    service_name="my-agent",
    otlp_endpoint="localhost:4317"
)

# Your agent code works as-is
agent = Agent(name="Assistant", model=model)
result = Runner.run_sync(agent, "Hello!")
```

## What Gets Traced

- ✅ **LLM Calls**: Request messages, responses, token counts
- ✅ **Tool Executions**: Function calls with inputs/outputs
- ✅ **Agent Workflows**: Complete execution flow
- ✅ **Errors**: Full error context for debugging
- ✅ **Custom Spans**: Support for custom_span()

## Configuration

### Basic Usage

```python
from agents_otel_bridge import setup_otel_tracing

setup_otel_tracing(
    service_name="my-service",
    otlp_endpoint="localhost:4317",
    service_namespace="production",
    service_version="1.0.0",
    insecure=True
)
```

### Environment Variables

```bash
export OTEL_SERVICE_NAME="my-agent"
export OTEL_EXPORTER_OTLP_ENDPOINT="localhost:4317"
export OTEL_SERVICE_NAMESPACE="production"
export SERVICE_VERSION="2.0.0"
```

```python
setup_otel_tracing()  # Uses environment variables
```

### Single Trace for Multi-Agent Workflows

```python
from agents.tracing import with_trace

# Group multiple agent runs in one trace
with with_trace("Blog Generation"):
    result1 = Runner.run_sync(research_agent, query)
    result2 = Runner.run_sync(writer_agent, result1.final_output)
```

### Custom Spans

```python
from agents.tracing import custom_span

with custom_span("Database Query", metadata={"table": "users"}):
    # Your code here
    pass
```

### Cost Tracking

```python
from agents_otel_bridge import EnhancedOTelBridgeProcessor

bridge = EnhancedOTelBridgeProcessor(
    cost_per_1k_input=0.0001,
    cost_per_1k_output=0.0002
)

# After workflow
print(f"Total cost: ${bridge.get_total_cost():.4f}")
```

## Viewing Traces

### Run Jaeger + OTel Collector with Podman

```bash
# Create a shared pod/network namespace
podman pod create --name observ -p 16686:16686 -p 4317:4317 -p 4318:4318

# Jaeger all-in-one with OTLP ingest enabled (gRPC 4317, HTTP 4318)
podman run -d --rm --name jaeger --pod observ \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:latest

# Prepare an OTel Collector config in your current dir:
cat > otelcol.yaml << 'YAML'
receivers:
  otlp:
    protocols:
      grpc:
      http:

exporters:
  # Forward traces to Jaeger's OTLP receiver inside the same pod
  otlp/jaeger:
    endpoint: "localhost:4317"   # from the Collector's POV (same pod)
    tls:
      insecure: true

processors:
  batch:

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/jaeger]
YAML

# Run the OTel Collector
podman run -d --rm --name otel-collector --pod observ \
  -v "$PWD/otelcol.yaml":/etc/otelcol/config.yaml:ro \
  otel/opentelemetry-collector:latest \
  --config=/etc/otelcol/config.yaml
```

**Access UI: http://localhost:16686**

## Requirements

Python >= 3.8
openai-agents-sdk >= 0.1.0
opentelemetry-api >= 1.20.0
opentelemetry-sdk >= 1.20.0
opentelemetry-exporter-otlp-proto-grpc >= 1.20.0

## Testing Your Package Locally

Before pushing to GitHub, test it works:

```bash
# Install in development mode
pip install -e .

# Test import
python -c "from agents_otel_bridge import setup_otel_tracing; print('Works!')"

# Build the package (optional - to check it builds correctly)
pip install build
python -m build
```

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please open an issue or PR.
