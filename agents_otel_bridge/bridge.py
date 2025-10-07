"""Core OpenTelemetry bridge implementation for Agents SDK."""

import os
import uuid
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from agents.tracing import TracingProcessor, set_trace_processors
except ImportError:
    raise ImportError(
        "openai-agents-sdk is required. Install it with: pip install openai-agents"
    )

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, Sampler
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


class OTelBridgeProcessor(TracingProcessor):
    """
    Bridges Agents SDK spans to OpenTelemetry.
    
    Captures:
    - LLM calls (requests, responses, token usage)
    - Tool executions (inputs, outputs, errors)
    - Agent workflows (full execution flow)
    - Custom spans (user-defined operations)
    """
    
    def __init__(self):
        self._stack = {}
        self._contexts = {}
        self._tracer = trace.get_tracer("agents-otel-bridge")

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: Optional[int] = None) -> bool:
        return True

    @staticmethod
    def _get_id(obj: Any) -> str:
        for key in ("id", "span_id", "trace_id"):
            if hasattr(obj, key):
                val = getattr(obj, key)
                if val:
                    return str(val)
        return f"obj-{id(obj)}"

    @staticmethod
    def _get_parent_id(span: Any) -> Optional[str]:
        for key in ("parent_span_id", "parent_id", "parent"):
            if hasattr(span, key):
                val = getattr(span, key)
                if val is None:
                    continue
                if not isinstance(val, (str, int)):
                    return OTelBridgeProcessor._get_id(val)
                return str(val)
        return None

    @staticmethod
    def _extract_span_data(span: Any) -> Dict[str, Any]:
        if not hasattr(span, 'span_data'):
            return {}
        
        span_data = span.span_data
        data = {'span_type': span_data.__class__.__name__}
        
        for attr in dir(span_data):
            if not attr.startswith('_'):
                try:
                    val = getattr(span_data, attr)
                    if not callable(val):
                        data[attr] = val
                except:
                    pass
        
        return data

    def on_trace_start(self, trace_obj: Any):
        rid = self._get_id(trace_obj) or f"run-{uuid.uuid4()}"
        
        span = self._tracer.start_span(
            name="Agent Workflow",
            attributes={
                "agent.run_id": rid,
                "agent.root": True,
            },
        )
        ctx = trace.set_span_in_context(span)
        self._stack[rid] = span
        self._contexts[rid] = ctx

    def on_span_start(self, span: Any):
        sid = self._get_id(span)
        parent_id = self._get_parent_id(span)
        
        span_data = self._extract_span_data(span)
        span_type = span_data.get('span_type', 'Unknown')
        
        parent_ctx = self._contexts.get(parent_id)
        if not parent_ctx:
            for ctx in self._contexts.values():
                parent_ctx = ctx
                break

        name, attrs = self._get_span_details(span_type, span_data)
        
        otel_span = self._tracer.start_span(
            name=name,
            context=parent_ctx,
            attributes=attrs,
        )
        ctx = trace.set_span_in_context(otel_span, parent_ctx)
        self._stack[sid] = otel_span
        self._contexts[sid] = ctx

    def on_span_end(self, span: Any):
        sid = self._get_id(span)
        
        otel_span = self._stack.get(sid)
        if not otel_span:
            return

        span_data = self._extract_span_data(span)
        span_type = span_data.get('span_type')
        
        # Add input at end if not already present
        if span_type == 'GenerationSpanData' and span_data.get('input'):
            llm_input = span_data['input']
            try:
                existing_attrs = otel_span.attributes if hasattr(otel_span, 'attributes') else {}
                
                if isinstance(llm_input, list) and 'llm.request.messages' not in existing_attrs:
                    otel_span.set_attribute("llm.request.messages", json.dumps(llm_input, default=str)[:3000])
                    
                    for msg in reversed(llm_input):
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            content = msg.get('content', '')
                            if content:
                                otel_span.set_attribute("llm.prompt", str(content)[:1000])
                                break
                    
                    msg_counts = {}
                    for msg in llm_input:
                        if isinstance(msg, dict):
                            role = msg.get('role', 'unknown')
                            msg_counts[role] = msg_counts.get(role, 0) + 1
                    
                    otel_span.set_attribute("llm.request.message_count", str(len(llm_input)))
                    for role, count in msg_counts.items():
                        otel_span.set_attribute(f"llm.request.{role}_messages", str(count))
                        
            except Exception as e:
                otel_span.set_attribute("llm.request.input", str(llm_input)[:2000])
        
        self._add_completion_attributes(otel_span, span_data)
        
        if hasattr(span, 'error') and span.error:
            error = span.error
            otel_span.set_attribute("error", True)
            if isinstance(error, dict):
                otel_span.set_attribute("error.message", str(error.get('message', '')))
                if 'data' in error:
                    otel_span.set_attribute("error.data", str(error['data'])[:500])
            else:
                otel_span.set_attribute("error.message", str(error)[:500])

        otel_span.end()
        self._stack.pop(sid, None)
        self._contexts.pop(sid, None)

    def on_trace_end(self, trace_obj: Any):
        rid = self._get_id(trace_obj)
        
        root = self._stack.get(rid)
        if root:
            root.end()
            self._stack.pop(rid, None)
            self._contexts.pop(rid, None)
        
        trace.get_tracer_provider().force_flush()

    def _get_span_details(self, span_type: str, data: Dict[str, Any]):
        """Determine span name and attributes based on span type."""
        
        if span_type == 'GenerationSpanData':
            model_name = data.get('model', 'unknown')
            name = f"llm.{model_name}"
            attrs = {
                "event.type": "llm",
                "llm.model": model_name,
            }
            
            if data.get('input'):
                llm_input = data['input']
                try:
                    if isinstance(llm_input, list):
                        attrs["llm.request.messages"] = json.dumps(llm_input, default=str)[:3000]
                        
                        for msg in reversed(llm_input):
                            if isinstance(msg, dict) and msg.get('role') == 'user':
                                content = msg.get('content', '')
                                if content:
                                    attrs["llm.prompt"] = str(content)[:1000]
                                    break
                        
                        msg_counts = {}
                        for msg in llm_input:
                            if isinstance(msg, dict):
                                role = msg.get('role', 'unknown')
                                msg_counts[role] = msg_counts.get(role, 0) + 1
                        
                        attrs["llm.request.message_count"] = str(len(llm_input))
                        for role, count in msg_counts.items():
                            attrs[f"llm.request.{role}_messages"] = str(count)
                    
                    elif isinstance(llm_input, dict):
                        attrs["llm.request.input"] = json.dumps(llm_input, default=str)[:3000]
                    else:
                        attrs["llm.request.input"] = str(llm_input)[:2000]
                        
                except Exception as e:
                    attrs["llm.request.input"] = str(llm_input)[:2000]
            
            if data.get('model_config'):
                model_config = data['model_config']
                if isinstance(model_config, dict):
                    for key in ['temperature', 'max_tokens', 'top_p']:
                        if key in model_config:
                            attrs[f"llm.config.{key}"] = str(model_config[key])
            
        elif span_type == 'FunctionSpanData':
            func_name = data.get('name', 'unknown')
            name = f"tool.{func_name}"
            attrs = {
                "event.type": "tool",
                "tool.name": func_name,
            }
            
            if data.get('input'):
                attrs["tool.input"] = json.dumps(data['input'], default=str)[:1000]
            
        elif span_type == 'AgentSpanData':
            agent_name = data.get('agent_name', 'agent')
            name = f"agent.{agent_name}"
            attrs = {
                "event.type": "agent",
                "agent.name": agent_name,
            }
            
            if data.get('input'):
                attrs["agent.input"] = str(data['input'])[:1000]
        
        elif span_type == 'CustomSpanData':
            span_name = data.get('name', 'custom-span')
            name = f"custom.{span_name}"
            attrs = {
                "event.type": "custom",
                "custom.name": span_name,
            }
            
            if data.get('metadata'):
                metadata = data['metadata']
                if isinstance(metadata, dict):
                    for key, val in list(metadata.items())[:10]:
                        attrs[f"custom.{key}"] = str(val)[:500]
            
        else:
            name = "agent.span"
            attrs = {"event.type": "unknown"}
        
        return name, attrs

    def _add_completion_attributes(self, otel_span, data: Dict[str, Any]):
        """Add completion/result attributes to span."""
        
        span_type = data.get('span_type')
        
        if span_type == 'GenerationSpanData':
            if data.get('output'):
                output = data['output']
                try:
                    otel_span.set_attribute("llm.response.raw", json.dumps(output, default=str)[:3000])
                    
                    if isinstance(output, list) and len(output) > 0:
                        message = output[0]
                        if isinstance(message, dict):
                            content = message.get('content', '')
                            if content:
                                otel_span.set_attribute("llm.response.content", str(content)[:2000])
                            
                            tool_calls = message.get('tool_calls', [])
                            if tool_calls:
                                otel_span.set_attribute("llm.response.tool_calls", json.dumps(tool_calls, default=str)[:2000])
                                otel_span.set_attribute("llm.response.tool_calls_count", str(len(tool_calls)))
                                
                                for i, tc in enumerate(tool_calls[:3]):
                                    if isinstance(tc, dict):
                                        func = tc.get('function', {})
                                        if isinstance(func, dict):
                                            otel_span.set_attribute(f"llm.response.tool_call.{i}.name", func.get('name', ''))
                                            otel_span.set_attribute(f"llm.response.tool_call.{i}.args", str(func.get('arguments', ''))[:500])
                            
                            role = message.get('role', '')
                            if role:
                                otel_span.set_attribute("llm.response.role", role)
                    
                    elif isinstance(output, dict):
                        choices = output.get('choices', [])
                        if choices and len(choices) > 0:
                            message = choices[0].get('message', {})
                            content = message.get('content', '')
                            if content:
                                otel_span.set_attribute("llm.response.content", str(content)[:2000])
                    
                    elif isinstance(output, str):
                        otel_span.set_attribute("llm.response.content", output[:2000])
                except Exception as e:
                    otel_span.set_attribute("llm.response", str(output)[:2000])
            
            if data.get('usage'):
                usage = data['usage']
                if isinstance(usage, dict):
                    for key in ['input_tokens', 'output_tokens', 'total_tokens', 'prompt_tokens', 'completion_tokens']:
                        if key in usage:
                            otel_span.set_attribute(f"llm.usage.{key}", str(usage[key]))
            
            if data.get('model'):
                otel_span.set_attribute("llm.model_name", str(data['model']))
            
            if data.get('type'):
                otel_span.set_attribute("llm.generation_type", str(data['type']))
                
        elif span_type == 'FunctionSpanData':
            if data.get('output') is not None:
                result = data['output']
                otel_span.set_attribute("tool.result", str(result)[:1000])
            
            if data.get('name'):
                otel_span.set_attribute("tool.function_name", str(data['name']))
            
            if data.get('input'):
                otel_span.set_attribute("tool.input", json.dumps(data['input'], default=str)[:1000])
        
        elif span_type == 'AgentSpanData':
            if data.get('output'):
                otel_span.set_attribute("agent.output", str(data['output'])[:2000])


class EnhancedOTelBridgeProcessor(OTelBridgeProcessor):
    """
    Extended bridge with cost tracking and analytics.
    
    Example:
        bridge = EnhancedOTelBridgeProcessor(
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0002
        )
        
        # After workflow
        print(f"Total cost: ${bridge.get_total_cost():.4f}")
    """
    
    def __init__(self, cost_per_1k_input: float = 0.0001, cost_per_1k_output: float = 0.0002):
        super().__init__()
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.total_cost = 0.0
        self.request_count = 0
    
    def _add_completion_attributes(self, otel_span, data: Dict[str, Any]):
        super()._add_completion_attributes(otel_span, data)
        
        if data.get('span_type') == 'GenerationSpanData' and data.get('usage'):
            usage = data['usage']
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            
            cost = (
                (input_tokens / 1000 * self.cost_per_1k_input) +
                (output_tokens / 1000 * self.cost_per_1k_output)
            )
            
            self.total_cost += cost
            self.request_count += 1
            
            otel_span.set_attribute("llm.cost.usd", f"{cost:.6f}")
            otel_span.set_attribute("llm.cost.cumulative_usd", f"{self.total_cost:.6f}")
            
            if cost > 0.10:
                logger.warning(f"💰 Expensive LLM request: ${cost:.4f}")
            
            otel_span.add_event(
                "cost.calculated",
                attributes={
                    "cost.usd": f"{cost:.6f}",
                    "cost.input_tokens": input_tokens,
                    "cost.output_tokens": output_tokens
                }
            )
    
    def get_total_cost(self) -> float:
        """Get total cost across all LLM calls."""
        return self.total_cost
    
    def get_average_cost(self) -> float:
        """Get average cost per LLM call."""
        return self.total_cost / self.request_count if self.request_count > 0 else 0.0


def setup_otel_tracing(
    service_name: Optional[str] = None,
    otlp_endpoint: str = "localhost:4317",
    service_namespace: str = "default",
    service_version: str = "1.0.0",
    insecure: bool = True,
    sampler: Optional[Sampler] = None,
    use_enhanced: bool = False,
    cost_per_1k_input: float = 0.0001,
    cost_per_1k_output: float = 0.0002,
) -> OTelBridgeProcessor:
    """
    Setup OpenTelemetry tracing for Agents SDK.
    
    Args:
        service_name: Name of your service (defaults to OTEL_SERVICE_NAME env var)
        otlp_endpoint: OTLP gRPC endpoint (defaults to localhost:4317)
        service_namespace: Namespace for organizing services
        service_version: Version of your service
        insecure: Whether to use insecure gRPC connection
        sampler: Custom sampler (defaults to ALWAYS_ON)
        use_enhanced: Use EnhancedOTelBridgeProcessor with cost tracking
        cost_per_1k_input: Cost per 1000 input tokens (for enhanced mode)
        cost_per_1k_output: Cost per 1000 output tokens (for enhanced mode)
    
    Returns:
        The bridge processor instance (for accessing metrics in enhanced mode)
    
    Example:
        setup_otel_tracing(
            service_name="my-agent",
            otlp_endpoint="localhost:4317"
        )
    """
    service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "agent-service")
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", otlp_endpoint)
    service_namespace = os.getenv("OTEL_SERVICE_NAMESPACE", service_namespace)
    service_version = os.getenv("SERVICE_VERSION", service_version)
    
    provider = TracerProvider(
        sampler=sampler or ALWAYS_ON,
        resource=Resource.create({
            "service.name": service_name,
            "service.namespace": service_namespace,
            "service.version": service_version,
        })
    )
    
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=insecure,
            )
        )
    )
    
    trace.set_tracer_provider(provider)
    
    if use_enhanced:
        bridge = EnhancedOTelBridgeProcessor(
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output
        )
    else:
        bridge = OTelBridgeProcessor()
    
    set_trace_processors([bridge])
    
    logger.info(f"✅ OpenTelemetry tracing enabled for service: {service_name}")
    logger.info(f"   Exporting to: {otlp_endpoint}")
    
    return bridge
