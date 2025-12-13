import inspect
import asyncio
import logging
from typing import Callable, Coroutine, Dict, Any, List, Optional, Type
from pydantic import BaseModel, ValidationError
from google.genai import types
from src.tools.schemas import ToolContext, ToolResult, ToolError

logger = logging.getLogger(__name__)

class ToolSpec(BaseModel):
    name: str
    description: str
    args_model: Type[BaseModel]
    executor: Any # Callable[[BaseModel, ToolContext], Coroutine[Any, Any, ToolResult]]
    timeout_seconds: float = 10.0
    side_effect: bool = False # If True, requires idempotency check (future)

    class Config:
        arbitrary_types_allowed = True

class ToolRegistry:
    """
    Instance-based registry for voice tools.
    """
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self, 
        name: str, 
        description: str, 
        args_model: Type[BaseModel], 
        side_effect: bool = False,
        timeout: float = 10.0
    ):
        """Decorator to register a tool implementation."""
        def decorator(func):
            spec = ToolSpec(
                name=name,
                description=description,
                args_model=args_model,
                executor=func,
                side_effect=side_effect,
                timeout_seconds=timeout
            )
            self._tools[name] = spec
            return func
        return decorator

    def register_manual(self, spec: ToolSpec):
        self._tools[spec.name] = spec

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    async def execute(self, name: str, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        """
        Execute a tool by name with validation and timeout.
        """
        spec = self._tools.get(name)
        if not spec:
            return ToolResult.error_result("TOOL_NOT_FOUND", f"Tool '{name}' not found")

        try:
            # 1. Pydantic Validation
            validated_args = spec.args_model(**args)
        except ValidationError as e:
            logger.warning(f"Tool {name} validation failed: {e}")
            return ToolResult.error_result("VALIDATION_ERROR", str(e))

        try:
            # 2. Execution with Timeout
            # We assume executor is async
            if asyncio.iscoroutinefunction(spec.executor):
                result = await asyncio.wait_for(
                    spec.executor(validated_args, context), 
                    timeout=spec.timeout_seconds
                )
            else:
                # Synchronous fallback (discouraged for I/O)
                result = spec.executor(validated_args, context)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Tool {name} timed out after {spec.timeout_seconds}s")
            return ToolResult.error_result("TIMEOUT", "Tool execution timed out", retryable=True)
            
        except Exception as e:
            logger.exception(f"Tool {name} execution failed")
            return ToolResult.error_result("EXECUTION_ERROR", str(e), retryable=False)

    def _pydantic_to_genai_schema(self, schema: Dict[str, Any]) -> types.Schema:
        """
        Recursively converts a Pydantic JSON schema to a google.genai.types.Schema.
        """
        # Map JSON schema types to GenAI types
        type_map = {
            "string": types.Type.STRING,
            "number": types.Type.NUMBER,
            "integer": types.Type.INTEGER,
            "boolean": types.Type.BOOLEAN,
            "object": types.Type.OBJECT,
            "array": types.Type.ARRAY,
            "null": None
        }

        # Handle 'anyOf' (usually for Optional fields in Pydantic: type + null)
        if "anyOf" in schema:
             # Find the non-null type
             valid_types = [x for x in schema["anyOf"] if x.get("type") != "null"]
             if valid_types:
                 # Use the first valid type found
                 return self._pydantic_to_genai_schema(valid_types[0])
        
        json_type = schema.get("type")
        genai_type = type_map.get(json_type, types.Type.OBJECT) # Default to OBJECT
        
        genai_schema = types.Schema(
            type=genai_type,
            description=schema.get("description"),
            nullable=False # Simplified
        )

        # Handle Object Properties
        if json_type == "object" and "properties" in schema:
            properties = {}
            for k, v in schema["properties"].items():
                properties[k] = self._pydantic_to_genai_schema(v)
            genai_schema.properties = properties
            genai_schema.required = schema.get("required", [])

        # Handle Array Items
        if json_type == "array" and "items" in schema:
            genai_schema.items = self._pydantic_to_genai_schema(schema["items"])
            
        return genai_schema

    def get_gemini_declarations(self) -> List[types.FunctionDeclaration]:
        """
        Convert registered tools to Gemini API SDK FunctionDeclaration objects.
        """
        declarations = []
        for name, spec in self._tools.items():
            
            # Generate JSON Schema from Pydantic
            schema = spec.args_model.model_json_schema()
            
            # Convert to SDK Schema
            # Pydantic schema root is always an object with properties
            sdk_schema = self._pydantic_to_genai_schema(schema)
            
            function_decl = types.FunctionDeclaration(
                name=name,
                description=spec.description,
                parameters=sdk_schema
            )
            declarations.append(function_decl)
            
        return declarations
