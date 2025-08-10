# opensloth/cli/autogen.py
"""
Dynamically generates Typer CLI commands from Pydantic models.
This allows Pydantic models to be the single source of truth for configuration.
"""
import inspect
from functools import wraps
from typing import Any, Dict, List, Type, get_origin, get_args, Union

import typer
from pydantic import BaseModel
from pathlib import Path

def _get_params_from_models(*models: Type[BaseModel]) -> List[Dict[str, Any]]:
    """
    Generates a unique list of CLI parameters from Pydantic models, handling nesting.
    """
    params = []
    seen_param_keys = set()

    for model_cls in models:
        for field_name, field in model_cls.model_fields.items():
            # Check if this is a nested Pydantic model
            field_type = field.annotation
            origin_type = get_origin(field_type)
            
            # Handle Optional[SomeModel] and Union[SomeModel, None] cases
            if origin_type is Union:
                args = get_args(field_type)
                if len(args) == 2 and type(None) in args:
                    # This is Optional[T] which is Union[T, None]
                    field_type = args[0] if args[1] is type(None) else args[1]
            
            is_nested = inspect.isclass(field_type) and issubclass(field_type, BaseModel)

            if is_nested:
                # Recursively process nested model
                for nested_name, nested_field in field_type.model_fields.items():
                    # The unique key for kwargs, e.g., 'model_name', 'r', 'lora_alpha'  
                    param_key = nested_name
                    if param_key in seen_param_keys:
                        continue
                    seen_param_keys.add(param_key)
                    
                    alias_info = nested_field.json_schema_extra or {}
                    cli_flags = []
                    
                    # Add the short, user-friendly alias if it exists
                    if 'cli_alias' in alias_info:
                        cli_flags.append(f"--{alias_info['cli_alias']}")
                    
                    # Add the full, nested name for backwards compatibility (hidden)
                    full_cli_name = f"--{field_name}-{nested_name}".replace("_", "-")
                    if full_cli_name not in cli_flags:
                        cli_flags.append(full_cli_name)
                    
                    # Determine the help panel for grouping
                    panel_name = f"{field_name.replace('_', ' ').title()} Options"
                    
                    # Handle boolean flags specially
                    is_bool = nested_field.annotation == bool or nested_field.annotation == bool | None
                    
                    if is_bool:
                        # For boolean fields, create --flag/--no-flag pairs
                        main_flag = cli_flags[0] if cli_flags else f"--{param_key.replace('_', '-')}"
                        negative_flag = main_flag.replace('--', '--no-') if main_flag.startswith('--') else f"--no-{main_flag[2:]}"
                        option = typer.Option(
                            nested_field.default,
                            main_flag + "/" + negative_flag,
                            help=nested_field.description,
                            rich_help_panel=panel_name,
                        )
                    else:
                        option = typer.Option(
                            nested_field.default,
                            *cli_flags,
                            help=nested_field.description,
                            rich_help_panel=panel_name,
                            hidden=len(cli_flags) > 1,  # Hide the long name if an alias exists
                        )
                    
                    params.append({
                        "param_key": param_key,
                        "annotation": nested_field.annotation,
                        "option": option,
                        "nested_parent": field_name,  # Remember which parent this belongs to
                    })
            else:
                # Process top-level field
                param_key = field_name
                if param_key in seen_param_keys:
                    continue
                seen_param_keys.add(param_key)

                alias_info = field.json_schema_extra or {}
                cli_flag = f"--{alias_info.get('cli_alias', param_key.replace('_', '-'))}"

                # Handle boolean flags specially
                is_bool = field.annotation == bool or field.annotation == bool | None
                
                if is_bool:
                    negative_flag = cli_flag.replace('--', '--no-') if cli_flag.startswith('--') else f"--no-{cli_flag[2:]}"
                    option = typer.Option(
                        field.default,
                        cli_flag + "/" + negative_flag,
                        help=field.description,
                        rich_help_panel="Core Options",
                    )
                else:
                    option = typer.Option(
                        field.default,
                        cli_flag,
                        help=field.description,
                        rich_help_panel="Core Options",
                    )

                params.append({
                    "param_key": param_key,
                    "annotation": field.annotation,
                    "option": option,
                    "nested_parent": None,  # Top-level field
                })
    return params

def cli_from_pydantic(*models: Type[BaseModel]):
    """
    A decorator that transforms a function into a Typer command with options
    generated from the provided Pydantic models.
    """
    def decorator(func):
        params = _get_params_from_models(*models)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config-related kwargs and reconstruct config
            config_kwargs = {}
            other_kwargs = {}
            
            # Get all field names from the models to identify config parameters
            config_field_names = set()
            for model_cls in models:
                for field_name, field in model_cls.model_fields.items():
                    field_type = field.annotation
                    origin_type = get_origin(field_type)
                    
                    if origin_type is Union:
                        args = get_args(field_type)
                        if len(args) == 2 and type(None) in args:
                            field_type = args[0] if args[1] is type(None) else args[1]
                    
                    is_nested = inspect.isclass(field_type) and issubclass(field_type, BaseModel)
                    
                    if is_nested:
                        for nested_name in field_type.model_fields:
                            config_field_names.add(nested_name)
                    else:
                        config_field_names.add(field_name)
            
            # Separate config and non-config kwargs
            for k, v in kwargs.items():
                if k in config_field_names:
                    config_kwargs[k] = v
                else:
                    other_kwargs[k] = v
            
            # Reconstruct the config from config kwargs
            if config_kwargs:
                config = reconstruct_config_from_kwargs(config_kwargs, *models)
                other_kwargs['cli_overrides'] = config
            
            return func(*args, **other_kwargs)

        # Create new function signature with all the parameters
        sig = inspect.signature(func)
        existing_params = list(sig.parameters.values())
        existing_names = {p.name for p in existing_params}

        # Separate positional, keyword-only, and VAR_KEYWORD parameters
        positional_params = []
        keyword_only_params = []
        var_keyword_param = None
        
        for param in existing_params:
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                var_keyword_param = param
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                keyword_only_params.append(param)
            else:
                positional_params.append(param)
        
        # Add generated parameters as keyword-only
        for p_info in params:
            if p_info["param_key"] not in existing_names:
                keyword_only_params.append(inspect.Parameter(
                    p_info["param_key"], 
                    inspect.Parameter.KEYWORD_ONLY,
                    default=p_info["option"], 
                    annotation=p_info["annotation"]
                ))
        
        # Reconstruct the parameter list in correct order
        new_params = positional_params + keyword_only_params
        if var_keyword_param:
            new_params.append(var_keyword_param)
        
        wrapper.__signature__ = sig.replace(parameters=new_params)
        return wrapper
    return decorator

def reconstruct_config_from_kwargs(kwargs: Dict[str, Any], *models: Type[BaseModel]) -> Dict[str, Any]:
    """
    Reconstructs the nested config dict from flat kwargs using model schemas.
    """
    config: Dict[str, Any] = {}
    
    # Map model class names to keys in the final config dict
    model_key_map = {
        "OpenSlothConfig": "opensloth_config",
        "TrainingArguments": "training_args",
    }

    for model_cls in models:
        model_key = model_key_map.get(model_cls.__name__)
        if not model_key: 
            continue
        
        config[model_key] = {}
        
        for field_name, field in model_cls.model_fields.items():
            # Check if this is a nested Pydantic model
            field_type = field.annotation
            origin_type = get_origin(field_type)
            
            # Handle Optional[SomeModel] cases
            if origin_type is Union:
                args = get_args(field_type)
                if len(args) == 2 and type(None) in args:
                    field_type = args[0] if args[1] is type(None) else args[1]
            
            is_nested = inspect.isclass(field_type) and issubclass(field_type, BaseModel)
            
            if is_nested:
                # Collect all nested fields that belong to this parent
                nested_data = {}
                for nested_name in field_type.model_fields:
                    if nested_name in kwargs and kwargs[nested_name] is not None:
                        nested_data[nested_name] = kwargs[nested_name]
                
                if nested_data:
                    config[model_key][field_name] = nested_data
            else:
                # Top-level field
                if field_name in kwargs and kwargs[field_name] is not None:
                    config[model_key][field_name] = kwargs[field_name]
    
    return config

__all__ = ["cli_from_pydantic", "reconstruct_config_from_kwargs"]
