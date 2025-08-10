# opensloth/cli/autogen.py

import inspect
import typing
from functools import wraps
from typing import Any, Type, Union, get_args, get_origin

import typer
from pydantic import BaseModel


def _process_model_fields(
    model_cls: type[BaseModel],
    seen_keys: set[str],
    prefix: list[str] | None = None,
    panel_prefix: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Recursively processes a Pydantic model's fields to generate CLI parameter info.

    This function traverses nested Pydantic models to create a flat list of
    parameters, generating unique keys and CLI flags for each field.
    """
    if prefix is None:
        prefix = []
    if panel_prefix is None:
        panel_prefix = []
    
    params = []
    
    for field_name, field in model_cls.model_fields.items():
        field_type = field.annotation
        
        # Resolve Optional[T] into T for type checking
        origin_type = get_origin(field_type)
        if origin_type is Union or str(origin_type) == "<class 'types.UnionType'>":
            union_args = get_args(field_type)
            if len(union_args) == 2 and type(None) in union_args:
                field_type = next(arg for arg in union_args if arg is not type(None))

        is_nested = inspect.isclass(field_type) and issubclass(field_type, BaseModel)
        
        if is_nested:
            # Recurse into the nested model
            nested_prefix = [*prefix, field_name]
            nested_panel_prefix = [*panel_prefix, field_name.replace('_', ' ').title()]
            params.extend(
                _process_model_fields(field_type, seen_keys, nested_prefix, nested_panel_prefix)
            )
        else:
            # Process a regular, non-nested field
            param_key = "_".join([*prefix, field_name])
            if param_key in seen_keys:
                continue
            seen_keys.add(param_key)
            
            # --- Generate CLI Flags ---
            cli_flags = []
            alias_info = field.json_schema_extra or {}
            
            # 1. Short, user-friendly alias (e.g., --model)
            if 'cli_alias' in alias_info:
                cli_flags.append(f"--{alias_info['cli_alias']}")
            
            # 2. Full, unambiguous nested name (e.g., --fast-model-args-model-name)
            full_cli_name = f"--{'-'.join([*prefix, field_name]).replace('_', '-')}"
            if full_cli_name not in cli_flags:
                cli_flags.append(full_cli_name)

            # --- Generate Help Panel Name ---
            panel_name = " > ".join(panel_prefix) if panel_prefix else "Core Options"
            
            # --- Handle Types and Defaults for Typer.Option ---
            is_bool = field_type in (bool, bool | None)
            is_list = (get_origin(field.annotation) is list) or str(field.annotation).startswith('list[')
            
            # Handle Literal types by converting them to choices
            is_literal = str(field_type).startswith('typing.Literal')
            
            # Initialize field_annotation for later use
            field_annotation = field.annotation
            
            # Safely get the default value, calling factory if needed
            default_val = field.get_default(call_default_factory=True)
            
            # FIX: Ensure list-type options have a list as a default value for Typer/Click
            if is_list and default_val is None:
                default_val = []

            # Create the Typer.Option object
            if is_bool:
                # Use --flag/--no-flag syntax for booleans
                main_flag = cli_flags[0]
                no_flag = f"--no-{main_flag.lstrip('-')}"
                option = typer.Option(
                    default_val,
                    main_flag,
                    no_flag,
                    help=field.description,
                    rich_help_panel=panel_name,
                )
            elif is_literal:
                # For Literal types, convert annotation to str and let typer handle the choices
                # Typer will automatically create choices from the Literal values
                option = typer.Option(
                    default_val,
                    *cli_flags,
                    help=field.description,
                    rich_help_panel=panel_name,
                )
                # Override the annotation to be str so typer doesn't complain
                field_annotation = str
            else:
                option = typer.Option(
                    default_val,
                    *cli_flags,
                    help=field.description,
                    rich_help_panel=panel_name,
                )

            params.append({
                "param_key": param_key,
                "annotation": field_annotation if is_literal else field.annotation,
                "option": option,
                "path": prefix, # Path for reconstruction
            })
            
    return params

def _get_params_from_models(*models: type[BaseModel]) -> list[dict[str, Any]]:
    """
    Generates a unique list of CLI parameters from one or more Pydantic models.
    """
    all_params = []
    seen_keys = set()
    
    model_key_map = {
        "OpenSlothConfig": "opensloth_config",
        "TrainingArguments": "training_args",
        "DatasetPrepConfig": "dataset_prep_config",
    }
    
    for model_cls in models:
        top_level_key = model_key_map.get(model_cls.__name__)
        if not top_level_key:
             raise ValueError(f"Unknown model class for CLI generation: {model_cls.__name__}")

        model_params = _process_model_fields(model_cls, seen_keys)
        for p in model_params:
            p['top_level_key'] = top_level_key
        all_params.extend(model_params)
        
    return all_params

def reconstruct_config_from_kwargs(kwargs: dict[str, Any], params_info: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstructs the nested config dict from flat kwargs using the param info map."""
    config: dict[str, Any] = {}

    for p_info in params_info:
        param_key = p_info["param_key"]
        
        # If the value was provided via CLI (is not None, which is Typer's default for not-provided)
        if param_key in kwargs and kwargs[param_key] is not None:
            # Create nested dictionaries as needed
            current_level = config.setdefault(p_info["top_level_key"], {})
            path = p_info["path"]
            
            for key in path:
                current_level = current_level.setdefault(key, {})
            
            # Set the final value
            field_name = param_key.split("_")[-1]
            current_level[field_name] = kwargs[param_key]
            
    return config

def cli_from_pydantic(*models: type[BaseModel]):
    """Decorator to transform a function into a Typer command with options from Pydantic models."""
    def decorator(func):
        params_info = _get_params_from_models(*models)
        dynamic_param_names = {p["param_key"] for p in params_info}

        @wraps(func)
        def wrapper(**kwargs):
            # Separate the dynamic config kwargs from the original function's static kwargs
            config_kwargs = {k: v for k, v in kwargs.items() if k in dynamic_param_names}
            static_kwargs = {k: v for k, v in kwargs.items() if k not in dynamic_param_names}
            
            # Reconstruct the nested config from the dynamic kwargs
            cli_overrides = reconstruct_config_from_kwargs(config_kwargs, params_info)
            
            # Call the original function with its static args and the new overrides dict
            return func(cli_overrides=cli_overrides, **static_kwargs)

        # Build new function signature for Typer
        sig = inspect.signature(func)
        existing_params = list(sig.parameters.values())
        
        # Keep original arguments (like 'dataset'), remove the placeholder for overrides
        new_params = [p for p in existing_params if p.name != 'cli_overrides']
        
        for p_info in params_info:
            # Add generated parameters as keyword-only
            new_params.append(inspect.Parameter(
                name=p_info["param_key"], 
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=p_info["option"], 
                annotation=p_info["annotation"]
            ))
        
        wrapper.__signature__ = sig.replace(parameters=new_params)
        return wrapper
    return decorator

__all__ = ["cli_from_pydantic", "reconstruct_config_from_kwargs"]
