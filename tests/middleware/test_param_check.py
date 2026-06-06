from __future__ import annotations

import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
PARAM_CHECK_PATH = ROOT / "ascend-vllm" / "middleware" / "param_check.py"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeURL:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeRequest:
    def __init__(
        self,
        payload: dict[str, Any],
        path: str = "/v1/chat/completions",
        method: str = "POST",
    ) -> None:
        self.method = method
        self.url = _FakeURL(path)
        self.headers: dict[str, str] = {}
        self.state = types.SimpleNamespace()
        self._body = json.dumps(payload).encode("utf-8")

    async def body(self) -> bytes:
        return self._body


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_external_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    fastapi = _make_package("fastapi")
    fastapi_responses = types.ModuleType("fastapi.responses")

    class Request:
        pass

    class Response:
        def __init__(
            self,
            content: Any = None,
            headers: dict[str, str] | None = None,
            media_type: str | None = None,
            status_code: int = 200,
        ) -> None:
            self.content = content
            self.headers = {} if headers is None else dict(headers)
            self.media_type = media_type
            self.status_code = status_code

    class JSONResponse(Response):
        def __init__(self, status_code: int, content: Any) -> None:
            super().__init__(content=content, status_code=status_code)

    class StreamingResponse(Response):
        def __init__(
            self,
            body_iterator: Any,
            status_code: int = 200,
            headers: dict[str, str] | None = None,
            media_type: str | None = None,
        ) -> None:
            super().__init__(headers=headers, media_type=media_type, status_code=status_code)
            self.body_iterator = body_iterator

    fastapi.Request = Request
    fastapi.Response = Response
    fastapi_responses.JSONResponse = JSONResponse
    fastapi_responses.StreamingResponse = StreamingResponse

    starlette = _make_package("starlette")
    starlette_middleware = _make_package("starlette.middleware")
    starlette_middleware_base = types.ModuleType("starlette.middleware.base")

    class BaseHTTPMiddleware:
        def __init__(self, app: Any = None) -> None:
            self.app = app

    starlette_middleware_base.BaseHTTPMiddleware = BaseHTTPMiddleware

    vllm = _make_package("vllm")
    vllm_entrypoints = _make_package("vllm.entrypoints")
    vllm_openai = _make_package("vllm.entrypoints.openai")
    vllm_engine = _make_package("vllm.entrypoints.openai.engine")
    vllm_protocol = types.ModuleType("vllm.entrypoints.openai.engine.protocol")
    vllm_logger = types.ModuleType("vllm.logger")

    class OpenAIBaseModel:
        pass

    class ErrorInfo:
        def __init__(self, message: str, type: str, code: int) -> None:
            self.message = message
            self.type = type
            self.code = code

    class ErrorResponse:
        def __init__(self, error: ErrorInfo) -> None:
            self.error = error

        def model_dump(self) -> dict[str, dict[str, Any]]:
            return {
                "error": {
                    "message": self.error.message,
                    "type": self.error.type,
                    "code": self.error.code,
                }
            }

    class FakeLogger:
        def __init__(self) -> None:
            self.messages: list[tuple[str, tuple[Any, ...]]] = []

        def info(self, *args: Any, **kwargs: Any) -> None:
            self.messages.append(("info", args))

        def warning(self, *args: Any, **kwargs: Any) -> None:
            self.messages.append(("warning", args))

    fake_logger = FakeLogger()

    def init_logger(name: str) -> FakeLogger:
        return fake_logger

    vllm_protocol.OpenAIBaseModel = OpenAIBaseModel
    vllm_protocol.ErrorInfo = ErrorInfo
    vllm_protocol.ErrorResponse = ErrorResponse
    vllm_logger.init_logger = init_logger

    vllm_ascend = _make_package("vllm_ascend")
    vllm_ascend_envs = types.ModuleType("vllm_ascend.envs")
    vllm_ascend_envs.ENABLE_TRACE_LOG = False
    vllm_ascend.envs = vllm_ascend_envs

    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fastapi_responses)
    monkeypatch.setitem(sys.modules, "starlette", starlette)
    monkeypatch.setitem(sys.modules, "starlette.middleware", starlette_middleware)
    monkeypatch.setitem(sys.modules, "starlette.middleware.base", starlette_middleware_base)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", vllm_entrypoints)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", vllm_openai)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.engine", vllm_engine)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.engine.protocol", vllm_protocol)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.envs", vllm_ascend_envs)


def _load_param_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: dict[str, Any] | None = None,
) -> Any:
    _install_external_dependency_stubs(monkeypatch)
    monkeypatch.delenv("ROLE", raising=False)
    monkeypatch.delenv("NOT_ALLOWED_COMPLETIONS", raising=False)

    config_path = tmp_path / "validators.json"
    config_path.write_text(
        json.dumps(config or {"validators": {}, "validators_json": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("VALIDATORS_CONFIG_PATH", str(config_path))

    module_name = f"param_check_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PARAM_CHECK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _middleware_config() -> dict[str, Any]:
    return {
        "validators": {
            "max_tokens": {"validator_type": "supported"},
            "messages": [
                {
                    "validator_type": "supported",
                    "subfield": [
                        "messages",
                        "messages.role",
                        "messages.tool_calls",
                        "messages.tool_calls.function",
                        "messages.tool_calls.type",
                    ],
                },
                {
                    "validator_type": "nested_value",
                    "subfield": ["messages.role"],
                    "target_value": ["developer", "system", "user", "assistant", "function", "tool"],
                },
            ],
            "tool_choice": {"validator_type": "supported"},
            "tools": {
                "validator_type": "supported",
                "subfield": [
                    "tools",
                    "tools.function",
                    "tools.function.parameters",
                    "tools.type",
                ],
                "skip_check_subfield": ["tools.function.parameters"],
            },
            "top_k": [
                {"validator_type": "supported"},
                {"validator_type": "range", "type_": "int", "max_val": 100},
            ],
        },
        "validators_json": {
            "tool_choice": {
                "validator_type": "value",
                "target_value": ["auto"],
            }
        },
    }


async def _dispatch_payload(module: Any, payload: dict[str, Any]) -> Any:
    middleware = module.ValidateSamplingParams(app=object())
    request = _FakeRequest(payload)

    async def call_next(next_request: _FakeRequest) -> Any:
        return module.JSONResponse(
            status_code=200,
            content={
                "received": json.loads((await next_request.body()).decode("utf-8")),
                "message": "Success",
            },
        )

    return await middleware.dispatch(request, call_next)


def test_range_validator_converts_values_and_reports_bounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path)
    validator = module.RangeValidator("top_p", min_val=0.0, max_val=1.0, type_=float)

    assert validator.validate("0.5") is None
    assert "`top_p` must be greater than 0.0" in validator.validate("-0.1")
    assert "`top_p` must be smaller than 1.0" in validator.validate("1.5")
    assert "must belong to float" in validator.validate("not-a-number")


def test_supported_validator_removes_unknown_nested_fields_but_honors_skip_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path)
    validator = module.create_validator(
        "metadata",
        {
            "validator_type": "supported",
            "subfield": ["metadata.allowed"],
            "skip_check_subfield": ["metadata.allowed"],
        },
    )
    payload = {
        "allowed": {"unknown_inner": "kept by skip list"},
        "unsupported": "removed",
    }

    assert validator.validate(payload) is None

    assert payload == {"allowed": {"unknown_inner": "kept by skip list"}}


def test_value_and_incompatibility_validators_edit_request_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path)
    request_json = {"mode": "unsupported", "legacy": "removed", "keep": True}

    assert module.ValueValidator("mode", target_value=["supported"]).validate_json(request_json) is None
    assert module.IncompatibilityValidator("unused", subfield=["legacy"]).validate_json(request_json) is None

    assert request_json == {"keep": True}


def test_nested_value_validator_reports_disallowed_nested_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path)
    validator = module.NestedValueValidator(
        "tools",
        subfield=["tools.type"],
        target_values=["function"],
    )

    assert validator.validate([{"type": "function"}]) is None
    assert "tools.type only support the value in ['function']" in str(
        validator.validate([{"type": "code_interpreter"}])
    )


def test_load_validators_from_json_accepts_single_and_list_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path)
    config_path = tmp_path / "custom_validators.json"
    config_path.write_text(
        json.dumps(
            {
                "validators": {
                    "temperature": [
                        {"validator_type": "range", "type_": "float", "min_val": 0, "max_val": 2},
                        {"validator_type": "value", "target_value": [0.5, 1.0]},
                    ]
                },
                "validators_json": {
                    "legacy": {"validator_type": "incompatibility", "subfield": ["legacy"]}
                },
            }
        ),
        encoding="utf-8",
    )

    validators, validators_json = module.load_validators_from_json(str(config_path))

    assert len(validators["temperature"]) == 2
    assert isinstance(validators["temperature"][0], module.RangeValidator)
    assert isinstance(validators["temperature"][1], module.ValueValidator)
    assert len(validators_json["legacy"]) == 1
    assert isinstance(validators_json["legacy"][0], module.IncompatibilityValidator)


def test_validator_check_drops_unknown_params_and_runs_json_validators(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path)
    module.VALIDATORS = module.defaultdict(
        list,
        {
            "temperature": [module.RangeValidator("temperature", min_val=0, max_val=1, type_=float)],
            "guided": [module.SupportedValidator("guided")],
        },
    )
    module.VALIDATORS_JSON = module.defaultdict(
        list,
        {"guided": [module.ValueValidator("guided", target_value=["json"])]},
    )
    middleware = module.ValidateSamplingParams(app=object())
    payload = {"temperature": "0.25", "guided": "regex", "ignored": "defaulted"}

    assert middleware.validator_check(payload) is None

    assert payload == {"temperature": "0.25"}


def test_validator_check_returns_error_response_for_invalid_param(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path)
    module.VALIDATORS = module.defaultdict(
        list,
        {"temperature": [module.RangeValidator("temperature", min_val=0, max_val=1, type_=float)]},
    )
    module.VALIDATORS_JSON = module.defaultdict(list)
    middleware = module.ValidateSamplingParams(app=object())

    response = middleware.validator_check({"temperature": "2"})

    assert response.status_code == 400
    assert response.content["error"]["type"] == "BadRequestError"
    assert "`temperature` must be smaller than 1" in response.content["error"]["message"]


@pytest.mark.anyio
async def test_middleware_passes_supported_post_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())
    payload = {"max_tokens": "100"}

    response = await _dispatch_payload(module, payload)

    assert response.status_code == 200
    assert response.content == {"received": payload, "message": "Success"}


@pytest.mark.anyio
async def test_middleware_removes_unsupported_top_level_param(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())

    response = await _dispatch_payload(module, {"max_tokens": "100", "best_of": 3})

    assert response.status_code == 200
    assert response.content["received"] == {"max_tokens": "100"}


@pytest.mark.anyio
async def test_middleware_removes_unsupported_nested_subfield(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())
    payload = {
        "max_tokens": "100",
        "messages": {"tool_calls": {"function": None, "type": None, "others": None}},
    }

    response = await _dispatch_payload(module, payload)

    assert response.status_code == 200
    assert response.content["received"] == {
        "max_tokens": "100",
        "messages": {"tool_calls": {"function": None, "type": None}},
    }


@pytest.mark.anyio
async def test_middleware_skips_configured_subfield_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())
    payload = {
        "max_tokens": "100",
        "tools": {"function": {"parameters": {"others": None}}},
    }

    response = await _dispatch_payload(module, payload)

    assert response.status_code == 200
    assert response.content["received"] == payload


@pytest.mark.anyio
async def test_middleware_returns_400_for_unsupported_nested_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())

    response = await _dispatch_payload(module, {"max_tokens": "100", "messages": {"role": "others"}})

    assert response.status_code == 400


@pytest.mark.anyio
async def test_middleware_removes_unsupported_json_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())

    response = await _dispatch_payload(module, {"max_tokens": "100", "tool_choice": "others"})

    assert response.status_code == 200
    assert response.content["received"] == {"max_tokens": "100"}


@pytest.mark.anyio
async def test_middleware_removes_guided_decoding_when_function_calling_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())
    payload = {
        "max_tokens": "100",
        "response_format": {"type": "json_object"},
        "tools": None,
    }

    response = await _dispatch_payload(module, payload)

    assert response.status_code == 200
    assert response.content["received"] == {"max_tokens": "100", "tools": None}


@pytest.mark.anyio
async def test_middleware_returns_400_for_invalid_range_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())

    response = await _dispatch_payload(module, {"max_tokens": "100", "top_k": "a"})

    assert response.status_code == 400


@pytest.mark.anyio
async def test_middleware_validates_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_param_check(monkeypatch, tmp_path, _middleware_config())

    invalid_response = await _dispatch_payload(module, {"max_tokens": "100", "top_k": 122})
    valid_payload = {"max_tokens": "100", "top_k": 10}
    valid_response = await _dispatch_payload(module, valid_payload)

    assert invalid_response.status_code == 400
    assert valid_response.status_code == 200
    assert valid_response.content["received"] == valid_payload
