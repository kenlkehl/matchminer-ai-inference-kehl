"""LLM settings shared by the ctgov screen (``trials.screen_trials``) and ctgov cohort extraction
(``trials.extract_ctgov_cohorts``): the package's cohort-workflow configuration, its endpoints and its limits, so
both steps send their requests the same way."""
import os
from pathlib import Path
import resource


def add_llm_arguments(p):
    """The model, endpoint and limit options of both steps."""
    p.add_argument('--model', default='nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4')
    p.add_argument('--tokenizer-model', help='Underlying model family when --model is a served alias')
    endpoints = p.add_mutually_exclusive_group(required=True)
    endpoints.add_argument('--server_urls', '--server-urls', nargs='+')
    endpoints.add_argument('--server_urls_file', '--server-urls-file', type=Path)
    p.add_argument('--max_concurrent_per_server', type=int, default=768,
                   help='Adaptive ceiling when registry entries do not specify upstream_concurrency')
    p.add_argument('--workers', type=int, default=1536, help='Total in-flight public API calls')
    p.add_argument('--request_timeout', type=float, default=7200)
    p.add_argument('--recovery-timeout', type=float, default=86400)
    p.add_argument('--max-attempts', type=int, choices=range(1, 11), default=3)
    p.add_argument('--max-tokens', type=int, default=50000)
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    return p


def prepare_runtime(args):
    """Check the limits, set the API key and raise the open-file limit."""
    if min(args.workers, args.max_concurrent_per_server, args.max_tokens,
           args.request_timeout, args.recovery_timeout) <= 0:
        raise ValueError('Concurrency, output budget and timeouts must be positive')
    if args.api_key_env != 'OPENAI_API_KEY':
        os.environ['OPENAI_API_KEY'] = os.environ.get(args.api_key_env, '')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 4 * args.workers + 1024
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed if hard == resource.RLIM_INFINITY else min(needed, hard), hard))


def make_config(args):
    from matchminer_ai.config import load_default_preset
    from matchminer_ai.llm.model_profiles import apply_model_profile
    config = load_default_preset()
    config.debug_mode = True
    family = args.tokenizer_model or args.model
    if apply_model_profile(config, family, sections=['trial']) is None:
        raise ValueError('No vendor sampling profile; supply a recognized --tokenizer-model')
    config.trial['cohorts'] = {'max_attempts': args.max_attempts}
    config.trial['remote'].update(model_name=args.model, tokenizer_name=family)
    config.trial['remote']['request_params']['max_tokens'] = args.max_tokens
    config.remote.update(enabled=True, server_urls=args.server_urls or [],
        server_urls_file=str(args.server_urls_file) if args.server_urls_file else None,
        max_concurrent_requests=args.max_concurrent_per_server, pool_max_workers=args.workers,
        request_timeout=args.request_timeout, recovery_timeout=args.recovery_timeout)
    config.remote['adaptive_concurrency'].update(enabled=True, metrics_interval=10)
    config.model_metadata_cache_dir = str(args.work_dir / 'model_metadata')
    return config
