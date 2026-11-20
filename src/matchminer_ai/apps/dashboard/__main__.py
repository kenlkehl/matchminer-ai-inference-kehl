"""``mmai-dashboard`` (``python -m matchminer_ai.apps.dashboard``): serve the dashboard on 127.0.0.1;
``mmai-dashboard build``: build its store once."""
import os
import sys
import threading

from matchminer_ai.apps.common.security import is_loopback

from .settings import parse_settings


def main(argv=None):
    settings = parse_settings(argv)
    if settings.command == "build":
        from .pipeline import Builder, Sources, write_store

        store = Builder(Sources(settings.trial_data, settings.oa_db, settings.grouping,
                                settings.good_options_catalog)).build()
        write_store(store, settings.store_path)
        print(f"{store['counts']} -> {settings.store_path} ({store['build_seconds']} s)")
        return
    if not is_loopback(settings.host):
        sys.exit(f"Refusing to serve on {settings.host}: the dashboard shows confidential protocol content and serves "
                 "on 127.0.0.1 only (forward the port with ssh -L to reach it from another machine).")
    if not settings.allow_downloads:  # before transformers or huggingface_hub is imported
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import uvicorn

    from .server import create_app
    from .state import DashState

    token = None
    if settings.require_token:
        from matchminer_ai.apps.common.security import load_access_token

        from .settings import ACCESS_TOKEN_ENV, ACCESS_TOKEN_FILE

        token = load_access_token(ACCESS_TOKEN_ENV, ACCESS_TOKEN_FILE)
        url = f"http://{settings.host}:{settings.port}/login"
        if sys.stdout.isatty():
            print(f"Sign in: {url}?token={token}", flush=True)
        else:  # redirected output may be readable by others: no token here
            source = ACCESS_TOKEN_ENV if os.getenv(ACCESS_TOKEN_ENV) else ACCESS_TOKEN_FILE
            print(f"Sign in at {url} with the access token from {source}.", flush=True)
    state = DashState(settings, log=lambda message: print(message, flush=True))
    decisions = state.decisions
    if settings.device_notice:
        print(settings.device_notice, flush=True)
    print(f"Decision model: {decisions.model} on {settings.device} ({decisions.why})", flush=True)

    def preload(progress=None):
        try:
            result = decisions.preload(progress)
        except Exception as exc:
            print(f"Decision model failed to load on {settings.device}: {type(exc).__name__}: {exc}", flush=True)
            raise
        print(f"Decision model loaded on {decisions.device} in {result['load_seconds']} s", flush=True)
        return result

    state.jobs.submit("load", "decision model", preload, device=decisions.device)
    state.embed_cohorts()
    threading.Thread(target=state.corpus.warm_search, name="search-index", daemon=True).start()
    stop = threading.Event()
    interval = settings.rebuild_poll_seconds
    if interval > 0:  # new extraction results, a later TrialSpace round, edited grouping rules
        threading.Thread(target=state.watch, args=(interval, stop), name="watch", daemon=True).start()
    print(f"Ready in {state.startup_seconds} s. Serving on http://{settings.host}:{settings.port}/ "
          f"(research use; protocol content is confidential). Ctrl-C to stop.", flush=True)
    try:
        uvicorn.run(create_app(state, access_token=token), host=settings.host, port=settings.port, log_level="warning")
    finally:
        stop.set()
        state.encoder.close()
        state.jobs.shutdown()
        state.background.shutdown()
        state.llm.jobs.shutdown()


if __name__ == "__main__":
    main()
