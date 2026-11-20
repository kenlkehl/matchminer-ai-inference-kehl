"""Start the guideline fetcher (``mmai-guideline-fetcher`` or ``python -m matchminer_ai.apps.guideline_fetcher``).

By default this is the non-PHI app on all interfaces (port 8034). --phi starts the PHI
deployment for real patient records: loopback only (port 8044), an access token, DuckDB
populations and population caches, and the data root's phi/ cache. --site-config and
--data-root (or MMAI_SITE_CONFIG and MMAI_DATA_ROOT) name the site config and the data root.
"""

import argparse
import ipaddress
import os
import sys
import multiprocessing

from matchminer_ai.site import DATA_ROOT_ENV, SITE_CONFIG_ENV


def loopback(host):
    """Only this machine may connect to the PHI app: real patient records are served."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site-config", help=f"site config YAML (default ${SITE_CONFIG_ENV})")
    parser.add_argument("--data-root", help=f"data root holding phi/ and no_phi/ (default ${DATA_ROOT_ENV} or the "
                        "site config's paths.data_root)")
    parser.add_argument("--phi", action="store_true",
                        help="start the PHI deployment (real patient records; loopback and access token)")
    parser.add_argument("--host", help="bind address (default: 0.0.0.0, or 127.0.0.1 with --phi, "
                        "which accepts only loopback addresses)")
    parser.add_argument("--port", type=int,
                        help="port (default: 8034, or 8044 with --phi)")
    args = parser.parse_args(argv)
    # Before the settings module reads them.
    if args.site_config:
        os.environ[SITE_CONFIG_ENV] = args.site_config
    if args.data_root:
        os.environ[DATA_ROOT_ENV] = args.data_root
    from .settings import DEFAULT_PORT, PHI_PORT, Settings, access_token_file, load_access_token

    host = args.host or ("127.0.0.1" if args.phi else "0.0.0.0")
    port = args.port if args.port is not None else PHI_PORT if args.phi else DEFAULT_PORT
    if args.phi and not loopback(host):
        parser.error("The PHI app binds only to a loopback address such as 127.0.0.1.")
    # The mode comes only from this command line, never from the environment.
    try:
        settings = Settings(phi=args.phi)
    except ValueError as exc:
        parser.error(str(exc))
    token = None
    if args.phi:
        token = load_access_token()
        shown = f"[{host}]" if ":" in host else host
        url = f"http://{shown}:{port}/login"
        if sys.stdout.isatty():
            print(f"Sign in: {url}?token={token}", flush=True)
        else:
            # Redirected output (nohup, log files) may be readable by others: no token here.
            source = ("GUIDELINE_ACCESS_TOKEN" if os.getenv("GUIDELINE_ACCESS_TOKEN")
                      else access_token_file())
            print(f"Sign in at {url} with the access token from {source}.", flush=True)
    # Forking after tokenizer/Torch threads initialize can inherit locked mutexes.
    multiprocessing.set_start_method("spawn", force=True)
    # Use Hugging Face's standard local cache and honor explicit HF_HOME /
    # HF_HUB_CACHE settings. Do not prefer a machine-specific network mount.
    import uvicorn

    from .app import create_app

    app = create_app(settings, access_token=token)
    uvicorn.run(app, host=host, port=port, access_log=False)


if __name__ == "__main__":
    main()
