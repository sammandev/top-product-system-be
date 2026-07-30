from app.routers import iplas_proxy


def test_iter_active_iplas_sites_uses_only_app_config_active_site(monkeypatch) -> None:
    monkeypatch.setenv("IPLAS_API_TOKEN_PXD", "pxd-env-token")
    active_config = {
        "base_url": "http://ptb.example",
        "token": "ptb-token",
        "v1_url": "http://ptb.example/api/v1",
        "v2_url": "http://ptb.example/api/v2",
    }
    monkeypatch.setattr(iplas_proxy, "_get_active_app_config_iplas_site", lambda: ("PTB", active_config))

    configured_sites = iplas_proxy._iter_active_iplas_sites()

    assert configured_sites == [("PTB", active_config)]


def test_iter_active_iplas_sites_does_not_fallback_to_env_tokens(monkeypatch) -> None:
    monkeypatch.setenv("IPLAS_API_TOKEN_PTB", "ptb-env-token")
    monkeypatch.setenv("IPLAS_API_TOKEN_PSZ", "psz-env-token")
    monkeypatch.setattr(iplas_proxy, "_get_active_app_config_iplas_site", lambda: None)

    configured_sites = iplas_proxy._iter_active_iplas_sites()

    assert configured_sites == []