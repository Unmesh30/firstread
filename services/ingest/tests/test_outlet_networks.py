"""Outlet-network config and the network finder (ADR 0013)."""

import pytest

from firstread_ingest.config import OutletNetworks, load_outlet_networks
from firstread_ingest.networks import networks


def test_unapplied_networks_change_nothing() -> None:
    nets = OutletNetworks(apply=False, networks={"n1": ["a.com", "b.com"]})
    assert nets.mapping() == {}
    applied = OutletNetworks(apply=True, networks={"n1": ["a.com", "b.com"]})
    assert applied.mapping() == {"a.com": "n1", "b.com": "n1"}


def test_a_domain_belongs_to_one_network() -> None:
    with pytest.raises(ValueError, match="both"):
        OutletNetworks(apply=True, networks={"n1": ["a.com"], "n2": ["a.com"]})


def test_the_committed_config_loads() -> None:
    load_outlet_networks.cache_clear()
    nets = load_outlet_networks()
    assert isinstance(nets.apply, bool)


def test_finder_links_domains_that_share_most_stories() -> None:
    cov = [(f"s{i}", d) for i in range(10) for d in ("a.uk", "b.uk", "c.uk")]
    cov += [(f"t{i}", "indie.com") for i in range(10)]
    cov += [("s0", "indie.com")]
    found, _ = networks(cov, min_stories=5, jaccard=0.5, min_size=3)
    assert found == [["a.uk", "b.uk", "c.uk"]]
