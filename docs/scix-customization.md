# SciX customization and migration

This fork uses SciX as its astronomy discovery layer while keeping the upstream retrieval and ranking architecture. SciX currently exposes the same production API, endpoints, authentication, and tokens as ADS. The migration is therefore a compatibility migration, not an API-host cutover.

## Compatibility contract

| Concern | Current canonical value | Compatibility behavior |
| --- | --- | --- |
| Retriever name | `scix` | `ads` remains a legacy alias |
| Token | `SCIX_API_TOKEN` | Falls back to `ADS_API_TOKEN`; existing tokens do not need rotation |
| API base | `https://api.adsabs.harvard.edu/v1` | Configurable; change only after SciX publishes a new production server |
| Record URL | `https://scixplorer.org/abs/<bibcode>/abstract` | New user-facing links use SciX |
| Stable ID | `ads:<bibcode>` | Preserved to prevent duplicate delivery |
| State path | `.runtime/state/ads.json` | Preserved so existing private caches continue to work |
| Actions cache key | `ads-state-*` | Preserved across the migration |

Do not configure both `scix` and `ads`: they query the same service, and the executor rejects the pair before making a request.

## Runtime flow

1. Read the Zotero corpus and apply collection filters.
2. Query a bounded rolling window through the SciX/ADS-compatible Search API.
3. Remove records already present in the private delivery state.
4. Rank licensed SciX abstracts against Zotero abstracts with the local embedding model.
5. Keep the configured top results.
6. For selected records with an explicit arXiv identifier, fetch the matching arXiv record and replace the working title, author, abstract, and full text with arXiv-sourced content.
7. Generate optional TLDR, affiliation, and title-translation fields only for successfully arXiv-enriched or otherwise remote-eligible content.
8. Apply the Atom publication policy, send the private email, and save only the SciX records actually delivered after the email succeeds.

The state commit point is successful email delivery. A Pages validation or deployment failure after that point does not cause the same private email to be sent again, but it can leave that day's public feed undeployed. Persisting feed history is a separate feature and is not part of this migration.

## Content boundary

| Data | Local ranking | Remote LLM | Private email | Public Atom, default |
| --- | --- | --- | --- | --- |
| Zotero abstracts | Yes | No | No | No |
| SciX-only abstract | Yes | No | No | No |
| SciX-only title/authors/link | Yes | No | Yes | No |
| Successfully enriched arXiv content | Yes | Optional | Yes | Yes |
| Native open-source records | Yes | Optional | Yes | Yes |
| Delivery state | Yes | No | No | No |

`output.atom.scix_only_policy: exclude` is the fail-closed default. A SciX-discovered record is publishable only after `content_source` becomes `arxiv`; merely containing an arXiv identifier is insufficient because enrichment can fail. `include` is an explicit metadata-only override and should be used only after confirming redistribution permission with SciX.

SciX-only records are intended for a private personal digest. Confirm multi-recipient or public redistribution with `help@scixplorer.org`. This operational safeguard is not legal advice.

## API rules

- Authentication uses `Authorization: Bearer ...`.
- The adapter prefers `SCIX_API_TOKEN`, then falls back to `ADS_API_TOKEN`.
- The scientific query does not contain a moving date clause. The adapter appends `entdate:[NOW-NDAYS TO *]` from `lookback_days`.
- `max_results` is a hard bound. A larger result count fails instead of silently truncating or bulk-harvesting.
- Debug mode requests at most five rows.
- Rate-limit response headers are logged without logging the token.
- Network and server failures are retried; authentication, contract, and rate-limit failures remain actionable errors.
- The requested response contract includes `identifier`, `entry_date`, and `keyword_schema`. The opt-in live test confirms that `identifier` remains retrievable.

Official references:

- [SciX API](https://scixplorer.org/scixhelp/api-scix/)
- [SciX search syntax](https://scixplorer.org/scixhelp/search-scix/search-syntax/)
- [ADS to SciX transition](https://scixplorer.org/adstoscix/)
- [SciX terms](https://scixplorer.org/scixhelp/policies-scix/terms)
- [SciX AI policy](https://scixplorer.org/help/policies/ai-policy)
- [Public OpenAPI specification](https://github.com/adsabs/adsabs-dev-api/blob/master/openapi/openapi_public.yaml)

## Validation

The normal test suite uses canned responses and never calls SciX. It covers API query construction and bounds, new and legacy source aliases, token fallback, response fields, record links, state compatibility, arXiv enrichment, public-feed exclusion, and the end-to-end remote-processing boundary.

An explicit one-row live contract test is available without sending email or writing state:

```bash
SCIX_LIVE_TEST=1 SCIX_API_TOKEN=... \
  uv run pytest -m live tests/retriever/test_ads_retriever.py
```

Run it manually during the transition or after an announced API change; it is skipped by default and is not required in CI.

## Upstream merge policy

Expected future conflict hotspots remain limited to:

- `protocol.py`: additive paper provenance and privacy fields.
- `executor.py`: post-ranking arXiv enrichment, delivery ordering, and compatibility state.
- `retriever/arxiv_retriever.py`: reusable ID fetch and conversion helpers.
- `retriever/scix_retriever.py`: the fork-owned SciX adapter.
- `config/base.yaml`: additive SciX and Atom schema.
- `.github/workflows/main.yml`: fork-owned schedule, private state cache, and Pages deployment.

The numerical reranking interface remains generic: candidates expose `ranking_text`, while concrete retrievers remain registered adapters. For a future upstream sync, preserve the content boundary and compatibility identifiers above rather than choosing either conflict side wholesale.
