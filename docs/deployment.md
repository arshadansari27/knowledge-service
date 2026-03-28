# Production Deployment

The knowledge-service runs as part of the **AEGIS stack** on a Docker Swarm cluster managed by Ansible.

**Infrastructure repo:** `Workspace/infrastructure/homelab-gitops`
**Ansible role:** `ansible/roles/aegis/`
**Stack template:** `ansible/roles/aegis/templates/aegis-stack.yml.j2`

---

## Cluster Topology

| Node | Hostname | Role | Hardware |
|------|----------|------|----------|
| 10.20.0.20 | meem | Swarm worker (GPU) | RTX 2070 SUPER 8GB |
| 10.20.0.16 | qaf | Swarm manager | — |
| 10.20.0.15 | lam | Swarm worker | — |
| 10.20.0.229 | pk | Swarm worker | — |
| — | asif | Ollama-2 host | RTX 5060 Ti 16GB |

---

## Knowledge Service Configuration

**Service name:** `knowledge` (internal hostname: `aegis_knowledge`)
**Image:** `arshadansari27/knowledge-service:latest`
**Deployed on:** meem (GPU node)
**External URL:** `https://knowledge.hikmahtech.in` (via Traefik)

### Environment Variables (Production)

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | `postgresql://aegis:<password>@aegis_postgres:5432/knowledge` |
| `LLM_BASE_URL` | `https://litellm.hikmahtech.in` |
| `LLM_API_KEY` | Via `aegis_knowledge_llm_api_key` secret |
| `LLM_EMBED_MODEL` | `nomic-embed-text` |
| `LLM_CHAT_MODEL` | `qwen3:14b` |
| `OXIGRAPH_DATA_DIR` | `/app/data/oxigraph` |
| `FEDERATION_ENABLED` | `true` |
| `FEDERATION_TIMEOUT` | `3.0` |
| `ADMIN_PASSWORD` | Via `aegis_knowledge_admin_password` secret |
| `SECRET_KEY` | Via `aegis_knowledge_secret_key` secret |
| `SPACY_DATA_DIR` | `/app/data/spacy` |

### Resources

| | Limit | Reservation |
|---|-------|-------------|
| Memory | 3G | 512M |

### Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `aegis_knowledge_oxigraph` | `/app/data/oxigraph` | RDF-star triple store (pyoxigraph/RocksDB) |
| `aegis_knowledge_spacy` | `/app/data/spacy` | spaCy Wikidata KB (~1GB, downloaded on first start) |

### Health Check

```
CMD: python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
Interval: 30s | Timeout: 10s | Retries: 5 | Start period: 90s
```

---

## AEGIS Stack Services

The knowledge-service is one of 11 services in the AEGIS stack:

| Service | Node | Port | Purpose |
|---------|------|------|---------|
| `core` | meem | 8080 | AEGIS API server |
| `worker` | meem | — | Background job processor |
| `telegram` | meem | 8081 | Telegram bot |
| **`knowledge`** | **meem** | **8000** | **Knowledge graph service** |
| `postgres` | meem | 5432 (internal), 5433 (host) | PostgreSQL 16 + pgvector |
| `n8n` | meem | 5678 | Workflow automation |
| `redis` | qaf | 6379 | Cache |
| `temporal` | qaf | 7233 | Workflow orchestration |
| `temporal-ui` | lam | 8080 | Temporal dashboard |
| `searxng` | lam | 8080 | Meta search engine |

### Networking

- **aegis_network** — overlay network (attachable), all services communicate here
- **traefik_public** — ingress via Traefik reverse proxy (HTTPS with Let's Encrypt)
- **monitoring** — Prometheus metrics collection

---

## Database

**Image:** `pgvector/pgvector:pg16`
**User:** `aegis`
**Databases:** `aegis` (primary), `knowledge`, `n8n`

The `knowledge` database is auto-created at deploy time via `init-extra-dbs.sh.j2` (mounted as an init script). No manual DB creation needed.

Data directory: `/opt/aegis/postgres/data` on meem.

Resource limits: 4G memory limit, 1G reservation.

---

## LLM Access via LiteLLM

Knowledge-service connects to LLMs through a **LiteLLM proxy** (`https://litellm.hikmahtech.in`), which routes to:

| Model | Backend | Node |
|-------|---------|------|
| `nomic-embed-text` | Ollama | meem (RTX 2070 SUPER) |
| `qwen3:14b` | Ollama-2 | asif (RTX 5060 Ti) |

LiteLLM also provides fallback chains — if `qwen3:14b` fails, it falls back to `claude-haiku`.

---

## Reverse Proxy (Traefik)

All external access goes through Traefik v2.11:

| Subdomain | Service | Port |
|-----------|---------|------|
| `knowledge.hikmahtech.in` | knowledge | 8000 |
| `aegis-api.hikmahtech.in` | core | 8080 |
| `aegis-bot.hikmahtech.in` | telegram | 8081 |
| `n8n.hikmahtech.in` | n8n | 5678 |
| `temporal.hikmahtech.in` | temporal-ui | 8080 |
| `litellm.hikmahtech.in` | litellm | 4000 |

Management services are IP-whitelisted to internal networks (192.168.1.0/24, 10.20.0.0/24) and VPN (192.168.255.0/24).

---

## Data Directories on meem

```
/opt/aegis/
├── postgres/data/              # PostgreSQL pgdata
├── knowledge-oxigraph/         # RDF-star store (mounted as Docker volume)
├── knowledge-spacy/            # spaCy Wikidata KB (~1GB, downloaded on first start)
├── config/                     # Shared config files
├── n8n-data/                   # n8n workflow data
├── personalities/              # AEGIS personality definitions
├── reasoning-rules/            # AEGIS reasoning rules
└── init-scripts/
    └── init-extra-dbs.sh       # Creates knowledge + n8n databases
```

---

## Deploying

From the `homelab-gitops/ansible/` directory:

```bash
source .env && ansible-playbook -i inventory/hosts.yml playbooks/deploy-aegis.yml
```

This:
1. Creates directories on target nodes
2. Syncs config files and templates
3. Deploys the Docker Swarm stack
4. Waits 45s for stabilisation
5. Prints service status and access URLs

### Updating Knowledge Service Only

Push to `main` on GitHub → CI builds and pushes `arshadansari27/knowledge-service:latest` → then re-deploy the stack:

```bash
# On meem, or via ansible
docker service update --image arshadansari27/knowledge-service:latest aegis_knowledge
```

---

## Secrets

Secrets are injected via environment variables at deploy time. Required for knowledge-service:

| Secret | Ansible Variable |
|--------|-----------------|
| PostgreSQL password | `aegis_postgres_password` |
| LiteLLM API key | `aegis_knowledge_llm_api_key` |

All secrets are sourced from `.env` in the ansible directory (not committed).
