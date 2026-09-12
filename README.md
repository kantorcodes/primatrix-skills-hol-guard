# Primatrix Skills Plugin Marketplace

A [Claude Code plugin marketplace](https://code.claude.com/docs/en/plugin-marketplaces) containing reusable Agent Skills and workflow plugins. The same repository can also be used by Codex through `codex plugin` and by Gemini CLI through individual skill installs.

## What is a Skill?

A Skill is a set of structured instructions, usually defined in a `SKILL.md` file, that teaches an AI coding agent how to perform a specific workflow. A plugin can bundle skills together with slash commands, hooks, agents, and supporting scripts.

## Available Plugins

| Plugin | Version | Contents | Description |
|--------|---------|----------|-------------|
| `agent-recap` | `0.1.0` | 1 skill | Mine local Claude/Codex session history and produce daily or weekly recaps. |
| `beaver` | `3.3.0` | 9 commands, 2 support skills | GitHub-native issue lifecycle and project workflow commands. |
| `exec-remote` | `1.0.0` | 3 skills | Run Python scripts, tests, or benchmarks on remote GPU/TPU clusters via SkyPilot. |
| `gcloud-support-case` | `1.0.0` | 1 skill | Open, list, and inspect Google Cloud support cases (Cloud TPU / GKE / GCS) via the Cloud Support API v2. |
| `gke-tpu` | `2.0.0` | 1 skill | Plan nodepool actions and render GKE TPU Job manifests with explicit context/namespace safety. |
| `hol-guard` | `0.1.0` | 1 skill | Install and operate HOL Guard local runtime security for supported AI coding agents before tool execution. |
| `lint-fix` | `1.0.0` | 1 skill | Check and fix lint issues for changed Python files. |
| `service-integrations` | `1.0.0` | 3 skills | Work with Plane, GitLab, and Outline through credential-safe agent workflows. |
| `session-recorder` | `1.0.0` | 1 skill | Record complete session content into dated work logs. |
| `superpowers` | `6.0.3` | 14 skills, 3 commands, hooks | Core workflow skills: TDD, debugging, brainstorming, review, planning, and collaboration patterns. Based on official Superpowers v6.0.3 with Primatrix RFC workflow customizations. |
| `tpu-perf` | `0.3.0` | 4 skills | Systematic TPU pretraining profile analysis: anatomy, communication, compute, and HBM memory. |
| `xprof-cli` | `1.0.0` | 1 skill | Terminal-first XProf analysis with the official CLI and a portable XSpace PB fallback. |
| `xprof-profiling-analysis` | `2.0.0` | 1 skill | TPU/XLA profiling methodology plus XProf MCP-oriented analysis workflows. |

`tpu-perf` replaces the older `tpu-perf-model` plugin.

## Skill Inventory

| Plugin | Skill | Use when |
|--------|-------|----------|
| `agent-recap` | `agent-recap` | Summarizing recent local agent work into a structured report. |
| `beaver` | `beaver-engine` | Internal engine for Beaver commands. |
| `beaver` | `spec-document-reviewer` | Internal reviewer prompt for Beaver design RFCs. |
| `exec-remote` | `exec-remote` | Running code, tests, or benchmarks on a provisioned remote GPU/TPU cluster. |
| `exec-remote` | `deploy-cluster` | Deploying a SkyPilot-managed TPU cluster on GKE. |
| `exec-remote` | `apply-resource` | Creating, deleting, or listing GKE TPU nodepool resources through xpk. |
| `gcloud-support-case` | `gcloud-support-case` | Opening or checking a Google Cloud support case when a Cloud TPU, GKE, or GCS production job is failing. |
| `gke-tpu` | `gke-tpu` | Planning GKE TPU nodepool actions and rendering batch/interactive TPU Job manifests. |
| `hol-guard` | `hol-guard` | Installing and operating HOL Guard local runtime protection for supported AI coding agents. |
| `lint-fix` | `lint-fix` | Linting or auto-fixing changed Python files with isort, ruff, black, and codespell. |
| `service-integrations` | `plane-api` | Reading or managing Plane projects, work items, cycles, modules, pages, members, and workspace settings through configured MCP tools. |
| `service-integrations` | `gitlab-cli` | Inspecting or managing GitLab projects, merge requests, CI/CD pipelines, jobs, runners, variables, or permissions through glab CLI or the GitLab REST API. |
| `service-integrations` | `outline-api` | Inspecting, creating, updating, publishing, searching, moving, or managing Outline knowledge-base documents through its API or MCP endpoint. |
| `session-recorder` | `session-recorder` | Recording full session history for progress tracking and documentation. |
| `superpowers` | `using-superpowers` | Establishing skill usage rules at the start of a conversation. |
| `superpowers` | `brainstorming` | Exploring requirements before creative implementation work. |
| `superpowers` | `writing-plans` | Writing an implementation plan from a spec or requirements. |
| `superpowers` | `executing-plans` | Executing a written implementation plan with review checkpoints. |
| `superpowers` | `subagent-driven-development` | Splitting independent implementation tasks across subagents. |
| `superpowers` | `dispatching-parallel-agents` | Dispatching independent read/search/review tasks in parallel. |
| `superpowers` | `systematic-debugging` | Diagnosing bugs or unexpected behavior before proposing fixes. |
| `superpowers` | `test-driven-development` | Implementing features or fixes through a red-green-refactor loop. |
| `superpowers` | `verification-before-completion` | Verifying evidence before claiming work is complete or passing. |
| `superpowers` | `requesting-code-review` | Checking work before merge or handoff. |
| `superpowers` | `receiving-code-review` | Handling review feedback rigorously before changing code. |
| `superpowers` | `finishing-a-development-branch` | Deciding how to finish, merge, PR, or clean up completed work. |
| `superpowers` | `using-git-worktrees` | Starting isolated feature work or plan execution. |
| `superpowers` | `writing-skills` | Creating, editing, or verifying skills before deployment. |
| `tpu-perf` | `profile-anatomy` | Reading TPU pretraining profile layouts, xplane.pb, and trace.json.gz. |
| `tpu-perf` | `comm-analysis` | Analyzing TPU communication primitives, axis bandwidth, and compute/comm overlap. |
| `tpu-perf` | `compute-breakdown` | Producing HLO duration breakdowns, layer scopes, non-compute audits, and roofline shortfall reports. |
| `tpu-perf` | `memory-profile` | Analyzing HBM peak occupancy and alive-buffer attribution from profile directories. |
| `xprof-cli` | `xprof-cli` | Inspecting local XProf captures from the terminal with bounded raw-event, HLO, memory, LLO, and fallback PB analysis. |
| `xprof-profiling-analysis` | `xprof-profiling-analysis` | Analyzing TPU/GPU profiles with XProf APIs and offline trace methodology. |

## Slash Commands

| Plugin | Commands |
|--------|----------|
| `beaver` | `/beaver-create`, `/beaver-design`, `/beaver-decompose`, `/beaver-dev`, `/beaver-fix`, `/beaver-focus`, `/beaver-pr`, `/beaver-setup`, `/beaver-tracker` |
| `superpowers` | `/brainstorm`, `/write-plan`, `/execute-plan` |

## Installation

### Claude Code

Add this repository as a marketplace, then install the plugins you need:

```text
/plugin marketplace add primatrix/skills

/plugin install agent-recap@primatrix-skills
/plugin install beaver@primatrix-skills
/plugin install exec-remote@primatrix-skills
/plugin install gcloud-support-case@primatrix-skills
/plugin install gke-tpu@primatrix-skills
/plugin install hol-guard@primatrix-skills
/plugin install lint-fix@primatrix-skills
/plugin install session-recorder@primatrix-skills
/plugin install superpowers@primatrix-skills
/plugin install tpu-perf@primatrix-skills
/plugin install xprof-cli@primatrix-skills
/plugin install xprof-profiling-analysis@primatrix-skills
```

For project-local installs, add `--scope project`:

```text
/plugin install superpowers@primatrix-skills --scope project
/plugin install tpu-perf@primatrix-skills --scope project
```

Verify from a Claude Code session:

```text
/plugin list
```

### Codex

Codex can consume this repository as a plugin marketplace:

```bash
codex plugin marketplace add primatrix/skills --ref main
codex plugin list --marketplace primatrix-skills

codex plugin add agent-recap@primatrix-skills
codex plugin add beaver@primatrix-skills
codex plugin add exec-remote@primatrix-skills
codex plugin add gcloud-support-case@primatrix-skills
codex plugin add gke-tpu@primatrix-skills
codex plugin add hol-guard@primatrix-skills
codex plugin add lint-fix@primatrix-skills
codex plugin add session-recorder@primatrix-skills
codex plugin add superpowers@primatrix-skills
codex plugin add tpu-perf@primatrix-skills
codex plugin add xprof-cli@primatrix-skills
codex plugin add xprof-profiling-analysis@primatrix-skills
```

Refresh an already-added marketplace snapshot before installing newly-added plugins:

```bash
codex plugin marketplace upgrade primatrix-skills
codex plugin list --marketplace primatrix-skills
```

Verify installed plugins:

```bash
codex plugin list
```

### Gemini CLI

Gemini CLI installs individual skill directories. Use the paths from [Skill Inventory](#skill-inventory):

```bash
gemini skills install https://github.com/primatrix/skills.git --path plugins/superpowers/skills/using-superpowers
gemini skills install https://github.com/primatrix/skills.git --path plugins/superpowers/skills/brainstorming
gemini skills install https://github.com/primatrix/skills.git --path plugins/tpu-perf/skills/profile-anatomy
gemini skills install https://github.com/primatrix/skills.git --path plugins/tpu-perf/skills/comm-analysis
gemini skills install https://github.com/primatrix/skills.git --path plugins/exec-remote/skills/exec-remote
gemini skills install https://github.com/primatrix/skills.git --path plugins/hol-guard/skills/hol-guard
gemini skills install https://github.com/primatrix/skills.git --path plugins/xprof-cli/skills/xprof-cli
```

Install into the current workspace instead of user scope:

```bash
gemini skills install https://github.com/primatrix/skills.git --path plugins/tpu-perf/skills/profile-anatomy --scope workspace
```

Verify:

```bash
gemini skills list
```

## Repository Structure

```text
.claude-plugin/
└── marketplace.json              # Marketplace definition and plugin registry
plugins/
└── <plugin-name>/
    ├── .claude-plugin/
    │   └── plugin.json           # Claude/Codex marketplace plugin manifest
    ├── .codex-plugin/            # Optional Codex-specific plugin manifest
    ├── .cursor-plugin/           # Optional Cursor-specific plugin manifest
    ├── commands/                 # Optional slash commands
    ├── agents/                   # Optional agent prompts
    ├── hooks/                    # Optional runtime hooks
    └── skills/
        └── <skill-name>/
            ├── SKILL.md          # Skill frontmatter and instructions
            └── scripts/          # Optional supporting scripts
```

## Local Development

When changing a plugin locally, test it from a local marketplace checkout before publishing.

Codex can install directly from the local repository:

```bash
cd /path/to/skills
codex plugin marketplace add /path/to/skills
codex plugin add superpowers@primatrix-skills
```

If the remote `primatrix-skills` marketplace is already registered, remove it first or use a temporary copy with a different `.claude-plugin/marketplace.json` name.

Claude Code can point at a local marketplace path through the same marketplace flow:

```text
/plugin marketplace add /path/to/skills
/plugin install superpowers@primatrix-skills
```

After editing skills, restart the agent session or reload plugins so the updated `SKILL.md` files are loaded.

## Contributing

To add or update a plugin:

1. Create or update `plugins/<plugin-name>/`.
2. Keep `.claude-plugin/plugin.json` in sync with the plugin name, description, and version.
3. Add skills under `skills/<skill-name>/SKILL.md`.
4. Add commands, agents, hooks, or scripts only when the plugin actually needs them.
5. Register the plugin in `.claude-plugin/marketplace.json`.
6. Update this README's plugin table, skill inventory, and install commands.
7. Submit a pull request.

## License

[Apache License 2.0](LICENSE)
